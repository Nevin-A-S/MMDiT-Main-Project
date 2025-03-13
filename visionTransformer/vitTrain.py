import os
import gc
import cv2
import wandb
import torch
import random
import numpy as np
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from typing import Dict, Tuple, List, Optional, Union, Any
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
import timm
from timm.data.auto_augment import rand_augment_transform

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# Advanced Data Augmentation
# ------------------------------
class RandAugment:
    def __init__(self, n=2, m=9):
        self.transform = rand_augment_transform(
            config_str=f'rand-m{m}-n{n}',
            hparams={'translate_const': 100, 'img_mean': (124, 116, 104)}
        )
    
    def __call__(self, img):
        return self.transform(img)

# ------------------------------
# MixUp and CutMix Advanced Augmentations
# ------------------------------
def mixup_data(x, y, alpha=0.4, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """Applies CutMix: https://arxiv.org/abs/1905.04899"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    # Get bbox for CutMix
    w, h = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------------
# Advanced Loss Functions
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, input, target):
        logp = -self.ce(input, target)
        p = torch.exp(logp)
        loss = -((1 - p) ** self.gamma) * logp
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# ------------------------------
# Exponential Moving Average for model weights
# ------------------------------
class ModelEMA:
    """ Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = {}
        self.decay = decay
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema[name] = param.data.clone().detach().to(device=self.device)
        
        for name, buffer in model.named_buffers():
            self.ema[name] = buffer.data.clone().detach().to(device=self.device)
    
    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema[name] = self.ema[name] * self.decay + param.data.to(device=self.device) * (1.0 - self.decay)
            
            for name, buffer in model.named_buffers():
                self.ema[name] = self.ema[name] * self.decay + buffer.data.to(device=self.device) * (1.0 - self.decay)
    
    def apply(self, model):
        # Store original parameters
        stored_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                stored_params[name] = param.data.clone()
                param.data.copy_(self.ema[name])
        
        for name, buffer in model.named_buffers():
            stored_params[name] = buffer.data.clone()
            buffer.data.copy_(self.ema[name])
        
        return stored_params
    
    def restore(self, model, stored_params):
        # Restore original parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in stored_params:
                param.data.copy_(stored_params[name])
        
        for name, buffer in model.named_buffers():
            if name in stored_params:
                buffer.data.copy_(stored_params[name])

# ------------------------------
# SAM Optimizer (Sharpness-Aware Minimization)
# ------------------------------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        
        # The closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)
        
        # First forward-backward pass
        loss = closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        closure()
        self.second_step()
        
        return loss
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach()).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

# ------------------------------
# Custom Dataset with Balanced Sampler
# ------------------------------
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform=None, cache_size: int = 1000):
        self.df = pd.read_csv(csv_path, sep=',')
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_size = cache_size
        self.cache: Dict[int, Tuple] = {}  # Image cache

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        if idx in self.cache:
            return self.cache[idx]

        row = self.df.iloc[idx]
        img_path = self.root_dir / Path(row['image'].replace("\\", "/"))

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            caption = row['caption']

            if self.transform:
                image_transformed = self.transform(image)
            else:
                image_transformed = image

            result = (image_transformed, caption)
            
            if len(self.cache) < self.cache_size:
                self.cache[idx] = result

            return result

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy sample (ensure consistent tensor shape)
            return torch.zeros((3, 224, 224)), "error loading image"

    @staticmethod
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(captions)

# ------------------------------
# Label Encoder for handling captions
# ------------------------------
class LabelEncoder:
    def __init__(self):
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_classes = 0
        self.is_fitted = False
    
    def fit(self, labels: List[str]) -> 'LabelEncoder':
        """Fit the encoder on a list of labels."""
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        self.is_fitted = True
        return self
    
    def transform(self, labels: List[str]) -> List[int]:
        """Transform labels to indices."""
        if not self.is_fitted:
            raise ValueError("LabelEncoder is not fitted yet. Call fit() first.")
        
        # Handle unknown labels gracefully
        return [self.label_to_idx.get(label, -1) for label in labels]
    
    def fit_transform(self, labels: List[str]) -> List[int]:
        """Fit the encoder and transform labels to indices."""
        self.fit(labels)
        return self.transform(labels)
    
    def inverse_transform(self, indices: List[int]) -> List[str]:
        """Transform indices back to labels."""
        if not self.is_fitted:
            raise ValueError("LabelEncoder is not fitted yet. Call fit() first.")
        
        return [self.idx_to_label.get(idx, "unknown") for idx in indices]
    
    def save(self, path: str) -> None:
        """Save the encoder to a JSON file."""
        data = {
            "label_to_idx": self.label_to_idx,
            "idx_to_label": {int(k): v for k, v in self.idx_to_label.items()},
            "num_classes": self.num_classes,
            "is_fitted": self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> 'LabelEncoder':
        """Load the encoder from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.label_to_idx = data["label_to_idx"]
        self.idx_to_label = {int(k): v for k, v in data["idx_to_label"].items()}
        self.num_classes = data["num_classes"]
        self.is_fitted = data["is_fitted"]
        
        return self

# ------------------------------
# Classification Dataset with Label Encoder
# ------------------------------
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, label_encoder=None, label_extractor=None):
        self.original_dataset = original_dataset
        
        if label_extractor is None:
            def label_extractor(caption):
                return caption
        
        self.label_extractor = label_extractor
        
        # Extract all labels
        all_labels = []
        for i in range(len(original_dataset)):
            _, caption = original_dataset[i]
            label = label_extractor(caption)
            all_labels.append(label)
        
        # Create or use label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder().fit(all_labels)
        else:
            self.label_encoder = label_encoder
            
        self.num_classes = self.label_encoder.num_classes
    
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx):
        image, caption = self.original_dataset[idx]
        label = self.label_extractor(caption)
        label_idx = self.label_encoder.transform([label])[0]
        
        # Handle unknown labels
        if label_idx == -1:
            print(f"Warning: Unknown label '{label}' encountered. Using fallback class 0.")
            label_idx = 0
            
        return image, label_idx
    
    def get_label_encoder(self):
        return self.label_encoder

# ------------------------------
# Balanced Batch Sampler for handling class imbalance
# ------------------------------
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, labels=None, batch_size=16):
        if labels is None:
            self.labels = self._get_labels(dataset)
        else:
            self.labels = labels
            
        self.indices = self._build_indices()
        self.batch_size = batch_size
        self.n_classes = len(self.indices)
        self.samples_per_class = batch_size // self.n_classes if self.n_classes < batch_size else 1
        self.count = 0
        
    def _get_labels(self, dataset):
        if isinstance(dataset, Subset):
            return [dataset.dataset[i][1] for i in dataset.indices]
        return [sample[1] for sample in dataset]
        
    def _build_indices(self):
        labels = np.array(self.labels)
        indices = {}
        for label in np.unique(labels):
            indices[label] = np.where(labels == label)[0]
        return indices
    
    def __iter__(self):
        count = 0
        while count + self.batch_size <= len(self.labels):
            classes = list(self.indices.keys())
            batch_indices = []
            
            for class_idx in classes:
                if self.samples_per_class <= len(self.indices[class_idx]):
                    batch_indices.extend(np.random.choice(
                        self.indices[class_idx], self.samples_per_class, replace=False))
                else:
                    batch_indices.extend(np.random.choice(
                        self.indices[class_idx], self.samples_per_class, replace=True))
            
            # If we didn't fill the batch with balanced samples, add random ones
            if len(batch_indices) < self.batch_size:
                extra_needed = self.batch_size - len(batch_indices)
                batch_indices.extend(np.random.choice(
                    np.arange(len(self.labels)), extra_needed, replace=False))
                
            # Shuffle the batch indices
            np.random.shuffle(batch_indices)
            batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
            count += self.batch_size
    
    def __len__(self):
        return len(self.labels) // self.batch_size

# ------------------------------
# Advanced LR Scheduler with Warmup
# ------------------------------
class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-7, 
                 eta_min=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * 
                    (1 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]

# ------------------------------
# Test Time Augmentation
# ------------------------------
class TestTimeAugmentation:
    def __init__(self, model, transforms_list, device):
        self.model = model
        self.transforms_list = transforms_list
        self.device = device
        
    def __call__(self, images):
        self.model.eval()
        batch_size = images.size(0)
        probs = torch.zeros((batch_size, self.model.config.num_labels)).to(self.device)
        
        with torch.no_grad():
            # Original prediction
            outputs = self.model(images).logits
            probs += torch.softmax(outputs, dim=1)
            
            # Augmented predictions
            for transform in self.transforms_list:
                aug_images = transform(images)
                outputs = self.model(aug_images).logits
                probs += torch.softmax(outputs, dim=1)
                
        # Average predictions
        probs /= (len(self.transforms_list) + 1)
        return probs

# ------------------------------
# ViT Fine-Tuner with Advanced Methods
# ------------------------------
class ViTFineTuner:
    def __init__(
        self, 
        model_name="google/vit-base-patch16-224-in21k", 
        num_classes=None,
        learning_rate=2e-5, 
        mixed_precision=True,
        gradient_accumulation_steps=1,
        checkpoint_dir="checkpoints",
        use_wandb=False,
        weight_decay=0.01,
        project_name="vit-finetuning",
        # Advanced training options
        use_mixup=False,
        use_cutmix=False,
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
        use_focal_loss=False,
        use_label_smoothing=False,
        label_smoothing=0.1,
        use_ema=False,
        ema_decay=0.9999,
        use_sam_optimizer=False,
        sam_rho=0.05,
        warmup_epochs=5,
        gradient_clip_val=1.0,
        # Data augmentation options
        use_randaugment=False,
        randaugment_n=2,
        randaugment_m=9,
        # Regularization options
        dropout_rate=0.1,
        balanced_sampling=False,
        # Test-time augmentation
        use_tta=False
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.weight_decay = weight_decay
        self.project_name = project_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Advanced training options
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.use_focal_loss = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_sam_optimizer = use_sam_optimizer
        self.sam_rho = sam_rho
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_val = gradient_clip_val
        
        # Data augmentation options
        self.use_randaugment = use_randaugment
        self.randaugment_n = randaugment_n
        self.randaugment_m = randaugment_m
        
        # Regularization options
        self.dropout_rate = dropout_rate
        self.balanced_sampling = balanced_sampling
        
        # Test-time augmentation
        self.use_tta = use_tta
        self.tta_transforms = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler() if mixed_precision else None
        self.ema_model = None
        
        if self.use_wandb:
            wandb.init(project=project_name)
    
    def setup_model(self, num_classes=None):
        if num_classes is not None:
            self.num_classes = num_classes
        
        if self.num_classes is None:
            raise ValueError("Number of classes must be specified")
            
        print(f"Loading pre-trained model: {self.model_name}")
        model_config = ViTConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            hidden_dropout_prob=self.dropout_rate,
            attention_probs_dropout_prob=self.dropout_rate
        )
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        self.model = self.model.to(self.device)
        
        # Initialize EMA model if needed
        if self.use_ema:
            self.ema_model = ModelEMA(self.model, decay=self.ema_decay, device=self.device)
        
        # Initialize TTA transforms if needed
        if self.use_tta:
            self.tta_transforms = [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ]
        
        # ------------------------------
        # Advanced: Layer-Wise Learning Rate Decay (LLRD)
        # ------------------------------
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        # Set different learning rates for different parts of the model
        # Use a lower learning rate when using SAM optimizer to improve stability
        base_lr = self.learning_rate
        if self.use_sam_optimizer:
            base_lr = self.learning_rate * 0.5  # Reduce learning rate for SAM
            print(f"Using reduced learning rate for SAM optimizer: {base_lr}")
        
        optimizer_grouped_parameters = [
            # Attention & hidden layers
            {'params': [p for n, p in param_optimizer if 'encoder.layer' in n and not any(nd in n for nd in no_decay)],
             'lr': base_lr, 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if 'encoder.layer' in n and any(nd in n for nd in no_decay)],
             'lr': base_lr, 'weight_decay': 0.0},
            
            # Embedding layers
            {'params': [p for n, p in param_optimizer if 'embeddings' in n and not any(nd in n for nd in no_decay)],
             'lr': base_lr * 0.1, 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if 'embeddings' in n and any(nd in n for nd in no_decay)],
             'lr': base_lr * 0.1, 'weight_decay': 0.0},
            
            # Classification head (highest learning rate)
            {'params': [p for n, p in param_optimizer if 'classifier' in n and not any(nd in n for nd in no_decay)],
             'lr': base_lr * 10, 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if 'classifier' in n and any(nd in n for nd in no_decay)],
             'lr': base_lr * 10, 'weight_decay': 0.0},
            
            # Everything else
            {'params': [p for n, p in param_optimizer if not any(x in n for x in ['encoder.layer', 'embeddings', 'classifier']) and not any(nd in n for nd in no_decay)],
             'lr': base_lr, 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if not any(x in n for x in ['encoder.layer', 'embeddings', 'classifier']) and any(nd in n for nd in no_decay)],
             'lr': base_lr, 'weight_decay': 0.0}
        ]
        
        # Use SAM optimizer or standard AdamW
        if self.use_sam_optimizer:
            print("Using SAM optimizer with AdamW base")
            self.optimizer = SAM(
                optimizer_grouped_parameters,
                base_optimizer=optim.AdamW,
                rho=self.sam_rho,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # ------------------------------
        # Advanced: Loss Function Setup
        # ------------------------------
        if self.use_focal_loss:
            self.criterion = FocalLoss(gamma=2, reduction="mean")
        elif self.use_label_smoothing:
            self.criterion = LabelSmoothingLoss(classes=self.num_classes, smoothing=self.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=10, 
        save_interval=1,
        eval_interval=1,
        resume_from=None,
        label_encoder=None
    ):
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model first.")
        
        # Initialize scheduler
        scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=epochs,
            warmup_start_lr=1e-7,
            eta_min=1e-7
        )
        
        best_val_acc = 0.0
        early_stopping_patience = 10
        early_stopping_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            print(f"Resuming training from checkpoint: {resume_from}")
            checkpoint = self.load_checkpoint(resume_from)
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            
            # Resume scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.4f}")
        
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.model.train()
            train_loss = 0.0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for batch_idx, (images, labels) in progress_bar:
                # Convert string labels to numeric if needed
                if isinstance(labels[0], str):
                    if label_encoder is None:
                        raise ValueError("Label encoder is required for string labels")
                    numeric_labels = torch.tensor(label_encoder.transform(labels))
                else:
                    numeric_labels = labels

                images = images.to(self.device)
                numeric_labels = numeric_labels.to(self.device)
                
                # Randomly apply MixUp or CutMix
                do_mixup = self.use_mixup and random.random() < 0.5
                do_cutmix = self.use_cutmix and random.random() < 0.5 and not do_mixup
                
                if do_mixup:
                    images, targets_a, targets_b, lam = mixup_data(images, numeric_labels, self.mixup_alpha, device=self.device)
                elif do_cutmix:
                    images, targets_a, targets_b, lam = cutmix_data(images, numeric_labels, self.cutmix_alpha, device=self.device)
                
                # Disable SAM optimizer for the first few epochs to stabilize training
                use_sam_this_batch = self.use_sam_optimizer and epoch >= 2
                
                if use_sam_this_batch:
                    # For SAM with mixed precision, we need a completely custom approach
                    if self.mixed_precision:
                        # Zero gradients to start
                        self.optimizer.zero_grad()
                        
                        # First forward pass
                        with autocast(device_type='cuda'):
                            outputs = self.model(images).logits
                            if do_mixup or do_cutmix:
                                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                            else:
                                loss = self.criterion(outputs, numeric_labels)
                            loss = loss / self.gradient_accumulation_steps
                        
                        # Check for NaN loss and skip this batch if found
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping batch.")
                            continue
                        
                        # First backward pass
                        self.scaler.scale(loss).backward()
                        
                        # First step of SAM (perturb weights)
                        self.scaler.unscale_(self.optimizer)
                        
                        # Check for NaN gradients
                        has_nan_grad = False
                        for param in self.model.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    has_nan_grad = True
                                    break
                        
                        if has_nan_grad:
                            print(f"Warning: NaN or Inf gradient detected in batch {batch_idx}. Skipping SAM steps.")
                            # Skip SAM steps but still update with regular optimizer
                            self.optimizer.base_optimizer.step()
                            self.optimizer.zero_grad()
                            self.scaler.update()
                            continue
                        
                        # Apply SAM first step
                        self.optimizer.first_step(zero_grad=True)
                        
                        # Second forward pass with no autocast to avoid mixed precision issues
                        outputs = self.model(images).logits
                        if do_mixup or do_cutmix:
                            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = self.criterion(outputs, numeric_labels)
                        loss = loss / self.gradient_accumulation_steps
                        
                        # Check for NaN loss again
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f"Warning: NaN or Inf loss detected in second forward pass. Skipping batch.")
                            continue
                        
                        # Second backward pass
                        loss.backward()
                        
                        # Check for NaN gradients again
                        has_nan_grad = False
                        for param in self.model.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    has_nan_grad = True
                                    break
                        
                        if has_nan_grad:
                            print(f"Warning: NaN or Inf gradient detected in second backward pass. Skipping second SAM step.")
                            continue
                        
                        # Second step of SAM (update weights)
                        self.optimizer.second_step(zero_grad=True)
                        
                        # Update scaler state
                        self.scaler.update()
                    else:
                        # Standard SAM optimization without mixed precision
                        def closure():
                            self.optimizer.zero_grad()
                            outputs = self.model(images).logits
                            if do_mixup or do_cutmix:
                                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                            else:
                                loss = self.criterion(outputs, numeric_labels)
                            loss = loss / self.gradient_accumulation_steps
                            
                            # Check for NaN loss
                            if torch.isnan(loss).any() or torch.isinf(loss).any():
                                return loss.detach()  # Return detached loss to avoid NaN propagation
                            
                            loss.backward()
                            return loss
                        
                        loss = self.optimizer.step(closure)
                        
                        # Check if loss is NaN and skip if it is
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping batch.")
                            continue
                else:
                    # Standard optimization
                    self.optimizer.zero_grad()
                    if self.mixed_precision:
                        with autocast(device_type='cuda'):
                            outputs = self.model(images).logits
                            if do_mixup or do_cutmix:
                                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                            else:
                                loss = self.criterion(outputs, numeric_labels)
                            loss = loss / self.gradient_accumulation_steps
                        
                        # Check for NaN loss
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping batch.")
                            continue
                        
                        self.scaler.scale(loss).backward()
                        
                        # Apply gradient clipping
                        if self.gradient_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                    else:
                        outputs = self.model(images).logits
                        if do_mixup or do_cutmix:
                            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = self.criterion(outputs, numeric_labels)
                        loss = loss / self.gradient_accumulation_steps
                        
                        # Check for NaN loss
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping batch.")
                            continue
                        
                        loss.backward()
                        
                        # Apply gradient clipping
                        if self.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.optimizer.step()
                
                # Update EMA model if enabled
                if self.use_ema and (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.ema_model.update(self.model)
                
                # Only add non-NaN losses to the running average
                if not torch.isnan(loss).any() and not torch.isinf(loss).any():
                    train_loss += loss.item() * self.gradient_accumulation_steps
                
                # For training metrics, we only consider the non-mixup case for simplicity
                if not do_mixup and not do_cutmix:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(numeric_labels.cpu().numpy())
                
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Compute average loss, handling the case where all batches might have been skipped
            if len(train_loader) > 0:
                train_loss /= len(train_loader)
            else:
                train_loss = float('nan')
            
            # Only compute metrics if we have predictions
            if len(all_preds) > 0:
                train_acc = accuracy_score(all_labels, all_preds)
                train_f1 = f1_score(all_labels, all_preds, average='weighted')
                print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_f1': train_f1,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            
            # Validation phase
            if val_loader is not None and (epoch + 1) % eval_interval == 0:
                # Use EMA model for validation if available
                if self.use_ema:
                    stored_params = self.ema_model.apply(self.model)
                
                val_loss, val_acc, val_f1 = self.evaluate(val_loader, use_tta=True, label_encoder=label_encoder)
                print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
                
                # Restore original model parameters
                if self.use_ema:
                    self.ema_model.restore(self.model, stored_params)
                
                if self.use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1
                    })
                
                # Early stopping and model saving logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stopping_counter = 0
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_dir, "best_model.pth"),
                        epoch=epoch,
                        best_val_acc=best_val_acc,
                        early_stopping_counter=early_stopping_counter,
                        scheduler=scheduler,
                        label_encoder=label_encoder
                    )
                    print(f"New best model saved with validation accuracy: {val_acc:.4f}")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {early_stopping_counter} epochs without improvement")
                        break
            
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    early_stopping_counter=early_stopping_counter,
                    scheduler=scheduler,
                    label_encoder=label_encoder
                )
            
            scheduler.step()  # Update learning rate scheduler
            gc.collect()
        
        self.save_checkpoint(
            os.path.join(self.checkpoint_dir, "final_model.pth"),
            epoch=epochs-1,
            best_val_acc=best_val_acc,
            early_stopping_counter=early_stopping_counter,
            scheduler=scheduler,
            label_encoder=label_encoder
        )
        
        if self.use_wandb:
            wandb.finish()
        
        return best_val_acc
    
    def evaluate(self, val_loader, use_tta=False, label_encoder=None):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Setup TTA if enabled
        tta = None
        if use_tta or self.use_tta:
            tta = TestTimeAugmentation(self.model, self.tta_transforms, self.device)
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                if isinstance(labels[0], str):
                    if label_encoder is None:
                        raise ValueError("Label encoder is required for string labels")
                    numeric_labels = torch.tensor(label_encoder.transform(labels))
                else:
                    numeric_labels = labels
                
                images = images.to(self.device)
                numeric_labels = numeric_labels.to(self.device)
                
                if self.mixed_precision:
                    with autocast(device_type='cuda'):
                        if tta:
                            probs = tta(images)
                            outputs = torch.log(probs)  # Convert to logits
                        else:
                            outputs = self.model(images).logits
                        
                        loss = self.criterion(outputs, numeric_labels)
                else:
                    if tta:
                        probs = tta(images)
                        outputs = torch.log(probs)  # Convert to logits
                    else:
                        outputs = self.model(images).logits
                    
                    loss = self.criterion(outputs, numeric_labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(numeric_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        return val_loss, val_acc, val_f1
    
    def save_checkpoint(self, path, epoch=0, best_val_acc=0.0, early_stopping_counter=0, scheduler=None, label_encoder=None):
        """Save a comprehensive checkpoint for resuming training"""
        checkpoint = {
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'early_stopping_counter': early_stopping_counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add EMA state if available
        if self.use_ema:
            stored_params = self.ema_model.apply(self.model)
            checkpoint['ema_state_dict'] = {k: v.clone() for k, v in self.ema_model.ema.items()}
            self.ema_model.restore(self.model, stored_params)
        
        # Save the checkpoint
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # Save label encoder if provided
        if label_encoder is not None:
            encoder_path = os.path.splitext(path)[0] + "_label_encoder.json"
            label_encoder.save(encoder_path)
            print(f"Label encoder saved to {encoder_path}")
    
    def load_checkpoint(self, path):
        """Load a checkpoint for resuming training"""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load EMA state if available
        if self.use_ema and 'ema_state_dict' in checkpoint:
            for name, param in checkpoint['ema_state_dict'].items():
                self.ema_model.ema[name] = param.to(self.device)
        
        return checkpoint
    
    def save_model(self, path):
        """Legacy method for backward compatibility"""
        self.save_checkpoint(path)
    
    def load_model(self, path):
        """Legacy method for backward compatibility"""
        return self.load_checkpoint(path)
    
    def predict(self, test_loader, use_tta=False, label_encoder=None):
        self.model.eval()
        all_preds = []
        all_probs = []
        
        # Setup TTA if enabled
        tta = None
        if use_tta or self.use_tta:
            tta = TestTimeAugmentation(self.model, self.tta_transforms, self.device)
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Predicting"):
                images = images.to(self.device)
                
                if self.mixed_precision:
                    with autocast(device_type='cuda'):
                        if tta:
                            probs = tta(images)
                        else:
                            outputs = self.model(images).logits
                            probs = torch.softmax(outputs, dim=1)
                else:
                    if tta:
                        probs = tta(images)
                    else:
                        outputs = self.model(images).logits
                        probs = torch.softmax(outputs, dim=1)
                
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs.cpu().numpy())
        
        # Convert numeric predictions to original labels if encoder is provided
        if label_encoder is not None:
            original_labels = label_encoder.inverse_transform(all_preds)
            return np.array(all_preds), np.array(all_probs), original_labels
        
        return np.array(all_preds), np.array(all_probs)

# ------------------------------
# Helper Functions for Data Processing
# ------------------------------
def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    
    # Set seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices

def load_or_create_label_encoder(dataset, label_extractor=None, encoder_path=None):
    """Load an existing label encoder or create a new one from the dataset"""
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading label encoder from {encoder_path}")
        label_encoder = LabelEncoder().load(encoder_path)
    else:
        print("Creating new label encoder from dataset")
        if label_extractor is None:
            def label_extractor(caption):
                return caption
        
        # Extract all labels
        all_labels = []
        for i in range(len(dataset)):
            _, caption = dataset[i]
            label = label_extractor(caption)
            all_labels.append(label)
        
        # Create and fit label encoder
        label_encoder = LabelEncoder().fit(all_labels)
        
        # Save the encoder if path is provided
        if encoder_path:
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            label_encoder.save(encoder_path)
            print(f"Label encoder saved to {encoder_path}")
    
    return label_encoder

def convert_caption_dataset_to_classification(dataset, label_encoder=None, label_extractor=None):
    """Convert a caption dataset to a classification dataset using a label encoder"""
    return ClassificationDataset(dataset, label_encoder, label_extractor)

# ------------------------------
# Example usage
# ------------------------------
def main():
    # Set global random seed for reproducibility
    set_seed(42)
    print("Hello")
    
    # Get image processor for normalization values
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    del processor
    
    # Define advanced augmentations
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandAugment(n=2, m=9),  # Apply RandAugment after basic augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std, inplace=True),
    ])
    
    # Simpler transform for validation
    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std, inplace=True),
    ])
    
    # Create dataset
    full_dataset = ImageCaptionDataset(
        csv_path="visionTransformer/new_synthetic_dataset.csv",
        root_dir="",
        transform=None,  # We'll apply transforms later
        cache_size=2000  # Increased cache size
    )
    
    # Define label extractor function based on your caption format
    def label_extractor(caption):
        # Extract the disease type from the caption
        # Example: "Non Demented Alzheimers" -> "Non"
        return caption.split()[0] if ' ' in caption else caption
    
    # Setup paths for label encoder
    encoder_path = "checkpoints/label_encoder.json"
    
    # Load or create label encoder
    label_encoder = load_or_create_label_encoder(
        full_dataset,
        label_extractor=label_extractor,
        encoder_path=encoder_path
    )
    
    # Convert to classification dataset
    classification_dataset = convert_caption_dataset_to_classification(
        full_dataset, 
        label_encoder=label_encoder,
        label_extractor=label_extractor
    )
    
    # Print class information
    print(f"Number of classes: {classification_dataset.num_classes}")
    print(f"Class mapping: {label_encoder.label_to_idx}")
    
    # Split datasets
    train_indices, val_indices = create_train_val_split(
        classification_dataset, 
        val_ratio=0.15,  # Slightly smaller validation set
        seed=42
    )
    
    # Create train and validation datasets with different transforms
    train_dataset = Subset(classification_dataset, train_indices)
    val_dataset = Subset(classification_dataset, val_indices)
    
    # Custom collate function to apply transforms
    def collate_fn_with_transform(batch, transform):
        images, labels = zip(*[(transform(item[0]), item[1]) for item in batch])
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels
    
    # Create dataloaders with balanced sampling for training
    train_labels = [classification_dataset[i][1] for i in train_indices]
    
    batch_size = 32
    num_workers = 4
    
    # Use balanced sampler for training
    train_sampler = BalancedBatchSampler(
        train_dataset, 
        labels=train_labels,
        batch_size=batch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler if classification_dataset.num_classes < batch_size else None,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn_with_transform(b, train_transform),
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn_with_transform(b, val_transform),
        persistent_workers=True
    )

    # Check for existing checkpoints to resume from
    resume_from = None
    checkpoint_dir = "checkpoints"
    if os.path.exists(os.path.join(checkpoint_dir, "best_model.pth")):
        resume_from = os.path.join(checkpoint_dir, "best_model.pth")
        print(f"Found existing checkpoint: {resume_from}")

    # Initialize the ViT fine-tuner with advanced techniques
    # Use a more stable configuration to avoid NaN issues
    fine_tuner = ViTFineTuner(
        model_name="google/vit-base-patch16-224-in21k",
        num_classes=classification_dataset.num_classes,
        learning_rate=1e-5,  # Lower learning rate for stability
        mixed_precision=True,
        gradient_accumulation_steps=1,
        checkpoint_dir=checkpoint_dir,
        use_wandb=False,  # Set to True if using wandb
        weight_decay=0.01,
        project_name="vit-finetuning",
        # Advanced training options
        use_mixup=False,  # Disable mixup initially for stability
        use_cutmix=False,  # Disable cutmix initially for stability
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
        use_focal_loss=False,
        use_label_smoothing=True,
        label_smoothing=0.1,
        use_ema=True,
        ema_decay=0.9999,
        use_sam_optimizer=False,  # Disable SAM initially for stability
        sam_rho=0.05,
        warmup_epochs=5,
        gradient_clip_val=1.0,
        # Data augmentation options
        use_randaugment=True,
        randaugment_n=2,
        randaugment_m=9,
        # Regularization options
        dropout_rate=0.1,
        balanced_sampling=True,
        # Test-time augmentation
        use_tta=True
    )

    # Setup the model
    fine_tuner.setup_model(num_classes=classification_dataset.num_classes)

    # Train the model with resume capability
    best_val_acc = fine_tuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        save_interval=5,
        eval_interval=1,
        resume_from=resume_from,
        label_encoder=label_encoder
    )

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

    # Load the best model for final evaluation
    fine_tuner.load_checkpoint(os.path.join(fine_tuner.checkpoint_dir, "best_model.pth"))

    # Final evaluation with test-time augmentation
    val_loss, val_acc, val_f1 = fine_tuner.evaluate(
        val_loader, 
        use_tta=True,
        label_encoder=label_encoder
    )
    print(f"Final evaluation results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    # Make predictions and convert back to original labels
    predictions, probabilities, original_labels = fine_tuner.predict(
        val_loader, 
        use_tta=True,
        label_encoder=label_encoder
    )
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(min(10, len(original_labels))):
        print(f"Predicted: {original_labels[i]}")

    # Close wandb run if it was used
    if fine_tuner.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()