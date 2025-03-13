import os
import gc
import cv2
import wandb
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from typing import Dict, Tuple
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor

# ------------------------------
# Advanced Data Augmentation: MixUp
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

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------------
# Advanced Loss: Focal Loss
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

# ------------------------------
# Warm-Up Scheduler (wraps a base scheduler)
# ------------------------------
class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, base_scheduler, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: start at 0 and go to base LR
            return [base_lr * float(self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return self.base_scheduler.get_lr()
    
    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            self.last_epoch += 1
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            self.base_scheduler.step(epoch)
            self.last_epoch = self.base_scheduler.last_epoch

# ------------------------------
# Custom Dataset
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
            return torch.zeros((3, 256, 256)), "error loading image"

    @staticmethod
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(captions)

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
        use_mixup=False,       # Advanced: flag to enable MixUp
        mixup_alpha=0.4,
        use_focal_loss=False,  # Advanced: flag to enable Focal Loss
        warmup_steps=5         # Advanced: warmup steps for LR scheduler
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
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_focal_loss = use_focal_loss
        self.warmup_steps = warmup_steps
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler() if mixed_precision else None
        
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
            num_labels=self.num_classes
        )
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            config=model_config,
            ignore_mismatched_sizes=True
        )
        self.model = self.model.to(self.device)
        
        # ------------------------------
        # Advanced: Layer-Wise Learning Rate Decay (LLRD)
        # ------------------------------
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # Set a base LR multiplier for each layer (example: deeper layers get lower LR)
        optimizer_grouped_parameters = []
        for n, p in param_optimizer:
            lr = self.learning_rate
            if "encoder.layer" in n:
                # Extract layer index and decay: lower layers get lr * decay_factor^(L - i)
                try:
                    layer_id = int(n.split("encoder.layer.")[1].split(".")[0])
                    total_layers = 12  # Adjust based on model depth
                    decay_factor = 0.95
                    lr = self.learning_rate * (decay_factor ** (total_layers - layer_id))
                except Exception as e:
                    pass
            if any(nd in n for nd in no_decay):
                optimizer_grouped_parameters.append({'params': p, 'lr': lr, 'weight_decay': 0.0})
            else:
                optimizer_grouped_parameters.append({'params': p, 'lr': lr, 'weight_decay': self.weight_decay})
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # ------------------------------
        # Advanced: Loss Function (Focal Loss or standard CrossEntropy)
        # ------------------------------
        if self.use_focal_loss:
            self.criterion = FocalLoss(gamma=2, reduction="mean")
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=10, 
        save_interval=1,
        eval_interval=1
    ):
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model first.")
        
        # Base scheduler (cosine annealing) and warmup scheduler wrapper
        base_scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - self.warmup_steps)
        scheduler = WarmupScheduler(self.optimizer, base_scheduler, warmup_steps=self.warmup_steps)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.model.train()
            train_loss = 0.0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for batch_idx, (images, labels) in progress_bar:
                # Convert string labels to numeric if needed
                if isinstance(labels[0], str):
                    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
                    numeric_labels = torch.tensor([label_mapping[label] for label in labels])
                else:
                    numeric_labels = labels

                images = images.to(self.device)
                numeric_labels = numeric_labels.to(self.device)
                
                # Apply MixUp if enabled
                if self.use_mixup:
                    images, targets_a, targets_b, lam = mixup_data(images, numeric_labels, self.mixup_alpha, device=self.device)
                
                # Mixed precision training
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images).logits
                        if self.use_mixup:
                            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                        else:
                            loss = self.criterion(outputs, numeric_labels)
                        loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    outputs = self.model(images).logits
                    if self.use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = self.criterion(outputs, numeric_labels)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                train_loss += loss.item() * self.gradient_accumulation_steps
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(numeric_labels.cpu().numpy())
                
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            train_loss /= len(train_loader)
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
            
            if val_loader is not None and (epoch + 1) % eval_interval == 0:
                val_loss, val_acc, val_f1 = self.evaluate(val_loader)
                print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
                if self.use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1
                    })
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(os.path.join(self.checkpoint_dir, "best_model.pth"))
            
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
            scheduler.step()  # Update learning rate scheduler
            gc.collect()
        
        self.save_model(os.path.join(self.checkpoint_dir, "final_model.pth"))
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                if isinstance(labels[0], str):
                    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
                    numeric_labels = torch.tensor([label_mapping[label] for label in labels])
                else:
                    numeric_labels = labels
                
                images = images.to(self.device)
                numeric_labels = numeric_labels.to(self.device)
                
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images).logits
                        loss = self.criterion(outputs, numeric_labels)
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
        return val_loss, val_acc, val_f1
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
    
    def predict(self, test_loader):
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Predicting"):
                images = images.to(self.device)
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images).logits
                else:
                    outputs = self.model(images).logits
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_preds), np.array(all_probs)

# ------------------------------
# Helper Functions for Data Splitting
# ------------------------------
def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    random.seed(seed)
    random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices

# ------------------------------
# Example usage
# ------------------------------
def main():
    from torch.utils.data import Subset

    # Using a ViT image processor to get normalization statistics
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    del processor

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std, inplace=True),
        # You can also add additional augmentations here (e.g., RandAugment)
    ])

    batch_size = 32
    num_workers = 4
    
    full_dataset = ImageCaptionDataset(
        csv_path="visionTransformer/new_synthetic_dataset.csv",
        root_dir="",
        transform=transform,
        cache_size=1000 
    )
    
    train_indices, val_indices = create_train_val_split(full_dataset, val_ratio=0.1)
    
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    num_classes = 10  # Replace with actual number of classes
    
    fine_tuner = ViTFineTuner(
        model_name="google/vit-base-patch16-224-in21k",
        learning_rate=3e-4,
        mixed_precision=True,
        gradient_accumulation_steps=2,
        checkpoint_dir="vit_checkpoints",
        use_wandb=True,
        weight_decay=0.01,
        use_mixup=True,        # Enable MixUp
        mixup_alpha=0.4,
        use_focal_loss=True,   # Enable Focal Loss (or set False for standard CrossEntropy)
        warmup_steps=5
    )
    
    fine_tuner.setup_model(num_classes=num_classes)
    fine_tuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        save_interval=10,
        eval_interval=1
    )

if __name__ == "__main__":
    main()

# Additional helper function for converting caption datasets to classification if needed
def convert_caption_dataset_to_classification(dataset, label_extractor=None):
    if label_extractor is None:
        def label_extractor(caption):
            return caption.split()[0]
    
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, label_extractor):
            self.original_dataset = original_dataset
            self.label_extractor = label_extractor
            all_labels = []
            for i in range(len(original_dataset)):
                _, caption = original_dataset[i]
                label = label_extractor(caption)
                all_labels.append(label)
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            self.num_classes = len(unique_labels)
        
        def __len__(self):
            return len(self.original_dataset)
            
        def __getitem__(self, idx):
            image, caption = self.original_dataset[idx]
            label = self.label_extractor(caption)
            label_idx = self.label_to_idx[label]
            return image, label_idx
            
    return ClassificationDataset(dataset, label_extractor)
