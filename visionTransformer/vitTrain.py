import os
import gc
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
from torch.utils.data import DataLoader,Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor

class ImageCaptionDataset(Dataset):
    def init(self, csv_path: str, root_dir: str, transform=None, cache_size: int = 1000):
        """
        Args:
            csv_path: Path to the CSV file with annotations.
            root_dir: Base directory for image paths in CSV.
            transform: Optional transform to be applied on images.
            cache_size: Number of images to cache in memory.
        """
        self.df = pd.read_csv(csv_path, sep=',')
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_size = cache_size
        self.cache: Dict[int, Tuple] = {}  # Image cache

    def len(self) -> int:
        return len(self.df)

    def getitem(self, idx: int) -> Tuple:
        # Check cache first
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
                image = self.transform(image)

            # Cache the result if cache isn't full
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (image, caption)

            return image, caption

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy sample in case of error
            return torch.zeros((3, 256, 256)), "error loading image"

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that:
         - Stacks image tensors.
         - Leaves captions as a list of strings.
        """
        images, captions = zip(*batch)
        # Stack images (assuming they are all tensors of the same shape)
        images = torch.stack(images, dim=0)
        # Return images as a tensor and captions as a list
        return images, list(captions)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]
del processor

transform = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[image_mean, image_mean, image_mean], std=[image_std, image_std, image_std], inplace=True),
])

def setup_dataloader(csv_location,root_dir,img_size,batch_size,num_workers):
    """Setup dataset and dataloader with optimized transforms"""

    dataset = ImageCaptionDataset(
        csv_path=csv_location,
        root_dir=root_dir,
        transform=transform,
        cache_size=1000  # Cache 1000 images in memory
    )

    dataloader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return dataloader

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
        project_name="vit-finetuning"
    ):
        """
        Initialize the ViT fine-tuning module.
        
        Args:
            model_name: Hugging Face model name/path
            num_classes: Number of classes for classification
            learning_rate: Learning rate for optimization
            mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps for gradient accumulation
            checkpoint_dir: Directory to save model checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            weight_decay: Weight decay for regularization
            project_name: Project name for W&B
        """
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
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize model, optimizer, and criterion later when num_classes is known
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler() if mixed_precision else None
        
        if self.use_wandb:
            wandb.init(project=project_name)
    
    def setup_model(self, num_classes=None):
        """
        Set up the ViT model, optimizer, and loss function.
        
        Args:
            num_classes: Number of classes for classification (if not specified at init)
        """
        if num_classes is not None:
            self.num_classes = num_classes
        
        if self.num_classes is None:
            raise ValueError("Number of classes must be specified")
            
        # Load pre-trained model
        print(f"Loading pre-trained model: {self.model_name}")
        
        # Use model_config to only keep necessary layers and reduce VRAM usage
        model_config = ViTConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            config=model_config,
            ignore_mismatched_sizes=True  # In case the classifier head doesn't match
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer with weight decay
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Set up cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()
    
    def train(
        self, 
        train_loader, 
        val_loader=None, 
        epochs=10, 
        save_interval=1,
        eval_interval=1
    ):
        """
        Train the ViT model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            epochs: Number of epochs to train
            save_interval: Save model checkpoint every N epochs
            eval_interval: Evaluate model every N epochs
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model first.")
        
        # Set up learning rate scheduler
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Use tqdm for progress bar
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for batch_idx, (images, labels) in progress_bar:
                # For your dataset, adapt this based on what it returns
                # Convert string labels to numeric if needed
                if isinstance(labels[0], str):
                    # Create a mapping for string labels if needed
                    # This is a simple example - adapt as necessary for your labels
                    label_mapping = {label: idx for idx, label in enumerate(set(labels))}
                    numeric_labels = torch.tensor([label_mapping[label] for label in labels])
                else:
                    numeric_labels = labels
                
                images = images.to(self.device)
                numeric_labels = numeric_labels.to(self.device)
                
                # Mixed precision training
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images).logits
                        loss = self.criterion(outputs, numeric_labels)
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Scale gradients and accumulate
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    outputs = self.model(images).logits
                    loss = self.criterion(outputs, numeric_labels)
                    loss = loss / self.gradient_accumulation_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # Track metrics
                train_loss += loss.item() * self.gradient_accumulation_steps
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(numeric_labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                
                # Clear cache to reduce VRAM usage
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
            # Validation phase
            if val_loader is not None and (epoch + 1) % eval_interval == 0:
                val_loss, val_acc, val_f1 = self.evaluate(val_loader)
                
                print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1
                    })
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(os.path.join(self.checkpoint_dir, "best_model.pth"))
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
            # Update learning rate
            scheduler.step()
            
            # Explicitly collect garbage to free memory
            gc.collect()
        
        # Save final model
        self.save_model(os.path.join(self.checkpoint_dir, "final_model.pth"))
        
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(self, val_loader):
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            val_loss: Validation loss
            val_acc: Validation accuracy
            val_f1: Validation F1 score
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                # Adapt based on your dataset
                if isinstance(labels[0], str):
                    # Create a mapping for string labels if needed
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
        """
        Save the model checkpoint.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model checkpoint.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
    
    def predict(self, test_loader):
        """
        Make predictions on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            predictions: Model predictions
            probabilities: Prediction probabilities
        """
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

# Function to create train/validation split
def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: PyTorch Dataset
        val_ratio: Validation set ratio
        seed: Random seed for reproducibility
        
    Returns:
        train_indices, val_indices: Indices for train and validation sets
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    return train_indices, val_indices

# Memory-efficient subset sampler
class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices
        
    def __iter__(self):
        return (i for i in self.indices)
        
    def __len__(self):
        return len(self.indices)

# Example usage
def main():
    """
    Example of how to use the ViTFineTuner with the provided dataset.
    """
    # Setup data loaders
    from torch.utils.data import DataLoader, Subset
    
    # Dataset parameters
    csv_location = "dataset/Flickr/captions.csv"
    root_dir = "dataset/Flickr/images"
    img_size = 224  # ViT default size
    batch_size = 16  # Adjust based on VRAM
    num_workers = 4  # Adjust based on CPU cores
    
    # Create full dataset - Replace with your own dataset
    # Note: For our example, we need to modify the existing dataset or create a new one for classification
    # This is just a mockup assuming we adapt the ImageCaptionDataset for classification
    full_dataset = ImageCaptionDataset(
        csv_path=csv_location,
        root_dir=root_dir,
        transform=transform,
        cache_size=1000
    )
    
    # Create train/val split
    train_indices, val_indices = create_train_val_split(full_dataset, val_ratio=0.2)
    
    # Create train and validation dataloaders
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size,
        sampler=SubsetSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=batch_size,
        sampler=SubsetSampler(val_indices),
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Count number of unique classes in your dataset
    # Adapt this according to how your labels are stored
    # For this example, assuming labels are the captions, we'd need to modify this
    # num_classes = len(set([label for _, label in full_dataset]))
    num_classes = 10  # Replace with actual number of classes
    
    # Initialize fine-tuner
    fine_tuner = ViTFineTuner(
        model_name="google/vit-base-patch16-224-in21k",
        learning_rate=2e-5,
        mixed_precision=True,
        gradient_accumulation_steps=2,  # Reduces VRAM usage
        checkpoint_dir="vit_checkpoints",
        use_wandb=True,  # Set to True to enable W&B logging
        weight_decay=0.01
    )
    
    # Setup model with number of classes
    fine_tuner.setup_model(num_classes=num_classes)
    
    # Train model
    fine_tuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        save_interval=1,
        eval_interval=1
    )
    
    # Optional: Make predictions on test data
    # test_predictions, test_probs = fine_tuner.predict(test_loader)

if __name__ == "__main__":
    main()

# Additional helper functions for adapting caption dataset to classification task

def convert_caption_dataset_to_classification(dataset, label_extractor=None):
    """
    Convert a caption dataset to a classification dataset.
    
    Args:
        dataset: Original ImageCaptionDataset
        label_extractor: Function to extract label from caption
        
    Returns:
        ClassificationDataset: Dataset for classification tasks
    """
    # Example label extractor - you would need to customize this
    # based on how your captions relate to class labels
    if label_extractor is None:
        def label_extractor(caption):
            # Example: extract first word of caption as class
            return caption.split()[0]
    
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, label_extractor):
            self.original_dataset = original_dataset
            self.label_extractor = label_extractor
            
            # Create label mapping
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