import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import logging
import time
import datetime
from tqdm.auto import tqdm
import wandb
import json
from PIL import Image

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PyTorch_ResNet50_Classifier")

# Config parameters (hyperparameters)
CONFIG = {
    "csv_path": "visionTransformer/new_synthetic_dataset.csv",  # Path to your CSV file
    "img_height": 224,  # ResNet50 default input height
    "img_width": 224,   # ResNet50 default input width
    "batch_size": 32,   # Batch size hyperparameter
    "epochs": 50,       # Number of epochs hyperparameter
    "fine_tune_epochs": 30,  # Number of fine-tuning epochs
    "learning_rate": 0.0001,
    "fine_tune_learning_rate": 0.00001,
    "validation_split": 0.2,
    "test_split": 0.1,
    "num_classes": 10,  # As per your specification
    "dropout_rate": 0.5,
    "early_stopping_patience": 5,
    "weight_decay": 1e-4,
    "model_checkpoint_dir": "resnet-model_checkpoints",
    "final_model_path": "final_model",
    "wandb_project": "resnet50_classifier_pytorch",  # WandB project name
    "wandb_entity": None,  # Your WandB username (None will use default)
    "wandb_tags": ["resnet50", "pytorch", "image-classification"],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def setup_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    logger.info("Initializing Weights & Biases")
    
    # Initialize WandB with config
    wandb.init(
        project=CONFIG["wandb_project"],
        entity=CONFIG["wandb_entity"],
        tags=CONFIG["wandb_tags"],
        config=CONFIG
    )
    
    # Log the model architecture as a graph
    wandb.run.name = f"resnet50_bs{CONFIG['batch_size']}_e{CONFIG['epochs']}_lr{CONFIG['learning_rate']}"
    
    logger.info(f"WandB initialized with run name: {wandb.run.name}")
    logger.info(f"Using device: {CONFIG['device']}")

def setup_directories():
    """Create necessary directories for model checkpoints and logs."""
    os.makedirs(CONFIG["model_checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["final_model_path"], exist_ok=True)
    logger.info(f"Created directories for model checkpoints and final model")

class ImageDataset(Dataset):
    """Custom Dataset for loading images from dataframe."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        # Create class to index mapping
        self.classes = sorted(dataframe['caption'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        logger.info(f"Dataset created with {len(self.dataframe)} samples and {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image']
        caption = self.dataframe.iloc[idx]['caption']
        label_idx = self.class_to_idx[caption]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label_idx
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the label
            placeholder = torch.zeros((3, CONFIG["img_height"], CONFIG["img_width"]))
            return placeholder, label_idx

def load_data():
    """Load the CSV file containing image paths and captions."""
    logger.info(f"Loading dataset from {CONFIG['csv_path']}")
    try:
        df = pd.read_csv(CONFIG["csv_path"])
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def preprocess_data(df):
    """Preprocess the dataframe and split into train, validation, and test sets."""
    logger.info("Starting data preprocessing")
    
    # Check if the expected columns exist
    required_columns = ['image', 'caption']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column {col} not found in the dataset")
            raise ValueError(f"Required column {col} not found in the dataset")
    
    # Check for missing image files
    missing_files = [path for path in df['image'] if not os.path.exists(path)]
    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing image files. First few: {missing_files[:5]}")
    
    # Split the data into train+val and test sets
    train_val_df, test_df = train_test_split(
        df, 
        test_size=CONFIG["test_split"], 
        stratify=df['caption'], 
        random_state=42
    )
    
    # Split the train+val into train and validation sets
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=CONFIG["validation_split"],
        stratify=train_val_df['caption'], 
        random_state=42
    )
    
    logger.info(f"Data split complete. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Log class distribution
    logger.info("Class distribution in training set:")
    class_dist = train_df['caption'].value_counts().to_dict()
    for cls, count in class_dist.items():
        logger.info(f"Class {cls}: {count} samples")
    
    return train_df, val_df, test_df

def create_data_loaders(train_df, val_df, test_df):
    """Create data loaders for training, validation, and testing."""
    logger.info("Setting up data transformations and loaders")
    
    # Define transformations for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_height"], CONFIG["img_width"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for validation and testing (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_height"], CONFIG["img_width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=val_test_transform)
    test_dataset = ImageDataset(test_df, transform=val_test_transform)
    
    # Get class mapping
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True if CONFIG["device"].type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if CONFIG["device"].type == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True if CONFIG["device"].type == "cuda" else False
    )
    
    logger.info(f"Created data loaders. Train: {len(train_loader)} batches, Validation: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, idx_to_class

def build_model():
    """Build a ResNet50 model fine-tuned for the classification task."""
    logger.info("Building ResNet50 model")
    
    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=CONFIG["dropout_rate"]),
        nn.Linear(num_ftrs, CONFIG["num_classes"])
    )
    
    # Move model to device
    model = model.to(CONFIG["device"])
    
    logger.info(f"Model built and moved to {CONFIG['device']}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Setup progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / total, 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    # Log metrics to wandb
    wandb.log({
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "epoch": epoch
    })
    
    logger.info(f"Train Epoch: {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, epoch, phase="val"):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Epoch {epoch+1}/{CONFIG['epochs']} [Validation]" if phase == "val" else "Testing"
    pbar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / total, 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    if phase == "val":
        # Log metrics to wandb
        wandb.log({
            "val_loss": epoch_loss,
            "val_acc": epoch_acc,
            "epoch": epoch
        })
        logger.info(f"Validation Epoch: {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
    else:
        wandb.log({
            "test_loss": epoch_loss,
            "test_acc": epoch_acc
        })
        logger.info(f"Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None):
    """Train the model."""
    logger.info("Starting model training")
    
    best_val_acc = 0.0
    early_stopping_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(CONFIG["epochs"]):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(CONFIG["model_checkpoint_dir"], f"best_model_epoch_{epoch+1}_acc_{val_acc:.2f}.pt"))
            
            logger.info(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        else:
            early_stopping_counter += 1
            
        # Check for early stopping
        if early_stopping_counter >= CONFIG["early_stopping_patience"]:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save the final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(CONFIG["final_model_path"], "final_model.pt"))
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, train_losses, train_accs, val_losses, val_accs

def fine_tune_model(model, train_loader, val_loader, criterion):
    """Fine-tune the model by unfreezing some of the base layers."""
    logger.info("Starting fine-tuning phase")
    
    # Unfreeze the last few layers of the base model
    # For ResNet50, we'll unfreeze the last layer4 block
    for name, child in model.named_children():
        if name == 'layer4':
            for param in child.parameters():
                param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Fine-tuning with {trainable_params} trainable parameters out of {total_params} total parameters")
    
    # Set up optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["fine_tune_learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Set up scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True
    )
    
    # Fine-tune for a few epochs
    model, ft_train_losses, ft_train_accs, ft_val_losses, ft_val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler
    )
    
    return model, ft_train_losses, ft_train_accs, ft_val_losses, ft_val_accs

def evaluate_model(model, test_loader, criterion, idx_to_class):
    """Evaluate the trained model on the test set."""
    logger.info("Evaluating model on test set")
    
    # Test the model
    test_loss, test_acc = validate(model, test_loader, criterion, 0, phase="test")
    
    # Get predictions and true labels for detailed metrics
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert numeric labels to class names
    pred_classes = [idx_to_class[idx] for idx in all_preds]
    true_classes = [idx_to_class[idx] for idx in all_labels]
    
    # Calculate classification report
    report = classification_report(all_labels, all_preds, target_names=list(idx_to_class.values()))
    logger.info(f"Classification Report:\n{report}")
    
    # Create and log confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(idx_to_class.values()), 
                yticklabels=list(idx_to_class.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix locally
    cm_path = os.path.join(log_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    
    # Log confusion matrix to wandb
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})
    
    # Save classification report to wandb
    wandb.log({"classification_report": wandb.Table(
        columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
        data=[[class_name, precision, recall, f1, support] 
              for class_name, (precision, recall, f1, support) in 
              zip(list(idx_to_class.values()), 
                  zip(*[list(row.values())[:-1] for row in classification_report(
                      all_labels, all_preds, target_names=list(idx_to_class.values()), 
                      output_dict=True).values()][:-3]))]
    )})
    
    return test_loss, test_acc, report, cm

def plot_training_history(train_losses, train_accs, val_losses, val_accs, 
                          ft_train_losses=None, ft_train_accs=None, 
                          ft_val_losses=None, ft_val_accs=None):
    """Plot and save the training history."""
    logger.info("Plotting training history")
    
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    
    if ft_train_accs and ft_val_accs:
        # Add fine-tuning results
        plt.plot(range(len(train_accs), len(train_accs) + len(ft_train_accs)), 
                 ft_train_accs, label='Fine-tune Train')
        plt.plot(range(len(val_accs), len(val_accs) + len(ft_val_accs)), 
                 ft_val_accs, label='Fine-tune Validation')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    
    if ft_train_losses and ft_val_losses:
        # Add fine-tuning results
        plt.plot(range(len(train_losses), len(train_losses) + len(ft_train_losses)), 
                 ft_train_losses, label='Fine-tune Train')
        plt.plot(range(len(val_losses), len(val_losses) + len(ft_val_losses)), 
                 ft_val_losses, label='Fine-tune Validation')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot locally
    history_path = os.path.join(log_dir, 'training_history.png')
    plt.savefig(history_path)
    
    # Log to wandb
    wandb.log({"training_history": wandb.Image(history_path)})
    
    logger.info("Training history plotted and saved")

def save_model_artifacts(model, idx_to_class):
    """Save model artifacts and information."""
    logger.info("Saving model artifacts")
    
    # Save class index mapping
    with open(os.path.join(CONFIG["final_model_path"], "class_mapping.json"), "w") as f:
        json.dump(idx_to_class, f)
    
    # Save model architecture summary
    summary = str(model)
    with open(os.path.join(CONFIG["final_model_path"], "model_summary.txt"), "w") as f:
        f.write(summary)
    
    # Save configuration
    with open(os.path.join(CONFIG["final_model_path"], "config.json"), "w") as f:
        # Convert torch.device to string for JSON serialization
        config_copy = CONFIG.copy()
        config_copy["device"] = str(config_copy["device"])
        json.dump(config_copy, f, indent=4)
    
    # Log model to wandb
    wandb.save(os.path.join(CONFIG["final_model_path"], "final_model.pt"))
    
    logger.info("Model artifacts saved successfully")

def main():
    """Main function to orchestrate the training pipeline."""
    try:
        logger.info("Starting PyTorch ResNet50 classification training pipeline")
        
        # Set up necessary directories
        setup_directories()
        
        # Setup wandb
        setup_wandb()
        
        # Load and preprocess data
        df = load_data()
        train_df, val_df, test_df = preprocess_data(df)
        
        # Create data loaders
        train_loader, val_loader, test_loader, idx_to_class = create_data_loaders(train_df, val_df, test_df)
        
        # Build the model
        model = build_model()
        
        # Log model graph to wandb
        sample_input = torch.randn(1, 3, CONFIG["img_height"], CONFIG["img_width"]).to(CONFIG["device"])
        wandb.watch(model, log="all", log_freq=10)
        
        # Set up loss function, optimizer, and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True
        )
        
        # Train the model
        model, train_losses, train_accs, val_losses, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        
        # Fine-tune the model
        model, ft_train_losses, ft_train_accs, ft_val_losses, ft_val_accs = fine_tune_model(
            model, train_loader, val_loader, criterion
        )
        
        # Evaluate the model
        test_loss, test_acc, report, cm = evaluate_model(model, test_loader, criterion, idx_to_class)
        
        # Plot training history
        plot_training_history(
            train_losses, train_accs, val_losses, val_accs,
            ft_train_losses, ft_train_accs, ft_val_losses, ft_val_accs
        )
        
        # Save model artifacts
        save_model_artifacts(model, idx_to_class)
        
        # Finish wandb run
        wandb.finish()
        
        logger.info(f"Training pipeline completed successfully with test accuracy: {test_acc:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main()