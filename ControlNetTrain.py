import argparse
import json
import os
import cv2
from glob import glob
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import lpips
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import vgg16
from tqdm.auto import tqdm
from lightning.fabric import Fabric

from opendit.diffusion import create_diffusion
from opendit.models.mmdit import MMDiT_models
from opendit.models.mmdit_control_net import MMdit_ControlNet

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable TF32 and cuda optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cudnn.deterministic = False

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features"""
    def __init__(self, layers=[3, 8, 15, 22], weights=[0.1, 0.2, 0.4, 0.8]):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.layers = layers
        self.weights = weights
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, y):
        # Normalize to VGG input range
        x = (x + 1) * 0.5  # Convert from [-1, 1] to [0, 1]
        y = (y + 1) * 0.5
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        x_features = []
        y_features = []
        
        # Extract VGG features
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                x_features.append(x)
                y_features.append(y)
        
        # Compute L1 loss between features
        loss = 0
        for i, (x_feat, y_feat) in enumerate(zip(x_features, y_features)):
            loss += self.weights[i] * F.l1_loss(x_feat, y_feat)
            
        return loss

class ControlNetLoss(nn.Module):
    """Combined loss function for ControlNet training"""
    def __init__(self, 
                 perceptual_weight=0.1,
                 control_weight=1.0,
                 lpips_weight=0.2,
                 edge_weight=0.5,
                 device='cuda'):
        super().__init__()
        self.perceptual = PerceptualLoss().to(device)
        self.lpips = lpips.LPIPS(net='vgg').to(device).eval()
        self.control_weight = control_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_weight = lpips_weight
        self.edge_weight = edge_weight
        
    def forward(self, diffusion_loss, pred, target, control_signal=None):
        """
        Args:
            diffusion_loss: Base diffusion loss
            pred: Model's prediction
            target: Ground truth
            control_signal: Control signal (edge map)
        """
        # Base diffusion loss (with increased weight for control)
        loss = self.control_weight * diffusion_loss
        
        # Add perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual(pred, target)
            loss = loss + self.perceptual_weight * p_loss
            
        # Add LPIPS loss for better perceptual quality
        if self.lpips_weight > 0:
            # LPIPS expects input in range [-1, 1]
            lpips_loss = self.lpips(pred, target).mean()
            loss = loss + self.lpips_weight * lpips_loss
            
        # Add edge-guided loss to emphasize control
        if control_signal is not None and self.edge_weight > 0:
            # Ensure control signal is properly sized for gradient calculation
            if control_signal.shape[2:] != target.shape[2:]:
                control_signal = F.interpolate(control_signal, size=target.shape[2:], mode='bilinear')
            
            # Calculate gradient of prediction and target
            pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            
            target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            
            # Resize control signal for gradient dimensions
            cs_x = control_signal[:, :, :, :-1]
            cs_y = control_signal[:, :, :-1, :]
            
            # Calculate edge-guided loss
            edge_loss_x = F.mse_loss(pred_grad_x * cs_x, target_grad_x * cs_x)
            edge_loss_y = F.mse_loss(pred_grad_y * cs_y, target_grad_y * cs_y)
            
            edge_loss = (edge_loss_x + edge_loss_y) * 0.5
            loss = loss + self.edge_weight * edge_loss
            
        return loss

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform=None, cache_size: int = 1000):
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
            
            image_np = np.array(image)
            
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_rgb = np.stack([edges, edges, edges], axis=2)
            
            edges_pil = Image.fromarray(edges_rgb)

            if self.transform:
                image_transformed = self.transform(image)
                edges_transformed = self.transform(edges_pil)
            else:
                image_transformed = image
                edges_transformed = edges_pil

            result = (image_transformed, caption, edges_transformed)
            
            if len(self.cache) < self.cache_size:
                self.cache[idx] = result

            return result

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return appropriate tensors and ensure 3 items in tuple
            return torch.zeros((3, 256, 256)), "error loading image", torch.zeros((3, 256, 256))

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Create multiple average meters for different loss components
class LossTracker:
    def __init__(self):
        self.total_loss = AverageMeter()
        self.diffusion_loss = AverageMeter()
        self.perceptual_loss = AverageMeter()
        self.lpips_loss = AverageMeter()
        self.edge_loss = AverageMeter()
    
    def update(self, loss_dict, batch_size=1):
        for key, value in loss_dict.items():
            if hasattr(self, key):
                getattr(self, key).update(value, batch_size)
    
    def reset(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, AverageMeter):
                attr.reset()

    def get_loss_dict(self):
        result = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, AverageMeter):
                result[attr_name] = attr.avg
        return result

@torch.compile  
def training_step(model, x, t, y, diffusion, edges, control_loss=None):
    """Compiled training step with improved loss functions"""
    model_kwargs = dict(c=y, edges=edges)
    
    # Get base diffusion loss
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    base_loss = loss_dict["loss"].mean()
    
    if control_loss is not None:
        # For obtaining the model prediction, we'll need to:
        # 1. Get noise prediction from the model
        # 2. Convert it to the x_0 prediction
        
        # Step 1: Get model output directly
        model_output = model(x, t, **model_kwargs)
        
        # Step 2: Use diffusion process to get x_0 prediction
        # This depends on your diffusion implementation, but generally:
        predicted_x0 = diffusion.predict_start_from_noise(x, t, model_output)
        
        # Compute the combined loss
        combined_loss = control_loss(base_loss, predicted_x0, x, edges)
        
        # Return both the combined loss and components for logging
        loss_components = {
            "total_loss": combined_loss.item(),
            "diffusion_loss": base_loss.item()
        }
        
        return combined_loss, loss_components
    
    return base_loss, {"total_loss": base_loss.item()}

def load_checkpoint(checkpoint_path: str, model, ema, optimizer, fabric):
    """Load checkpoint and return the starting epoch and global step"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = fabric.load(checkpoint_path)
    
    # Load model and EMA states
    model.load_state_dict(checkpoint['model'])
    ema.load_state_dict(checkpoint['ema'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Move optimizer states to GPU if needed
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    
    return checkpoint['epoch'], checkpoint['global_step']

def main(args):
    """Trains or resumes training of an MMDiT model with optimized performance."""
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    
    if args.resume_from:
        experiment_dir = str(Path(args.resume_from).parent)
        writer = SummaryWriter(f"{experiment_dir}/tensorboard", purge_step=0)
    else:
        experiment_dir = setup_experiment_dir(args)
        writer = SummaryWriter(f"{experiment_dir}/tensorboard")

    # Initialize Fabric for distributed training 
    fabric = Fabric(
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        precision="bf16-mixed",
    )
    fabric.launch()
    
    device = torch.device('cuda')

    # Setup models and optimizer
    model, ema, vae, optimizer = setup_models(args, fabric)
    diffusion = create_diffusion(timestep_respacing="")
    
    # Initialize the improved loss function
    control_loss = ControlNetLoss(
        perceptual_weight=0.2,  # Increased for stronger perceptual signal
        control_weight=1.5,     # Higher weight on control contribution
        lpips_weight=0.1,       # Moderate LPIPS for visual quality
        edge_weight=0.8,        # Strong edge guidance
        device=device
    ).to(device)
    
    # If resuming, ensure control_loss is moved to correct device
    control_loss = fabric.setup_module(control_loss)

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(args.resume_from, model, ema, optimizer, fabric)
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")

    # Setup dataset and dataloader
    train_loader = setup_dataloader(args, fabric)

    # Training loop
    loss_tracker = LossTracker()
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == "fp16" else None

    model.train()
    ema.eval()
    
    # Calculate total steps for weight annealing
    total_steps = args.epochs * len(train_loader)
    args.total_steps = total_steps

    print(f"Training for {args.epochs} epochs starting from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Beginning epoch {epoch}...")
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for x, y, edges in pbar:
                try:
                    # Dynamic weight adjustment based on training progress
                    progress = min(1.0, global_step / (total_steps * 0.8))
                    
                    # Increase control weight over time
                    control_loss.control_weight = 1.0 + progress * 1.0  # 1.0 to 2.0
                    
                    # Decrease auxiliary losses over time
                    control_loss.perceptual_weight = max(0.05, 0.2 * (1.0 - progress * 0.7))
                    control_loss.lpips_weight = max(0.02, 0.1 * (1.0 - progress * 0.8))
                    control_loss.edge_weight = max(0.2, 0.8 * (1.0 - progress * 0.5))
                    
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision == "fp16"):
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                        edges = vae.encode(edges).latent_dist.sample().mul_(0.18215)

                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            loss, loss_components = training_step(model, x, t, y, diffusion, edges, control_loss)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss, loss_components = training_step(model, x, t, y, diffusion, edges, control_loss)
                        fabric.backward(loss)
                        # Gradient clipping - helps stabilize training with complex loss functions
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, model, fabric)

                    # Update loss tracking
                    loss_tracker.total_loss.update(loss_components["total_loss"])
                    if "diffusion_loss" in loss_components:
                        loss_tracker.diffusion_loss.update(loss_components["diffusion_loss"])
                    
                    global_step += 1

                    # Update progress bar with additional metrics
                    pbar_dict = {
                        "loss": f"{loss_tracker.total_loss.avg:.4f}",
                        "step": global_step,
                        "ctrl_w": f"{control_loss.control_weight:.2f}"
                    }
                    pbar.set_postfix(pbar_dict)

                    if global_step % args.log_every == 0:
                        # Log all loss components
                        writer.add_scalar("loss/total", loss_tracker.total_loss.avg, global_step)
                        writer.add_scalar("loss/diffusion", loss_tracker.diffusion_loss.avg, global_step)
                        
                        # Log control weights
                        writer.add_scalar("weights/control", control_loss.control_weight, global_step)
                        writer.add_scalar("weights/perceptual", control_loss.perceptual_weight, global_step)
                        writer.add_scalar("weights/lpips", control_loss.lpips_weight, global_step)
                        writer.add_scalar("weights/edge", control_loss.edge_weight, global_step)
                        
                        loss_tracker.reset()

                    if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                        # Save captions for reference
                        with open(f"{experiment_dir}/{global_step}_Caption.txt", "w") as f:
                            f.write(f"Caption at global step : {global_step} \n {y}")
                        
                        # Save current loss weights
                        with open(f"{experiment_dir}/{global_step}_LossWeights.txt", "w") as f:
                            f.write(f"Control Weight: {control_loss.control_weight:.4f}\n")
                            f.write(f"Perceptual Weight: {control_loss.perceptual_weight:.4f}\n")
                            f.write(f"LPIPS Weight: {control_loss.lpips_weight:.4f}\n")
                            f.write(f"Edge Weight: {control_loss.edge_weight:.4f}\n")
                        
                        save_checkpoint(model, ema, optimizer, epoch, global_step, experiment_dir)

                except Exception as e:
                    print(f"Error at global step : {global_step}")
                    print(e)
                    f = open(f"{experiment_dir}/{global_step}_Error.txt", "w")
                    f.write(f"Error at global step : {global_step} \n {e}")
                    f.close()

    print("Training finished!")


def setup_experiment_dir(args):
    """Setup experiment directory and save config"""
    os.makedirs(args.outputs, exist_ok=True)
    experiment_index = len(glob(f"{args.outputs}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.outputs}/{experiment_index:03d}-{model_string_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    return experiment_dir

def setup_models(args, fabric):
    """Initialize and setup all models and optimizer"""
    device = torch.device('cuda')
    
    # Create VAE encoder
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.requires_grad_(False)  # Ensure VAE is frozen
    
    # Setup model
    input_size = args.image_size // 8
    model_config = {
        "input_size": input_size,
        "num_classes": args.num_classes,
        "clip_text_encoder": args.text_encoder,
        "t5_text_encoder": args.t5_text_encoder,
    }

    model = MMdit_ControlNet(args.model,
                             model_config,
                             args.pretrained_path,
                             fabric).to(device)
    
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create EMA model
    ema = MMdit_ControlNet(args.model,
                           model_config,
                           args.pretrained_path,
                           fabric).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)

    # Setup optimizer with gradient clipping and learning rate warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,  # Added small weight decay for regularization
        eps=1e-8,
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup with fabric
    model, optimizer = fabric.setup(model, optimizer)
    ema = fabric.setup_module(ema)
    vae = fabric.setup_module(vae)
    
    return model, ema, vae, optimizer

def setup_dataloader(args, fabric):
    """Setup dataset and dataloader with optimized transforms"""
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    csv = "dataset/MRI_DATASET/balanced_dataset.csv"
    dataset = ImageCaptionDataset(
        csv_path=csv,
        root_dir="",
        transform=transform,
        cache_size=1000  # Cache 1000 images in memory
    )
    print('Dataset Loaded from ', csv)
    # small_subset = Subset(dataset, indices=list(range(500)))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return fabric.setup_dataloaders(dataloader)

def save_checkpoint(model, ema, optimizer, epoch, global_step, experiment_dir):
    """Save training checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }
    torch.save(
        checkpoint,
        f"{experiment_dir}/checkpoint_control_net_{global_step:07d}.pt"
    )
    print(f"Saved checkpoint at global step {global_step}")

def update_ema(ema, model, fabric, decay=0.9999):
    """EMA weight update with Fabric support"""
    with fabric.no_backward_sync(model):
        for ema_param, model_param in zip(ema.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data.to(ema_param.dtype), alpha=1 - decay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add resume argument
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to checkpoint of Pretrained Model")
    
    # Original arguments remain the same
    parser.add_argument("--model", type=str, choices=list(MMDiT_models.keys()), default="MMDiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--outputs", type=str, default="./outputs")
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5_text_encoder", type=str, default="google-t5/t5-small")

    args = parser.parse_args()
    main(args)