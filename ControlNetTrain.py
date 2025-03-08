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
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.layers = [3, 8, 15, 22]
        self.weights = [0.1, 0.2, 0.4, 0.8]
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

class LossTracker:
    """Track multiple loss components"""
    def __init__(self):
        self.loss_meters = {
            'total_loss': AverageMeter(),
            'diffusion_loss': AverageMeter(),
            'perceptual_loss': AverageMeter(),
            'lpips_loss': AverageMeter(),
            'edge_loss': AverageMeter(),
        }
    
    def update(self, key, value, batch_size=1):
        if key in self.loss_meters:
            self.loss_meters[key].update(value, batch_size)
    
    def reset(self):
        for meter in self.loss_meters.values():
            meter.reset()
    
    def get_avg(self, key):
        if key in self.loss_meters:
            return self.loss_meters[key].avg
        return 0.0

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

# Using the original function signature but enhancing it internally
@torch.compile  
def training_step(model, x, t, y, diffusion, edges):
    """Compiled training step that maintains the original interface"""
    model_kwargs = dict(c=y, edges=edges)
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    return loss_dict["loss"].mean()

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
    
    # Initialize loss components - not connected to training_step yet
    perceptual_loss = PerceptualLoss().to(device)
    lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
    
    # These will be applied separately outside training_step to maintain compatibility
    perceptual_weight = 0.1
    lpips_weight = 0.1
    edge_weight = 0.5
    
    # If resuming, ensure models are moved to correct device
    perceptual_loss = fabric.setup_module(perceptual_loss)
    lpips_model = fabric.setup_module(lpips_model)

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(args.resume_from, model, ema, optimizer, fabric)
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")

    # Setup dataset and dataloader
    train_loader = setup_dataloader(args, fabric)

    # Training loop
    loss_meter = AverageMeter()  # Keep original loss meter
    loss_tracker = LossTracker()  # Add enhanced loss tracker
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == "fp16" else None

    model.train()
    ema.eval()
    
    # Calculate total steps for weight annealing
    total_steps = args.epochs * len(train_loader)

    print(f"Training for {args.epochs} epochs starting from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Beginning epoch {epoch}...")
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for x, y, edges in pbar:
                try:
                    # Dynamic weight adjustment based on training progress
                    progress = min(1.0, global_step / (total_steps * 0.8))
                    
                    # Update weights for additional losses based on progress
                    current_perceptual_weight = max(0.05, perceptual_weight * (1.0 - progress * 0.5))
                    current_lpips_weight = max(0.02, lpips_weight * (1.0 - progress * 0.5))
                    current_edge_weight = max(0.1, edge_weight * (1.0 - progress * 0.3))
                    
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision == "fp16"):
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                        edges = vae.encode(edges).latent_dist.sample().mul_(0.18215)

                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
                    
                    # Use the original training step function which has the correct signature
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            # Get diffusion loss from original training step
                            diffusion_loss = training_step(model, x, t, y, diffusion, edges)
                            
                            # Add enhanced losses (only if weights are > 0)
                            total_loss = diffusion_loss
                            
                            if current_perceptual_weight > 0 or current_lpips_weight > 0 or current_edge_weight > 0:
                                # Get model prediction for perceptual losses
                                with torch.no_grad():
                                    model_kwargs = dict(c=y, edges=edges)
                                    model_output = model(x, t, **model_kwargs)
                                    predicted_x0 = diffusion.predict_start_from_noise(x, t, model_output)
                                
                                # Apply additional losses
                                if current_perceptual_weight > 0:
                                    p_loss = perceptual_loss(predicted_x0, x)
                                    total_loss = total_loss + current_perceptual_weight * p_loss
                                    loss_tracker.update('perceptual_loss', p_loss.item())
                                
                                if current_lpips_weight > 0:
                                    l_loss = lpips_model(predicted_x0, x).mean()
                                    total_loss = total_loss + current_lpips_weight * l_loss
                                    loss_tracker.update('lpips_loss', l_loss.item())
                                
                                if current_edge_weight > 0:
                                    # Calculate edge-guided loss
                                    pred_grad_x = torch.abs(predicted_x0[:, :, :, 1:] - predicted_x0[:, :, :, :-1])
                                    pred_grad_y = torch.abs(predicted_x0[:, :, 1:, :] - predicted_x0[:, :, :-1, :])
                                    
                                    target_grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
                                    target_grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
                                    
                                    # Use edges for weighting
                                    cs_x = edges[:, :, :, :-1]
                                    cs_y = edges[:, :, :-1, :]
                                    
                                    edge_loss_x = F.mse_loss(pred_grad_x * cs_x, target_grad_x * cs_x)
                                    edge_loss_y = F.mse_loss(pred_grad_y * cs_y, target_grad_y * cs_y)
                                    
                                    edge_loss = (edge_loss_x + edge_loss_y) * 0.5
                                    total_loss = total_loss + current_edge_weight * edge_loss
                                    loss_tracker.update('edge_loss', edge_loss.item())
                            
                            # Track losses
                            loss_tracker.update('diffusion_loss', diffusion_loss.item())
                            loss_tracker.update('total_loss', total_loss.item())
                            
                        # Use scaler with the total loss
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Get diffusion loss from original training step
                        diffusion_loss = training_step(model, x, t, y, diffusion, edges)
                        
                        # Add enhanced losses (only if weights are > 0)
                        total_loss = diffusion_loss
                        
                        if current_perceptual_weight > 0 or current_lpips_weight > 0 or current_edge_weight > 0:
                            # Get model prediction for perceptual losses
                            with torch.no_grad():
                                model_kwargs = dict(c=y, edges=edges)
                                model_output = model(x, t, **model_kwargs)
                                predicted_x0 = diffusion.predict_start_from_noise(x, t, model_output)
                            
                            # Apply additional losses
                            if current_perceptual_weight > 0:
                                p_loss = perceptual_loss(predicted_x0, x)
                                total_loss = total_loss + current_perceptual_weight * p_loss
                                loss_tracker.update('perceptual_loss', p_loss.item())
                            
                            if current_lpips_weight > 0:
                                l_loss = lpips_model(predicted_x0, x).mean()
                                total_loss = total_loss + current_lpips_weight * l_loss
                                loss_tracker.update('lpips_loss', l_loss.item())
                            
                            if current_edge_weight > 0:
                                # Calculate edge-guided loss
                                pred_grad_x = torch.abs(predicted_x0[:, :, :, 1:] - predicted_x0[:, :, :, :-1])
                                pred_grad_y = torch.abs(predicted_x0[:, :, 1:, :] - predicted_x0[:, :, :-1, :])
                                
                                target_grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
                                target_grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
                                
                                # Use edges for weighting
                                cs_x = edges[:, :, :, :-1]
                                cs_y = edges[:, :, :-1, :]
                                
                                edge_loss_x = F.mse_loss(pred_grad_x * cs_x, target_grad_x * cs_x)
                                edge_loss_y = F.mse_loss(pred_grad_y * cs_y, target_grad_y * cs_y)
                                
                                edge_loss = (edge_loss_x + edge_loss_y) * 0.5
                                total_loss = total_loss + current_edge_weight * edge_loss
                                loss_tracker.update('edge_loss', edge_loss.item())
                        
                        # Track losses
                        loss_tracker.update('diffusion_loss', diffusion_loss.item())
                        loss_tracker.update('total_loss', total_loss.item())
                        
                        fabric.backward(total_loss)
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, model, fabric)

                    # For compatibility with original code, update the standard loss meter too
                    loss_meter.update(total_loss.item())
                    global_step += 1

                    # Update progress bar with additional metrics
                    pbar.set_postfix({
                        "loss": f"{loss_meter.avg:.4f}",
                        "step": global_step,
                        "p_w": f"{current_perceptual_weight:.2f}",
                        "e_w": f"{current_edge_weight:.2f}"
                    })

                    if global_step % args.log_every == 0:
                        # Log all loss components
                        writer.add_scalar("loss/total", loss_tracker.get_avg('total_loss'), global_step)
                        writer.add_scalar("loss/diffusion", loss_tracker.get_avg('diffusion_loss'), global_step)
                        writer.add_scalar("loss/perceptual", loss_tracker.get_avg('perceptual_loss'), global_step)
                        writer.add_scalar("loss/lpips", loss_tracker.get_avg('lpips_loss'), global_step)
                        writer.add_scalar("loss/edge", loss_tracker.get_avg('edge_loss'), global_step)
                        
                        # Log weights
                        writer.add_scalar("weights/perceptual", current_perceptual_weight, global_step)
                        writer.add_scalar("weights/lpips", current_lpips_weight, global_step)
                        writer.add_scalar("weights/edge", current_edge_weight, global_step)
                        
                        # Reset loss trackers
                        loss_meter.reset()
                        loss_tracker.reset()

                    if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                        # Save captions for reference
                        with open(f"{experiment_dir}/{global_step}_Caption.txt", "w") as f:
                            f.write(f"Caption at global step : {global_step} \n {y}")
                        
                        # Save current loss weights
                        with open(f"{experiment_dir}/{global_step}_LossWeights.txt", "w") as f:
                            f.write(f"Perceptual Weight: {current_perceptual_weight:.4f}\n")
                            f.write(f"LPIPS Weight: {current_lpips_weight:.4f}\n")
                            f.write(f"Edge Weight: {current_edge_weight:.4f}\n")
                        
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

    # Setup optimizer with gentle weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
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