import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader , Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from lightning.fabric import Fabric

from opendit.diffusion import create_diffusion
from opendit.models.mmdit import MMDiT_models

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable TF32 and cuda optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cudnn.deterministic = False

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

@torch.compile  
def training_step(model, x, t, y, diffusion):
    """Compiled training step for better performance"""
    model_kwargs = dict(c=y)
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

    # Setup models and optimizer
    model, ema, vae, optimizer = setup_models(args, fabric)
    diffusion = create_diffusion(timestep_respacing="")

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(args.resume_from, model, ema, optimizer, fabric)
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")

    # Setup dataset and dataloader
    train_loader = setup_dataloader(args, fabric)

    # Training loop
    loss_meter = AverageMeter()
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == "fp16" else None

    model.train()
    ema.eval()

    print(f"Training for {args.epochs} epochs starting from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"Beginning epoch {epoch}...")
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for x, y in pbar:

                try:
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision == "fp16"):
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)

                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            loss = training_step(model, x, t, y, diffusion)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = training_step(model, x, t, y, diffusion)
                        fabric.backward(loss)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, model, fabric)

                    loss_meter.update(loss.item())
                    global_step += 1

                    pbar.set_postfix({
                        "loss": f"{loss_meter.avg:.4f}",
                        "step": global_step
                    })

                    if global_step % args.log_every == 0:
                        writer.add_scalar("loss", loss_meter.avg, global_step)
                        loss_meter.reset()

                    if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:

                        f = open(f"{global_step}_Caption.txt", "w")
                        f.write(f"Caption at global step : {global_step} \n {y}")
                        f.close()
                        save_checkpoint(model, ema, optimizer, epoch, global_step, experiment_dir)

                except Exception as e:
                    print(e)
                    print(f"Error at global step : {global_step}")
                    f = open(f"{global_step}_Error.txt", "w")
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
    
    model = MMDiT_models[args.model](**model_config).to(device)
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create EMA model
    ema = MMDiT_models[args.model](**model_config).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)

    # Setup optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0,
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
        transforms.Resize((args.image_size,args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = ImageCaptionDataset(
        csv_path="dataset/MRI_DATASET/caption_large_cleaned_tumor.csv",
        root_dir="",
        transform=transform,
        cache_size=1000  # Cache 1000 images in memory
    )
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
        f"{experiment_dir}/checkpoint_{global_step:07d}.pt"
    )
    print(f"Saved checkpoint at global step {global_step}")

def update_ema(ema, model,fabric ,decay=0.9999  ):
    """EMA weight update with Fabric support"""
    with fabric.no_backward_sync(model):
        for ema_param, model_param in zip(ema.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data.to(ema_param.dtype), alpha=1 - decay)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add resume argument
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    
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