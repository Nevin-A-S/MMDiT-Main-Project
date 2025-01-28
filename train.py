import argparse
import json
import os
from glob import glob
import numpy as np

from PIL import Image
import torch
from pathlib import Path
from diffusers.models import AutoencoderKL

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from opendit.diffusion import create_diffusion
from opendit.models.mmdit import MMDiT_models
from opendit.utils.data_utils import get_transforms_image

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from SpawningDataset import SpawningPD12Dataset

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_path (string): Path to the CSV file with annotations
            root_dir (string): Base directory for image paths in CSV
            transform (callable, optional): Optional transform to be applied on images
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_rel_path = Path(row['File Path'].replace("\\", "/")) # Normalize to forward slashes
        img_full_path = self.root_dir / img_rel_path

        try:
            if not img_full_path.exists():
                raise FileNotFoundError(f"Image not found at: {img_full_path}")
            
        except FileNotFoundError:
            print(FileNotFoundError)
        image = Image.open(img_full_path).convert('RGB')
        
        caption = row['Caption']
        
        if self.transform:
            image = self.transform(image)

        return image, caption

def center_crop_arr(pil_image, image_size):


    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def requires_grad(model, flag=True):
    """Enable/disable gradients for a model's parameters."""
    for p in model.parameters():
        p.requires_grad = flag

def update_ema(ema, model, decay=0.9999):
    """Update EMA parameters."""
    with torch.no_grad():
        for ema_param, model_param in zip(ema.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

def main(args):
    """Trains a new MMDiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup directories
    os.makedirs(args.outputs, exist_ok=True)
    experiment_index = len(glob(f"{args.outputs}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.outputs}/{experiment_index:03d}-{model_string_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    # Setup tensorboard
    tensorboard_dir = f"{experiment_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    # Setup device and dtype
    device = torch.device('cuda')
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create VAE encoder
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    input_size = args.image_size // 8

    # Create model
    model_config = {
        "input_size": input_size,
        "num_classes": args.num_classes,
        "clip_text_encoder": args.text_encoder,
        "t5_text_encoder": args.t5_text_encoder,
    }

    # Initialize model
    model_class = MMDiT_models[args.model]
    model = model_class(**model_config).to(device, dtype=dtype)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")


    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create EMA model
    ema = MMDiT_models[args.model](**model_config).to(device)
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)

    # Create diffusion
    diffusion = create_diffusion(timestep_respacing="")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0
    )
    # Setup dataset
    # dataset = CIFAR10(
    #     args.data_path,
    #     transform=get_transforms_image(args.image_size),
    #     download=True
    # )

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    csv_path = "datasets/anime/image_labels.csv"
    root_dir = "datasets/anime"  

    # dataset = ImageCaptionDataset(
    # csv_path=csv_path,
    # root_dir=root_dir,
    # transform=transform
    # )
    
    dataset = SpawningPD12Dataset(transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,

    )

    print(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    print(f"Training for {args.epochs} epochs...")
    num_steps_per_epoch = len(dataloader)
    global_step = 0

    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")
        
        with tqdm(range(num_steps_per_epoch), desc=f"Epoch {epoch}") as pbar:
            for step in pbar:
                # Get batch
                x, y = next(iter(dataloader))
                x = x.to(device)

                # VAE encode
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

                # print('vae encode:', x.shape)

                # Diffusion training step
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(c=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update EMA
                update_ema(ema, model)

                # Logging
                global_step = epoch * num_steps_per_epoch + step
                pbar.set_postfix({"loss": loss.item(), "step": step, "global_step": global_step})

                if (global_step + 1) % args.log_every == 0:
                    writer.add_scalar("loss", loss.item(), global_step)

                # Save checkpoint
                if args.ckpt_every > 0 and (global_step + 1) % args.ckpt_every == 0:
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

    print("Training finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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