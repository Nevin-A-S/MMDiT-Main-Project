import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from torchvision import transforms
from typing import List, Optional, Dict
from tqdm import tqdm
import glob
import os

# Enable TF32 and cuda optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class ImageGenerator:
    def __init__(
        self,
        vae_type: str = "ema",
        image_size: int = 256,
        device: str = "cuda",
        cfg_scale: float = 1.0,
        num_inference_steps: int = 250,
    ):
        """Initialize the image generator with settings.
        
        Args:
            vae_type: Type of VAE to use ("ema" or "mse")
            image_size: Size of generated images
            device: Device to run inference on
            cfg_scale: Classifier free guidance scale
            num_inference_steps: Number of denoising steps
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.cfg_scale = cfg_scale
        self.num_inference_steps = num_inference_steps
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{vae_type}"
        ).to(self.device)
        self.vae.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_model(self, model_path: str):
        """Load the trained model from checkpoint."""
        print(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            model = checkpoint['model'] if 'model' in checkpoint else checkpoint['ema']
        else:
            model = checkpoint
        
        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        return model

    def _get_checkpoint_name(self, checkpoint_path: str) -> str:
        """Extract a clean name from checkpoint path for folder naming."""
        base_name = os.path.basename(checkpoint_path)
        # Remove extension and common checkpoint prefixes
        name = os.path.splitext(base_name)[0]
        name = name.replace('checkpoint_', '').replace('model_', '')
        return name

    @torch.no_grad()
    def generate_from_checkpoint(
        self,
        checkpoint_path: str,
        prompts: List[str],
        output_dir: str,
        batch_size: int = 4,
        seed: Optional[int] = None,
    ) -> Dict[str, List[Path]]:
        """Generate images using a specific checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            prompts: List of text prompts
            output_dir: Base directory for outputs
            batch_size: Number of images to generate in parallel
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping checkpoint names to lists of generated image paths
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Create checkpoint-specific output directory
        ckpt_name = self._get_checkpoint_name(checkpoint_path)
        ckpt_output_dir = Path(output_dir) / ckpt_name
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model for this checkpoint
        self.model = self._load_model(checkpoint_path)
        generated_paths = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Generate latent noise
            latents = torch.randn(
                len(batch_prompts),
                4,
                self.image_size // 8,
                self.image_size // 8,
                device=self.device
            )
            
            # Add classifier-free guidance
            latents = torch.cat([latents, latents], 0)
            
            # Create conditional inputs
            model_kwargs = {
                "prompts": batch_prompts * 2,  # Double for classifier-free guidance
                "cfg_scale": self.cfg_scale
            }
            
            # Generate images
            samples = self.model.sample(
                latents,
                self.num_inference_steps,
                model_kwargs=model_kwargs
            )
            
            # Remove guidance samples and decode
            samples = samples[:len(batch_prompts)]
            images = self.vae.decode(samples / 0.18215).sample
            
            # Save images
            for j, image in enumerate(images):
                prompt_slug = batch_prompts[j][:30].replace(" ", "_")  # First 30 chars of prompt
                save_path = ckpt_output_dir / f"{prompt_slug}_{i+j:04d}.png"
                save_image(
                    image,
                    save_path,
                    normalize=True,
                    value_range=(-1, 1)
                )
                generated_paths.append(save_path)
                
        return {ckpt_name: generated_paths}

    def generate_from_folder(
        self,
        checkpoint_dir: str,
        prompts: List[str],
        output_dir: str,
        batch_size: int = 4,
        seed: Optional[int] = None,
        checkpoint_pattern: str = "*.pt"
    ) -> Dict[str, List[Path]]:
        """Generate images using all checkpoints in a folder.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            prompts: List of text prompts
            output_dir: Base directory for outputs
            batch_size: Number of images to generate in parallel
            seed: Random seed for reproducibility
            checkpoint_pattern: Pattern to match checkpoint files
            
        Returns:
            Dictionary mapping checkpoint names to lists of generated image paths
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_paths = sorted(checkpoint_dir.glob(checkpoint_pattern))
        
        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found in {checkpoint_dir} matching pattern {checkpoint_pattern}")
        
        all_generated = {}
        for checkpoint_path in tqdm(checkpoint_paths, desc="Processing checkpoints"):
            try:
                generated = self.generate_from_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    prompts=prompts,
                    output_dir=output_dir,
                    batch_size=batch_size,
                    seed=seed
                )
                all_generated.update(generated)
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint_path}: {str(e)}")
                continue
                
        return all_generated

def main():
    parser = argparse.ArgumentParser(description="Generate images using trained MMDiT models")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--checkpoint_pattern", type=str, default="*.pt", help="Pattern to match checkpoint files")
    parser.add_argument("--prompts", type=str, nargs="+", required=True, help="Text prompts for generation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory for generated images")
    parser.add_argument("--vae_type", type=str, choices=["ema", "mse"], default="ema", help="Type of VAE to use")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256, help="Size of generated images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier free guidance scale")
    parser.add_argument("--steps", type=int, default=250, help="Number of inference steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ImageGenerator(
        vae_type=args.vae_type,
        image_size=args.image_size,
        device=args.device,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps
    )
    
    # Generate images from all checkpoints
    generated_paths = generator.generate_from_folder(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_pattern=args.checkpoint_pattern,
        prompts=args.prompts,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Print summary
    print("\nGeneration Summary:")
    for checkpoint_name, paths in generated_paths.items():
        print(f"\nCheckpoint: {checkpoint_name}")
        print(f"Generated {len(paths)} images in {args.output_dir}/{checkpoint_name}")
    
if __name__ == "__main__":
    main()