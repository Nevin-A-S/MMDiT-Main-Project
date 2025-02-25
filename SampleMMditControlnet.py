import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from opendit.utils.download import find_model
from opendit.models.mmdit import MMDiT_models
from opendit.diffusion import create_diffusion
from opendit.vae.wrapper import AutoencoderKLWrapper

# Enable TF32 for faster sampling
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        self.cache: Dict[int, Tuple] = {}  

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
            return torch.zeros((3, 256, 256)), "error loading image", torch.zeros((1, 256, 256))

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

def make_dataset(string_args , image_paths):
    caption = np.array(string_args)
    image = np.array(image_paths)
    dataFrame = pd.DataFrame({'image': image, 'caption': caption})
    os.makedirs('temp', exist_ok=True)
    dataFrame.to_csv('temp/data.csv', index=False)
    return 'temp/data.csv'

def main(args):
    global_count = 0

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None :
        raise ValueError("Please specify a checkpoint path with --ckpt.")


    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)


    
    input_size = args.image_size // 8

    model_config = {
        "input_size": input_size,
        "clip_text_encoder": args.text_encoder,
        "t5_text_encoder": args.t5_text_encoder,
        "use_video": args.use_video
    }

    model = MMDiT_models[args.model](**model_config).to(device)

    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))

    text_prompts =  ['Flair Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'T2-Weighted Brain MRI of a Patient with Tumour. ', 'T1-Weighted Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ', 'Flair Brain MRI of a Patient with Tumour. ']

    temp_csv_path = make_dataset(text_prompts, text_prompts)

    transform = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = ImageCaptionDataset(
        csv_path=temp_csv_path,
        root_dir="",
        transform=transform,
        cache_size=1000  
    )

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

    for i, (_, captions , edges) in enumerate(dataloader):
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision == "fp16"):

                        edges = vae.encode(edges).latent_dist.sample().mul_(0.18215)

                        n = len(captions)
                        z = torch.randn(n, 4, input_size, input_size, device=device)
                        y = text_prompts * 2

                        z = torch.cat([z, z], 0)
                        model_kwargs = dict(c=y, cfg_scale=args.cfg_scale,edges=edges)

                        samples = diffusion.p_sample_loop(
                            model.forward_with_cfg,
                            z.shape,
                            z,
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            progress=True,
                            device=device
                        )
                        samples, _ = samples.chunk(2, dim=0) 

                        samples = vae.decode(samples / 0.18215).sample

                        save_path = Path(args.output_dir) / f"samples_image_{ckpt_path}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        for i, (sample, prompt) in enumerate(zip(samples, text_prompts)):
                            image_path = save_path / f"sample_{global_count}.png"
                            save_image(sample, image_path, normalize=True, value_range=(-1, 1))
                            prompt_path = save_path / f"sample_{global_count}_prompt.txt"
                            prompt_path.write_text(prompt)
                            print(f"Saved image sample to {image_path}")
                            global_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MMDiT_models.keys()), default="MMDiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_sampling_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--use_video", action="store_true", help="Use video data instead of images")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5_text_encoder", type=str, default="google-t5/t5-small")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a MMDiT checkpoint",
    )
    args = parser.parse_args()
    main(args)
