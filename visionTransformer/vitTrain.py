import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple
from torchvision import transforms
from transformers import ViTImageProcessor
from torch.utils.data import DataLoader,Dataset

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
    
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

transform = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[image_mean, image_mean, image_mean], std=[image_std, image_std, image_std], inplace=True),
])

def setup_dataloader(csv_location,root_dir,img_size,batch_size,num_workers):
    """Setup dataset and dataloader with optimized transforms"""

    dataset = ImageCaptionDataset(
        csv_path="dataset/Flickr/captions.csv",
        root_dir="dataset/FLickr/images",
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


