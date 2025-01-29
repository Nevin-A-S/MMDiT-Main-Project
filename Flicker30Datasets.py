from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
from pathlib import Path

class Flicker30KDataset(Dataset):
    def __init__(self, csv_path : str = "datasets/captions.txt", root_dir : str = "datasets/Flicker30K", transform=None):
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
        
        img_rel_path = Path(row['image'].replace("\\", "/")) 
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
        
if __name__ == "__main__":

    import kagglehub
    
    # Download latest version
    path = kagglehub.dataset_download("adityajn105/flickr30k" , path="datasets/Flicker30K")

    print("Path to dataset files:", path)

    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])    

    dataset = Flicker30KDataset(transform=transform, max_images=60000)
    print(len(dataset))