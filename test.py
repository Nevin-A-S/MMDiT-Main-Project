from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image

class SpawningPD12Dataset(Dataset):
    def __init__(self, transform=None, max_images=60000):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on images
            max_images (int): Maximum number of images to include in the dataset
        """
        self.dataset = load_dataset("spawning/pd12", split="train")
        self.transform = transform
        self.max_images = min(max_images, len(self.dataset))
        
    def __len__(self):
        return self.max_images

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and convert image
        image = Image.open(item['image'].convert('RGB'))
        caption = item['text']
        
        if self.transform:
            image = self.transform(image)

        return image, caption

transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])    

dataset = SpawningPD12Dataset(transform=transform, max_images=60000)
