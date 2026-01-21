"""
dataset.py
Dataset class for loading images and watermarks for training and evaluation.
Handles preprocessing, normalization, and data augmentation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random


class WatermarkDataset(Dataset):
    """
    Dataset for image watermarking.
    
    Loads cover images from data/images and watermarks from data/watermarks.
    Preprocesses images to 256x256 RGB format and normalizes to [0, 1].
    """
    def __init__(self, image_dir='data/images', watermark_dir='data/watermarks', 
                 image_size=256, watermark_size=32, train=True):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing cover images
            watermark_dir (str): Directory containing watermark images
            image_size (int): Size to resize images to (default: 256x256)
            watermark_size (int): Size to resize watermarks to (default: 32x32)
            train (bool): Whether this is training dataset (enables augmentation)
        """
        self.image_dir = image_dir
        self.watermark_dir = watermark_dir
        self.image_size = image_size
        self.watermark_size = watermark_size
        self.train = train
        
        # Get list of image files
        self.image_files = []
        if os.path.exists(image_dir):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            for filename in os.listdir(image_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    self.image_files.append(os.path.join(image_dir, filename))
        
        # Get list of watermark files
        self.watermark_files = []
        if os.path.exists(watermark_dir):
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for filename in os.listdir(watermark_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    self.watermark_files.append(os.path.join(watermark_dir, filename))
        
        # If no watermarks exist, we'll generate random binary watermarks on-the-fly
        self.generate_watermarks = len(self.watermark_files) == 0
        
        print(f"Loaded {len(self.image_files)} images from {image_dir}")
        if self.generate_watermarks:
            print(f"No watermarks found in {watermark_dir}, will generate random watermarks")
        else:
            print(f"Loaded {len(self.watermark_files)} watermarks from {watermark_dir}")
        
        # Define transformations for images
        if train:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
                transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        
        # Define transformations for watermarks
        self.watermark_transform = transforms.Compose([
            transforms.Resize((watermark_size, watermark_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_files)
    
    def load_image(self, image_path):
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            torch.Tensor: Preprocessed image [3, H, W] with values in [0, 1]
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed (handles grayscale, RGBA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            image_tensor = self.image_transform(image)
            
            return image_tensor
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def load_watermark(self, watermark_path):
        """
        Load and preprocess a watermark image.
        
        Args:
            watermark_path (str): Path to the watermark file
        
        Returns:
            torch.Tensor: Preprocessed watermark [1, H, W] with values in [0, 1]
        """
        try:
            # Load watermark
            watermark = Image.open(watermark_path)
            
            # Convert to grayscale (1 channel)
            if watermark.mode != 'L':
                watermark = watermark.convert('L')
            
            # Apply transformations
            watermark_tensor = self.watermark_transform(watermark)
            
            # Binarize watermark (threshold at 0.5)
            watermark_tensor = (watermark_tensor > 0.5).float()
            
            return watermark_tensor
        
        except Exception as e:
            print(f"Error loading watermark {watermark_path}: {e}")
            # Return a blank watermark as fallback
            return torch.zeros(1, self.watermark_size, self.watermark_size)
    
    def generate_random_watermark(self):
        """
        Generate a random binary watermark.
        
        Returns:
            torch.Tensor: Random binary watermark [1, watermark_size, watermark_size]
        """
        # Generate random binary watermark (0 or 1)
        watermark = torch.rand(1, self.watermark_size, self.watermark_size)
        watermark = (watermark > 0.5).float()
        return watermark
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (image, watermark) where:
                - image: torch.Tensor [3, H, W] with values in [0, 1]
                - watermark: torch.Tensor [1, W, W] with values in [0, 1]
        """
        # Load image
        image_path = self.image_files[idx]
        image = self.load_image(image_path)
        
        # Load or generate watermark
        if self.generate_watermarks:
            watermark = self.generate_random_watermark()
        else:
            # Randomly select a watermark
            watermark_path = random.choice(self.watermark_files)
            watermark = self.load_watermark(watermark_path)
        
        return image, watermark


def create_dataloaders(image_dir='data/images', watermark_dir='data/watermarks',
                       image_size=256, watermark_size=32, batch_size=8,
                       train_split=0.9, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        image_dir (str): Directory containing cover images
        watermark_dir (str): Directory containing watermark images
        image_size (int): Size to resize images to
        watermark_size (int): Size to resize watermarks to
        batch_size (int): Batch size for dataloaders
        train_split (float): Fraction of data to use for training (rest for validation)
        num_workers (int): Number of workers for dataloaders
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = WatermarkDataset(
        image_dir=image_dir,
        watermark_dir=watermark_dir,
        image_size=image_size,
        watermark_size=watermark_size,
        train=True
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_watermark_images(watermark_dir='data/watermarks', num_watermarks=10, size=32):
    """
    Utility function to create random binary watermark images if none exist.
    
    Args:
        watermark_dir (str): Directory to save watermarks
        num_watermarks (int): Number of watermarks to generate
        size (int): Size of each watermark (size x size)
    """
    os.makedirs(watermark_dir, exist_ok=True)
    
    for i in range(num_watermarks):
        # Generate random binary watermark
        watermark = np.random.randint(0, 2, size=(size, size)) * 255
        
        # Save as image
        watermark_path = os.path.join(watermark_dir, f'watermark_{i+1}.png')
        Image.fromarray(watermark.astype(np.uint8), mode='L').save(watermark_path)
    
    print(f"Created {num_watermarks} random watermark images in {watermark_dir}")


if __name__ == '__main__':
    """
    Test the dataset class.
    """
    # Create sample watermarks if they don't exist
    if not os.path.exists('data/watermarks') or len(os.listdir('data/watermarks')) == 0:
        print("Creating sample watermarks...")
        create_watermark_images()
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(batch_size=4)
    
    # Test loading a batch
    print("\nTesting data loading...")
    for images, watermarks in train_loader:
        print(f"Image batch shape: {images.shape}")  # Should be [batch_size, 3, 256, 256]
        print(f"Watermark batch shape: {watermarks.shape}")  # Should be [batch_size, 1, 32, 32]
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Watermark value range: [{watermarks.min():.3f}, {watermarks.max():.3f}]")
        break
    
    print("\nDataset test completed successfully!")
