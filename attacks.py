"""
attacks.py
Implements various attack functions to simulate real-world scenarios on watermarked images.
These attacks test the robustness of the watermarking system.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random


def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    Add Gaussian noise to the image.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
    
    Returns:
        torch.Tensor: Noisy image clipped to [0, 1]
    """
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


def jpeg_compression(image, quality=75):
    """
    Simulate JPEG compression on the image.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        quality (int): JPEG quality factor (1-100, lower means more compression)
    
    Returns:
        torch.Tensor: Compressed image
    """
    # Simplified JPEG compression simulation using downsampling
    # In practice, you would use PIL or cv2 for actual JPEG compression
    # This is a differentiable approximation for training
    
    # Add quantization noise to simulate compression artifacts
    quantization_noise = torch.rand_like(image) * (1.0 / (quality + 1)) - (0.5 / (quality + 1))
    compressed = image + quantization_noise
    return torch.clamp(compressed, 0, 1)


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to the image.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation of the Gaussian kernel
    
    Returns:
        torch.Tensor: Blurred image
    """
    # Create Gaussian kernel
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Generate 1D Gaussian kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx = ax.repeat(kernel_size).view(kernel_size, kernel_size)
    yy = xx.t()
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    
    # Reshape kernel for conv2d: [out_channels, in_channels, H, W]
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(image.size(1), 1, 1, 1).to(image.device)
    
    # Apply convolution with padding
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=image.size(1))
    
    return blurred


def random_crop(image, crop_ratio=0.1):
    """
    Randomly crop the image and resize back to original size.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        crop_ratio (float): Ratio of the image to crop (0-1)
    
    Returns:
        torch.Tensor: Cropped and resized image
    """
    B, C, H, W = image.shape
    
    # Calculate crop size
    crop_h = int(H * (1 - crop_ratio))
    crop_w = int(W * (1 - crop_ratio))
    
    # Random crop position
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    
    # Crop and resize
    cropped = image[:, :, top:top+crop_h, left:left+crop_w]
    resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    
    return resized


def brightness_adjustment(image, factor=None):
    """
    Adjust the brightness of the image.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        factor (float): Brightness factor (None for random between 0.8 and 1.2)
    
    Returns:
        torch.Tensor: Brightness-adjusted image
    """
    if factor is None:
        factor = random.uniform(0.8, 1.2)
    
    adjusted = image * factor
    return torch.clamp(adjusted, 0, 1)


def contrast_adjustment(image, factor=None):
    """
    Adjust the contrast of the image.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        factor (float): Contrast factor (None for random between 0.8 and 1.2)
    
    Returns:
        torch.Tensor: Contrast-adjusted image
    """
    if factor is None:
        factor = random.uniform(0.8, 1.2)
    
    mean = torch.mean(image, dim=[2, 3], keepdim=True)
    adjusted = (image - mean) * factor + mean
    return torch.clamp(adjusted, 0, 1)


def apply_random_attack(image, attack_prob=0.5):
    """
    Apply a random attack to the image with given probability.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        attack_prob (float): Probability of applying an attack
    
    Returns:
        torch.Tensor: Attacked image
    """
    if random.random() > attack_prob:
        return image
    
    # List of available attacks
    attacks = [
        lambda x: add_gaussian_noise(x, std=random.uniform(0.01, 0.1)),
        lambda x: jpeg_compression(x, quality=random.randint(50, 90)),
        lambda x: gaussian_blur(x, kernel_size=random.choice([3, 5, 7]), sigma=random.uniform(0.5, 2.0)),
        lambda x: random_crop(x, crop_ratio=random.uniform(0.05, 0.15)),
        lambda x: brightness_adjustment(x),
        lambda x: contrast_adjustment(x),
    ]
    
    # Randomly select and apply an attack
    attack = random.choice(attacks)
    return attack(image)


def apply_combined_attacks(image, num_attacks=2):
    """
    Apply multiple random attacks sequentially to the image.
    
    Args:
        image (torch.Tensor): Input image tensor [B, C, H, W] with values in [0, 1]
        num_attacks (int): Number of attacks to apply
    
    Returns:
        torch.Tensor: Image after multiple attacks
    """
    attacked_image = image
    for _ in range(num_attacks):
        attacked_image = apply_random_attack(attacked_image, attack_prob=1.0)
    
    return attacked_image
