"""
encoder.py
CNN Encoder network that embeds a watermark into a cover image.
Takes a cover image and watermark as input, outputs a watermarked image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """
    Encoder network that embeds a watermark into an image.
    
    Architecture:
    - Input: Concatenation of cover image (3 channels) and watermark (1 channel) = 4 channels
    - Multiple convolutional layers to learn embedding
    - Output: Watermarked image (3 channels)
    
    The encoder learns to imperceptibly embed the watermark into the cover image.
    """
    def __init__(self, watermark_size=32):
        super(Encoder, self).__init__()
        self.watermark_size = watermark_size
        
        # Initial convolution block (4 channels: 3 for image + 1 for watermark)
        self.conv1 = ConvBlock(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Deeper encoding layers
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Middle layers for feature transformation
        self.conv5 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Decoding layers back to image space
        self.conv7 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Final output layer (no activation, output is the watermarked image)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, image, watermark):
        """
        Forward pass of the encoder.
        
        Args:
            image (torch.Tensor): Cover image [B, 3, H, W] with values in [0, 1]
            watermark (torch.Tensor): Watermark [B, 1, watermark_size, watermark_size] with values in [0, 1]
        
        Returns:
            torch.Tensor: Watermarked image [B, 3, H, W] with values in [0, 1]
        """
        B, C, H, W = image.shape
        
        # Resize watermark to match image dimensions
        watermark_resized = F.interpolate(watermark, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate image and watermark
        x = torch.cat([image, watermark_resized], dim=1)  # [B, 4, H, W]
        
        # Encoder pathway
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Middle transformation
        x = self.conv5(x)
        x = self.conv6(x)
        
        # Decoder pathway
        x = self.conv7(x)
        x = self.conv8(x)
        
        # Output watermarked image
        watermarked = self.conv_out(x)
        
        # Add residual connection: watermarked = image + learned_residual
        # This helps maintain image quality while embedding watermark
        watermarked = image + watermarked
        
        # Ensure output is in valid range [0, 1]
        watermarked = torch.clamp(watermarked, 0, 1)
        
        return watermarked


class EncoderDeep(nn.Module):
    """
    Deeper encoder network with more capacity for complex watermark embedding.
    Uses residual connections for better gradient flow.
    """
    def __init__(self, watermark_size=32):
        super(EncoderDeep, self).__init__()
        self.watermark_size = watermark_size
        
        # Initial processing
        self.initial = nn.Sequential(
            ConvBlock(4, 64, kernel_size=7, stride=1, padding=3),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        # Encoder blocks with increasing channels
        self.encoder1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        self.encoder2 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Middle bottleneck
        self.middle = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Decoder blocks with decreasing channels
        self.decoder1 = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        self.decoder2 = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        # Final output
        self.final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, image, watermark):
        """
        Forward pass of the deep encoder.
        
        Args:
            image (torch.Tensor): Cover image [B, 3, H, W]
            watermark (torch.Tensor): Watermark [B, 1, watermark_size, watermark_size]
        
        Returns:
            torch.Tensor: Watermarked image [B, 3, H, W]
        """
        B, C, H, W = image.shape
        
        # Resize watermark to match image dimensions
        watermark_resized = F.interpolate(watermark, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate image and watermark
        x = torch.cat([image, watermark_resized], dim=1)  # [B, 4, H, W]
        
        # Process through network
        x = self.initial(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.middle(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        residual = self.final(x)
        
        # Add residual to original image
        watermarked = image + residual
        watermarked = torch.clamp(watermarked, 0, 1)
        
        return watermarked


# Default encoder to use
def get_encoder(watermark_size=32, deep=False):
    """
    Factory function to get an encoder model.
    
    Args:
        watermark_size (int): Size of the watermark (default: 32x32)
        deep (bool): Whether to use the deeper encoder architecture
    
    Returns:
        nn.Module: Encoder model
    """
    if deep:
        return EncoderDeep(watermark_size)
    else:
        return Encoder(watermark_size)
