"""
decoder.py
CNN Decoder network that extracts the watermark from a watermarked image.
Takes a watermarked image (possibly attacked) as input, outputs the extracted watermark.
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


class Decoder(nn.Module):
    """
    Decoder network that extracts a watermark from a watermarked image.
    
    Architecture:
    - Input: Watermarked image (3 channels)
    - Multiple convolutional layers to extract watermark features
    - Output: Extracted watermark (1 channel, binary)
    
    The decoder learns to robustly extract the watermark even after attacks.
    """
    def __init__(self, watermark_size=32):
        super(Decoder, self).__init__()
        self.watermark_size = watermark_size
        
        # Initial feature extraction
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Deeper feature extraction
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Middle processing layers
        self.conv5 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Watermark extraction layers
        self.conv7 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = ConvBlock(64, 32, kernel_size=3, stride=1, padding=1)
        
        # Final output layer (1 channel for binary watermark)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, watermarked_image):
        """
        Forward pass of the decoder.
        
        Args:
            watermarked_image (torch.Tensor): Watermarked image [B, 3, H, W] with values in [0, 1]
        
        Returns:
            torch.Tensor: Extracted watermark [B, 1, watermark_size, watermark_size] with values in [0, 1]
        """
        # Feature extraction pathway
        x = self.conv1(watermarked_image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Middle processing
        x = self.conv5(x)
        x = self.conv6(x)
        
        # Watermark extraction
        x = self.conv7(x)
        x = self.conv8(x)
        
        # Output extracted watermark
        watermark = self.conv_out(x)
        
        # Resize to watermark size
        watermark = F.interpolate(watermark, size=(self.watermark_size, self.watermark_size), 
                                  mode='bilinear', align_corners=False)
        
        # Apply sigmoid to get values in [0, 1] (probability of each bit being 1)
        watermark = torch.sigmoid(watermark)
        
        return watermark


class DecoderDeep(nn.Module):
    """
    Deeper decoder network with more capacity for robust watermark extraction.
    Uses residual connections and attention mechanisms.
    """
    def __init__(self, watermark_size=32):
        super(DecoderDeep, self).__init__()
        self.watermark_size = watermark_size
        
        # Initial processing
        self.initial = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=1, padding=3),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )
        
        # Feature extraction blocks with increasing channels
        self.extract1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        self.extract2 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Middle bottleneck for processing
        self.middle = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Watermark reconstruction blocks with decreasing channels
        self.reconstruct1 = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
        )
        
        self.reconstruct2 = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
        )
        
        # Final output
        self.final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, watermarked_image):
        """
        Forward pass of the deep decoder.
        
        Args:
            watermarked_image (torch.Tensor): Watermarked image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Extracted watermark [B, 1, watermark_size, watermark_size]
        """
        # Process through network
        x = self.initial(watermarked_image)
        x = self.extract1(x)
        x = self.extract2(x)
        x = self.middle(x)
        x = self.reconstruct1(x)
        x = self.reconstruct2(x)
        watermark = self.final(x)
        
        # Resize to watermark size
        watermark = F.interpolate(watermark, size=(self.watermark_size, self.watermark_size), 
                                  mode='bilinear', align_corners=False)
        
        # Apply sigmoid for binary watermark
        watermark = torch.sigmoid(watermark)
        
        return watermark


class DecoderWithAttention(nn.Module):
    """
    Decoder with attention mechanism to focus on watermark-relevant features.
    """
    def __init__(self, watermark_size=32):
        super(DecoderWithAttention, self).__init__()
        self.watermark_size = watermark_size
        
        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Watermark extraction
        self.extraction = nn.Sequential(
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, watermarked_image):
        """
        Forward pass with attention.
        
        Args:
            watermarked_image (torch.Tensor): Watermarked image [B, 3, H, W]
        
        Returns:
            torch.Tensor: Extracted watermark [B, 1, watermark_size, watermark_size]
        """
        # Extract features
        features = self.features(watermarked_image)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Extract watermark
        watermark = self.extraction(attended_features)
        
        # Resize to watermark size
        watermark = F.interpolate(watermark, size=(self.watermark_size, self.watermark_size), 
                                  mode='bilinear', align_corners=False)
        
        # Apply sigmoid
        watermark = torch.sigmoid(watermark)
        
        return watermark


# Default decoder to use
def get_decoder(watermark_size=32, architecture='standard'):
    """
    Factory function to get a decoder model.
    
    Args:
        watermark_size (int): Size of the watermark (default: 32x32)
        architecture (str): Decoder architecture ('standard', 'deep', or 'attention')
    
    Returns:
        nn.Module: Decoder model
    """
    if architecture == 'deep':
        return DecoderDeep(watermark_size)
    elif architecture == 'attention':
        return DecoderWithAttention(watermark_size)
    else:
        return Decoder(watermark_size)
