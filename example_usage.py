"""
example_usage.py
Example script demonstrating how to use the watermarking system.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import custom modules
from models.encoder import get_encoder
from models.decoder import get_decoder
from attacks import add_gaussian_noise, jpeg_compression, gaussian_blur


def create_example_image():
    """Create a simple example image for demonstration."""
    # Create a 256x256 RGB image with gradient
    img = np.zeros((256, 256, 3), dtype=np.float32)
    
    # Red gradient
    img[:, :, 0] = np.linspace(0, 1, 256)[None, :].repeat(256, axis=0)
    # Green gradient
    img[:, :, 1] = np.linspace(0, 1, 256)[:, None].repeat(256, axis=1)
    # Blue constant
    img[:, :, 2] = 0.5
    
    return img


def create_example_watermark():
    """Create a simple binary watermark (32x32 checkerboard pattern)."""
    watermark = np.zeros((32, 32), dtype=np.float32)
    
    # Create checkerboard pattern
    for i in range(32):
        for j in range(32):
            if (i // 4 + j // 4) % 2 == 0:
                watermark[i, j] = 1.0
    
    return watermark


def numpy_to_torch(image, watermark):
    """Convert numpy arrays to torch tensors."""
    # Image: [H, W, 3] -> [1, 3, H, W]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    # Watermark: [H, W] -> [1, 1, H, W]
    watermark_tensor = torch.from_numpy(watermark).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, watermark_tensor


def torch_to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if tensor.dim() == 4:
        # [1, C, H, W] -> [H, W, C]
        return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    elif tensor.dim() == 3:
        # [1, H, W] -> [H, W]
        return tensor.squeeze().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def visualize_example(original_image, watermarked_image, extracted_watermark, 
                     original_watermark, attack_name='No Attack'):
    """Visualize the watermarking results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Watermarked image
    axes[0, 1].imshow(watermarked_image)
    axes[0, 1].set_title('Watermarked Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference (amplified)
    diff = np.abs(original_image - watermarked_image) * 10
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title('Difference (10x)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Original watermark
    axes[1, 0].imshow(original_watermark, cmap='gray')
    axes[1, 0].set_title('Original Watermark', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Extracted watermark
    axes[1, 1].imshow(extracted_watermark, cmap='gray')
    axes[1, 1].set_title('Extracted Watermark', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Binarized extracted watermark
    extracted_binary = (extracted_watermark > 0.5).astype(np.float32)
    axes[1, 2].imshow(extracted_binary, cmap='gray')
    axes[1, 2].set_title('Binarized Extracted', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Example Watermarking ({attack_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'example_{attack_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function to demonstrate watermarking."""
    print("="*60)
    print("Image Watermarking Example")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create models
    print("\n1. Creating encoder and decoder models...")
    encoder = get_encoder(watermark_size=32).to(device)
    decoder = get_decoder(watermark_size=32).to(device)
    encoder.eval()
    decoder.eval()
    print("   ✓ Models created")
    
    # Create example data
    print("\n2. Creating example image and watermark...")
    image_np = create_example_image()
    watermark_np = create_example_watermark()
    print("   ✓ Example data created")
    
    # Convert to torch tensors
    image_tensor, watermark_tensor = numpy_to_torch(image_np, watermark_np)
    image_tensor = image_tensor.to(device)
    watermark_tensor = watermark_tensor.to(device)
    
    # Test 1: No attack
    print("\n3. Testing watermark embedding and extraction (no attack)...")
    with torch.no_grad():
        watermarked = encoder(image_tensor, watermark_tensor)
        extracted = decoder(watermarked)
    
    watermarked_np = torch_to_numpy(watermarked)
    extracted_np = torch_to_numpy(extracted)
    
    # Calculate BER
    extracted_binary = (extracted_np > 0.5).astype(np.float32)
    ber = np.mean(np.abs(watermark_np - extracted_binary))
    print(f"   BER (no attack): {ber:.4f} ({ber*100:.2f}%)")
    
    # Visualize
    visualize_example(image_np, watermarked_np, extracted_np, watermark_np, 'No Attack')
    
    # Test 2: With Gaussian noise
    print("\n4. Testing with Gaussian noise attack...")
    with torch.no_grad():
        attacked = add_gaussian_noise(watermarked, std=0.05)
        extracted_attacked = decoder(attacked)
    
    attacked_np = torch_to_numpy(attacked)
    extracted_attacked_np = torch_to_numpy(extracted_attacked)
    
    extracted_attacked_binary = (extracted_attacked_np > 0.5).astype(np.float32)
    ber_attacked = np.mean(np.abs(watermark_np - extracted_attacked_binary))
    print(f"   BER (with noise): {ber_attacked:.4f} ({ber_attacked*100:.2f}%)")
    
    # Visualize
    visualize_example(image_np, attacked_np, extracted_attacked_np, watermark_np, 'Gaussian Noise')
    
    # Test 3: With JPEG compression
    print("\n5. Testing with JPEG compression attack...")
    with torch.no_grad():
        compressed = jpeg_compression(watermarked, quality=70)
        extracted_compressed = decoder(compressed)
    
    compressed_np = torch_to_numpy(compressed)
    extracted_compressed_np = torch_to_numpy(extracted_compressed)
    
    extracted_compressed_binary = (extracted_compressed_np > 0.5).astype(np.float32)
    ber_compressed = np.mean(np.abs(watermark_np - extracted_compressed_binary))
    print(f"   BER (with compression): {ber_compressed:.4f} ({ber_compressed*100:.2f}%)")
    
    # Visualize
    visualize_example(image_np, compressed_np, extracted_compressed_np, watermark_np, 'JPEG Compression')
    
    print("\n" + "="*60)
    print("Example completed!")
    print("\nNOTE: This uses untrained models, so the watermark extraction")
    print("      will be poor. After training, BER should be < 0.1 (10%).")
    print("\nNext steps:")
    print("1. Download DIV2K dataset and place in data/images/")
    print("2. Run: python train.py --num_epochs 100 --batch_size 8 --use_attacks")
    print("3. After training, evaluate on real images with evaluate.py")
    print("="*60)


if __name__ == '__main__':
    main()
