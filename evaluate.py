"""
evaluate.py
Evaluation script to test the trained watermarking system.
Calculates metrics (PSNR, SSIM, BER) and visualizes results.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import custom modules
from models.encoder import get_encoder
from models.decoder import get_decoder
from attacks import (
    add_gaussian_noise,
    jpeg_compression,
    gaussian_blur,
    random_crop,
    brightness_adjustment,
    contrast_adjustment,
    apply_combined_attacks
)


def load_model(checkpoint_path, device):
    """
    Load trained encoder and decoder models from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device: Torch device
    
    Returns:
        tuple: (encoder, decoder, config)
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    
    # Create models
    encoder = get_encoder(
        watermark_size=config['watermark_size'],
        deep=config.get('deep_encoder', False)
    ).to(device)
    
    decoder = get_decoder(
        watermark_size=config['watermark_size'],
        architecture=config.get('decoder_arch', 'standard')
    ).to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return encoder, decoder, config


def load_and_preprocess_image(image_path, size=256):
    """
    Load and preprocess an image.
    
    Args:
        image_path (str): Path to image file
        size (int): Size to resize image to
    
    Returns:
        torch.Tensor: Preprocessed image [1, 3, H, W]
    """
    image = Image.open(image_path)
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((size, size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor


def load_and_preprocess_watermark(watermark_path, size=32):
    """
    Load and preprocess a watermark.
    
    Args:
        watermark_path (str): Path to watermark file
        size (int): Size to resize watermark to
    
    Returns:
        torch.Tensor: Preprocessed watermark [1, 1, H, W]
    """
    watermark = Image.open(watermark_path)
    
    # Convert to grayscale
    if watermark.mode != 'L':
        watermark = watermark.convert('L')
    
    # Resize
    watermark = watermark.resize((size, size), Image.BILINEAR)
    
    # Convert to tensor and binarize
    watermark_array = np.array(watermark).astype(np.float32) / 255.0
    watermark_array = (watermark_array > 0.5).astype(np.float32)
    watermark_tensor = torch.from_numpy(watermark_array).unsqueeze(0).unsqueeze(0)
    
    return watermark_tensor


def generate_random_watermark(size=32):
    """
    Generate a random binary watermark.
    
    Args:
        size (int): Size of watermark
    
    Returns:
        torch.Tensor: Random watermark [1, 1, size, size]
    """
    watermark = torch.rand(1, 1, size, size)
    watermark = (watermark > 0.5).float()
    return watermark


def calculate_psnr(original, modified):
    """
    Calculate PSNR between two images.
    
    Args:
        original (torch.Tensor): Original image [1, 3, H, W]
        modified (torch.Tensor): Modified image [1, 3, H, W]
    
    Returns:
        float: PSNR value
    """
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    modified_np = modified.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return psnr(original_np, modified_np, data_range=1.0)


def calculate_ssim(original, modified):
    """
    Calculate SSIM between two images.
    
    Args:
        original (torch.Tensor): Original image [1, 3, H, W]
        modified (torch.Tensor): Modified image [1, 3, H, W]
    
    Returns:
        float: SSIM value
    """
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    modified_np = modified.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return ssim(original_np, modified_np, data_range=1.0, channel_axis=2)


def calculate_ber(original_watermark, extracted_watermark):
    """
    Calculate Bit Error Rate between original and extracted watermark.
    
    Args:
        original_watermark (torch.Tensor): Original watermark [1, 1, H, W]
        extracted_watermark (torch.Tensor): Extracted watermark [1, 1, H, W]
    
    Returns:
        float: BER value (0 to 1, lower is better)
    """
    # Binarize extracted watermark
    extracted_binary = (extracted_watermark > 0.5).float()
    
    # Calculate bit errors
    errors = torch.sum(torch.abs(original_watermark - extracted_binary))
    total_bits = original_watermark.numel()
    
    ber = errors.item() / total_bits
    return ber


def visualize_results(original_image, watermarked_image, extracted_watermark, 
                     original_watermark, attack_name='No Attack', metrics=None, 
                     save_path=None):
    """
    Visualize original image, watermarked image, and extracted watermark.
    
    Args:
        original_image (torch.Tensor): Original image
        watermarked_image (torch.Tensor): Watermarked image
        extracted_watermark (torch.Tensor): Extracted watermark
        original_watermark (torch.Tensor): Original watermark
        attack_name (str): Name of the attack applied
        metrics (dict): Dictionary of metrics
        save_path (str): Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    orig_img = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Watermarked image
    wm_img = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 1].imshow(wm_img)
    axes[0, 1].set_title('Watermarked Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference (amplified)
    diff = np.abs(orig_img - wm_img) * 10  # Amplify for visibility
    axes[0, 2].imshow(diff)
    axes[0, 2].set_title('Difference (10x)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Original watermark
    orig_wm = original_watermark.squeeze().cpu().numpy()
    axes[1, 0].imshow(orig_wm, cmap='gray')
    axes[1, 0].set_title('Original Watermark', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Extracted watermark
    ext_wm = extracted_watermark.squeeze().cpu().numpy()
    axes[1, 1].imshow(ext_wm, cmap='gray')
    axes[1, 1].set_title('Extracted Watermark', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Binarized extracted watermark
    ext_wm_binary = (ext_wm > 0.5).astype(np.float32)
    axes[1, 2].imshow(ext_wm_binary, cmap='gray')
    axes[1, 2].set_title('Binarized Extracted', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add metrics text
    if metrics:
        metrics_text = f"Attack: {attack_name}\n"
        metrics_text += f"PSNR: {metrics['psnr']:.2f} dB\n"
        metrics_text += f"SSIM: {metrics['ssim']:.4f}\n"
        metrics_text += f"BER: {metrics['ber']:.4f} ({metrics['ber']*100:.2f}%)"
        
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def evaluate_single_image(encoder, decoder, image_path, watermark, device, 
                         attack_fn=None, attack_name='No Attack', visualize=True, 
                         save_dir=None):
    """
    Evaluate the watermarking system on a single image.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        image_path: Path to cover image
        watermark: Watermark tensor
        device: Torch device
        attack_fn: Attack function to apply (optional)
        attack_name: Name of the attack
        visualize: Whether to visualize results
        save_dir: Directory to save results
    
    Returns:
        dict: Dictionary of metrics
    """
    # Load image
    image = load_and_preprocess_image(image_path).to(device)
    watermark = watermark.to(device)
    
    with torch.no_grad():
        # Encode watermark into image
        watermarked_image = encoder(image, watermark)
        
        # Apply attack if specified
        if attack_fn:
            attacked_image = attack_fn(watermarked_image)
        else:
            attacked_image = watermarked_image
        
        # Decode watermark
        extracted_watermark = decoder(attacked_image)
    
    # Calculate metrics
    psnr_value = calculate_psnr(image, watermarked_image)
    ssim_value = calculate_ssim(image, watermarked_image)
    ber_value = calculate_ber(watermark, extracted_watermark)
    
    metrics = {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'ber': ber_value
    }
    
    # Print metrics
    print(f"\n{attack_name}:")
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  SSIM: {ssim_value:.4f}")
    print(f"  BER: {ber_value:.4f} ({ber_value*100:.2f}%)")
    
    # Visualize
    if visualize:
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{attack_name.replace(' ', '_')}.png")
        
        visualize_results(image, attacked_image, extracted_watermark, watermark, 
                        attack_name, metrics, save_path)
    
    return metrics


def evaluate_robustness(encoder, decoder, image_path, watermark, device, save_dir=None):
    """
    Evaluate robustness against various attacks.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        image_path: Path to cover image
        watermark: Watermark tensor
        device: Torch device
        save_dir: Directory to save results
    
    Returns:
        dict: Dictionary of metrics for each attack
    """
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION")
    print("="*60)
    
    attacks = [
        (None, 'No Attack'),
        (lambda x: add_gaussian_noise(x, std=0.05), 'Gaussian Noise (std=0.05)'),
        (lambda x: add_gaussian_noise(x, std=0.1), 'Gaussian Noise (std=0.1)'),
        (lambda x: jpeg_compression(x, quality=90), 'JPEG Compression (Q=90)'),
        (lambda x: jpeg_compression(x, quality=70), 'JPEG Compression (Q=70)'),
        (lambda x: jpeg_compression(x, quality=50), 'JPEG Compression (Q=50)'),
        (lambda x: gaussian_blur(x, kernel_size=3, sigma=1.0), 'Gaussian Blur (3x3)'),
        (lambda x: gaussian_blur(x, kernel_size=5, sigma=1.5), 'Gaussian Blur (5x5)'),
        (lambda x: random_crop(x, crop_ratio=0.1), 'Random Crop (10%)'),
        (lambda x: brightness_adjustment(x, factor=1.2), 'Brightness +20%'),
        (lambda x: brightness_adjustment(x, factor=0.8), 'Brightness -20%'),
        (lambda x: contrast_adjustment(x, factor=1.2), 'Contrast +20%'),
        (lambda x: apply_combined_attacks(x, num_attacks=2), 'Combined Attacks (2)'),
    ]
    
    all_metrics = {}
    
    for attack_fn, attack_name in attacks:
        metrics = evaluate_single_image(
            encoder, decoder, image_path, watermark, device,
            attack_fn=attack_fn, attack_name=attack_name,
            visualize=False, save_dir=save_dir
        )
        all_metrics[attack_name] = metrics
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Attack':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'BER':<10}")
    print("-"*60)
    for attack_name, metrics in all_metrics.items():
        print(f"{attack_name:<30} {metrics['psnr']:>10.2f}  {metrics['ssim']:>8.4f}  {metrics['ber']:>8.4f}")
    
    return all_metrics


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description='Evaluate image watermarking system')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to test image')
    parser.add_argument('--watermark', type=str, default=None, 
                       help='Path to watermark image (optional, random if not provided)')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='Directory to save results')
    parser.add_argument('--robustness_test', action='store_true', 
                       help='Run full robustness evaluation')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Visualize results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    encoder, decoder, config = load_model(args.checkpoint, device)
    
    # Load or generate watermark
    if args.watermark and os.path.exists(args.watermark):
        print(f"Loading watermark from: {args.watermark}")
        watermark = load_and_preprocess_watermark(args.watermark, config['watermark_size'])
    else:
        print("Generating random watermark")
        watermark = generate_random_watermark(config['watermark_size'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    if args.robustness_test:
        evaluate_robustness(encoder, decoder, args.image, watermark, device, args.output_dir)
    else:
        evaluate_single_image(encoder, decoder, args.image, watermark, device,
                            visualize=args.visualize, save_dir=args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
