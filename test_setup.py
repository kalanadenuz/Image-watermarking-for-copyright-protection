"""
test_setup.py
Simple script to verify that everything is set up correctly.
Run this before starting training to ensure all components work.
"""

import os
import sys
import torch

def check_python_version():
    """Check Python version."""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (Need Python 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n2. Checking dependencies...")
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'skimage': 'scikit-image',
        'tqdm': 'tqdm'
    }
    
    all_good = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} (Not installed)")
            all_good = False
    
    return all_good

def check_pytorch_cuda():
    """Check if PyTorch can use CUDA."""
    print("\n3. Checking PyTorch and CUDA...")
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠ CUDA not available (will use CPU - training will be slower)")
    
    return True

def check_directories():
    """Check if required directories exist."""
    print("\n4. Checking directories...")
    dirs = {
        'data': 'Data directory',
        'data/images': 'Images directory',
        'data/watermarks': 'Watermarks directory',
        'models': 'Models directory'
    }
    
    all_good = True
    for dir_path, description in dirs.items():
        if os.path.exists(dir_path):
            if dir_path in ['data/images', 'data/watermarks']:
                num_files = len([f for f in os.listdir(dir_path) 
                               if os.path.isfile(os.path.join(dir_path, f))])
                print(f"   ✓ {description} ({num_files} files)")
            else:
                print(f"   ✓ {description}")
        else:
            print(f"   ⚠ {description} (will be created)")
            os.makedirs(dir_path, exist_ok=True)
    
    return all_good

def check_dataset():
    """Check if dataset can be loaded."""
    print("\n5. Checking dataset loading...")
    try:
        from dataset import WatermarkDataset
        
        dataset = WatermarkDataset(
            image_dir='data/images',
            watermark_dir='data/watermarks',
            image_size=256,
            watermark_size=32,
            train=True
        )
        
        num_images = len(dataset)
        if num_images > 0:
            print(f"   ✓ Dataset loaded: {num_images} images")
            
            # Try loading one sample
            image, watermark = dataset[0]
            print(f"   ✓ Sample loaded: image {image.shape}, watermark {watermark.shape}")
            return True
        else:
            print("   ✗ No images found in data/images/")
            print("   → Download DIV2K dataset or place your images in data/images/")
            return False
            
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return False

def check_models():
    """Check if models can be created."""
    print("\n6. Checking model creation...")
    try:
        from models.encoder import get_encoder
        from models.decoder import get_decoder
        
        encoder = get_encoder(watermark_size=32)
        decoder = get_decoder(watermark_size=32)
        
        encoder_params = sum(p.numel() for p in encoder.parameters())
        decoder_params = sum(p.numel() for p in decoder.parameters())
        
        print(f"   ✓ Encoder created: {encoder_params:,} parameters")
        print(f"   ✓ Decoder created: {decoder_params:,} parameters")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        test_image = torch.randn(1, 3, 256, 256).to(device)
        test_watermark = torch.randn(1, 1, 32, 32).to(device)
        
        with torch.no_grad():
            watermarked = encoder(test_image, test_watermark)
            extracted = decoder(watermarked)
        
        print(f"   ✓ Forward pass successful")
        print(f"     Watermarked image shape: {watermarked.shape}")
        print(f"     Extracted watermark shape: {extracted.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error creating models: {e}")
        return False

def main():
    """Run all checks."""
    print("="*60)
    print("Image Watermarking Setup Verification")
    print("="*60)
    
    results = []
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("PyTorch/CUDA", check_pytorch_cuda()))
    results.append(("Directories", check_directories()))
    results.append(("Dataset", check_dataset()))
    results.append(("Models", check_models()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, status in results:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{name:<20} {status_str}")
    
    all_passed = all(status for _, status in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Ensure you have images in data/images/ (download DIV2K dataset)")
        print("2. Run: python train.py --num_epochs 100 --batch_size 8 --use_attacks")
        print("3. Monitor training: tensorboard --logdir=logs")
    else:
        print("⚠ Some checks failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download images: Place training images in data/images/")
        print("3. Check Python version: Need Python 3.8 or higher")
    print("="*60)

if __name__ == '__main__':
    main()
