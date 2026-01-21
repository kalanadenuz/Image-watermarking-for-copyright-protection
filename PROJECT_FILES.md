# Project Files Summary

This document provides an overview of all files in the Image Watermarking project.

## Core Python Files

### 1. **attacks.py**
- **Purpose**: Attack simulation functions for robustness testing
- **Functions**:
  - `add_gaussian_noise()`: Add random noise to images
  - `jpeg_compression()`: Simulate JPEG compression artifacts
  - `gaussian_blur()`: Apply blurring
  - `random_crop()`: Crop and resize images
  - `brightness_adjustment()`: Change brightness
  - `contrast_adjustment()`: Modify contrast
  - `apply_random_attack()`: Randomly select and apply one attack
  - `apply_combined_attacks()`: Apply multiple attacks sequentially

### 2. **dataset.py**
- **Purpose**: Data loading and preprocessing
- **Key Classes**:
  - `WatermarkDataset`: PyTorch Dataset for loading images and watermarks
- **Functions**:
  - `create_dataloaders()`: Create train/validation data loaders
  - `create_watermark_images()`: Generate random watermark images
- **Features**:
  - Handles any image format (RGB, grayscale)
  - Automatic resizing to 256×256
  - Data augmentation (random flip)
  - Watermark binarization

### 3. **models/encoder.py**
- **Purpose**: CNN encoder for watermark embedding
- **Classes**:
  - `Encoder`: Standard CNN encoder (default)
  - `EncoderDeep`: Deeper encoder with more capacity
  - `ConvBlock`: Reusable convolutional block
- **Functions**:
  - `get_encoder()`: Factory function to create encoder
- **Architecture**:
  - Input: Image (3 channels) + Watermark (1 channel) = 4 channels
  - Multiple Conv → BatchNorm → ReLU layers
  - Output: Watermarked image (3 channels) with residual connection

### 4. **models/decoder.py**
- **Purpose**: CNN decoder for watermark extraction
- **Classes**:
  - `Decoder`: Standard CNN decoder (default)
  - `DecoderDeep`: Deeper decoder with more layers
  - `DecoderWithAttention`: Decoder with attention mechanism
  - `ConvBlock`: Reusable convolutional block
- **Functions**:
  - `get_decoder()`: Factory function to create decoder
- **Architecture**:
  - Input: Watermarked image (3 channels)
  - Feature extraction layers
  - Output: Extracted watermark (1 channel, binary)

### 5. **models/__init__.py**
- **Purpose**: Package initialization for cleaner imports
- **Exports**: All encoder and decoder classes

### 6. **train.py**
- **Purpose**: Main training script
- **Key Classes**:
  - `WatermarkingSystem`: Handles complete training pipeline
- **Features**:
  - Joint encoder-decoder training
  - Combined loss (MSE for images + BCE for watermarks)
  - Attack simulation during training
  - Checkpoint saving/loading
  - TensorBoard logging
  - Learning rate scheduling
- **Command-line Arguments**:
  - `--num_epochs`: Number of training epochs
  - `--batch_size`: Batch size
  - `--learning_rate`: Learning rate
  - `--use_attacks`: Enable attack simulation
  - `--deep_encoder`: Use deep encoder architecture
  - `--decoder_arch`: Decoder architecture (standard/deep/attention)
  - Many more (see `python train.py --help`)

### 7. **evaluate.py**
- **Purpose**: Model evaluation and testing
- **Key Functions**:
  - `load_model()`: Load trained model from checkpoint
  - `evaluate_single_image()`: Test on one image
  - `evaluate_robustness()`: Test against multiple attacks
  - `calculate_psnr()`: Calculate PSNR metric
  - `calculate_ssim()`: Calculate SSIM metric
  - `calculate_ber()`: Calculate Bit Error Rate
  - `visualize_results()`: Create visualization plots
- **Command-line Arguments**:
  - `--checkpoint`: Path to model checkpoint
  - `--image`: Path to test image
  - `--watermark`: Path to watermark (optional)
  - `--robustness_test`: Run full robustness evaluation
  - `--visualize`: Show visualization

### 8. **test_setup.py**
- **Purpose**: Verify setup before training
- **Checks**:
  - Python version (≥3.8)
  - Dependencies installation
  - PyTorch and CUDA availability
  - Directory structure
  - Dataset loading
  - Model creation
- **Usage**: `python test_setup.py`

## Configuration Files

### 9. **requirements.txt**
- **Purpose**: Python dependencies
- **Packages**:
  - torch (≥2.0.0): Deep learning framework
  - torchvision (≥0.15.0): Image transformations
  - numpy (≥1.24.0): Numerical operations
  - opencv-python (≥4.8.0): Image processing
  - Pillow (≥10.0.0): Image I/O
  - matplotlib (≥3.7.0): Visualization
  - scikit-image (≥0.21.0): Image quality metrics
  - tqdm (≥4.65.0): Progress bars

### 10. **.gitignore**
- **Purpose**: Specify files to ignore in Git
- **Ignores**:
  - Python cache files (`__pycache__/`, `*.pyc`)
  - Virtual environments
  - Data files (images, watermarks)
  - Model checkpoints
  - Logs and results
  - IDE files
  - OS-specific files

## Documentation Files

### 11. **README.md**
- **Purpose**: Main project documentation
- **Sections**:
  - Features overview
  - Project structure
  - Requirements and installation
  - Dataset setup instructions
  - Training guide
  - Evaluation guide
  - Architecture description
  - Metrics explanation
  - Expected results
  - Troubleshooting
  - Applications
  - Authors and acknowledgments

### 12. **QUICK_START.md**
- **Purpose**: Quick start guide for beginners
- **Sections**:
  - Step-by-step setup
  - Dataset preparation
  - Training commands
  - Evaluation commands
  - Common issues and solutions
  - Minimal test setup
  - Command cheatsheet

### 13. **PROJECT_FILES.md** (this file)
- **Purpose**: Complete overview of all project files
- **Contents**: Description of each file and its purpose

## Directory Structure

```
Image Watermarking for Copyright Protection/
├── data/                          # Data directory
│   ├── images/                   # Training images (DIV2K dataset)
│   └── watermarks/               # Watermark images
├── models/                       # Model definitions and checkpoints
│   ├── __init__.py              # Package initialization
│   ├── encoder.py               # Encoder architectures
│   ├── decoder.py               # Decoder architectures
│   ├── best_model.pth          # Best model checkpoint (created during training)
│   └── latest_checkpoint.pth   # Latest checkpoint (created during training)
├── logs/                         # TensorBoard logs (created during training)
├── results/                      # Evaluation results (created during evaluation)
├── attacks.py                    # Attack functions
├── dataset.py                    # Data loading
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── test_setup.py                 # Setup verification
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore file
├── README.md                     # Main documentation
├── QUICK_START.md               # Quick start guide
└── PROJECT_FILES.md             # This file
```

## Usage Flow

### 1. Setup Phase
```bash
pip install -r requirements.txt    # Install dependencies
python test_setup.py               # Verify setup
```

### 2. Data Preparation
- Download DIV2K dataset
- Place images in `data/images/`
- (Optional) Place watermarks in `data/watermarks/`

### 3. Training Phase
```bash
python train.py --num_epochs 100 --batch_size 8 --use_attacks
```
**Creates**:
- `models/best_model.pth`: Best performing model
- `models/latest_checkpoint.pth`: Latest checkpoint
- `models/config.json`: Training configuration
- `logs/`: TensorBoard event files

### 4. Evaluation Phase
```bash
python evaluate.py --checkpoint models/best_model.pth --image test.jpg --robustness_test
```
**Creates**:
- `results/`: Visualization images and results

### 5. Monitoring (Optional)
```bash
tensorboard --logdir=logs
```
View training progress at http://localhost:6006

## Key Design Decisions

### 1. **Modular Architecture**
- Separate files for encoder, decoder, attacks, dataset
- Easy to modify or extend individual components
- Clean imports via `models/__init__.py`

### 2. **Flexible Training**
- Support for multiple encoder/decoder architectures
- Configurable attack simulation
- Adjustable loss weights
- Checkpoint resume capability

### 3. **Comprehensive Evaluation**
- Multiple metrics (PSNR, SSIM, BER)
- Robustness testing against various attacks
- Visualization of results
- Batch evaluation support

### 4. **User-Friendly**
- Command-line interface for all scripts
- Detailed help messages (`--help`)
- Progress bars during training
- Clear error messages
- Setup verification script

### 5. **Production-Ready Features**
- TensorBoard integration for monitoring
- Checkpoint saving for resume
- Learning rate scheduling
- Gradient clipping
- Data augmentation

## File Dependencies

```
train.py
├── models/encoder.py
├── models/decoder.py
├── dataset.py
└── attacks.py

evaluate.py
├── models/encoder.py
├── models/decoder.py
└── attacks.py

test_setup.py
├── dataset.py
├── models/encoder.py
└── models/decoder.py
```

## Extending the Project

### Adding a New Attack
1. Add function to `attacks.py`
2. Update `apply_random_attack()` to include new attack

### Adding a New Architecture
1. Add class to `models/encoder.py` or `models/decoder.py`
2. Update `get_encoder()` or `get_decoder()` factory function

### Adding a New Metric
1. Add calculation function to `evaluate.py`
2. Update `evaluate_single_image()` to compute metric
3. Update `visualize_results()` to display metric

### Adding a New Dataset
1. Create new dataset class in `dataset.py`
2. Modify `create_dataloaders()` to use new dataset

## File Sizes (Approximate)

- **Code Files**: ~15 KB total
  - attacks.py: ~5 KB
  - dataset.py: ~10 KB
  - train.py: ~15 KB
  - evaluate.py: ~15 KB
  - models/encoder.py: ~7 KB
  - models/decoder.py: ~9 KB
  
- **Data** (not included):
  - DIV2K dataset: ~3 GB
  - Model checkpoint: ~10-50 MB
  - Logs: ~1-10 MB

## License and Credits

- **Project Type**: Academic/Educational
- **Authors**: WH Naveen, Kalana Denusha
- **License**: Educational use only
- **Based on**: Deep learning watermarking research

---

*Last Updated: January 2026*
