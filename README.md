# Image Watermarking for Copyright Protection

This project implements an AI-based invisible digital image watermarking system using deep learning (CNN encoder-decoder architecture) in PyTorch. The system embeds watermarks into images and can robustly extract them even after various attacks.

## Features
- **Deep Learning Approach**: CNN-based encoder-decoder architecture for watermark embedding and extraction
- **Invisible Watermarking**: Watermarks are imperceptible to human eyes while preserving image quality
- **Robust**: Resistant to various attacks (noise, compression, blur, cropping, brightness/contrast changes)
- **Flexible**: Works with any image format (RGB, grayscale) and any watermark (text, logo, binary image)
- **End-to-End Training**: Joint training of encoder and decoder with combined loss functions
- **Multiple Architectures**: Standard, deep, and attention-based decoder options

## Project Structure
```
├── data/
│   ├── images/          # Place DIV2K training images here (800 images)
│   └── watermarks/      # Place watermark images here (or generate random)
├── models/
│   ├── encoder.py       # CNN encoder for watermark embedding
│   └── decoder.py       # CNN decoder for watermark extraction
├── attacks.py           # Attack simulation functions
├── dataset.py           # Dataset loader and preprocessing
├── train.py            # Training script
├── evaluate.py         # Evaluation and visualization script
└── requirements.txt    # Python dependencies
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### 1. Download DIV2K Dataset
Download the DIV2K dataset (800 training images) from:
- [DIV2K Official Website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Or use the direct link: [DIV2K Train Data](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)

Extract and place images in `data/images/` folder.

### 2. Prepare Watermarks
Option 1: Place your own watermark images in `data/watermarks/` folder
Option 2: The system will automatically generate random binary watermarks if the folder is empty

To manually create sample watermarks:
```bash
python dataset.py
```

## Training

### Basic Training
```bash
python train.py --num_epochs 100 --batch_size 8
```

### Training with Attacks (Recommended for Robustness)
```bash
python train.py --num_epochs 100 --batch_size 8 --use_attacks --attack_probability 0.5
```

### Advanced Training Options
```bash
python train.py \
    --image_dir data/images \
    --watermark_dir data/watermarks \
    --num_epochs 100 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --image_size 256 \
    --watermark_size 32 \
    --image_loss_weight 1.0 \
    --watermark_loss_weight 1.0 \
    --use_attacks \
    --attack_probability 0.5 \
    --deep_encoder \
    --decoder_arch attention \
    --save_frequency 5
```

### Resume Training from Checkpoint
```bash
python train.py --resume models/latest_checkpoint.pth
```

## Evaluation

### Evaluate on a Single Image
```bash
python evaluate.py --checkpoint models/best_model.pth --image path/to/test/image.jpg --visualize
```

### Full Robustness Test (Multiple Attacks)
```bash
python evaluate.py --checkpoint models/best_model.pth --image path/to/test/image.jpg --robustness_test
```

### Evaluate with Custom Watermark
```bash
python evaluate.py --checkpoint models/best_model.pth --image path/to/test/image.jpg --watermark path/to/watermark.png
```

## Architecture

### Encoder Network
- **Input**: Cover image (3 channels) + Watermark (1 channel) = 4 channels
- **Architecture**: Multiple convolutional layers with BatchNorm and ReLU
- **Output**: Watermarked image (3 channels) with residual connection
- **Purpose**: Imperceptibly embed watermark into the cover image

### Decoder Network
- **Input**: Watermarked/attacked image (3 channels)
- **Architecture**: CNN with feature extraction and watermark reconstruction layers
- **Output**: Extracted watermark (1 channel, binary)
- **Purpose**: Robustly extract watermark even after attacks

### Loss Functions
- **Image Loss (MSE)**: Ensures watermarked image looks like original
- **Watermark Loss (BCE)**: Ensures extracted watermark matches original
- **Combined Loss**: `Total = α × Image_Loss + β × Watermark_Loss`

## Attack Simulation
The system is trained to be robust against:
- **Gaussian Noise**: Random noise addition
- **JPEG Compression**: Lossy compression artifacts
- **Gaussian Blur**: Image blurring
- **Random Cropping**: Partial image removal
- **Brightness Adjustment**: Lighting changes
- **Contrast Adjustment**: Contrast modifications
- **Combined Attacks**: Multiple attacks sequentially

## Evaluation Metrics

### Image Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality (higher is better)
  - Excellent: > 40 dB
  - Good: 30-40 dB
  - Acceptable: 20-30 dB

- **SSIM (Structural Similarity Index)**: Measures perceptual similarity (0-1, higher is better)
  - Excellent: > 0.95
  - Good: 0.9-0.95
  - Acceptable: 0.8-0.9

### Watermark Extraction Metrics
- **BER (Bit Error Rate)**: Ratio of incorrectly extracted bits (0-1, lower is better)
  - Excellent: < 0.05 (5%)
  - Good: 0.05-0.10 (5-10%)
  - Acceptable: 0.10-0.20 (10-20%)

## Expected Results
After training on DIV2K dataset for 100 epochs:
- **PSNR**: 35-42 dB (watermarked vs original)
- **SSIM**: 0.95-0.99 (imperceptible watermark)
- **BER**: 0.01-0.10 (robust extraction)
- **Training Time**: ~2-4 hours on GPU (depends on hardware)

## Tips for Better Results
1. **Dataset**: Use all 800 DIV2K images for better generalization
2. **Training**: Enable `--use_attacks` during training for robustness
3. **Batch Size**: Increase if you have more GPU memory (16 or 32)
4. **Epochs**: Train for 100-150 epochs for convergence
5. **Architecture**: Try `--deep_encoder` and `--decoder_arch attention` for better performance
6. **Loss Weights**: Adjust `--image_loss_weight` and `--watermark_loss_weight` if one metric is poor

## Troubleshooting

### Out of Memory Error
```bash
python train.py --batch_size 4 --image_size 128
```

### Poor Image Quality (Low PSNR)
Increase image loss weight:
```bash
python train.py --image_loss_weight 2.0 --watermark_loss_weight 1.0
```

### Poor Watermark Extraction (High BER)
Increase watermark loss weight:
```bash
python train.py --image_loss_weight 1.0 --watermark_loss_weight 2.0
```

### Model Not Learning
- Check that images are in `data/images/` folder
- Verify image formats are supported (.jpg, .png, etc.)
- Try lower learning rate: `--learning_rate 0.00005`

## Applications
- **Copyright Protection**: Protect ownership of digital images
- **Content Authentication**: Verify image authenticity
- **Digital Media Security**: Secure distribution of copyrighted content
- **Watermark-based Tracking**: Track unauthorized usage

## Tools & Technologies
- **PyTorch**: Deep learning framework
- **TorchVision**: Image transformations
- **NumPy**: Numerical operations
- **OpenCV**: Image processing
- **Matplotlib**: Visualization
- **Scikit-image**: Image quality metrics
- **TensorBoard**: Training visualization

## Methodology
1. **Data Loading**: Load cover images and watermarks, preprocess to 256×256 RGB
2. **Encoder Training**: Learn to embed watermark imperceptibly into images
3. **Attack Simulation**: Apply random attacks during training for robustness
4. **Decoder Training**: Learn to extract watermark from attacked images
5. **Joint Optimization**: Train encoder and decoder together with combined loss
6. **Evaluation**: Test on unseen images with various attacks

## Future Enhancements
- [ ] Support for different watermark sizes dynamically
- [ ] Text-to-watermark conversion for text watermarking
- [ ] Real-time watermark embedding and extraction
- [ ] Web interface for easy usage
- [ ] Pre-trained model weights for quick testing
- [ ] Support for video watermarking

## Project Type
Academic/Student project for Digital Image Processing module demonstrating deep learning for image watermarking.

## Authors
- WH Naveen
- Kalana Denusha

## License
This project is for educational purposes. Please ensure you have rights to any images you use for training or testing.

## Acknowledgments
- DIV2K dataset providers
- PyTorch community
- Deep learning watermarking research papers

## Citation
If you use this project, please cite:
```
@misc{image-watermarking-2026,
  title={AI-Based Image Watermarking for Copyright Protection},
  author={Naveen, WH and Denusha, Kalana},
  year={2026},
  institution={Digital Image Processing Module}
}
```
