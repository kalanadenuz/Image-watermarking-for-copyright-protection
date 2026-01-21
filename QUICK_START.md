# Quick Start Guide

This guide will help you get started with the Image Watermarking system quickly.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Dataset

### Option A: Download DIV2K Dataset (Recommended)
1. Download DIV2K training images: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Extract the ZIP file
3. Copy all images to `data/images/` folder

### Option B: Use Your Own Images
- Place at least 100-800 images in `data/images/` folder
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp

## Step 3: Prepare Watermarks (Optional)

### Option A: Auto-generate Random Watermarks
The system will automatically generate random binary watermarks if none are found.

### Option B: Use Custom Watermarks
- Place your watermark images (logos, binary images) in `data/watermarks/` folder
- Watermarks will be converted to grayscale and binarized automatically

### Option C: Generate Sample Watermarks
```bash
python dataset.py
```

## Step 4: Start Training

### Quick Training (10 epochs for testing)
```bash
python train.py --num_epochs 10 --batch_size 8
```

### Full Training (100 epochs, with attacks for robustness)
```bash
python train.py --num_epochs 100 --batch_size 8 --use_attacks --attack_probability 0.5
```

### Monitor Training with TensorBoard
Open a new terminal and run:
```bash
tensorboard --logdir=logs
```
Then open http://localhost:6006 in your browser.

## Step 5: Evaluate the Model

### Test on a Single Image
```bash
python evaluate.py --checkpoint models/best_model.pth --image path/to/test/image.jpg --visualize
```

### Full Robustness Test
```bash
python evaluate.py --checkpoint models/best_model.pth --image path/to/test/image.jpg --robustness_test --output_dir results
```

## Common Issues and Solutions

### Issue 1: No images found in data/images
**Solution**: Download DIV2K dataset or place your own images in `data/images/` folder

### Issue 2: Out of memory error
**Solution**: Reduce batch size and image size
```bash
python train.py --batch_size 4 --image_size 128 --num_epochs 10
```

### Issue 3: CUDA out of memory
**Solution**: Use CPU instead
```bash
python train.py --num_epochs 10
# The code automatically detects if GPU is available
```

### Issue 4: Training is too slow
**Solution**: 
- Reduce number of workers: Add `--num_workers 0` to train.py
- Reduce image size: Add `--image_size 128`
- Use GPU if available (CUDA)

## Minimal Test Setup (For Quick Testing)

If you just want to test the system quickly without downloading large datasets:

1. **Create a small test dataset**:
   ```bash
   # Create directories
   mkdir -p data/images data/watermarks
   
   # Download a few sample images (you can use any images you have)
   # Place 5-10 images in data/images/
   ```

2. **Generate watermarks**:
   ```bash
   python dataset.py
   ```

3. **Quick training** (2 epochs):
   ```bash
   python train.py --num_epochs 2 --batch_size 4 --image_size 128
   ```

4. **Test the model**:
   ```bash
   python evaluate.py --checkpoint models/latest_checkpoint.pth --image data/images/0001.png --visualize
   ```

## Expected Timeline

- **Dataset Download**: 5-10 minutes (DIV2K ~3GB)
- **Installation**: 2-5 minutes
- **Quick Training** (10 epochs): 15-30 minutes (GPU) or 1-2 hours (CPU)
- **Full Training** (100 epochs): 2-4 hours (GPU) or 10-20 hours (CPU)
- **Evaluation**: 1-2 minutes per image

## Next Steps

After successful training:
1. Check training logs in TensorBoard
2. Evaluate on multiple test images
3. Test robustness against different attacks
4. Adjust hyperparameters for better results
5. Try different architectures (deep encoder, attention decoder)

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review the code comments in each file
- Adjust loss weights if one metric (PSNR or BER) is poor
- Try different watermark sizes (16x16, 32x32, 64x64)

## Useful Commands Cheatsheet

```bash
# Train with default settings
python train.py --num_epochs 100 --batch_size 8

# Train with attacks (robust)
python train.py --num_epochs 100 --batch_size 8 --use_attacks

# Train with deep architecture
python train.py --num_epochs 100 --deep_encoder --decoder_arch attention

# Resume training
python train.py --resume models/latest_checkpoint.pth

# Evaluate single image
python evaluate.py --checkpoint models/best_model.pth --image test.jpg

# Robustness test
python evaluate.py --checkpoint models/best_model.pth --image test.jpg --robustness_test

# Test dataset loading
python dataset.py

# View tensorboard
tensorboard --logdir=logs
```

Happy Watermarking! 🎨🔒
