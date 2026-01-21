# Installation Guide

Complete step-by-step guide to set up the Image Watermarking project.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Storage**: 5 GB free space (for code, dataset, and models)
- **Internet**: Required for downloading dependencies and dataset

### Recommended Requirements (for faster training)
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **CUDA**: Version 11.7 or higher
- **RAM**: 16 GB or more
- **Storage**: 10 GB free space

## Installation Steps

### Step 1: Check Python Version

Open a terminal/command prompt and check your Python version:

```bash
python --version
```

or

```bash
python3 --version
```

If Python is not installed or version is below 3.8, download and install from: https://www.python.org/downloads/

### Step 2: Clone or Download the Project

If using Git:
```bash
cd "C:\Users\kalan\Desktop\DIP"
cd "Image Watermarking for Copyright Protection"
```

Or simply navigate to the project folder if you already have it.

### Step 3: Create Virtual Environment (Recommended)

Creating a virtual environment keeps dependencies isolated:

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 4: Install PyTorch

PyTorch installation depends on whether you have a CUDA-capable GPU.

#### Option A: With CUDA GPU (Recommended for faster training)

1. Check if you have NVIDIA GPU:
   ```bash
   nvidia-smi
   ```
   If this works, you have an NVIDIA GPU.

2. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### Option B: CPU Only (Slower but works on all systems)

```bash
pip install torch torchvision torchaudio
```

### Step 5: Install Other Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- opencv-python
- Pillow
- matplotlib
- scikit-image
- tqdm

### Step 6: Verify Installation

Run the setup verification script:

```bash
python test_setup.py
```

This will check:
- ✓ Python version
- ✓ All dependencies
- ✓ PyTorch and CUDA
- ✓ Directory structure
- ✓ Model creation

If all checks pass, you're ready to go!

### Step 7: Download Dataset

#### Option A: DIV2K Dataset (Recommended)

1. Download DIV2K training images (~3 GB):
   - Direct link: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
   - Or visit: https://data.vision.ee.ethz.ch/cvl/DIV2K/

2. Extract the ZIP file

3. Copy all images to `data/images/` folder:
   ```bash
   # Create directory if it doesn't exist
   mkdir -p data/images
   
   # Copy images (adjust path to where you extracted the ZIP)
   cp /path/to/DIV2K_train_HR/*.png data/images/
   ```

#### Option B: Use Your Own Images

- Place at least 100 images in `data/images/` folder
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- Recommended: 500+ images for better results

### Step 8: Generate Watermarks (Optional)

The system can auto-generate random watermarks, but you can create custom ones:

```bash
python dataset.py
```

This creates 10 sample watermarks in `data/watermarks/`.

### Step 9: Run Example (Optional)

Test the system with an example (uses untrained models):

```bash
python example_usage.py
```

This creates visualization images showing how the system works.

## Verification Checklist

Before starting training, verify:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip list` should show torch, numpy, etc.)
- [ ] PyTorch working (`python -c "import torch; print(torch.__version__)"`)
- [ ] Images in `data/images/` folder (at least 100 images)
- [ ] `python test_setup.py` passes all checks

## Troubleshooting Installation

### Issue 1: "python: command not found"

**Solution**: Try `python3` instead of `python`, or install Python from python.org

### Issue 2: "pip: command not found"

**Solution**: 
```bash
python -m pip install --upgrade pip
```

### Issue 3: PyTorch installation fails

**Solution**: 
1. Try installing CPU version: `pip install torch torchvision torchaudio`
2. Visit https://pytorch.org/get-started/locally/ for platform-specific instructions

### Issue 4: "No module named 'torch'"

**Solution**: 
1. Make sure virtual environment is activated
2. Reinstall PyTorch: `pip install torch torchvision`
3. Check installation: `pip list | grep torch`

### Issue 5: CUDA not available but I have NVIDIA GPU

**Solution**:
1. Install NVIDIA drivers from nvidia.com
2. Install CUDA toolkit from nvidia.com
3. Reinstall PyTorch with CUDA version matching your CUDA toolkit

### Issue 6: Out of memory during installation

**Solution**:
1. Close other applications
2. Install packages one by one:
   ```bash
   pip install torch
   pip install torchvision
   pip install numpy
   # ... etc
   ```

### Issue 7: Permission denied errors

**Solution**:
- On Windows: Run terminal as Administrator
- On Linux/Mac: Use `sudo pip install ...` or install to user directory with `pip install --user ...`

### Issue 8: SSL certificate errors

**Solution**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package_name
```

## Platform-Specific Notes

### Windows
- Use Command Prompt or PowerShell
- Backslashes in paths: `data\images\`
- Virtual environment activation: `venv\Scripts\activate`

### Linux/Mac
- Use Terminal
- Forward slashes in paths: `data/images/`
- Virtual environment activation: `source venv/bin/activate`
- May need to use `python3` and `pip3` instead of `python` and `pip`

## Post-Installation

After successful installation:

1. **Test the setup**:
   ```bash
   python test_setup.py
   ```

2. **Run example** (optional):
   ```bash
   python example_usage.py
   ```

3. **Start training** (after downloading dataset):
   ```bash
   python train.py --num_epochs 10 --batch_size 8
   ```

4. **Monitor training**:
   ```bash
   tensorboard --logdir=logs
   ```
   Open http://localhost:6006 in browser

## Getting Help

If you encounter issues not covered here:

1. Check error messages carefully
2. Search for the error online
3. Ensure all requirements are met
4. Try reinstalling in a fresh virtual environment
5. Check PyTorch installation guide: https://pytorch.org/get-started/locally/

## Uninstallation

To remove the project and dependencies:

1. **Deactivate virtual environment**:
   ```bash
   deactivate
   ```

2. **Delete virtual environment**:
   ```bash
   rm -rf venv  # Linux/Mac
   rmdir /s venv  # Windows
   ```

3. **Delete project folder** (optional):
   ```bash
   rm -rf "Image Watermarking for Copyright Protection"
   ```

## Next Steps

After successful installation, proceed to:
- [QUICK_START.md](QUICK_START.md) for usage instructions
- [README.md](README.md) for detailed documentation
- Train your first model!

---

*Installation Guide - Last Updated: January 2026*
