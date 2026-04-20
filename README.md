# WatermarkShield

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/Model-PyTorch-ee4c2c.svg)](https://pytorch.org/)

WatermarkShield is a Digital Image Processing (DIP) + ML system for **visible watermark detection** in images.
It combines handcrafted signal analysis with a deep model to classify images as:

- **Watermarked (WM)**
- **Clean / No Watermark (NWM)**

Built and maintained by **Kalana DenuZ**.

## Live App

[Open WatermarkShield Live](https://watermarkshield.streamlit.app/)

---

## Highlights

- Modern Streamlit dashboard for interactive image analysis
- Step-by-step DIP visualization pipeline
- Detection support for:
  - Text/logo-like visible watermarks
  - Tiled watermark patterns
  - Color-tint overlay watermark patterns
- Deep model inference with local cached weights (ConvNeXt-based)
- Dataset-aware workflow with train/test/demo folder structure

---

## Demo Screenshots

### Main Interface

![Main Dashboard](Screenshots/Screenshot%202026-04-20%20185519.png)

### Pipeline Overview

![Pipeline Overview](Screenshots/Screenshot%202026-04-20%20190134.png)

### Step 1 - Image Acquisition & Preprocessing

![Step 1](Screenshots/Screenshot%202026-04-20%20190134.png)

### Step 2 - Visible Watermark Signal Analysis

![Step 2](Screenshots/Screenshot%202026-04-20%20190142.png)

### Step 3 - Color Watermark Detection

![Step 3](Screenshots/Screenshot%202026-04-20%20190148.png)

### Step 4 - Final ML Decision Layer

![Step 4](Screenshots/Screenshot%202026-04-20%20190156.png)

### Additional UI Views

![UI View 1](Screenshots/Screenshot%202026-04-20%20185531.png)
![UI View 2](Screenshots/Screenshot%202026-04-20%20185543.png)
![UI View 3](Screenshots/Screenshot%202026-04-20%20185603.png)
![UI View 4](Screenshots/Screenshot%202026-04-20%20185613.png)
![UI View 5](Screenshots/Screenshot%202026-04-20%20185635.png)
![UI View 6](Screenshots/Screenshot%202026-04-20%20185651.png)
![UI View 7](Screenshots/Screenshot%202026-04-20%20185701.png)
![UI View 8](Screenshots/Screenshot%202026-04-20%20185717.png)
![UI View 9](Screenshots/Screenshot%202026-04-20%20185756.png)
![UI View 10](Screenshots/Screenshot%202026-04-20%20190123.png)

---

## Tech Stack

### Core

- Python
- NumPy
- SciPy
- Pandas

### Computer Vision / DIP

- OpenCV
- scikit-image
- PyWavelets
- Matplotlib
- Plotly

### Machine Learning / Deep Learning

- scikit-learn
- PyTorch

### App / Runtime

- Streamlit
- Pillow
- requests
- tqdm

---

## Project Structure

```text
DIP/
├── app.py                         # Streamlit app entry point
├── watermark.py                   # DIP + watermark signal detection pipeline
├── deep_watermark.py              # Deep model loading/inference (ConvNeXt-based)
├── features.py                    # Feature extraction helpers
├── requirements.txt               # Python dependencies
├── run_app_safe.ps1               # Windows-safe app launcher
├── models/
│   └── boomb0om_convnext_tiny_watermarks_detector.pth
├── dataset/
│   ├── demo/
│   ├── train/
│   │   ├── clean/
│   │   └── watermarked/
│   └── test/
│       ├── clean/
│       └── watermarked/
├── Screenshots/
├── assets/
└── results/
```

---

## Dataset Notes

The repository includes paired clean/watermarked image sets for training, testing, and quick demo checks.

### Example demo files

- `dataset/demo/demo_0010_clean.jpg`
- `dataset/demo/demo_0010_watermarked.jpg`
- `dataset/demo/demo_0020_clean.jpg`
- `dataset/demo/demo_0020_watermarked.jpg`
- `dataset/demo/demo_0237_clean.jpg`
- `dataset/demo/demo_0237_watermarked.jpg`

### Typical test naming patterns

- `dataset/test/clean/kaggle_test_clean_*.jpeg|jpg|png`
- `dataset/test/watermarked/kaggle_test_wm_*.jpeg|jpg|png`
- `dataset/test/clean/synthetic_test_clean_*.jpg`
- `dataset/test/watermarked/synthetic_test_wm_*.png`

---

## Setup & Run

## 1) Clone

```bash
git clone https://github.com/kalanadenuz/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

## 2) Create virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Run app

### Standard

```bash
streamlit run app.py
```

### Windows safe launcher (recommended in this repo)

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app_safe.ps1
```

Open in browser:

- http://localhost:8501

---

## Model Behavior

- The deep model utility in `deep_watermark.py` uses a local weight file at:
  - `models/boomb0om_convnext_tiny_watermarks_detector.pth`
- If the weight file is missing, the code can attempt to download it from Hugging Face on first load.
- For fully offline deployment, keep the `.pth` file in `models/` before running.

---

## Usage Tips

- Use high-resolution images when possible for stronger watermark signal visibility
- Test with both `dataset/demo` samples and your own real-world images
- If classification confidence is borderline, inspect the step panels in the UI before final judgement

---

## Publish To GitHub (Recommended Checklist)

1. Confirm `.gitignore` is present (this repo includes one).
2. Keep only needed assets, model files, and source code.
3. Add repository topics: `computer-vision`, `watermark-detection`, `streamlit`, `digital-image-processing`.
4. Add this README as your project landing page.
5. Optionally add a project release tag after first stable upload.

---

## Can This Be Live On GitHub Pages?

**Short answer: not as a full Streamlit app.**

GitHub Pages is for static sites (HTML/CSS/JS). This project needs a Python runtime and server process.

### Best options for live deployment

- Streamlit Community Cloud
- Render
- Railway
- Hugging Face Spaces (with Streamlit)
- VPS (Ubuntu + Nginx + systemd)

### GitHub Pages can still be used for

- A static landing page
- Documentation site
- Demo media gallery (screenshots, architecture notes)

---

## Support & Contact

**Kalana DenuZ**

- GitHub: https://github.com/kalanadenuz/
- WhatsApp: https://wa.me/+94783761329
- Email: kalanadenuz2002@gmail.com

---

## License

If you plan to open-source publicly, add a `LICENSE` file (MIT is common for project demos).

---

## Acknowledgment

This project integrates classic DIP analysis and deep learning to improve practical visible watermark detection in mixed real-world image conditions.
