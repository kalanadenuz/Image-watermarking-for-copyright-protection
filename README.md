# Image Watermarking for Copyright Protection

This project implements an invisible digital image watermarking system using the Discrete Wavelet Transform (DWT) in MATLAB. The main goal is to protect image copyright by embedding a hidden watermark that is robust against common image processing attacks.

## Features
- Frequency-domain watermark embedding using DWT
- Invisible and blind watermarking technique
- Watermark extraction without original image
- Performance evaluation using PSNR, MSE, and Normalized Correlation
- Robustness testing against noise and compression attacks

## Tools & Technologies
- MATLAB
- Image Processing Toolbox
- Wavelet Toolbox

## Methodology
1. Convert input image to grayscale
2. Apply 2-D Discrete Wavelet Transform (DWT)
3. Embed watermark into selected sub-band using a scaling factor
4. Perform inverse DWT to obtain watermarked image
5. Extract watermark from the watermarked image
6. Evaluate performance using quantitative metrics

## Evaluation Metrics
- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)
- Normalized Correlation (NC)

## Applications
- Copyright protection
- Ownership verification
- Digital media security

## Project Type
Academic project for Digital Image Processing module.

## Author
Kalana Denuz
