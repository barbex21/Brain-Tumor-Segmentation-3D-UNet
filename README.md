# Brain Tumor Segmentation with 3D U-Net

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains a **3D U-Net implementation** for segmenting brain tumors from multi-modal MRI scans (FLAIR, T1, T1ce, T2) using the **UPenn GBM dataset**. The model is trained with a combined BCE-Dice loss, supports Google Colab, and includes visualization utilities for predictions overlaid on FLAIR images.[file:1]

## Features
- **3D U-Net architecture** with double convolutions, instance normalization, and transpose convolutions for upsampling.
- **Custom dataset loader** (`UPennGBMDataset`) handling NIfTI files, resizing to 96x96x96, and normalization.
- **BCE-Dice loss** for improved segmentation performance on imbalanced tumor data.
- **Training loop** with validation metrics (Dice, IoU) and progress tracking.
- **Visualization function** to overlay predictions and ground truth on FLAIR slices.
- 80/67 train/validation split yielding ~1M parameters.[file:1]

## Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, NiBabel, Matplotlib, Pandas, Torchvision

Install via:
```bash
pip install torch torchvision nibabel matplotlib pandas numpy
