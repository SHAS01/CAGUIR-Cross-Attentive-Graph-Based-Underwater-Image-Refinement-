{This project paper is under review, Please Don't Use this code }
# Ori_CAGUIR

A Graph Neural Network-based underwater image enhancement system using computer vision and deep learning techniques.

## Overview

This project implements an underwater image enhancement model using graph neural networks with adjacency matrix construction based on image block similarity. The system processes underwater images to improve their quality using Sobel and Gaussian operators for feature extraction.

## Features

- Graph-based image enhancement using adjacency matrices
- Block-wise image processing with similarity calculations
- Multiple loss functions (Charbonnier, SSIM, VGG)
- Comprehensive evaluation metrics (PSNR, SSIM, FSIM, UIQM, UCIQE)
- Support for training and testing workflows
- Data preprocessing and splitting utilities

## Requirements

- Python 3.x
- PyTorch
- OpenCV (cv2)
- scikit-image
- NumPy
- PIL (Pillow)
- torchvision
- tqdm

## Installation

1. **Install dependencies:**
   ```bash
   pip install torch torchvision opencv-python scikit-image numpy pillow tqdm
   ```

2. **Clone or download the project files**

## Usage

### Training

```bash
python train.py --block_size 32 --batch_size 2 --cudaid 0 \
    --enhan_images_path /path/to/enhanced/images \
    --ori_images_path /path/to/original/images
```

### Testing

```bash
python test.py --block_size 32 --batch_size 1 --cudaid 0 \
    --enhan_images_path /path/to/test/enhanced \
    --ori_images_path /path/to/test/original \
    --epoch 100 --run 2
```

### Data Preprocessing

```bash
python image_transfer.py  # Resize and organize images
python split_data.py      # Split dataset into train/test
```

## Project Structure

```
Ori_CAGUIR/
├── train.py              # Training script
├── test.py               # Testing and evaluation script
├── utils.py              # Data loading utilities
├── loss.py               # Loss functions and metrics
├── graphmethods.py       # Graph construction methods
├── image_transfer.py     # Image preprocessing utilities
├── split_data.py         # Dataset splitting utility
├── networks/             # Neural network architectures
│   └── graph_network.py  # Graph neural network model
└── README.md             # This file
```

## Key Components

- **Graph Methods**: Builds adjacency matrices using image block similarity
- **Loss Functions**: Implements Charbonnier, SSIM, and VGG losses
- **Evaluation Metrics**: PSNR, SSIM, FSIM, UIQM, UCIQE for image quality assessment
- **Data Utilities**: Handles image loading, preprocessing, and dataset management

## Configuration

Key parameters in training/testing:
- `block_size`: Size of image blocks for graph construction (default: 32)
- `batch_size`: Training batch size (default: 2)
- `cudaid`: GPU device ID
- Image paths for original and enhanced datasets

## Output

- Enhanced images saved during testing
- Training metrics logged to file
- Comprehensive evaluation results with multiple quality metrics

