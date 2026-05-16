#  Deforestation Detection using U-Net on Sentinel-2 Satellite Imagery

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Pixel-level deforestation detection from Sentinel-2 multispectral satellite imagery using a custom U-Net architecture**

## Overview

This project implements a deep learning solution for automated deforestation detection using Sentinel-2 satellite data. The model processes 4 spectral bands (Red, Near-Infrared, SWIR-1, SWIR-2) at 10m resolution and outputs pixel-wise deforestation probability maps.

**Key Features:**
- Processes 4-band Sentinel-2 imagery (B04, B08, B11, B12)
-  U-Net architecture with 31M trainable parameters
-  Sliding window inference for large-scale prediction
-  GeoTIFF export for GIS integration

---

##  Results Summary

| Metric | Default (τ=0.5) | Optimized (τ=0.35) |
|--------|-----------------|---------------------|
| **Overall Accuracy** | 91.16% | 91.16% |
| **IoU (Jaccard Index)** | 1.40% | **49.92%** |
| **Precision** | 4.08% | 5.35% |
| **Recall** | 2.08% | 33.22% |
| **Deforested Area** | 1,228 ha | 14,949 ha |

>  **35.6× improvement** in IoU through threshold optimization!

---

##  Study Area

- **Location:** T44RKU (Sentinel-2 Tile)
- **Resolution:** 10m per pixel
- **Analysis Area:** 2,000 × 2,000 pixels (400 km²)
- **Detected Deforestation:** 1,228 hectares (3.07% of area)

---

##  Spectral Bands Used

| Band | Name | Wavelength | Resolution | Purpose |
|------|------|------------|------------|---------|
| **B04** | Red | ~665 nm | 10m | Vegetation analysis |
| **B08** | NIR (Near-Infrared) | ~842 nm | 10m | Vegetation health |
| **B11** | SWIR-1 | ~1610 nm | 20m (resampled) | Soil moisture, cloud detection |
| **B12** | SWIR-2 | ~2190 nm | 20m (resampled) | Bare soil, burn scars |

---

##  Model Architecture


**Architecture Details:**
- **Encoder:** 4 downsampling blocks (MaxPool, stride=2)
- **Bottleneck:** Deepest layer (8×8, 1024 channels)
- **Decoder:** 4 upsampling blocks (ConvTranspose2d, stride=2)
- **Skip Connections:** Preserve spatial details
- **Total Parameters:** 31,044,097

---

## Project Structure (Due to github limits actual bands have not been uploaded, instead GDRIVE link has been added)
deforestation-detection/
│
├── data/
│ ├── 2025/ # 2025 Sentinel-2 tiles
│ │ ├── T44RKU_B04_10m.jp2
│ │ ├── T44RKUB08_10m.jp2
│ │ ├── T44RKUB11_20m.jp2
│ │ └── T44RKUB12_20m.jp2
│ └── 2026/ # 2026 Sentinel-2 tiles
│ ├── T44RKUB04_10m.jp2
│ ├── T44RKUB08_10m.jp2
│ ├── T44RKUB11_20m.jp2
│ └── T44RKU_B12_20m.jp2
│
├── outputs/
│ ├── unet_model.pth # Trained model weights
│ ├── deforestation_probability_2026.tif
│ ├── deforestation_binary_2026.tif
│ ├── deforestation_map_2026.png
│ ├── deforestation_results_2026.txt
│ └── accuracy_metrics.txt
│
├── notebooks/
│ └── deforestation.ipynb # Main training notebook
│
├── requirements.txt
└── README.md

## Requirements
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
rasterio>=1.3.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
pillow>=9.5.0

import rasterio
import numpy as np

# Load Sentinel-2 bands
b04 = rasterio.open('data/2025/T44RKU_*_B04_10m.jp2').read(1)
b08 = rasterio.open('data/2025/T44RKU_*_B08_10m.jp2').read(1)
b11 = rasterio.open('data/2025/T44RKU_*_B11_20m.jp2').read(1)
b12 = rasterio.open('data/2025/T44RKU_*_B12_20m.jp2').read(1)

# Resample 20m bands to 10m
from rasterio.enums import Resampling

with rasterio.open('b11.jp2') as src:
    new_shape = (src.height * 2, src.width * 2)
    b11_resampled = src.read(1, out_shape=new_shape, resampling=Resampling.bilinear)

# Stack bands
image = np.stack([b04, b08, b11_resampled, b12_resampled], axis=-1)

# Normalize
for i in range(4):
    mi, ma = image[:,:,i].min(), image[:,:,i].max()
    image[:,:,i] = (image[:,:,i] - mi) / (ma - mi)

# Run the Jupyter notebook
jupyter notebook notebooks/deforestation.ipynb

Training Parameters:

Parameter	      Value
Patch size	   128×128 pixels
Number of patches	 300
Batch size	        16
Epochs	            20
Learning rate	       1e-3
Optimizer	           Adam
Loss function	    Binary Cross-Entropy
Train/Val split	     80/20

import torch
from model import UNet

# Load model
model = UNet(in_channels=4, out_channels=1)
model.load_state_dict(torch.load('outputs/unet_model.pth'))
model.eval()

# Sliding window prediction
patch_size = 128
stride = 64

for y in range(0, height - patch_size + 1, stride):
    for x in range(0, width - patch_size + 1, stride):
        patch = image[y:y+patch_size, x:x+patch_size]
        patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(patch_tensor).cpu().numpy()[0,0]
        
        result[y:y+patch_size, x:x+patch_size] += pred
        count[y:y+patch_size, x:x+patch_size] += 1

# Average overlapping predictions
result = result / (count + 1e-6)

Data Preprocessing
Band Resampling: 20m bands (B11, B12) → 10m using bilinear interpolation

Normalization: Min-max scaling per band to [0, 1]

Patch Extraction: 128×128 random patches (300 total)

Training Strategy
Train/Val Split: 80/20

Batch Size: 16

Epochs: 20

Loss: Binary Cross-Entropy

Optimizer: Adam (lr=1e-3)

Inference
Patch Size: 128×128

Stride: 64 (50% overlap)

Averaging: Mean of overlapping predictions

## Key Findings
Threshold optimization is critical - IoU improved from 1.40% to 49.92% (35.6×)

Model has high specificity (96.87%) - rarely misclassifies forest as deforestation

Recall improves dramatically from 2.08% to 33.22% at τ=0.35

Detected 1,228 hectares of deforestation (3.07% of study area)


---

##  Quick Setup Commands:

```bash
# Create repository structure
mkdir -p deforestation-detection/{data/2025,data/2026,outputs,notebooks}
cd deforestation-detection

# Create README
touch README.md

# Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
rasterio>=1.3.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
pillow>=9.5.0
EOF

# Create .gitignore
cat > .gitignore << EOF
__pycache__/
*.pyc
venv/
.env
*.jp2
*.tif
*.pth
.ipynb_checkpoints/
.DS_Store
EOF

# Initialize git
git init
git add .
git commit -m "Initial commit: Deforestation detection with U-Net"
