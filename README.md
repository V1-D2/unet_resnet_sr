# U-Net ResNet 2× Super-Resolution for AMSR2 Satellite Imagery

**A PyTorch implementation of deep learning-based super-resolution for thermal satellite data using U-Net architecture with ResNet backbone.**

---

## Overview

This repository implements a 2× super-resolution model specifically designed for AMSR2 (Advanced Microwave Scanning Radiometer 2) brightness temperature data. The model employs a U-Net architecture with ResNet-style encoder blocks to enhance the spatial resolution of single-channel thermal satellite imagery from 1024×104 to 2048×208 pixels.

### Key Features

- **U-Net architecture** with ResNet backbone for robust feature extraction
- **2× super-resolution** with cascadable design for higher scaling factors (4×, 8×)
- **Specialized loss function** combining pixel reconstruction, gradient preservation, and physical consistency
- **Optimized training pipeline** with mixed precision, gradient accumulation, and in-memory caching
- **Comprehensive evaluation** with PSNR/SSIM metrics and detailed visualizations

---

## Model Architecture

### Network Design

The model follows a **U-Net encoder-decoder architecture** enhanced with ResNet blocks for improved gradient flow and feature learning.

<div align="center">

```
Input (1×H×W)
    │
    ├─── [Bicubic 2× Upsample] ─────────────────────┐
    │                                               │
    ▼                                               │
[Encoder: ResNet Blocks]                            │
    │                                               │
    ├─ Level 1: 64 channels   ──┐                   │
    ├─ Level 2: 128 channels  ──┤                   │
    ├─ Level 3: 256 channels  ──┤ Skip Connections  │
    ├─ Level 4: 512 channels  ──┤                   │
    │                           │                   │
    ▼                           │                   │
[Decoder: Progressive Upsampling]                   │
    │                           │                   │
    └─── (Concatenate) ─────────┘                   │
    │                                               │
    ▼                                               │
[2× Upsampling Module]                              │
    │                                               │
    ▼                                               ▼
[Final Conv] ────────── (+) ◄─── [Residual Add] ────┘
    │
    ▼
Output (1×2H×2W)
```

</div>

### Encoder Architecture

The encoder consists of **ResNet blocks** organized in four hierarchical levels with progressive channel expansion:

**Initial Convolution:**
- Conv2d: 1 → 64 channels, kernel 7×7, stride 2
- BatchNorm2d + ReLU activation
- MaxPool2d: kernel 3×3, stride 2

**ResNet Layers:**
- **Layer 1:** 2 blocks, 64 → 64 channels, stride 1 (maintains resolution)
- **Layer 2:** 3 blocks, 64 → 128 channels, stride 2 (½ spatial resolution)
- **Layer 3:** 4 blocks, 128 → 256 channels, stride 2 (¼ spatial resolution)
- **Layer 4:** 2 blocks, 256 → 512 channels, stride 2 (⅛ spatial resolution)

**Total encoder blocks:** 11 ResNet blocks with skip connections at each level

#### ResNet Block Structure

Each ResNet block implements residual learning with the following components:

```
Input (C_in channels)
  │
  ├─────────────────────────────────┐
  │                                 │
  ▼                                 │
Conv 3×3 (C_in → C_out, stride=s)   │
  │                                 │
BatchNorm2d                         │
  │                                 │
ReLU + Dropout2d(p=0.1)             │
  │                                 │
Conv 3×3 (C_out → C_out, stride=1)  │
  │                                 │
BatchNorm2d                         │
  │                                 ▼
  │                          [Shortcut Path]
  │                          Conv 1×1 (if needed)
  │                          + BatchNorm2d
  │                                 │
  └──────────── (+) ◄───────────────┘
               │
            ReLU
               │
          Output
```

**Key features:**
- **Residual connections** prevent vanishing gradients
- **Batch normalization** stabilizes training
- **Dropout regularization** (10%) reduces overfitting
- **No bias terms** in convolutions (handled by BatchNorm)

### Decoder Architecture

The decoder progressively reconstructs spatial resolution through transposed convolutions and skip connections:

**Upsampling Blocks:**
- **Up4:** 512 → 256 channels, 2× upsample, concatenate with encoder skip (256ch)
- **Up3:** 512 → 128 channels, 2× upsample, concatenate with encoder skip (128ch)
- **Up2:** 256 → 64 channels, 2× upsample, concatenate with encoder skip (64ch)
- **Up1:** 128 → 64 channels, 2× upsample, concatenate with encoder skip (64ch)

Each upsampling block consists of:
```
ConvTranspose2d (2× upsample)
  → BatchNorm2d
  → ReLU
  → Conv2d 3×3 (feature refinement)
  → BatchNorm2d
  → ReLU
```

**Final Reconstruction:**
- ConvTranspose2d: 128 → 32 channels, 2× upsample
- Conv2d: 32 → 16 channels, kernel 3×3
- Conv2d: 16 → 1 channel, kernel 1×1

### Final Upsampling Module

Additional 2× upsampling with feature refinement:
```
ConvTranspose2d: 1 → 32 channels, kernel 4×4, stride 2
  → ReLU
  → Conv2d: 32 → 32 channels, kernel 3×3 (refinement)
  → ReLU
  → Conv2d: 32 → 1 channel, kernel 1×1 (projection)
```

### Residual Connection

A **global residual connection** adds bicubic-upsampled input to the network output:
```python
bicubic_upsampled = F.interpolate(input, scale_factor=2, mode='bicubic')
output = decoder_output + bicubic_upsampled
output = torch.clamp(output, -1.5, 1.5)
```

This design:
- Provides stable gradient flow from output to input
- Allows the network to learn residual corrections rather than absolute values
- Prevents temperature drift and color shift
- Improves training convergence

---

## Loss Function

The training objective combines three complementary loss components:

### Loss Formulation

```
Total Loss = α·L₁ + β·L_gradient + γ·L_physical
```

where α=1.0, β=0.1, γ=0.05

### 1. L1 Reconstruction Loss

```python
L₁ = Mean(|prediction - target|)
```

Direct pixel-wise reconstruction loss using L1 norm. More robust to outliers than L2/MSE and encourages sharp reconstructions.

### 2. Gradient Loss (Edge Preservation)

```python
∇_x = img[:, :, :-1, :] - img[:, :, 1:, :]  # Horizontal gradients
∇_y = img[:, :, :, :-1] - img[:, :, :, 1:]  # Vertical gradients

L_gradient = L₁(∇_x_pred, ∇_x_target) + L₁(∇_y_pred, ∇_y_target)
```

Preserves sharp boundaries and spatial coherence by matching gradient magnitudes. Critical for maintaining thermal fronts and cloud boundaries in satellite imagery.

### 3. Physical Consistency Loss

```python
L_physical = MSE(mean(pred), mean(target)) + 0.5·MSE(std(pred), std(target))
```

Ensures physical realism through:
- **Energy conservation:** Global temperature average must match
- **Distribution preservation:** Temperature variance must be realistic

This component prevents systematic bias and maintains radiometric calibration.

---

## Repository Structure

```
.
├── unet_resnet_model.py          # Main model definition and training
├── gpu_sequential_amsr2_optimized.py  # Optimized dataset and training utilities
├── cascaded_unet_resnet_8x.py    # Cascaded 8× super-resolution inference
├── test_unet_resnet.py           # Model evaluation and testing
├── test_amsr2_model.py           # Single-image testing script
├── utils/
│   ├── __init__.py
│   └── util_calculate_psnr_ssim.py  # PSNR/SSIM metrics calculation
├── run/
│   ├── unet_resnet.sbatch        # Training job script
│   ├── test/
│   │   └── test_enhanced.sbatch  # Testing job script
│   └── inference/
│       └── run_cascaded_unet_8x.sbatch  # 8× inference script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Training

### Data Preprocessing

**Input data:** NPZ files containing AMSR2 swath data with brightness temperature arrays

**Preprocessing pipeline:**
1. **Validation filtering:** Remove pixels outside 50-350K range
2. **Quality check:** Require ≥50% valid pixels per sample
3. **NaN filling:** Replace missing values with spatial mean
4. **Spatial processing:** Crop/pad to target size (2048×208)
5. **Normalization:** `(T - 200) / 150` to approximate [-1, 1] range

**Data augmentation:**
- Horizontal flips (50% probability)
- Vertical flips (50% probability)
- Applied to 30% of training batches

### Training Configuration

```bash
python unet_resnet_model.py \
    --npz-dir /path/to/data \
    --max-files 100 \
    --epochs 100 \
    --batch-size 8 \
    --num-workers 4 \
    --files-per-batch 5 \
    --max-swaths-per-file 1000 \
    --gradient-accumulation 2 \
    --lr 1e-4 \
    --save-dir ./models \
    --use-amp
```

**Key hyperparameters:**
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=20, eta_min=1e-6)
- **Gradient accumulation:** 2-4 steps (effective batch size: 16-32)
- **Gradient clipping:** max_norm=1.0
- **Mixed precision:** FP16 forward/backward, FP32 optimizer
- **Validation split:** 5% of files for independent evaluation

### Training Features

- **In-memory caching:** Preprocessed data cached in RAM for fast access
- **Progressive file loading:** Load files in batches to manage memory
- **Real-time metrics:** PSNR/SSIM calculated during training
- **Best model selection:** Saves model with highest validation PSNR
- **Training history:** JSON logs and visualization plots

---

## Cascaded Inference for Higher Scaling Factors

The 2× model can be applied cascadingly to achieve 4× or 8× super-resolution:

### 8× Super-Resolution Pipeline

```
Original (H×W)
    ↓ [Model 2×]
  2× (2H×2W)
    ↓ [Model 2×]
  4× (4H×4W)
    ↓ [Model 2×]
  8× (8H×8W)
```

### Patch-Based Processing

Large images are processed in overlapping patches with Gaussian-weighted blending:

```python
python cascaded_unet_resnet_8x.py \
    --npz-dir /path/to/data \
    --model-path models/best_model.pth \
    --num-samples 5 \
    --patch-size 1024,104 \
    --overlap-ratio 0.75 \
    --save-dir ./cascaded_8x_results
```

**Patch processing:**
- **Patch size:** 1024×104 pixels (model input size)
- **Overlap ratio:** 75% for smooth blending
- **Weight map:** 2D Gaussian weights for seamless stitching
- **Memory management:** Periodic cleanup every 50 patches

**Output formats:**
- Individual stage results (2×, 4×, 8×)
- Comparison visualizations with bicubic baseline
- Processing statistics (time, temperature ranges)
- Both NPZ arrays and PNG images

---

## Evaluation

### Testing on Validation Data

```bash
python test_unet_resnet.py --npz-dir /path/to/data
```

Evaluates model on 50 samples from the end of the last NPZ file, generating:
- Individual test images (exact 1:1 pixel mapping)
- Comparison grids with low-res, enhanced, and ground truth
- Difference maps highlighting reconstruction errors
- Detailed metrics (PSNR, SSIM) per sample

### Single Image Testing

```bash
python test_amsr2_model.py \
    --model-path models/best_model.pth \
    --test-file /path/to/test.npz \
    --output-dir test_results
```

Processes a single image and saves:
- Super-resolution result and ground truth
- Low-resolution input for comparison
- Absolute difference map
- Zoomed regions showing fine details
- Metrics summary

### Evaluation Metrics

**PSNR (Peak Signal-to-Noise Ratio):**
- Measures reconstruction quality in decibels
- Higher values indicate better quality
- Typical results: 39-40 dB

**SSIM (Structural Similarity Index):**
- Assesses perceptual similarity (0-1 scale)
- Considers luminance, contrast, and structure
- Typical results: 0.95-0.97

---

## Installation

### Requirements

```bash
# Core dependencies
torch==2.1.0
torchvision==0.16.0
numpy>=1.21.0,<1.24.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.60.0
Pillow>=9.0.0
psutil>=5.8.0
opencv-python-headless  # For SSIM calculation
```

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/unet-resnet-amsr2-sr.git
cd unet-resnet-amsr2-sr

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

---

## GPU Cluster Usage

### SLURM Training Script

```bash
sbatch run/unet_resnet.sbatch
```

The provided SLURM scripts configure:
- GPU allocation (Turing/A100)
- Memory management (32-64GB per GPU)
- Apptainer container execution
- Automatic package installation
- Data binding and environment setup

### Memory Optimization

For memory-constrained environments:
- Reduce `--batch-size` (default: 8)
- Increase `--gradient-accumulation` (effective batch size maintained)
- Limit `--max-swaths-per-file` (default: 1000)
- Reduce `--files-per-batch` (default: 5)

---

## Implementation Details

### Key Components

**`unet_resnet_model.py`:**
- `UNetResNet`: Main model class
- `UNetResNetEncoder`: ResNet-based encoder
- `UNetDecoder`: U-Net decoder with skip connections
- `ResNetBlock`: Basic residual block with dropout
- `SimpleLoss`: Multi-component loss function
- `MetricsCalculator`: PSNR/SSIM computation
- `SimpleTrainer`: Training loop with validation

**`gpu_sequential_amsr2_optimized.py`:**
- `OptimizedAMSR2Dataset`: In-memory cached dataset
- `AMSR2DataPreprocessor`: Data normalization and preparation
- `aggressive_cleanup()`: GPU/CPU memory management

**`cascaded_unet_resnet_8x.py`:**
- `PatchBasedSuperResolution`: Patch processing with blending
- Gaussian weight map generation
- Multi-stage cascading pipeline

### Data Format

**Input NPZ structure:**
```python
{
    'swath_array': [
        {
            'temperature': ndarray,  # Shape: (H, W), dtype: float32
            'metadata': {
                'scale_factor': float,
                'orbit_type': str,  # 'A' (ascending) or 'D' (descending)
                ...
            }
        },
        ...
    ]
}
```

**Normalized temperature range:** Approximately [-1, 1]
- Center: 200K (typical Earth temperature)
- Scale: 150K (covers ±3σ of typical range)
- Formula: `(T_kelvin - 200) / 150`

---

## Performance

### Typical Results

- **PSNR:** 35-40 dB (validation set)
- **SSIM:** 0.95-0.98 (validation set)
- **Inference time:** ~10-20s per 2048×208 image (RTX 2080 Ti)
- **Training time:** ~48 hours for 100 epochs (100 NPZ files, Turing GPU)

### Cascaded 8× Performance

- **Temperature drift:** 0.2K across three stages (0.07% error)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{unet-resnet-amsr2-sr,
  title={Advanced Deep Learning Models for Generating Super-resolution AMSR2 Imagery in Support of Sea Ice Forecasting and Analysis},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/unet-resnet-amsr2-sr}}
}
```

---

## Acknowledgments

This implementation builds upon:
- **U-Net architecture** ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597))
- **ResNet blocks** ([He et al., 2016](https://arxiv.org/abs/1512.03385))
- **PSNR/SSIM metrics** from [BasicSR](https://github.com/XPixelGroup/BasicSR)

AMSR2 data courtesy of JAXA (Japan Aerospace Exploration Agency).

---

## License

This project is released under the MIT License.
