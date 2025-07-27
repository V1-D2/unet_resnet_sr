#!/usr/bin/env python3
"""
Cascaded 8x Super-Resolution for AMSR2 with U-Net ResNet Model
Implements single approach: Model 2x → Model 2x → Model 2x = 8x total

Key features:
- Single cascading strategy for 8x upscaling
- Saves intermediate results (2x, 4x, 8x)
- Comparison with 8x bicubic baseline
- Processes 5 samples from last NPZ file
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import model components - ADAPTED FOR UNET RESNET
from unet_resnet_model import UNetResNet, AMSR2DataPreprocessor, MetricsCalculator
from gpu_sequential_amsr2_optimized import OptimizedAMSR2Dataset, aggressive_cleanup

try:
    from basicsr.utils import tensor2img, imwrite

    BASICSR_AVAILABLE = True
except ImportError:
    BASICSR_AVAILABLE = False
    print("BasicSR not available - using matplotlib for image saving")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cascaded_unet_8x_inference.log')
    ]
)
logger = logging.getLogger(__name__)


class PatchBasedSuperResolution:
    """Patch-based super-resolution processor with advanced blending"""

    def __init__(self, model: nn.Module, preprocessor: AMSR2DataPreprocessor,
                 device: torch.device = torch.device('cuda')):
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = device
        self.metrics_calc = MetricsCalculator()

    def create_gaussian_weight_map(self, shape: Tuple[int, int], sigma_ratio: float = 0.3) -> np.ndarray:
        """Create 2D Gaussian weight map for smooth blending"""
        h, w = shape

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Calculate Gaussian weights
        sigma_y = h * sigma_ratio
        sigma_x = w * sigma_ratio

        gaussian = np.exp(-((y - center_y) ** 2 / (2 * sigma_y ** 2) +
                            (x - center_x) ** 2 / (2 * sigma_x ** 2)))

        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        return gaussian.astype(np.float32)

    def calculate_patch_positions(self, image_shape: Tuple[int, int],
                                  patch_size: Tuple[int, int],
                                  overlap_ratio: float) -> List[Tuple[int, int, int, int]]:
        """Calculate optimal patch positions with adaptive overlap"""
        h, w = image_shape
        ph, pw = patch_size

        # Calculate stride based on overlap
        stride_h = int(ph * (1 - overlap_ratio))
        stride_w = int(pw * (1 - overlap_ratio))

        # Ensure minimum stride
        stride_h = max(1, stride_h)
        stride_w = max(1, stride_w)

        positions = []

        # Calculate positions
        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate patch boundaries
                y_end = min(y + ph, h)
                x_end = min(x + pw, w)

                # Adjust start position for edge patches to maintain size
                y_start = max(0, y_end - ph) if y_end == h else y
                x_start = max(0, x_end - pw) if x_end == w else x

                positions.append((y_start, y_end, x_start, x_end))

                # Move to next position
                if x_end >= w:
                    break
                x += stride_w

            if y_end >= h:
                break
            y += stride_h

        return positions

    def process_patch(self, patch: np.ndarray) -> np.ndarray:
        """Process single patch through model"""
        # Ensure patch is correct size by padding if necessary
        h, w = patch.shape
        target_h, target_w = 1024, 104

        if h < target_h or w < target_w:
            # Pad patch to target size
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            pad_top = pad_h // 2
            pad_left = pad_w // 2

            patch_padded = np.pad(patch,
                                  ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)),
                                  mode='reflect')
        else:
            patch_padded = patch
            pad_top = pad_left = 0

        # Convert to tensor
        patch_tensor = torch.from_numpy(patch_padded).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Run super-resolution
        with torch.no_grad():
            sr_tensor = self.model(patch_tensor)
            sr_tensor = torch.clamp(sr_tensor, -1.5, 1.5)  # Clamp as in model

        # Convert back to numpy
        sr_patch = sr_tensor.cpu().numpy()[0, 0]

        # Remove padding from output if we padded input
        if pad_top > 0 or pad_left > 0:
            out_h = h * 2  # Scale factor is 2
            out_w = w * 2
            pad_top_out = pad_top * 2
            pad_left_out = pad_left * 2
            sr_patch = sr_patch[pad_top_out:pad_top_out + out_h,
                       pad_left_out:pad_left_out + out_w]

        return sr_patch

    def patch_based_super_resolution(self, image: np.ndarray,
                                     patch_size: Tuple[int, int] = (1024, 104),
                                     overlap_ratio: float = 0.75,
                                     stage_name: str = "Stage") -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply super-resolution using patch-based approach with weighted blending

        Args:
            image: Input image (normalized)
            patch_size: Size of patches for model
            overlap_ratio: Overlap ratio (0.75 = 75% overlap)
            stage_name: Name for logging

        Returns:
            sr_image: Super-resolution result
            stats: Processing statistics
        """
        start_time = time.time()
        h, w = image.shape
        scale_factor = 2

        logger.info(f"\n{stage_name}: Processing image of size {h}×{w}")

        # Initialize output arrays
        output_h, output_w = h * scale_factor, w * scale_factor
        sr_accumulated = np.zeros((output_h, output_w), dtype=np.float64)
        weight_accumulated = np.zeros((output_h, output_w), dtype=np.float64)

        # Create Gaussian weight map for blending
        weight_map = self.create_gaussian_weight_map(
            (patch_size[0] * scale_factor, patch_size[1] * scale_factor)
        )

        # Calculate patch positions
        positions = self.calculate_patch_positions((h, w), patch_size, overlap_ratio)
        logger.info(f"{stage_name}: Created {len(positions)} patches")

        # Process patches with progress bar
        with tqdm(total=len(positions), desc=f"{stage_name} patches") as pbar:
            for i, (y_start, y_end, x_start, x_end) in enumerate(positions):
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end]

                # Process patch
                sr_patch = self.process_patch(patch)

                # Calculate output position
                out_y_start = y_start * scale_factor
                out_y_end = y_end * scale_factor
                out_x_start = x_start * scale_factor
                out_x_end = x_end * scale_factor

                # Get weight map for this patch size
                patch_h = out_y_end - out_y_start
                patch_w = out_x_end - out_x_start

                if patch_h != weight_map.shape[0] or patch_w != weight_map.shape[1]:
                    # Create custom weight map for edge patches
                    patch_weight = self.create_gaussian_weight_map((patch_h, patch_w))
                else:
                    patch_weight = weight_map[:patch_h, :patch_w]

                # Accumulate weighted result
                sr_accumulated[out_y_start:out_y_end, out_x_start:out_x_end] += sr_patch * patch_weight
                weight_accumulated[out_y_start:out_y_end, out_x_start:out_x_end] += patch_weight

                # Update progress
                pbar.update(1)

                # Periodic memory cleanup
                if i % 50 == 0:
                    torch.cuda.empty_cache()

        # Normalize by accumulated weights
        mask = weight_accumulated > 0
        sr_image = np.zeros_like(sr_accumulated)
        sr_image[mask] = sr_accumulated[mask] / weight_accumulated[mask]

        # Calculate statistics
        stats = {
            'min_val': float(np.min(sr_image)),
            'max_val': float(np.max(sr_image)),
            'mean_val': float(np.mean(sr_image)),
            'processing_time': time.time() - start_time,
            'num_patches': len(positions)
        }

        logger.info(f"{stage_name} completed in {stats['processing_time']:.2f}s")

        return sr_image, stats


def cascaded_super_resolution_8x_unet(npz_dir: str, model_path: str,
                                      num_samples: int = 5,
                                      save_dir: str = "./cascaded_unet_8x_results") -> List[Dict]:
    """
    Process samples using cascaded 8x super-resolution with U-Net ResNet

    Single approach: Model 2x → Model 2x → Model 2x = 8x

    Args:
        npz_dir: Directory containing NPZ files
        model_path: Path to trained model
        num_samples: Number of samples to process (default: 5)
        save_dir: Directory to save results

    Returns:
        List of results
    """
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    for subdir in ['arrays', 'images', 'visualizations']:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Load model - ADAPTED FOR UNET RESNET
    logger.info(f"Loading U-Net ResNet model from: {model_path}")
    model = UNetResNet(in_channels=1, out_channels=1, scale_factor=2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_psnr' in checkpoint:
            logger.info(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")
    else:
        model.load_state_dict(checkpoint)

    # Create preprocessor and patch processor
    preprocessor = AMSR2DataPreprocessor(target_height=2048, target_width=208)
    patch_processor = PatchBasedSuperResolution(model, preprocessor, device)

    # Find NPZ files - EXACT SAME AS cascaded_8x_inference.py
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    # Use last file - EXACT SAME AS cascaded_8x_inference.py
    last_file = npz_files[-1]
    logger.info(f"\nProcessing last NPZ file: {os.path.basename(last_file)}")
    logger.info(f"Processing {num_samples} samples with cascaded 8x super-resolution")

    # Load samples - EXACT SAME LOGIC AS cascaded_8x_inference.py
    results = []
    processed_count = 0

    with np.load(last_file, allow_pickle=True) as data:
        swath_array = data['swath_array']
        total_swaths = len(swath_array)
        logger.info(f"Total swaths in file: {total_swaths}")

        # Process from the end of file - EXACT SAME AS cascaded_8x_inference.py
        for idx in range(total_swaths - 1, max(0, total_swaths - 100), -1):
            if processed_count >= num_samples:
                break

            try:
                swath = swath_array[idx].item() if hasattr(swath_array[idx], 'item') else swath_array[idx]

                if 'temperature' not in swath:
                    continue

                temperature = swath['temperature'].astype(np.float32)
                metadata = swath.get('metadata', {})
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor

                # Filter invalid values
                temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)
                valid_ratio = np.sum(~np.isnan(temperature)) / temperature.size

                if valid_ratio < 0.5:
                    continue

                # Fill NaN values
                valid_mask = ~np.isnan(temperature)
                if np.sum(valid_mask) > 0:
                    mean_temp = np.mean(temperature[valid_mask])
                    temperature = np.where(np.isnan(temperature), mean_temp, temperature)

                # Store original shape and data
                original_shape = temperature.shape
                original_temp = temperature.copy()

                # Normalize (don't crop/pad - work with original size)
                normalized = (temperature - 200) / 150

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing sample {processed_count + 1}/{num_samples}")
                logger.info(f"Original shape: {original_shape}")
                logger.info(f"Swath index: {idx}")

                # === CASCADED 8X: Model 2x → Model 2x → Model 2x ===
                logger.info(f"\n--- Cascaded 8x Super-Resolution ---")

                # Stage 1: Original → 2x (first application of model)
                stage1_normalized, stage1_stats = patch_processor.patch_based_super_resolution(
                    normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="Stage 1 (2x)"
                )
                stage1_temp = stage1_normalized * 150 + 200
                logger.info(f"Stage 1: {normalized.shape} → {stage1_normalized.shape}")

                # Stage 2: 2x → 4x (second application of model)
                stage2_normalized, stage2_stats = patch_processor.patch_based_super_resolution(
                    stage1_normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="Stage 2 (4x)"
                )
                stage2_temp = stage2_normalized * 150 + 200
                logger.info(f"Stage 2: {stage1_normalized.shape} → {stage2_normalized.shape}")

                # Stage 3: 4x → 8x (third application of model)
                stage3_normalized, stage3_stats = patch_processor.patch_based_super_resolution(
                    stage2_normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="Stage 3 (8x)"
                )
                stage3_temp = stage3_normalized * 150 + 200
                logger.info(f"Stage 3: {stage2_normalized.shape} → {stage3_normalized.shape}")
                logger.info(f"Final: {original_shape} → {stage3_temp.shape} (8x)")

                # Create 8x bicubic baseline from ORIGINAL
                h, w = original_shape
                bicubic_8x = cv2.resize(original_temp, (w * 8, h * 8), interpolation=cv2.INTER_CUBIC)

                # Calculate total processing time
                total_time = stage1_stats['processing_time'] + stage2_stats['processing_time'] + stage3_stats[
                    'processing_time']

                # Store results
                result = {
                    'original': original_temp,
                    'sr_2x': stage1_temp,  # After first model application
                    'sr_4x': stage2_temp,  # After second model application
                    'sr_8x': stage3_temp,  # After third model application
                    'bicubic_8x': bicubic_8x,
                    'stats': {
                        'stage1': stage1_stats,
                        'stage2': stage2_stats,
                        'stage3': stage3_stats,
                        'total_time': total_time
                    },
                    'metadata': {
                        'original_shape': original_shape,
                        'final_shape': stage3_temp.shape,
                        'swath_index': idx,
                        'scale_factor': metadata.get('scale_factor', 1.0)
                    }
                }

                results.append(result)
                processed_count += 1

                # Log summary
                logger.info(f"\nProcessing Summary:")
                logger.info(f"  Total processing time: {total_time:.2f}s")
                logger.info(f"  Temperature range (8x): [{np.min(stage3_temp):.1f}, {np.max(stage3_temp):.1f}] K")

            except Exception as e:
                logger.warning(f"Error processing swath {idx}: {e}")
                continue

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Successfully processed {len(results)} samples with 8x SR")

    # Save all results
    save_cascaded_results(results, save_dir)

    # Create visualizations
    create_cascaded_visualization(results, save_dir)

    # Save processing statistics
    save_processing_stats(results, save_dir)

    return results


def save_cascaded_results(results: List[Dict], save_dir: str):
    """Save cascaded results including intermediate stages"""

    arrays_dir = os.path.join(save_dir, 'arrays')
    images_dir = os.path.join(save_dir, 'images')

    for i, result in enumerate(results):
        # Save arrays with all stages
        array_path = os.path.join(arrays_dir, f'sample_{i + 1:03d}.npz')
        np.savez_compressed(
            array_path,
            original=result['original'],
            sr_2x=result['sr_2x'],
            sr_4x=result['sr_4x'],
            sr_8x=result['sr_8x'],
            bicubic_8x=result['bicubic_8x'],
            stats=result['stats'],
            metadata=result['metadata']
        )

        # Save temperature images for all stages
        save_temperature_image(result['original'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_original.png'))
        save_temperature_image(result['sr_2x'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_sr_2x.png'))
        save_temperature_image(result['sr_4x'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_sr_4x.png'))
        save_temperature_image(result['sr_8x'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_sr_8x.png'))
        save_temperature_image(result['bicubic_8x'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_8x.png'))

        # Save grayscale versions if BasicSR available
        if BASICSR_AVAILABLE:
            # Percentile normalization для grayscale
            p_low, p_high = 1, 99
            temp_min, temp_max = np.percentile(result['original'], [p_low, p_high])

            # Нормализация с percentile для всех изображений
            def normalize_with_percentile(data):
                data_clipped = np.clip(data, temp_min, temp_max)
                return (data_clipped - temp_min) / (temp_max - temp_min)

            orig_norm = normalize_with_percentile(result['original'])
            sr_2x_norm = normalize_with_percentile(result['sr_2x'])
            sr_4x_norm = normalize_with_percentile(result['sr_4x'])
            sr_8x_norm = normalize_with_percentile(result['sr_8x'])
            bicubic_8x_norm = normalize_with_percentile(result['bicubic_8x'])

            # Convert to tensors
            orig_tensor = torch.from_numpy(orig_norm).unsqueeze(0).unsqueeze(0).float()
            sr_2x_tensor = torch.from_numpy(sr_2x_norm).unsqueeze(0).unsqueeze(0).float()
            sr_4x_tensor = torch.from_numpy(sr_4x_norm).unsqueeze(0).unsqueeze(0).float()
            sr_8x_tensor = torch.from_numpy(sr_8x_norm).unsqueeze(0).unsqueeze(0).float()
            bicubic_8x_tensor = torch.from_numpy(bicubic_8x_norm).unsqueeze(0).unsqueeze(0).float()

            # Convert to images
            orig_img = tensor2img([orig_tensor])
            sr_2x_img = tensor2img([sr_2x_tensor])
            sr_4x_img = tensor2img([sr_4x_tensor])
            sr_8x_img = tensor2img([sr_8x_tensor])
            bicubic_8x_img = tensor2img([bicubic_8x_tensor])

            # Save grayscale images
            imwrite(orig_img, os.path.join(images_dir, f'sample_{i + 1:03d}_original_gray.png'))
            imwrite(sr_2x_img, os.path.join(images_dir, f'sample_{i + 1:03d}_sr_2x_gray.png'))
            imwrite(sr_4x_img, os.path.join(images_dir, f'sample_{i + 1:03d}_sr_4x_gray.png'))
            imwrite(sr_8x_img, os.path.join(images_dir, f'sample_{i + 1:03d}_sr_8x_gray.png'))
            imwrite(bicubic_8x_img, os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_8x_gray.png'))


def save_temperature_image(temperature: np.ndarray, save_path: str, dpi: int = 100):
    """Save temperature array as image with exact pixel mapping"""
    import matplotlib.cm as cm
    from PIL import Image

    # Используем percentile normalization
    p_low, p_high = 1, 99
    temp_min, temp_max = np.percentile(temperature, [p_low, p_high])
    temperature_clipped = np.clip(temperature, temp_min, temp_max)
    temp_norm = (temperature_clipped - temp_min) / (temp_max - temp_min)

    # Apply turbo colormap
    turbo_cmap = cm.get_cmap('turbo')
    turbo_rgb = (turbo_cmap(temp_norm)[:, :, :3] * 255).astype(np.uint8)

    # Save as PNG
    Image.fromarray(turbo_rgb).save(save_path)


def create_cascaded_visualization(results: List[Dict], save_dir: str):
    """Create visualization showing all stages of cascaded processing"""

    n_samples = len(results)

    # Create comparison figure showing all stages
    fig, axes = plt.subplots(5, n_samples, figsize=(4 * n_samples, 20))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        # Original
        im0 = axes[0, i].imshow(result['original'], cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'Original {i + 1}\n{result["original"].shape}')
        axes[0, i].axis('off')
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046)

        # Stage 1: 2x
        im1 = axes[1, i].imshow(result['sr_2x'], cmap='turbo', aspect='auto')
        axes[1, i].set_title(f'Stage 1: 2x\n{result["sr_2x"].shape}')
        axes[1, i].axis('off')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046)

        # Stage 2: 4x
        im2 = axes[2, i].imshow(result['sr_4x'], cmap='turbo', aspect='auto')
        axes[2, i].set_title(f'Stage 2: 4x\n{result["sr_4x"].shape}')
        axes[2, i].axis('off')
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046)

        # Stage 3: 8x
        im3 = axes[3, i].imshow(result['sr_8x'], cmap='turbo', aspect='auto')
        axes[3, i].set_title(f'Stage 3: 8x\nTime: {result["stats"]["total_time"]:.1f}s')
        axes[3, i].axis('off')
        plt.colorbar(im3, ax=axes[3, i], fraction=0.046)

        # Bicubic 8x baseline
        im4 = axes[4, i].imshow(result['bicubic_8x'], cmap='turbo', aspect='auto')
        axes[4, i].set_title(f'Bicubic 8x\n{result["bicubic_8x"].shape}')
        axes[4, i].axis('off')
        plt.colorbar(im4, ax=axes[4, i], fraction=0.046)

    # Calculate average processing time
    avg_time = np.mean([r['stats']['total_time'] for r in results])

    plt.suptitle(f'Cascaded 8x Super-Resolution Results\n'
                 f'U-Net ResNet Model - {n_samples} samples\n'
                 f'Average processing time: {avg_time:.1f}s',
                 fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'comparison_8x.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison visualization: {save_path}")

    # Create detailed comparison for first sample
    if len(results) > 0:
        create_detailed_comparison(results[0], save_dir)


def create_detailed_comparison(result: Dict, save_dir: str):
    """Create detailed comparison with zoomed regions"""

    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    # Row 1: Full images
    images = [
        (result['original'], 'Original'),
        (result['sr_2x'], 'Stage 1: 2x'),
        (result['sr_4x'], 'Stage 2: 4x'),
        (result['sr_8x'], 'Stage 3: 8x'),
        (result['bicubic_8x'], 'Bicubic 8x')
    ]

    for i, (img, title) in enumerate(images):
        im = axes[0, i].imshow(img, cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'{title}\n{img.shape}')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

    # Row 2 & 3: Zoomed regions
    if result['sr_8x'].shape[0] >= 1024:
        h, w = result['sr_8x'].shape

        # Define two zoom regions
        regions = [
            (h // 4, w // 4, 512),  # Top-left region
            (3 * h // 4, 3 * w // 4, 512)  # Bottom-right region
        ]

        for row_idx, (center_y, center_x, size) in enumerate(regions):
            y1 = max(0, center_y - size // 2)
            y2 = min(h, center_y + size // 2)
            x1 = max(0, center_x - size // 2)
            x2 = min(w, center_x + size // 2)

            # Calculate zoom regions for each scale
            for col_idx, (scale, img, title) in enumerate([
                (1, result['original'], 'Original'),
                (2, result['sr_2x'], '2x'),
                (4, result['sr_4x'], '4x'),
                (8, result['sr_8x'], '8x'),
                (8, result['bicubic_8x'], 'Bicubic 8x')
            ]):
                if scale < 8:
                    # For lower scales, calculate corresponding region
                    scale_factor = 8 // scale
                    y1_scaled = y1 // scale_factor
                    y2_scaled = y2 // scale_factor
                    x1_scaled = x1 // scale_factor
                    x2_scaled = x2 // scale_factor

                    # Extract and upscale to match 8x size
                    zoom_region = img[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                    if scale < 8:
                        zoom_region = cv2.resize(zoom_region, (x2 - x1, y2 - y1),
                                                 interpolation=cv2.INTER_CUBIC)
                else:
                    # For 8x images, direct extraction
                    zoom_region = img[y1:y2, x1:x2]

                axes[row_idx + 1, col_idx].imshow(zoom_region, cmap='turbo', aspect='auto')
                axes[row_idx + 1, col_idx].set_title(f'{title} (Zoom {row_idx + 1})')
                axes[row_idx + 1, col_idx].axis('off')
    else:
        # If image too small for zooming, hide these subplots
        for row in range(1, 3):
            for col in range(5):
                axes[row, col].axis('off')

    plt.suptitle('Detailed Cascaded 8x Super-Resolution Comparison\nU-Net ResNet Model',
                 fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'detailed_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved detailed comparison: {save_path}")


def save_processing_stats(results: List[Dict], save_dir: str):
    """Save processing statistics to JSON"""

    stats = {
        'num_samples': len(results),
        'model_type': 'U-Net ResNet',
        'scale_factor': 8,
        'cascaded_stages': 3,
        'patch_size': [1024, 104],
        'overlap_ratio': 0.75,
        'samples': []
    }

    for i, result in enumerate(results):
        sample_stats = {
            'sample_id': i + 1,
            'swath_index': result['metadata']['swath_index'],
            'original_shape': list(result['metadata']['original_shape']),
            'final_shape': list(result['metadata']['final_shape']),
            'stage1_time': result['stats']['stage1']['processing_time'],
            'stage2_time': result['stats']['stage2']['processing_time'],
            'stage3_time': result['stats']['stage3']['processing_time'],
            'total_time': result['stats']['total_time'],
            'temperature_ranges': {
                'original': [float(np.min(result['original'])), float(np.max(result['original']))],
                'sr_2x': [float(np.min(result['sr_2x'])), float(np.max(result['sr_2x']))],
                'sr_4x': [float(np.min(result['sr_4x'])), float(np.max(result['sr_4x']))],
                'sr_8x': [float(np.min(result['sr_8x'])), float(np.max(result['sr_8x']))],
                'bicubic_8x': [float(np.min(result['bicubic_8x'])), float(np.max(result['bicubic_8x']))]
            }
        }
        stats['samples'].append(sample_stats)

    # Calculate averages
    avg_times = {
        'avg_stage1_time': np.mean([s['stage1_time'] for s in stats['samples']]),
        'avg_stage2_time': np.mean([s['stage2_time'] for s in stats['samples']]),
        'avg_stage3_time': np.mean([s['stage3_time'] for s in stats['samples']]),
        'avg_total_time': np.mean([s['total_time'] for s in stats['samples']])
    }
    stats['averages'] = avg_times

    # Save to JSON
    json_path = os.path.join(save_dir, 'processing_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved processing statistics: {json_path}")


def main():
    """Main function for cascaded U-Net ResNet 8x inference"""
    import argparse

    parser = argparse.ArgumentParser(description='Cascaded 8x Super-Resolution for U-Net ResNet Model')
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to process (default: 5)')
    parser.add_argument('--save-dir', type=str, default='./cascaded_unet_8x_results',
                        help='Directory to save results')
    parser.add_argument('--patch-size', type=str, default='1024,104',
                        help='Patch size as "height,width" (default: 1024,104)')
    parser.add_argument('--overlap-ratio', type=float, default=0.75,
                        help='Overlap ratio for patches (default: 0.75)')

    args = parser.parse_args()

    # Parse patch size
    try:
        patch_h, patch_w = map(int, args.patch_size.split(','))
        patch_size = (patch_h, patch_w)
    except:
        logger.error("Invalid patch size format. Use 'height,width' format.")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("CASCADED 8X SUPER-RESOLUTION - U-NET RESNET MODEL")
    logger.info("=" * 80)
    logger.info(f"NPZ directory: {args.npz_dir}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Patch size: {patch_size}")
    logger.info(f"Overlap ratio: {args.overlap_ratio}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info("=" * 80)

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Process samples
    try:
        results = cascaded_super_resolution_8x_unet(
            npz_dir=args.npz_dir,
            model_path=args.model_path,
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )

        logger.info("\n" + "=" * 80)
        logger.info("CASCADED 8X PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Processed {len(results)} samples with 8x upscaling")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("Files created:")
        logger.info(f"  - Arrays: {args.save_dir}/arrays/")
        logger.info(f"  - Images: {args.save_dir}/images/")
        logger.info(f"  - Visualizations: {args.save_dir}/*.png")
        logger.info(f"  - Statistics: {args.save_dir}/processing_stats.json")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            aggressive_cleanup()


if __name__ == "__main__":
    main()