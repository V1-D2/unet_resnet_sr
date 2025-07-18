#!/usr/bin/env python3
"""
Enhanced AMSR2 Sequential Trainer with Spatial Attention
Includes PSNR/SSIM tracking and improved monitoring

Key improvements:
1. Spatial Attention Module (CBAM-style)
2. Real-time PSNR/SSIM calculation
3. Rich progress bars with metrics
4. Optimized hyperparameters
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import glob
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import time
import json
import argparse
from pathlib import Path
import psutil
import sys
from tqdm import tqdm
import gc
import warnings
from collections import defaultdict
import cv2
from datetime import datetime
from gpu_sequential_amsr2_optimized import OptimizedAMSR2Dataset, AMSR2DataPreprocessor, aggressive_cleanup
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim



warnings.filterwarnings('ignore', category=UserWarning)

# GPU-optimized thread settings
if torch.cuda.is_available():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
else:
    torch.set_num_threads(min(8, os.cpu_count()))


# Enhanced logging with color support
class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32;21m"
    blue = "\x1b[34;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        log_color = self.FORMATS.get(record.levelno, self.grey)
        record.msg = f"{log_color}{record.msg}{self.reset}"
        return super().format(record)


# Setup simple logging without duplication
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplication
logger.handlers.clear()

# Simple console handler without colors (colors cause issues in some terminals)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# File handler
file_handler = logging.FileHandler('amsr2_enhanced_training.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False  # Prevent propagation to root logger


class ResNetBlock(nn.Module):
    """Simple ResNet block without attention"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ====== PSNR/SSIM CALCULATION MODULE ======
class MetricsCalculator:
    """Calculate PSNR and SSIM using standard implementation from utils"""

    @staticmethod
    def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
        tensor_01 = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
        tensor_255 = (tensor_01 * 255.0).clamp(0, 255)

        if tensor_255.dim() == 4:  # Batch
            numpy_array = tensor_255.detach().cpu().numpy().transpose(0, 2, 3, 1)
        elif tensor_255.dim() == 3:  # Single image
            numpy_array = tensor_255.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {tensor_255.shape}")

        return numpy_array.astype(np.uint8)

    @staticmethod
    def calculate_psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_255 = MetricsCalculator.tensor_to_uint8(pred)
        target_255 = MetricsCalculator.tensor_to_uint8(target)

        psnr_values = []
        batch_size = pred_255.shape[0]

        for i in range(batch_size):
            pred_img = pred_255[i]
            target_img = target_255[i]

            if pred_img.shape[-1] == 1:
                pred_img = pred_img.squeeze(-1)[..., np.newaxis]
                target_img = target_img.squeeze(-1)[..., np.newaxis]

            try:
                psnr = calculate_psnr(pred_img, target_img, crop_border=0, input_order='HWC')
                if np.isinf(psnr):
                    psnr = 100.0
                psnr_values.append(psnr)
            except:
                psnr_values.append(0.0)

        return np.mean(psnr_values) if psnr_values else 0.0

    @staticmethod
    def calculate_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_255 =MetricsCalculator.tensor_to_uint8(pred)
        target_255 = MetricsCalculator.tensor_to_uint8(target)

        ssim_values = []
        batch_size = pred_255.shape[0]

        for i in range(batch_size):
            pred_img = pred_255[i]
            target_img = target_255[i]

            if pred_img.shape[-1] == 1:
                pred_img = pred_img.squeeze(-1)[..., np.newaxis]
                target_img = target_img.squeeze(-1)[..., np.newaxis]

            try:
                ssim = calculate_ssim(pred_img, target_img, crop_border=0, input_order='HWC')
                ssim = np.clip(ssim, 0.0, 1.0)
                ssim_values.append(ssim)
            except:
                ssim_values.append(0.0)

        return np.mean(ssim_values) if ssim_values else 0.0


'''class MetricsCalculator:
    """Calculate PSNR and SSIM metrics using standard implementations"""
    
    @staticmethod
    def calculate_psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """Calculate PSNR using standard implementation"""
        # Convert from [-1, 1] to [0, 255]
        pred_255 = ((pred + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        target_255 = ((target + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        
        # Calculate PSNR for each image in batch
        psnr_values = []
        for i in range(pred_255.shape[0]):
            psnr = calculate_psnr(
                pred_255[i, 0],  # Remove channel dimension
                target_255[i, 0],
                crop_border=0,
                input_order='HWC' if pred_255[i, 0].ndim == 2 else 'CHW',
                test_y_channel=False
            )
            psnr_values.append(psnr)
        
        return torch.tensor(np.mean(psnr_values))
    
    @staticmethod
    def calculate_ssim_torch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Calculate SSIM using standard implementation"""
        # Convert from [-1, 1] to [0, 255]
        pred_255 = ((pred + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        target_255 = ((target + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        
        # Calculate SSIM for each image in batch
        ssim_values = []
        for i in range(pred_255.shape[0]):
            # Add channel dimension if needed
            pred_img = pred_255[i, 0, :, :, np.newaxis]
            target_img = target_255[i, 0, :, :, np.newaxis]
            
            ssim = calculate_ssim(
                pred_img,
                target_img,
                crop_border=0,
                input_order='HWC',
                test_y_channel=False
            )
            ssim_values.append(ssim)
        
        return torch.tensor(np.mean(ssim_values))
    
    @staticmethod
    def denormalize_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
        """Convert from normalized [-1, 1] to [0, 1] range for metrics"""
        return (tensor + 1.0) / 2.0'''


# ====== ENHANCED MODEL ARCHITECTURE ======
# ЗАМЕНИТЬ EnhancedUNetResNetEncoder НА:
class UNetResNetEncoder(nn.Module):
    """Simple U-Net encoder with ResNet blocks"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Simple ResNet layers without attention
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # Reduced from 3 blocks
        self.layer2 = self._make_layer(64, 128, 3, stride=2)  # Reduced from 4 blocks
        self.layer3 = self._make_layer(128, 256, 4, stride=2)  # Reduced from 6 blocks
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # Reduced from 3 blocks

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = []
        # Use simple ResNetBlock instead of AttentionResNetBlock
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return x, features


# ЗАМЕНИТЬ EnhancedUNetDecoder НА:
class UNetDecoder(nn.Module):
    """Simple U-Net decoder without attention"""

    def __init__(self, out_channels: int = 1):
        super().__init__()

        self.up4 = self._make_upconv_block(512, 256)
        self.up3 = self._make_upconv_block(256 + 256, 128)
        self.up2 = self._make_upconv_block(128 + 128, 64)
        self.up1 = self._make_upconv_block(64 + 64, 64)

        self.final_up = nn.ConvTranspose2d(64 + 64, 32, 2, 2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1)
        )

    def _make_upconv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        # Simple skip connections without attention
        # Up4
        x = self.up4(x)
        if x.shape[2] != skip_features[3].shape[2] or x.shape[3] != skip_features[3].shape[3]:
            diff_h = skip_features[3].shape[2] - x.shape[2]
            diff_w = skip_features[3].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[3]], dim=1)

        # Up3
        x = self.up3(x)
        if x.shape[2] != skip_features[2].shape[2] or x.shape[3] != skip_features[2].shape[3]:
            diff_h = skip_features[2].shape[2] - x.shape[2]
            diff_w = skip_features[2].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[2]], dim=1)

        # Up2
        x = self.up2(x)
        if x.shape[2] != skip_features[1].shape[2] or x.shape[3] != skip_features[1].shape[3]:
            diff_h = skip_features[1].shape[2] - x.shape[2]
            diff_w = skip_features[1].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[1]], dim=1)

        # Up1
        x = self.up1(x)
        if x.shape[2] != skip_features[0].shape[2] or x.shape[3] != skip_features[0].shape[3]:
            diff_h = skip_features[0].shape[2] - x.shape[2]
            diff_w = skip_features[0].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[0]], dim=1)

        x = self.final_up(x)
        x = self.final_conv(x)
        return x


# ЗАМЕНИТЬ EnhancedUNetWithAttention НА:
class UNetResNet(nn.Module):
    """Simple U-Net with ResNet encoder"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = UNetResNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

        # Simple upsampling for 2x
        if scale_factor == 2:
            self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(out_channels, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, 1)
            )
        else:
            self.upsampling = nn.Identity()

    def forward(self, x):
        # Store input for residual connection
        input_upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        # Main forward pass
        encoded, skip_features = self.encoder(x)
        decoded = self.decoder(encoded, skip_features)
        output = self.upsampling(decoded)

        # Residual connection
        output = output + input_upsampled

        # Clamp output
        output = torch.clamp(output, -1.5, 1.5)

        return output


# УДАЛИТЬ класс PerceptualLoss полностью

# ЗАМЕНИТЬ EnhancedAMSR2Loss НА:
class SimpleLoss(nn.Module):
    """Simplified loss function"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta  # Gradient loss weight
        self.gamma = gamma  # Physical consistency weight

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.metrics_calc = MetricsCalculator()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Edge-aware gradient loss"""

        def compute_gradients(x):
            grad_x = x[:, :, :-1, :] - x[:, :, 1:, :]
            grad_y = x[:, :, :, :-1] - x[:, :, :, 1:]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradients(pred)
        target_grad_x, target_grad_y = compute_gradients(target)

        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def physical_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Physical consistency for brightness temperature"""
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        energy_loss = self.mse_loss(pred_mean, target_mean)

        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        distribution_loss = self.mse_loss(pred_std, target_std)

        return energy_loss + 0.5 * distribution_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        # Calculate losses
        l1_loss = self.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        phys_loss = self.physical_consistency(pred, target)

        total_loss = (self.alpha * l1_loss +
                      self.beta * grad_loss +
                      self.gamma * phys_loss)

        # Calculate metrics
        with torch.no_grad():
            pred_clamped = torch.clamp(pred, -1.0, 1.0)
            target_clamped = torch.clamp(target, -1.0, 1.0)

            psnr = self.metrics_calc.calculate_psnr_batch(pred_clamped, target_clamped)
            ssim = self.metrics_calc.calculate_ssim_batch(pred_clamped, target_clamped)

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'gradient_loss': grad_loss.item(),
            'physical_loss': phys_loss.item(),
            'total_loss': total_loss.item(),
            'psnr': psnr if isinstance(psnr, float) else psnr.item(),
            'ssim': ssim if isinstance(ssim, float) else ssim.item()
        }

# ====== TRAINING HISTORY TRACKER ======
class TrainingHistory:
    """Track and visualize training progress"""

    def __init__(self):
        self.history = defaultdict(list)
        self.best_metrics = {
            'psnr': 0,
            'ssim': 0,
            'loss': float('inf')
        }

    def update(self, metrics: dict, epoch: int):
        """Update history with new metrics"""
        for key, value in metrics.items():
            self.history[key].append(value)

        # Update best metrics
        if 'psnr' in metrics and metrics['psnr'] > self.best_metrics['psnr']:
            self.best_metrics['psnr'] = metrics['psnr']

        if 'ssim' in metrics and metrics['ssim'] > self.best_metrics['ssim']:
            self.best_metrics['ssim'] = metrics['ssim']

        if 'total_loss' in metrics and metrics['total_loss'] < self.best_metrics['loss']:
            self.best_metrics['loss'] = metrics['total_loss']

    def plot_progress(self, save_path: str):
        """Create training progress plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Loss components
        ax = axes[0, 0]
        for loss_type in ['l1_loss', 'gradient_loss', 'physical_loss']:  # Убрать 'perceptual_loss', 'ssim_loss'
            if loss_type in self.history:
                ax.plot(self.history[loss_type], label=loss_type)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True)

        # Total loss
        ax = axes[0, 1]
        if 'total_loss' in self.history:
            ax.plot(self.history['total_loss'], 'b-', linewidth=2)
            ax.axhline(y=self.best_metrics['loss'], color='r', linestyle='--',
                       label=f'Best: {self.best_metrics["loss"]:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True)

        # PSNR
        ax = axes[0, 2]
        if 'psnr' in self.history:
            ax.plot(self.history['psnr'], 'g-', linewidth=2, label='Train PSNR')
        if 'val_psnr' in self.history:
            ax.plot(self.history['val_psnr'], 'b-', linewidth=2, label='Val PSNR')
            ax.axhline(y=self.best_metrics['psnr'], color='r', linestyle='--',
                       label=f'Best Val: {self.best_metrics["psnr"]:.2f} dB')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Peak Signal-to-Noise Ratio')
        ax.legend()
        ax.grid(True)

        # SSIM
        ax = axes[1, 0]
        if 'ssim' in self.history:
            ax.plot(self.history['ssim'], 'm-', linewidth=2)
            ax.axhline(y=self.best_metrics['ssim'], color='r', linestyle='--',
                       label=f'Best: {self.best_metrics["ssim"]:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity Index')
        ax.legend()
        ax.grid(True)

        # Learning rate
        ax = axes[1, 1]
        if 'learning_rate' in self.history:
            ax.semilogy(self.history['learning_rate'], 'c-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True)

        # Best metrics summary
        ax = axes[1, 2]
        ax.text(0.1, 0.8, f"Best PSNR: {self.best_metrics['psnr']:.2f} dB", fontsize=12)
        ax.text(0.1, 0.6, f"Best SSIM: {self.best_metrics['ssim']:.4f}", fontsize=12)
        ax.text(0.1, 0.4, f"Best Loss: {self.best_metrics['loss']:.4f}", fontsize=12)
        ax.text(0.1, 0.2, f"Total Epochs: {len(self.history.get('total_loss', []))}", fontsize=12)
        ax.axis('off')
        ax.set_title('Best Metrics')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_history(self, save_path: str):
        """Save training history to JSON"""
        history_dict = dict(self.history)
        history_dict['best_metrics'] = self.best_metrics

        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)


# ====== ENHANCED TRAINER WITH RICH PROGRESS ======
# В классе EnhancedTrainer (переименовать в SimpleTrainer):
class SimpleTrainer:
    """Simple trainer with proper validation split"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 5e-4,
                 use_amp: bool = True, gradient_accumulation_steps: int = 1):

        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Simple optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Cosine scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=1,
            eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.criterion = SimpleLoss()
        self.history = TrainingHistory()
        self.best_psnr = 0
        self.best_ssim = 0

    def train_on_files(self, npz_files: List[str], preprocessor,
                      epochs: int = 50,
                      batch_size: int = 16,
                      num_workers: int = 4,
                      files_per_batch: int = 5,
                      max_swaths_per_file: int = 1000,
                      save_dir: str = "./models",
                      val_split: float = 0.05):  # Добавляем val_split
        """Training with proper validation split"""

        os.makedirs(save_dir, exist_ok=True)

        # Split files for train/validation
        num_files = len(npz_files)
        num_val = max(1, int(num_files * val_split))

        # Shuffle и split
        import random
        random.shuffle(npz_files)
        val_files = npz_files[:num_val]
        train_files = npz_files[num_val:]

        logger.info("=" * 60)
        logger.info(f"📊 Data split:")
        logger.info(f"   Training files: {len(train_files)}")
        logger.info(f"   Validation files: {len(val_files)}")
        logger.info("=" * 60)

        # Save file lists for reproducibility
        with open(os.path.join(save_dir, 'train_files.txt'), 'w') as f:
            f.write('\n'.join(train_files))
        with open(os.path.join(save_dir, 'val_files.txt'), 'w') as f:
            f.write('\n'.join(val_files))

        total_batches = (len(train_files) + files_per_batch - 1) // files_per_batch

        for epoch in range(epochs):
            epoch_start_time = time.time()
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📈 EPOCH {epoch + 1}/{epochs}")
            logger.info(f"{'=' * 60}")

            # Training phase
            self.model.train()
            epoch_metrics = defaultdict(list)

            for batch_idx in range(total_batches):
                start_idx = batch_idx * files_per_batch
                end_idx = min(start_idx + files_per_batch, len(train_files))
                batch_files = train_files[start_idx:end_idx]

                if batch_idx % 5 == 0:
                    logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

                # Load datasets
                datasets = []
                for file_path in batch_files:
                    dataset = OptimizedAMSR2Dataset(
                        file_path,
                        preprocessor,
                        degradation_scale=2,
                        augment=True,
                        max_swaths_in_memory=max_swaths_per_file
                    )
                    if len(dataset) > 0:
                        datasets.append(dataset)

                if not datasets:
                    continue

                # Create dataloader
                combined_dataset = ConcatDataset(datasets)
                dataloader = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True if num_workers > 0 else False,
                    drop_last=True
                )

                # Training loop
                progress_bar = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch + 1}/{epochs} Batch {batch_idx + 1}/{total_batches}",
                    leave=True,
                    ncols=120
                )

                for data_idx, (low_res, high_res) in enumerate(progress_bar):
                    low_res = low_res.to(self.device, non_blocking=True)
                    high_res = high_res.to(self.device, non_blocking=True)

                    # Forward pass
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(low_res)
                        loss, metrics = self.criterion(pred, high_res)
                        loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Update weights
                    if (data_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()

                        self.optimizer.zero_grad()

                    # Update metrics
                    for key, value in metrics.items():
                        epoch_metrics[key].append(value)

                    if data_idx % 5 == 0:  # Update more frequently
                        progress_bar.set_postfix({
                            'Loss': f"{metrics['total_loss']:.4f}",
                            'PSNR': f"{metrics['psnr']:.1f}dB",
                            'SSIM': f"{metrics['ssim']:.3f}",
                            'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                        })

                    if data_idx % 50 == 0:
                        torch.cuda.empty_cache()

                # Clean up
                del dataloader, combined_dataset, datasets
                aggressive_cleanup()

            # Calculate training metrics
            train_metrics = {}
            for key, values in epoch_metrics.items():
                train_metrics[f'train_{key}'] = np.mean(values)

            # Validation phase
            val_metrics = self._validate(val_files, preprocessor, batch_size, num_workers, max_swaths_per_file)

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history.update(all_metrics, epoch)

            # Step scheduler with validation loss
            self.scheduler.step()

            # Log summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1:3d} | "
                       f"Train Loss: {train_metrics['train_total_loss']:.4f} | "
                       f"Val Loss: {val_metrics['val_total_loss']:.4f} | "
                       f"Val PSNR: {val_metrics['val_psnr']:.2f} dB | "
                       f"Val SSIM: {val_metrics['val_ssim']:.4f} | "
                       f"Time: {epoch_time:.0f}s")

            # Save best model based on validation PSNR
            if val_metrics['val_psnr'] > self.best_psnr:
                self.best_psnr = val_metrics['val_psnr']
                self.best_ssim = val_metrics['val_ssim']

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_psnr': self.best_psnr,
                    'best_ssim': self.best_ssim,
                    'metrics': all_metrics,
                    'scale_factor': 2
                }

                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(checkpoint, save_path)
                logger.info(f"   💾 Saved best model! Val PSNR: {self.best_psnr:.2f} dB")

            # Save training progress
            if (epoch + 1) % 5 == 0:
                plot_path = os.path.join(save_dir, 'training_progress.png')
                self.history.plot_progress(plot_path)

                history_path = os.path.join(save_dir, 'training_history.json')
                self.history.save_history(history_path)

        # Final summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🎉 Training Completed!")
        logger.info(f"   🏆 Best Val PSNR: {self.best_psnr:.2f} dB")
        logger.info(f"   🎯 Best Val SSIM: {self.best_ssim:.4f}")
        logger.info(f"   💾 Model saved to: {save_dir}")
        logger.info(f"{'=' * 60}")

    def _validate(self, val_files: List[str], preprocessor, batch_size: int,
                  num_workers: int, max_swaths_per_file: int) -> dict:
        """Validation on separate files"""
        self.model.eval()
        val_metrics = defaultdict(list)

        # Process validation files
        datasets = []
        for file_path in val_files[:3]:  # Limit validation files
            dataset = OptimizedAMSR2Dataset(
                file_path,
                preprocessor,
                degradation_scale=2,
                augment=False,  # No augmentation for validation
                max_swaths_in_memory=min(100, max_swaths_per_file)  # Less samples for validation
            )
            if len(dataset) > 0:
                datasets.append(dataset)

        if datasets:
            combined_dataset = ConcatDataset(datasets)
            dataloader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )

            with torch.no_grad():
                for low_res, high_res in dataloader:
                    low_res = low_res.to(self.device, non_blocking=True)
                    high_res = high_res.to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(low_res)
                        loss, metrics = self.criterion(pred, high_res)

                    for key, value in metrics.items():
                        val_metrics[key].append(value)

            del dataloader, combined_dataset, datasets
            aggressive_cleanup()

        # Average validation metrics
        val_results = {}
        for key, values in val_metrics.items():
            val_results[f'val_{key}'] = np.mean(values)

        return val_results

# ====== KEEP EXISTING DATASET AND PREPROCESSOR ======
# (Using the OptimizedAMSR2Dataset and AMSR2DataPreprocessor from your original code)
from gpu_sequential_amsr2_optimized import OptimizedAMSR2Dataset, AMSR2DataPreprocessor, aggressive_cleanup


# ====== MAIN FUNCTION ======
def main():
    """Enhanced main function with better parameter defaults"""

    parser = argparse.ArgumentParser(
        description='U-Net ResNet Super-Resolution Training'
    )

    # Required
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Path to directory with NPZ files')

    # Data parameters
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to train on')
    parser.add_argument('--max-swaths-per-file', type=int, default=500,
                        help='Maximum swaths per file (default: 500)')

    # Training parameters - OPTIMIZED
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')

    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--files-per-batch', type=int, default=5,
                        help='Files to load per batch (default: 5)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    # Model parameters
    parser.add_argument('--target-height', type=int, default=2048,
                        help='Target height (default: 2048)')
    parser.add_argument('--target-width', type=int, default=208,
                        help='Target width (default: 208)')

    # Output
    parser.add_argument('--save-dir', type=str, default='./models_enhanced',
                        help='Directory to save models and results')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Validate every N epochs')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("🛰️  U-net ResNet")
    print("=" * 60)

    logger.info(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Find files
    npz_files = find_npz_files(args.npz_dir, args.max_files)
    if not npz_files:
        logger.error("❌ No NPZ files found")
        sys.exit(1)

    # Estimate data requirements
    estimated_examples = len(npz_files) * args.max_swaths_per_file
    logger.info(f"\n📊 Data Estimation:")
    logger.info(f"   Files: {len(npz_files)}")
    logger.info(f"   Estimated examples: ~{estimated_examples:,}")

    if estimated_examples < 100000:
        logger.warning("   ⚠️  For SOTA results (34+ PSNR), consider using 100k+ examples")
        logger.warning("   Current estimate may achieve 30-32 PSNR")

    # Create enhanced model
    logger.info("\n🧠 Creating U-Net ResNet Model...")
    model = UNetResNet(  # Изменить с EnhancedUNetWithAttention
        in_channels=1,
        out_channels=1,
        scale_factor=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Model size: {total_params * 4 / 1024 ** 2:.1f} MB")

    # Create preprocessor
    preprocessor = AMSR2DataPreprocessor(
        target_height=args.target_height,
        target_width=args.target_width
    )


    trainer = SimpleTrainer(  # Изменить с EnhancedTrainer
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=1e-4,  # Упростить weight_decay
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    # Start training
    logger.info("\n🚀 Starting Enhanced Training...")
    start_time = time.time()

    try:
        trainer.train_on_files(
            npz_files=npz_files,
            preprocessor=preprocessor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            files_per_batch=args.files_per_batch,
            max_swaths_per_file=args.max_swaths_per_file,
            save_dir=args.save_dir,
            val_split=0.05  # Добавить это
        )

        training_time = time.time() - start_time
        logger.info(f"\n⏱️  Total training time: {training_time / 3600:.2f} hours")

    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device.type == 'cuda':
            aggressive_cleanup()
        logger.info("✅ Program finished")


# ====== HELPER FUNCTION ======
def find_npz_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Find NPZ files"""
    if not os.path.exists(directory):
        logger.error(f"❌ Directory does not exist: {directory}")
        return []

    pattern = os.path.join(directory, "*.npz")
    all_files = glob.glob(pattern)

    if not all_files:
        logger.error(f"❌ No NPZ files found")
        return []

    all_files.sort()

    if max_files is not None and max_files > 0:
        selected_files = all_files[:max_files]
    else:
        selected_files = all_files

    logger.info(f"📁 Found {len(selected_files)} NPZ files")

    # Check total size
    total_size_gb = sum(os.path.getsize(f) / 1024 ** 3 for f in selected_files)
    logger.info(f"📊 Total data size: {total_size_gb:.2f} GB")

    return selected_files


if __name__ == "__main__":
    main()