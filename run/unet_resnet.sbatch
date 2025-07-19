#!/bin/bash
#SBATCH --job-name=unet_resnet_sr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=48:00:00

echo "============================================"
echo "U-Net ResNet SR Training"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/vdidur/unet_resnet_sr

# Install packages
echo "============================================"
echo "Installing packages..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --no-deps \
    torch==2.1.0 \
    torchvision==0.16.0 \
    'numpy>=1.21.0,<1.24.0' \
    tqdm \
    psutil

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user \
    matplotlib \
    scikit-learn \
    Pillow \
    opencv-python-headless

# Create directories
mkdir -p ./models ./logs ./results

# Run training
echo "============================================"
echo "Starting U-Net ResNet training..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/unet_resnet_sr:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python unet_resnet_model.py \
    --npz-dir /data \
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

echo "============================================"
echo "Training Finished: $(date)"
echo "============================================"