#!/bin/bash
#SBATCH --job-name=test_unet_resnet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01:00:00

echo "============================================"
echo "U-Net ResNet Model Testing Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /home/vdidur/unet_resnet_sr

# Create test results directory
mkdir -p test_results test_new_results

echo "============================================"
echo "Installing required packages..."
echo "============================================"

# Install essential packages
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

echo "============================================"
echo "Verifying model and data paths..."
echo "============================================"

# Check if model exists
if [ -f "/home/vdidur/unet_resnet_sr/models/best_model.pth" ]; then
    echo "✅ Model found: /home/vdidur/unet_resnet_sr/models/best_model.pth"
    ls -lh /home/vdidur/unet_resnet_sr/models/best_model.pth
else
    echo "❌ Model not found at: /home/vdidur/unet_resnet_sr/models/best_model.pth"
    echo "Available files in models directory:"
    ls -la /home/vdidur/unet_resnet_sr/models/ 2>/dev/null || echo "Models directory not found"
    exit 1
fi

# Check if data directory exists
if [ -d "/home/vdidur/temperature_sr_project/data" ]; then
    echo "✅ Data directory found: /home/vdidur/temperature_sr_project/data"
    echo "NPZ files in data directory:"
    ls -1 /home/vdidur/temperature_sr_project/data/*.npz 2>/dev/null | wc -l | xargs echo "Found NPZ files:"
else
    echo "❌ Data directory not found: /home/vdidur/temperature_sr_project/data"
    exit 1
fi

echo "============================================"
echo "Running U-Net ResNet model test..."
echo "============================================"

# Run the test
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/unet_resnet_sr:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_unet_resnet.py --npz-dir /data

echo "============================================"
echo "U-Net ResNet Testing Finished: $(date)"
echo "============================================"

# Check results
if [ -f "./test_results/enhanced_test_results.png" ] || [ -f "./test_new_results/enhanced_test_results.png" ]; then
    echo "✅ Testing completed successfully!"
    echo "Results saved in:"
    ls -la test_results/ 2>/dev/null || echo "No test_results directory"
    ls -la test_new_results/ 2>/dev/null || echo "No test_new_results directory"

    # Show individual test results if they exist
    if [ -d "./test_results/individual_images" ]; then
        echo "Individual test images:"
        ls -la test_results/individual_images/
    fi
    if [ -d "./test_new_results/individual_images" ]; then
        echo "Individual test images (new):"
        ls -la test_new_results/individual_images/
    fi
else
    echo "❌ Testing may have failed - no output files found"
    echo "Check the log above for errors"
fi

# Show GPU status
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo "============================================"
echo "Test job completed: $(date)"
echo "=================================ß==========="