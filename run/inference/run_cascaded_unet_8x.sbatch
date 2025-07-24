#!/bin/bash
#SBATCH --job-name=cascaded_unet_8x
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00

echo "============================================"
echo "CASCADED 8X INFERENCE FOR U-NET RESNET SR"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 32GB per GPU"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to U-Net ResNet project directory
cd /home/vdidur/unet_resnet_sr

# Step 1: Verify environment
echo "============================================"
echo "Verifying environment..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
import numpy as np
import cv2
print('PyTorch:', torch.__version__)
print('NumPy:', np.__version__)
print('OpenCV:', cv2.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"

# Step 2: Create output directories
echo "============================================"
echo "Creating output directories for 8x results..."
echo "============================================"

mkdir -p ./cascaded_unet_8x_results
mkdir -p ./cascaded_unet_8x_results/arrays
mkdir -p ./cascaded_unet_8x_results/images
mkdir -p ./cascaded_unet_8x_results/visualizations

# Step 3: Check for model file
echo "============================================"
echo "Checking for trained U-Net ResNet model..."
echo "============================================"

MODEL_PATH=""
if [ -f "./models/best_model.pth" ]; then
    MODEL_PATH="./models/best_model.pth"
    echo "Found model: $MODEL_PATH"
elif [ -f "./models_enhanced/best_model.pth" ]; then
    MODEL_PATH="./models_enhanced/best_model.pth"
    echo "Found model: $MODEL_PATH"
else
    echo "ERROR: No model found!"
    echo "Checked paths:"
    echo "  - ./models/best_model.pth"
    echo "  - ./models_enhanced/best_model.pth"
    exit 1
fi

ls -lh $MODEL_PATH

# Step 4: Run cascaded 8x inference
echo "============================================"
echo "Starting cascaded 8x inference..."
echo "Processing 5 samples with single approach:"
echo "  Model 2x → Model 2x → Model 2x = 8x"
echo "============================================"

# CRITICAL: Use exact same data binding as patch_inference_8x.sbatch
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/unet_resnet_sr:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python cascaded_unet_resnet_8x.py \
    --npz-dir /data \
    --model-path $MODEL_PATH \
    --num-samples 5 \
    --save-dir ./cascaded_unet_8x_results \
    --patch-size 1024,104 \
    --overlap-ratio 0.75

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Cascaded 8x inference completed successfully!"
    echo "============================================"

    # Show results summary
    echo ""
    echo "Results summary:"
    echo "----------------"

    echo "Arrays saved:"
    ls -la ./cascaded_unet_8x_results/arrays/ | head -10

    echo ""
    echo "Images saved:"
    ls -la ./cascaded_unet_8x_results/images/ | head -10

    echo ""
    echo "Visualizations:"
    ls -la ./cascaded_unet_8x_results/*.png

    echo ""
    echo "Processing statistics:"
    if [ -f "./cascaded_unet_8x_results/processing_stats.json" ]; then
        head -20 ./cascaded_unet_8x_results/processing_stats.json
    fi

else
    echo "============================================"
    echo "ERROR: Cascaded 8x inference failed!"
    echo "Check the error log above for details"
    echo "============================================"
    exit 1
fi

echo ""
echo "============================================"
echo "Cascaded 8x Inference Finished: $(date)"
echo "============================================"

# Create archive of results
echo ""
echo "Creating compressed archive of 8x results..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf cascaded_unet_8x_results_${TIMESTAMP}.tar.gz ./cascaded_unet_8x_results/
echo "Archive created: cascaded_unet_8x_results_${TIMESTAMP}.tar.gz"
ls -lh cascaded_unet_8x_results_${TIMESTAMP}.tar.gz

# Show final GPU status
echo ""
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "All processing completed successfully!"