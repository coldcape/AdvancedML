#!/bin/bash
#SBATCH --job-name=cnn_cifar10_jit
#SBATCH --output=single_GPU_cnn_cifar10_jit_output.log
#SBATCH --error=single_GPU_cnn_cifar10_jit_error.log
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hpc7

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

mkdir -p experiments
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
GPU_LOGFILE="experiments/${TIMESTAMP}_single_gpu_usage.csv"
echo "timestamp,gpu_index,gpu_name,utilization_gpu,memory_used,power_draw" > "$GPU_LOGFILE"

# Start GPU logging in background
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader -l 1 >> "$GPU_LOGFILE" &

GPU_MONITOR_PID=$!  # Capture PID of background logger

# Run your training script
python -u single_gpu_CNN.py

# Stop GPU logging when training is done
kill $GPU_MONITOR_PID
