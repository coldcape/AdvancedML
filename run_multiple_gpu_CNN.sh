#!/bin/bash
#SBATCH --job-name=cnn_cifar10_multi
#SBATCH --output=multi_GPU_cnn_cifar10_multi_output.log
#SBATCH --error=multi_GPU_cnn_cifar10_multi_error.log
#SBATCH --time=01:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=hpc7

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

mkdir -p experiments

# GPU logging
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
GPU_LOGFILE="experiments/${TIMESTAMP}_multi_gpu_usage.csv"
echo "timestamp,gpu_index,gpu_name,utilization_gpu,memory_used,power_draw" > "$GPU_LOGFILE"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader -l 15 >> "$GPU_LOGFILE" &

GPU_MONITOR_PID=$!

# Run the multi-GPU training
python -u multiple_gpu_CNN.py

# Stop GPU monitoring
kill $GPU_MONITOR_PID
