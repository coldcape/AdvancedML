#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar10_pmap    # Job name
#SBATCH --output=deep_fnn_pmap_output.log   # Standard output log file
#SBATCH --error=deep_fnn_pmap_error.log     # Standard error log file
#SBATCH --time=02:00:00                     # Maximum runtime
#SBATCH --partition=normal                  # Partition/queue
#SBATCH --nodes=1                           # Single node
#SBATCH --ntasks=1                          # One task (process)
#SBATCH --cpus-per-task=16                  # CPU cores
#SBATCH --mem=64G                           # Memory
#SBATCH --gres=gpu:2                        # Request 2 GPUs

# Remove 'module load cuda' since it's not recognized in your environment
# If you have a conda environment, you can source it here:
# source activate your_env

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 10 > gpu_usage.log &

# Run the pmap script with unbuffered output
python -u deep_FNN_cifar10_pmap.py
