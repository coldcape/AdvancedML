#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar10_single    # Job name
#SBATCH --output=deep_fnn_single_output.log   # Standard output log file
#SBATCH --error=deep_fnn_single_error.log     # Standard error log file
#SBATCH --time=02:00:00                       # Max runtime (hh:mm:ss)
#SBATCH --partition=normal                    # Partition (queue) name
#SBATCH --nodes=1                             # Run on a single node
#SBATCH --ntasks=1                            # Single task (process)
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --mem=64G                             # Memory
#SBATCH --gres=gpu:2                          # Requests 2 GPUs (but script uses only 1 effectively)

# We remove "nodelist=hpc7" and "module load cuda" to avoid environment issues.

# If you have a conda environment, activate it here:
# source activate myenv

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0  # Force the script to see only GPU 0 for single-GPU training
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

# Optionally, monitor GPU usage (though script is effectively single-device).
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 10 > gpu_usage.log &

python -u deep_FNN_cifar10.py
