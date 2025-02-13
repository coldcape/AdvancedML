#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar100_pmap    # Job name
#SBATCH --output=deep_fnn_cifar100_pmap_output.log   # Standard output log file
#SBATCH --error=deep_fnn_cifar100_pmap_error.log     # Standard error log file
#SBATCH --time=02:00:00                     # Maximum runtime
#SBATCH --partition=normal                  # Partition
#SBATCH --nodes=1                           # Single node
#SBATCH --ntasks=1                          # One process (task)
#SBATCH --cpus-per-task=16                  # CPU cores
#SBATCH --mem=64G                           # Memory
#SBATCH --gres=gpu:2                        # Request 2 GPUs

# Allowed nodes
#SBATCH --nodelist=hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8

# We remove "module load cuda" because your environment doesn't have it.
# If you need a conda environment, you could do:
# source activate my_conda_env

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

# Optionally track GPU usage
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 10 > gpu_usage.log &

# Run the pmap-based CIFAR-100 script
python -u deep_FNN_cifar100_pmap.py
