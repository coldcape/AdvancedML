#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar100_sharding
#SBATCH --output=deep_fnn_cifar100_sharding_output.log
#SBATCH --error=deep_fnn_cifar100_sharding_error.log
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

# Allowed nodes
#SBATCH --nodelist=hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8

# Remove 'module load cuda' to avoid environment issues
# If you have a conda env: source activate my_conda_env

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

# Optionally log GPU usage
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 10 > gpu_usage.log &

# Run the pjit-based CIFAR-100 script with unbuffered output
python -u deep_FNN_cifar100_sharding.py
