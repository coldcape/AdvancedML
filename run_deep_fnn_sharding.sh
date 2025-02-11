#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar10_sharding    # Job name
#SBATCH --output=deep_fnn_sharding_output.log     # Standard output log file
#SBATCH --error=deep_fnn_sharding_error.log       # Standard error log file
#SBATCH --time=02:00:00                           # Maximum runtime (hh:mm:ss)
#SBATCH --partition=normal                        # Partition (queue) name
#SBATCH --nodes=1                                 # Run on a single node
#SBATCH --ntasks=1                                # Single task (process)
#SBATCH --cpus-per-task=16                        # Number of CPU cores for the task
#SBATCH --mem=64G                                 # Memory allocation
#SBATCH --gres=gpu:2                              # Request 2 GPUs
# (No node name is forced here; Slurm will choose an available node.)

# Load the CUDA module to ensure GPU drivers and libraries are available.
module load cuda

# Set up the correct Python environment.
export PATH="/cluster/datastore/anaconda3/bin:$PATH"

# Set environment variables to force JAX to use GPUs.
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Print the Python version for verification.
echo "Using Python from: $(which python)"
python --version

# Optionally, log GPU usage every 10 seconds to monitor utilization.
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 10 > gpu_usage.log &

# Run the sharding-enabled deep FNN script with unbuffered output.
python -u deep_FNN_cifar10_sharding.py
