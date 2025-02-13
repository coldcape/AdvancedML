#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar100_single   # Job name
#SBATCH --output=deep_fnn_cifar100_out.log    # Std output log file
#SBATCH --error=deep_fnn_cifar100_err.log     # Std error log file
#SBATCH --time=02:00:00                       # Max runtime (hh:mm:ss)
#SBATCH --partition=normal                    # Partition
#SBATCH --nodes=1                             # Single node
#SBATCH --ntasks=1                            # One task
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --mem=64G                             # Memory
#SBATCH --gres=gpu:1                          # Request a single GPU for single-device training
#SBATCH --nodelist=hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8  # Allowed nodes

# Remove references to specific node or module load CUDA if not available
# If you have a conda environment, you can source it here
# source activate myenv

export PATH="/cluster/datastore/anaconda3/bin:$PATH"

# Force using GPU 0 if you want to ensure single GPU usage
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

# Optionally, log GPU usage every 10 seconds
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 10 > gpu_usage.log &

# Run the single-device CIFAR-100 script with unbuffered output
python -u deep_FNN_cifar100.py
