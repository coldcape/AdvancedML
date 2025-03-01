#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar10_single    # Job name
#SBATCH --output=deep_fnn_cifar10_single_output.log   # Standard output log file
#SBATCH --error=deep_fnn_cifar10_single_error.log     # Standard error log file
#SBATCH --time=02:00:00                       # Max runtime (hh:mm:ss)
#SBATCH --partition=normal                    # Partition (queue) name
#SBATCH --nodes=1                             # Run on a single node
#SBATCH --ntasks=1                            # Single task (process)
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --mem=64G                             # Memory
#SBATCH --gres=gpu:2                          # Requests 2 GPUs (but script uses only 1 effectively)

# Allowed nodes
#SBATCH --nodelist=hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0  # Force the script to see only GPU 0
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

# Create "experiments" directory if it doesn't exist
mkdir -p experiments

# Generate a timestamp
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# GPU usage CSV
GPU_LOGFILE="experiments/${TIMESTAMP}_deep_fnn_cifar10_single_gpu_usage.csv"
echo "timestamp,gpu_index,gpu_name,utilization_gpu,memory_used,power_draw" > "$GPU_LOGFILE"

# Training log CSV
TRAINING_LOGFILE="experiments/${TIMESTAMP}_deep_fnn_cifar10_single_training_log.csv"

# Start GPU logging every 1 second
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader -l 1 >> "$GPU_LOGFILE" &

# Run the single-GPU Python script, passing the training CSV path
python -u deep_fnn_cifar10_single.py --log_csv "$TRAINING_LOGFILE"

# Optionally kill nvidia-smi background process
# kill %1
