#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar10_pmap    # Job name
#SBATCH --output=deep_fnn_cifar10_pmap_output.log   # Standard output log file
#SBATCH --error=deep_fnn_cifar10_pmap_error.log     # Standard error log file
#SBATCH --time=02:00:00                     # Maximum runtime
#SBATCH --partition=normal                  # Partition/queue
#SBATCH --nodes=1                           # Single node
#SBATCH --ntasks=1                          # One task (process)
#SBATCH --cpus-per-task=16                  # CPU cores
#SBATCH --mem=64G                           # Memory
#SBATCH --gres=gpu:2                        # Request 2 GPUs

# Allowed nodes
#SBATCH --nodelist=hpc2,hpc3,hpc4,hpc5,hpc6,hpc7,hpc8

export PATH="/cluster/datastore/anaconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using Python from: $(which python)"
python --version

# Create "experiments" directory if it doesn't exist
mkdir -p experiments

# Generate a timestamp for this run
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# GPU usage CSV
GPU_LOGFILE="experiments/${TIMESTAMP}_deep_fnn_cifar10_pmap_gpu_usage.csv"
echo "timestamp,gpu_index,gpu_name,utilization_gpu,memory_used,power_draw" > "$GPU_LOGFILE"

# Training log CSV
TRAINING_LOGFILE="experiments/${TIMESTAMP}_deep_fnn_cifar10_pmap_training_log.csv"

# Log GPU usage every 1 second to the CSV (appending noheader lines)
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader -l 1 >> "$GPU_LOGFILE" &

# Run the pmap Python script, passing the training CSV path
python -u deep_fnn_cifar10_pmap.py --log_csv "$TRAINING_LOGFILE"

# Optionally stop the background nvidia-smi
# kill %1
