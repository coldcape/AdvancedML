#!/bin/bash
#SBATCH --job-name=deep_fnn_cifar10  
#SBATCH --output=deep_fnn_output.log  
#SBATCH --error=deep_fnn_error.log    
#SBATCH --time=02:00:00               
#SBATCH --partition=normal            
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=16            
#SBATCH --mem=64G                     
#SBATCH --gres=gpu:2                  
#SBATCH --nodelist=hpc7               

# Ensure the correct Python environment is used
export PATH="/cluster/datastore/anaconda3/bin:$PATH"

# Print Python version for verification
echo "Using Python from: $(which python)"
python --version

# Run script
python deep_FNN_cifar10.py
