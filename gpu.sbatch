#!/bin/bash
#
#SBATCH --job-name=motifcaller_gpu    # Job name
#SBATCH --partition=gecko             # GPU partition to use
#SBATCH --ntasks=1                    # Number of tasks (usually 1 for deep learning jobs)
#SBATCH --cpus-per-task=16            # Number of CPU threads
#SBATCH --gres=gpu:1                 # Request 3 GPU
#SBATCH --time=2-00:00:00             # Time limit (2 days)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80  # Email notifications
#SBATCH --output=joboutput_%j.out     # Standard output
#SBATCH --error=joboutput_%j.err      # Standard error

# Load necessary modules
source /etc/profile.d/modules.sh
module load CUDA

# Activate your local virtual environment
source squigulator/myenv/bin/activate

# Run the Python script using Python 3.12
python3.12 method_1.py
