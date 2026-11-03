#!/bin/bash
#SBATCH --job-name=fit_gpu
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/fit_%x_r%a_%j.out
#SBATCH --error=../logs/fit_%x_r%a_%j.err
# One array task per repeat. Submit with --array=0-20.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_assay.py "$ASSAY" "$FAMILY" fit "$SLURM_ARRAY_TASK_ID"
