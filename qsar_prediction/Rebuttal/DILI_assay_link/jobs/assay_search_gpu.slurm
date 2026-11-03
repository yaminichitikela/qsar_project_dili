#!/bin/bash
#SBATCH --job-name=srch_gpu
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --output=../logs/search_%x_%j.out
#SBATCH --error=../logs/search_%x_%j.err
# Optuna search, graph family. 25 trials with median pruning.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_assay.py "$ASSAY" "$FAMILY" search
