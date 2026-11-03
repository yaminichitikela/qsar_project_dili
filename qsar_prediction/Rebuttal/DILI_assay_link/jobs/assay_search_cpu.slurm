#!/bin/bash
#SBATCH --job-name=srch_cpu
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=../logs/search_%x_%j.out
#SBATCH --error=../logs/search_%x_%j.err
# Optuna search, classical family (RF/SVM/XGB). 50 trials, CPU only.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_assay.py "$ASSAY" "$FAMILY" search
