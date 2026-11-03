#!/bin/bash
#SBATCH --job-name=ls_cpu
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/lsearch_%x_%j.out
#SBATCH --error=../logs/lsearch_%x_%j.err
# Optuna search for one (arm, family, stage). Needs ARM, FAMILY, STAGE.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_dili_link.py "$ARM" "$FAMILY" "$STAGE" search
