#!/bin/bash
#SBATCH --job-name=s1_up_rf_random_repeat
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=../logs/rf_random_repeat_%j.out
#SBATCH --error=../logs/rf_random_repeat_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_classical.py random_repeat rf
