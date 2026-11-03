#!/bin/bash
#SBATCH --job-name=s3_up_xgb_random_repeat
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=../logs/xgb_random_repeat_%j.out
#SBATCH --error=../logs/xgb_random_repeat_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_classical.py random_repeat xgb
