#!/bin/bash
#SBATCH --job-name=c_rf_s1
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/classical_rf_s1_%j.out
#SBATCH --error=../logs/classical_rf_s1_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_classical_cv.py RF 1
