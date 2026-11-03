#!/bin/bash
#SBATCH --job-name=s2_prep_mvgnn_loo
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/prep_mvgnn_loo_%j.out
#SBATCH --error=../logs/prep_mvgnn_loo_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python prepare_mvgnn_bonds_loo.py
