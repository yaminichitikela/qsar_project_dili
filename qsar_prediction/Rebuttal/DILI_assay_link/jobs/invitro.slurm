#!/bin/bash
#SBATCH --job-name=invitro
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/invitro_%j.out
#SBATCH --error=../logs/invitro_%j.err
# Out-of-bag aggregation -> results/invitro_features.csv, then re-audit.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python build_invitro_features.py
python audit_leakage.py
