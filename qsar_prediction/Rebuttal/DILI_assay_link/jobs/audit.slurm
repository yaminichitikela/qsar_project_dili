#!/bin/bash
#SBATCH --job-name=link_audit
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/audit_%j.out
#SBATCH --error=../logs/audit_%j.err
# Gates training: a non-zero exit here stops the dependent jobs via afterok.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python audit_leakage.py
