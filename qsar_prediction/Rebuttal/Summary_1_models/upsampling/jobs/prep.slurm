#!/bin/bash
#SBATCH --job-name=mmp_up_prep
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=../logs/prep_%j.out
#SBATCH --error=../logs/prep_%j.err

# ── Environment setup (fill in once you have it) ────────────────────────────
module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python prepare_split_upsample.py
