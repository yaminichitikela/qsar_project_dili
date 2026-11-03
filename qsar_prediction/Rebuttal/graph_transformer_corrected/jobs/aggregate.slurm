#!/bin/bash
#SBATCH --job-name=gtc_agg
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/agg_%j.out
#SBATCH --error=../logs/agg_%j.err
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python aggregate_corrected.py
