#!/bin/bash
#SBATCH --job-name=tx17_agg2
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:30:00
#SBATCH --output=../logs/agg2_%j.out
#SBATCH --error=../logs/agg2_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python -u aggregate_results.py 2
