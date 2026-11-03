#!/bin/bash
#SBATCH --job-name=mmp_agg_loo
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=../logs/aggregate_loo_%j.out
#SBATCH --error=../logs/aggregate_loo_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python aggregate_optuna.py loo
