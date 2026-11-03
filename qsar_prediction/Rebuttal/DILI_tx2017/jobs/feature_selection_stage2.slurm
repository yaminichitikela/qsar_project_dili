#!/bin/bash
#SBATCH --job-name=tx17_fs2
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/fs2_%j.out
#SBATCH --error=../logs/fs2_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python feature_selection.py 2
