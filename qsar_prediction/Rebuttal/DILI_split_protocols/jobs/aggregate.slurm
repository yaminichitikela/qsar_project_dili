#!/bin/bash
#SBATCH --job-name=dsp_agg
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/dsp_agg_%j.out
#SBATCH --error=../logs/dsp_agg_%j.err
# Runs even if some fits failed (afterany), so partial results are still summarized.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python aggregate.py
