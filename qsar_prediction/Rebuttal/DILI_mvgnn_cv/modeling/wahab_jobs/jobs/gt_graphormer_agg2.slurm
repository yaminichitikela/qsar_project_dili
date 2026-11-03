#!/bin/bash
#SBATCH --job-name=t_graphormer_a2
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=../logs/gt_graphormer_agg2_%j.out
#SBATCH --error=../logs/gt_graphormer_agg2_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python aggregate_gt_stage2_cv.py Graphormer
