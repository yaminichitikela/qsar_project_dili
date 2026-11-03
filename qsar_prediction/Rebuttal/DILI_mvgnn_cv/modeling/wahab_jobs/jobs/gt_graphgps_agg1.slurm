#!/bin/bash
#SBATCH --job-name=t_graphgps_a1
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=../logs/gt_graphgps_agg1_%j.out
#SBATCH --error=../logs/gt_graphgps_agg1_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python aggregate_gt_stage1_cv.py GraphGPS
