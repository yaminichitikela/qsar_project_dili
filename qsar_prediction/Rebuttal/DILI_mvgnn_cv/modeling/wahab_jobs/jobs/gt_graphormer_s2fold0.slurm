#!/bin/bash
#SBATCH --job-name=t_graphormer_2f0
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/gt_graphormer_s2fold0_%j.out
#SBATCH --error=../logs/gt_graphormer_s2fold0_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_stage2_fold.py Graphormer 0
