#!/bin/bash
#SBATCH --job-name=t_graphormer_1f2
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=../logs/gt_graphormer_s1fold2_%j.out
#SBATCH --error=../logs/gt_graphormer_s1fold2_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_stage1_fold.py Graphormer 2
