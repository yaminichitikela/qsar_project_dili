#!/bin/bash
#SBATCH --job-name=t_graphormer_c1
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --output=../logs/gt_graphormer_cfg1_%j.out
#SBATCH --error=../logs/gt_graphormer_cfg1_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_cv_config.py Graphormer 1
