#!/bin/bash
#SBATCH --job-name=mmp_graphormer_scaffold
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --array=0-4
#SBATCH --output=../logs/graphormer_scaffold_%A_%a.out
#SBATCH --error=../logs/graphormer_scaffold_%A_%a.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_optuna.py Graphormer scaffold $SLURM_ARRAY_TASK_ID
