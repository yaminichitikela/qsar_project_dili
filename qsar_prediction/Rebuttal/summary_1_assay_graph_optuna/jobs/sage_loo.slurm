#!/bin/bash
#SBATCH --job-name=mmp_sage_loo
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --array=0-9
#SBATCH --output=../logs/sage_loo_%A_%a.out
#SBATCH --error=../logs/sage_loo_%A_%a.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gnn_optuna.py SAGE loo $SLURM_ARRAY_TASK_ID
