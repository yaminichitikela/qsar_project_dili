#!/bin/bash
#SBATCH --job-name=g_gin_c1
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=../logs/gnn_gin_cfg1_%j.out
#SBATCH --error=../logs/gnn_gin_cfg1_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gnn_cv_config.py GIN 1
