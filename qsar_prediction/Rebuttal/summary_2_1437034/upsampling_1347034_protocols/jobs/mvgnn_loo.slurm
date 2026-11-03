#!/bin/bash
#SBATCH --job-name=s2o_up_mvgnn_loo
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00:00
#SBATCH --output=../logs/mvgnn_loo_%j.out
#SBATCH --error=../logs/mvgnn_loo_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_mvgnn.py loo
