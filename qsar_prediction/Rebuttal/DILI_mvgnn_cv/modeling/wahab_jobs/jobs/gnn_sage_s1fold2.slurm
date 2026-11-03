#!/bin/bash
#SBATCH --job-name=g_sage_1f2
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/gnn_sage_s1fold2_%j.out
#SBATCH --error=../logs/gnn_sage_s1fold2_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gnn_stage1_fold.py SAGE 2
