#!/bin/bash
#SBATCH --job-name=g_gat_2f4
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=../logs/gnn_gat_s2fold4_%j.out
#SBATCH --error=../logs/gnn_gat_s2fold4_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gnn_stage2_fold.py GAT 4
