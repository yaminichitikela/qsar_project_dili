#!/bin/bash
#SBATCH --job-name=s1_up_gat_random_repeat
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/gat_random_repeat_%j.out
#SBATCH --error=../logs/gat_random_repeat_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gnn.py random_repeat GAT
