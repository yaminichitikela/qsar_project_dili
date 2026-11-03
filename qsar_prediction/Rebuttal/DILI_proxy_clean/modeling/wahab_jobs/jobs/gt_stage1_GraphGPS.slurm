#!/bin/bash
#SBATCH --job-name=pc_graphgps_s1
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=../logs/graphgps_s1_%j.out
#SBATCH --error=../logs/graphgps_s1_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_stage1.py GraphGPS 20
