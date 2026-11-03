#!/bin/bash
#SBATCH --job-name=hepg2_graphgps_random_repeat
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00:00
#SBATCH --output=../logs/graphgps_random_repeat_%j.out
#SBATCH --error=../logs/graphgps_random_repeat_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_random_repeat.py GraphGPS
