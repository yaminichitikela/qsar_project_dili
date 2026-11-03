#!/bin/bash
#SBATCH --job-name=s1_up_graphgps_random_repeat
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00:00
#SBATCH --output=../logs/graphgps_random_repeat_%j.out
#SBATCH --error=../logs/graphgps_random_repeat_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt.py random_repeat GraphGPS
