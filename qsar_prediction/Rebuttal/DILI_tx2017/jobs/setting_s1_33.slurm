#!/bin/bash
#SBATCH --job-name=tx17_s1_33
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=../logs/setting_s1_33_%j.out
#SBATCH --error=../logs/setting_s1_33_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_one_setting.py 1 33
