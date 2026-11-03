#!/bin/bash
#SBATCH --job-name=tx17_s2_16
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=../logs/setting_s2_16_%j.out
#SBATCH --error=../logs/setting_s2_16_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_one_setting.py 2 16
