#!/bin/bash
#SBATCH --job-name=hepg2_prep_gt_random_repeat
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=../logs/prep_gt_random_repeat_%j.out
#SBATCH --error=../logs/prep_gt_random_repeat_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python prepare_gt_pe_random_repeat.py
