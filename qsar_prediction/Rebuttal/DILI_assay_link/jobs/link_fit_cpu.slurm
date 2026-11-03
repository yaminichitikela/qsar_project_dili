#!/bin/bash
#SBATCH --job-name=lf_cpu
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/lfit_%x_r%a_%j.out
#SBATCH --error=../logs/lfit_%x_r%a_%j.err
# One array task per repeat. Submit with --array=0-4.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_dili_link.py "$ARM" "$FAMILY" "$STAGE" fit "$SLURM_ARRAY_TASK_ID"
