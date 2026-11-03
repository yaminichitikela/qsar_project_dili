#!/bin/bash
#SBATCH --job-name=gtc_train
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=../logs/train_%x_u%a_%j.out
#SBATCH --error=../logs/train_%x_u%a_%j.err
# One array task per split unit. Needs ASSAY, ARCH, PROTOCOL via --export.
# Trains the three published configs and keeps the best on validation ROC-AUC.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_gt_corrected.py "$ASSAY" "$ARCH" "$PROTOCOL" "$SLURM_ARRAY_TASK_ID"
