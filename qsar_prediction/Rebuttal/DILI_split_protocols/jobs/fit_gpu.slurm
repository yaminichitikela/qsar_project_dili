#!/bin/bash
#SBATCH --job-name=dsp_gpu
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=../logs/%x_u%a_%j.out
#SBATCH --error=../logs/%x_u%a_%j.err
# One array task per split unit. Needs PROTOCOL, FAMILY, STAGE exported.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_protocol.py "$PROTOCOL" "$FAMILY" "$STAGE" "$SLURM_ARRAY_TASK_ID"
