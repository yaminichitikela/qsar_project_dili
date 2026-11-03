#!/bin/bash
#SBATCH --job-name=dsp_cpu
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=../logs/%x_u%a_%j.out
#SBATCH --error=../logs/%x_u%a_%j.err
# One array task per split unit. Needs PROTOCOL, FAMILY, STAGE exported.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python train_protocol.py "$PROTOCOL" "$FAMILY" "$STAGE" "$SLURM_ARRAY_TASK_ID"
