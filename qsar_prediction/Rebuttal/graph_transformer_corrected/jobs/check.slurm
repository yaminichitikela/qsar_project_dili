#!/bin/bash
#SBATCH --job-name=gtc_check
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/check_%j.out
#SBATCH --error=../logs/check_%j.err
# Verifies every published split and prediction the retrain reuses.
# A non-zero exit here must stop the GPU jobs (afterok dependency).
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python check_inputs.py ${ASSAYS:-}
