#!/bin/bash
#SBATCH --job-name=link_prep
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=../logs/prep_%x_%j.out
#SBATCH --error=../logs/prep_%x_%j.err
# WHAT: $TARGET is mmp | caspase | dili
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python prepare_assay_oob.py "$TARGET"
