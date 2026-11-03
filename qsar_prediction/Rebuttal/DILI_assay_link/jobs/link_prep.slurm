#!/bin/bash
#SBATCH --job-name=link_prep
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=../logs/linkprep_%j.out
#SBATCH --error=../logs/linkprep_%j.err
# Builds the DILI side: clinical pool / NCV-888 / Held-223 with the in vitro
# feature block attached, plus 5 repeated train/val splits.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python prepare_dili_link_data.py
