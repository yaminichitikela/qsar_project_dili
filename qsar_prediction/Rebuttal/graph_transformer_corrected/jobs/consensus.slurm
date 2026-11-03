#!/bin/bash
#SBATCH --job-name=gtc_cons
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/cons_%x_%j.out
#SBATCH --error=../logs/cons_%x_%j.err
# Recomputes the GT-containing consensus ensembles for one assay x protocol,
# reusing the eight published non-transformer families.
module load python3/2025.1-py312
source ~/qsar_env/bin/activate
cd "$SLURM_SUBMIT_DIR/.."
python compute_consensus_corrected.py "$ASSAY" "$PROTOCOL"
