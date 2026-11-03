#!/bin/bash
#SBATCH --job-name=s3_1to5_consensus_scaffold
#SBATCH --account=odu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=../logs/consensus_scaffold_%j.out
#SBATCH --error=../logs/consensus_scaffold_%j.err

module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python compute_consensus.py scaffold
