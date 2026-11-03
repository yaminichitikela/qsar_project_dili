#!/bin/bash
#SBATCH --job-name=casp_graphormer
#SBATCH --account=odu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

# usage: sbatch --job-name=<label> graphormer_sampling.slurm  $1 $2
module load python3/2025.1-py312
source ~/qsar_env/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python train_gt_sampling.py Graphormer $1 $2
