#!/bin/bash
#SBATCH --job-name=vae_analysis
#SBATCH --account=aip-lindell
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

cd /project/6101839/jw954/DiffSynth-Studio-SPAD
source ./env_py311.sh

# scipy is used for SSIM (uniform_filter), should be available via system packages
pip install --quiet scipy 2>/dev/null
pip install --quiet scikit-learn 2>/dev/null

echo "============================================"
echo "VAE Domain Gap Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python analyze_vae_domain_gap.py --n_scenes 25 --device cuda

echo "Done. Results in /scratch/jw954/vae_analysis/"
