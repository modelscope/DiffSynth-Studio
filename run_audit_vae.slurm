#!/bin/bash
#SBATCH --job-name=vae_audit
#SBATCH --account=aip-lindell
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

cd /project/6101839/jw954/DiffSynth-Studio-SPAD
source ./env_py311.sh
pip install --quiet scipy scikit-learn 2>/dev/null
python audit_vae_roundtrip.py
