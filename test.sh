#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

NGPUS=${NGPUS:-1}

srun --partition=medai_p --mpi=pmi2 --gres=gpu:${NGPUS} --quotatype=reserved \
     -n1 --ntasks-per-node=1 --cpus-per-task=8 \
     --job-name=lpips --kill-on-bad-exit=1 \
     python examples/image_quality_metric/lpips.py
