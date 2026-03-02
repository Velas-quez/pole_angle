#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1

cd ~/projects/tactile_pole_balancing/pole_angle

uv run python src/train.py \
  --config-name config_cnn_tcn_quat \
  strategies.materialize_dense_evtac=false