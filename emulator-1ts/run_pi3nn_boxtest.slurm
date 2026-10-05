#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pi3nn_boxtest
#SBATCH -o logs/pi3nn_boxtest-%j.o
#SBATCH -e logs/pi3nn_boxtest-%j.e
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

# First REAL-data smoke test for the PI3NN/emulator-1ts integration --
# CPU only, mirroring run_conus1_training.slurm's conventions but for
# main_pi3nn.py. Run this (and confirm it completes cleanly) before
# attempting a GPU/multi-GPU run, same escalation path as
# run_pi3nn_validate.slurm's synthetic checks before this.
#
# pi3nn_boxtest_config.yaml still has two FILL_IN placeholders
# (model_def.in_channels, data_def.scaler_yaml) that need the output of
# compute_conus1_scalers.py -- the exact same requirement
# conus1_boxtest_config.yaml already has for the regular emulator-1ts
# training, so if you've already filled those in there, reuse the same
# values here.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/groundwater-emulator

mkdir -p logs

# Needed for torch.distributed's rendezvous (env:// init method), even
# though this is launched as a single task -- main_pi3nn.py falls back to
# ordinary non-distributed training automatically whenever SLURM_NTASKS
# is 1, no script changes needed to go from 1 process to many, just the
# #SBATCH lines above (same convention as run_conus1_training.slurm).
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=3442

# main_pi3nn.py uses bare local imports (from dataset import ..., from
# model import ..., etc.) so it must be run with emulator-1ts/ as the
# working directory, not invoked by path from elsewhere.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable/emulator-1ts

srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 \
    python3 -u main_pi3nn.py --config pi3nn_boxtest_config.yaml
