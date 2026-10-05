#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pi3nn_boxtest
#SBATCH -o logs/pi3nn_boxtest-%j.o
#SBATCH -e logs/pi3nn_boxtest-%j.e
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # up to 8 for more GPUs on this node (Frontier has 8 GCDs/node); raise -N too for multiple nodes

# First real-data (CONUS1 boxtest) run for the PI3NN/emulator-1ts
# integration, GPU, mirroring run_conus1_training.slurm's conventions
# but for main_pi3nn.py. GPU/distributed mechanics are already validated
# (run_pi3nn_validate.slurm's ddp_loss check passed on real Frontier
# hardware) -- the open question here is real data shapes/paths, not
# the training mechanics, so this skips the CPU-only intermediate step.
#
# pi3nn_boxtest_config.yaml still has two FILL_IN placeholders
# (model_def.in_channels, data_def.scaler_yaml) that need the output of
# compute_conus1_scalers.py -- the exact same requirement
# conus1_boxtest_config.yaml already has for the regular emulator-1ts
# training, so reuse the same values here if already computed.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/groundwater-emulator

# /tmp is node-local and can carry permission-mismatched leftovers from
# other jobs that previously ran on the same compute node -- Lustre
# project space avoids that class of problem entirely. Shared with the
# emulator training scripts since MIOpen's cache isn't per-model.
export MIOPEN_USER_DB_PATH="/lustre/orion/stf006/proj-shared/irl1/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# Needed for torch.distributed's rendezvous (env:// init method), even
# when running a single task/GPU -- main_pi3nn.py falls back to ordinary
# non-distributed training automatically whenever SLURM_NTASKS is 1, no
# script changes needed to go from 1 GPU to many, just the #SBATCH lines
# above.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=3442

# main_pi3nn.py uses bare local imports (from dataset import ..., from
# model import ..., etc.) so it must be run with emulator-1ts/ as the
# working directory, not invoked by path from elsewhere.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable/emulator-1ts

# --gpus-per-task=1 --gpu-bind=closest (not --gpus-per-node): this
# hand-rolled DDP (same as main.py) expects each task to see exactly one
# GPU -- the opposite of emulator_configurable's Lightning-managed
# convention. See pi3nn/distributed-setup notes elsewhere in this
# project for why mixing the two is a known failure mode.
srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -u main_pi3nn.py --config pi3nn_boxtest_config.yaml
