#!/bin/bash
#SBATCH -A stf006
#SBATCH -J conus1_forcedstrnn
#SBATCH -o logs/conus1_forcedstrnn-%j.o
#SBATCH -e logs/conus1_forcedstrnn-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # up to 8 for more GPUs on this node (Frontier has 8 GCDs/node); raise -N too for multiple nodes

# Same modules used to build the environment -- must be reloaded here since
# each SLURM job starts with a fresh environment, not whatever was loaded
# interactively when the conda env was created.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/groundwater-emulator

# /tmp is node-local and can carry permission-mismatched leftovers from
# other jobs that previously ran on the same compute node (this is exactly
# what broke the first interactive run of this model: "Permission denied
# [/tmp/.cache/miopen/...]") -- Lustre project space avoids that entirely.
# Shared with emulator-1ts's script since MIOpen's cache isn't per-model.
export MIOPEN_USER_DB_PATH="/lustre/orion/stf006/proj-shared/irl1/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# train.py derives devices=/num_nodes= from SLURM_NTASKS_PER_NODE/SLURM_NNODES
# directly, falling back to 1/1 (today's working single-process behavior)
# whenever these aren't set at all -- no code changes needed to go from 1 GPU
# to many, just the #SBATCH lines above. Lightning's own SLURM-aware DDP
# strategy (auto-detected from being launched via srun with SLURM_NTASKS set)
# handles rank/process-group setup and the one-visible-GPU-per-task device
# mapping itself -- none of emulator-1ts's hand-rolled main.py correction
# work applies here.
#
# MASTER_ADDR/MASTER_PORT: Lightning's SLURMEnvironment resolves the main
# node's address itself via the same scontrol-based approach, but setting it
# explicitly here is a harmless, cheap safety net consistent with
# emulator-1ts's script, so keeping it rather than relying solely on
# Lightning's auto-detection, which hasn't actually been verified end-to-end
# at >1 node yet.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=3442

# Run from the repo root, not from inside emulator_configurable/ -- unlike
# emulator-1ts's bare local imports, this is a proper package
# (python -m emulator_configurable.main) and needs the repo root on the
# module search path.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable

srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -u -m emulator_configurable.main \
        --mode train --domain subsurface \
        --config emulator_configurable/conus1_boxtest_config.json
