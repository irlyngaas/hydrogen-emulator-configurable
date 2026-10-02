#!/bin/bash
#SBATCH -A stf006
#SBATCH -J conus1_forcedstrnn
#SBATCH -o logs/conus1_forcedstrnn-%j.o
#SBATCH -e logs/conus1_forcedstrnn-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=8   # up to 8 for more GPUs on this node (Frontier has 8 GCDs/node)
#SBATCH --gpus-per-node=8     # keep this equal to --ntasks-per-node above -- see note below on why this differs from emulator-1ts's script

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
# to many, just the #SBATCH lines above.
#
# IMPORTANT, and the opposite of emulator-1ts's script: do NOT use
# --gpus-per-task/--gpu-bind=closest here. emulator-1ts's hand-rolled DDP
# code specifically needs SLURM to restrict each task to seeing only 1 GPU
# (so it can do its own local_rank % device_count() mapping), but Lightning's
# SLURM integration is designed the opposite way -- it expects every process
# to see ALL GPUs on the node and picks the correct one itself internally via
# SLURM_LOCALID. Mixing the two conventions produces two mutually exclusive
# Lightning-internal errors (confirmed by actually running both ways):
#   --gpus-per-task=1 --gpu-bind=closest -> CUDAAccelerator.parse_devices()
#     fails ("You requested gpu: [0..7] But your machine only has: [0]")
#   devices=1 without per-task GPU restriction -> SLURMEnvironment.
#     validate_settings() fails ("devices=1 ... does not match ... HINT:
#     Set devices=8")
# This is a known Lightning/SLURM compatibility conflict (Lightning-AI/
# pytorch-lightning#16828), not something specific to this setup --
# requesting GPUs at the node level instead (--gpus-per-node=N at both
# #SBATCH and srun) keeps all N visible to every task, resolving both.
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

srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 --gpus-per-node=$SLURM_GPUS_PER_NODE \
    python3 -u -m emulator_configurable.main \
        --mode train --domain subsurface \
        --config emulator_configurable/conus1_boxtest_config.json
