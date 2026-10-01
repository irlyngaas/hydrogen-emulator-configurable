#!/bin/bash
#SBATCH -A stf006
#SBATCH -J conus1_boxtest
#SBATCH -o logs/conus1_boxtest-%j.o
#SBATCH -e logs/conus1_boxtest-%j.e
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
# other jobs that previously ran on the same compute node (hit exactly this
# running emulator_configurable interactively: "Permission denied
# [/tmp/.cache/miopen/...]") -- Lustre project space avoids that class of
# problem entirely, at some cost to kernel-cache read/write latency.
export MIOPEN_USER_DB_PATH="/lustre/orion/stf006/proj-shared/irl1/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# Needed for torch.distributed's rendezvous (env:// init method), even when
# running a single task/GPU -- main.py falls back to ordinary non-distributed
# training automatically whenever SLURM_NTASKS is 1, no script changes needed
# to go from 1 GPU to many, just the #SBATCH lines above.
# `hostname -i` is unreliable across cluster network configs (returned an
# address PyTorch's TCP rendezvous rejected as an unsupported family here) --
# resolving the allocation's first node name via scontrol is the more robust
# standard pattern, and works the same whether this is 1 node or several.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=3442

# main.py uses bare local imports (from dataset import ..., from model
# import ..., etc.) so it must be run with emulator-1ts/ as the working
# directory, not invoked by path from elsewhere.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable/emulator-1ts

srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -u main.py --config conus1_boxtest_config.yaml --mode train
