#!/bin/bash
#SBATCH -A stf006
#SBATCH -J conus1_boxtest
#SBATCH -o logs/conus1_boxtest-%j.o
#SBATCH -e logs/conus1_boxtest-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1

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

# Avoids MIOpen disk-cache contention/I/O errors on shared filesystems
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# main.py uses bare local imports (from dataset import ..., from model
# import ..., etc.) so it must be run with emulator-1ts/ as the working
# directory, not invoked by path from elsewhere.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable/emulator-1ts

srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -u main.py --config conus1_boxtest_config.yaml --mode train
