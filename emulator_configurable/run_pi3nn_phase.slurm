#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pi3nn_phase
#SBATCH -o logs/pi3nn_phase-%j.o
#SBATCH -e logs/pi3nn_phase-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # up to 8 for more GPUs on this node (Frontier has 8 GCDs/node)
#SBATCH --gpus-per-node=1     # keep this equal to --ntasks-per-node above -- see run_conus1_forcedstrnn_training.slurm's note on why

# One phase per job (mean, up, down, or calibrate), chained via
# --dependency=afterok by submit_pi3nn_chain.sh -- NOT a single script
# calling train_model()/trainer.fit() three times in one process.
# Constructing a fresh pl.Trainer(strategy='ddp', ...) multiple times in
# one running process is an untested combination here and a known
# source of friction in some Lightning versions -- not worth risking on
# a real GPU allocation without testing it first. See
# pi3nn/run_pi3nn_phase.py's module docstring for the same rationale.
#
# Reads PI3NN_PHASE (mean|up|down|calibrate) and PI3NN_MODE
# (stateless_output|stateless_hidden|recurrent_clone) from the
# environment -- set these via sbatch --export=... (submit_pi3nn_chain.sh
# does this for you).
: "${PI3NN_PHASE:?Set PI3NN_PHASE via sbatch --export=PI3NN_PHASE=mean|up|down|calibrate}"

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
# project space avoids that entirely. Shared with the other training
# scripts since MIOpen's cache isn't per-model.
export MIOPEN_USER_DB_PATH="/lustre/orion/stf006/proj-shared/irl1/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# IMPORTANT: --gpus-per-node (not --gpus-per-task/--gpu-bind=closest) --
# Lightning's SLURM integration expects every process to see ALL GPUs on
# the node and picks the correct one itself internally via SLURM_LOCALID.
# See run_conus1_forcedstrnn_training.slurm's longer note for the exact
# mutually-exclusive errors mixing the two conventions produces.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=3442

# Run from the repo root -- proper package (python -m
# emulator_configurable.pi3nn.run_pi3nn_phase), needs the repo root on
# the module search path, same as the other emulator_configurable scripts.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable

MODE_ARGS=""
if [ -n "$PI3NN_MODE" ]; then
    MODE_ARGS="--up-down-mode $PI3NN_MODE"
fi

srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 --gpus-per-node=$SLURM_GPUS_PER_NODE \
    python3 -u -m emulator_configurable.pi3nn.run_pi3nn_phase \
        --config emulator_configurable/conus1_boxtest_pi3nn_config.json \
        --phase ${PI3NN_PHASE} ${MODE_ARGS}
