#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pi3nn_validate
#SBATCH -o logs/pi3nn_validate-%j.o
#SBATCH -e logs/pi3nn_validate-%j.e
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

# CPU-only, synthetic-data correctness checks for the PI3NN/emulator-1ts
# integration -- run this BEFORE any real CONUS1-scale run via
# main_pi3nn.py, same escalation path the flat PI3NN port followed
# (synthetic/boston-housing correctness checks first, then real data).
# See pi3nn/validate_synthetic.py's module docstring for what each mode
# actually checks.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/groundwater-emulator

mkdir -p logs

# Run from emulator-1ts/ as a module (pi3nn is a package with relative
# imports) -- NOTE: this path assumes the same Lustre project location
# convention as the other scripts here. Update it if your checkout
# lives elsewhere.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable/emulator-1ts

echo '=== full pipeline check (single process) ==='
python3 -u -m pi3nn.validate_synthetic --mode full

echo '=== DDP masked-loss correctness check (2 processes, gloo/CPU) ==='
# --standalone sets up its own local rendezvous automatically (no
# MASTER_ADDR/MASTER_PORT needed) -- this is a fixed 2-process,
# single-node CPU check regardless of how many nodes/GPUs this job
# actually requested above.
python3 -u -m torch.distributed.run --nproc_per_node=2 --standalone \
    -m pi3nn.validate_synthetic --mode ddp_loss
