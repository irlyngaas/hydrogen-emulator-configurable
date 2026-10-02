#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pi3nn_boston
#SBATCH -o logs/pi3nn_boston-%j.o
#SBATCH -e logs/pi3nn_boston-%j.e
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

# CPU-only: run_boston_comparison.py never touches a GPU (full-batch
# gradient descent on 506 rows), so unlike the emulator training scripts
# this doesn't load rocm/craype-accel-amd-gfx90a or request --gpus-per-node.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

conda activate /lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/groundwater-emulator

mkdir -p logs

# Run from the repo root: pi3nn_torch is a proper package (relative imports
# in run_boston_comparison.py), so it must be invoked as
# `python -m pi3nn_torch.run_boston_comparison`, not by file path, and the
# repo root must be on the module search path -- same convention as
# emulator_configurable/run_conus1_forcedstrnn_training.slurm.
cd /lustre/orion/stf006/proj-shared/irl1/hydrogen-emulator-configurable

srun -N1 -n1 -c1 python3 -u -m pi3nn_torch.run_boston_comparison --quantile 0.95
