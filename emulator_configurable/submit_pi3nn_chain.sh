#!/bin/bash
# Submits PI3NN's phased training as chained SLURM jobs: one "mean" job,
# then for each requested up_down_mode, "up" and "down" (submitted in
# parallel -- they're independent, both only depend on net_mean's
# checkpoint, no reason to run them sequentially), then "calibrate"
# (depends on both). net_mean trains once and is reused across every
# mode sweep -- pass --skip-mean if you already have a trained
# checkpoint from a previous run (run_pi3nn_phase.py's run_mean() writes
# its path to <logging_location>/<run_name>_mean_ckpt.txt; the up/down
# phases read it automatically).
#
# Usage:
#   ./submit_pi3nn_chain.sh                                 # mean, then all 3 modes
#   ./submit_pi3nn_chain.sh stateless_output                 # mean, then just this mode
#   ./submit_pi3nn_chain.sh --skip-mean stateless_hidden recurrent_clone
set -euo pipefail

SLURM_SCRIPT="emulator_configurable/run_pi3nn_phase.slurm"

SKIP_MEAN=false
if [ "${1:-}" == "--skip-mean" ]; then
    SKIP_MEAN=true
    shift
fi
MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    MODES=(stateless_output stateless_hidden recurrent_clone)
fi

if [ "$SKIP_MEAN" == false ]; then
    MEAN_JOB=$(sbatch --parsable --export=ALL,PI3NN_PHASE=mean "$SLURM_SCRIPT")
    echo "Submitted mean phase: $MEAN_JOB"
    MEAN_DEP="--dependency=afterok:$MEAN_JOB"
else
    MEAN_DEP=""
    echo "Skipping mean phase (--skip-mean) -- up/down phases will read the existing checkpoint file."
fi

for MODE in "${MODES[@]}"; do
    UP_JOB=$(sbatch --parsable $MEAN_DEP --export=ALL,PI3NN_PHASE=up,PI3NN_MODE=$MODE "$SLURM_SCRIPT")
    DOWN_JOB=$(sbatch --parsable $MEAN_DEP --export=ALL,PI3NN_PHASE=down,PI3NN_MODE=$MODE "$SLURM_SCRIPT")
    echo "[$MODE] Submitted up=$UP_JOB down=$DOWN_JOB (parallel, both depend only on mean)"
    CAL_JOB=$(sbatch --parsable --dependency=afterok:$UP_JOB:$DOWN_JOB --export=ALL,PI3NN_PHASE=calibrate,PI3NN_MODE=$MODE "$SLURM_SCRIPT")
    echo "[$MODE] Submitted calibrate phase: $CAL_JOB"
done
