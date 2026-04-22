#!/bin/bash
#SBATCH --job-name=augmentum_sa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=standard
# ============================================================
# Simulated Annealing (SA) Tuning
#
# Usage:
#   sbatch sbatch_sa.sh [iterations] [max_targets]
#   e.g.: sbatch sbatch_sa.sh 6400 10
# ============================================================

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate augmentum
export PATH=${HOME}/miniconda3/envs/augmentum/bin:$PATH

# ============================================================
# Configuration — paths pointing to fsyang's copy
# ============================================================
AUGMENTUM_HOME=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/augmentum-main
WORKING_DIR=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/working_dir_sa
CFG_DIR=${AUGMENTUM_HOME}/driver/config
CONFIG=${AUGMENTUM_HOME}/evaluation_config.json
CPUS=${SLURM_CPUS_PER_TASK:-8}

# SA parameters
ITERATIONS=${1:-6400}
MAX_TARGETS=${2:-10}
MEM_LIMIT=1024

# SA-specific hyperparameters
TEMP_INIT=1.0
TEMP_MIN=0.01
COOLING_RATE=0.995

# Set USE_BEST_VALUE=1 in env to use composite-discovered best values
# (deterministic) instead of per-call random sampling from prior range.
USE_BEST_VALUE_FLAG=""
if [ "${USE_BEST_VALUE:-0}" = "1" ]; then
    USE_BEST_VALUE_FLAG="--use_best_value"
fi

# Use fsyang's backup composite DB (proven good, avoids copy corruption)
COMPOSITE_DB="sqlite:////scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/backup_5bench_20260410/working_dir_composite/subset_composite.sqlite"

# ============================================================
# Create working directory
# ============================================================
mkdir -p ${WORKING_DIR}

# ============================================================
# Cleanup old probe dirs (keeps sa_results.sqlite for resume)
# ============================================================
echo "Cleaning up old probe dirs in ${WORKING_DIR} ..."
rm -rf ${WORKING_DIR}/sa_probe_* 2>/dev/null
rm -rf ${WORKING_DIR}/benchmarks 2>/dev/null
echo "Cleanup done."

# ============================================================
# Run SA
# ============================================================
CMD="${AUGMENTUM_HOME}/driver/sa_tuning.py
        --config ${CONFIG}
        --composite_db ${COMPOSITE_DB}
        --working_dir ${WORKING_DIR}
        --function_cache ${CFG_DIR}/function_cache.pickle
        --baseline_cache /scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/backup_5bench_20260410/baseline_cache_5bench.pickle
        --iterations ${ITERATIONS}
        --max_targets ${MAX_TARGETS}
        --cpus ${CPUS}
        --probe_mem_limit ${MEM_LIMIT}
        --temp_init ${TEMP_INIT}
        --temp_min ${TEMP_MIN}
        --cooling_rate ${COOLING_RATE}
        ${USE_BEST_VALUE_FLAG}
        --verbose"

echo "================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Node:            ${SLURM_NODELIST}"
echo "CPUs:            ${CPUS}"
echo "Start time:      $(date)"
echo "Working dir:     ${WORKING_DIR}"
echo "Iterations:      ${ITERATIONS}"
echo "Max targets:     ${MAX_TARGETS}"
echo "Temp init:       ${TEMP_INIT}"
echo "Temp min:        ${TEMP_MIN}"
echo "Cooling rate:    ${COOLING_RATE}"
echo "Use best value:  ${USE_BEST_VALUE:-0}"
echo "Composite DB:    ${COMPOSITE_DB}"
echo "================================================"
echo "Running command:"
echo ${CMD}
echo "================================================"

$(which python) ${CMD}

EXIT_CODE=$?
echo "================================================"
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "================================================"
exit ${EXIT_CODE}