#!/bin/bash
#SBATCH --job-name=augmentum_rtp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=standard

# ============================================================
# Random Tuning Profile (RTP) — Paper Section VII-E
#
# Combines multiple tuning targets discovered by composite search
# to stack binary size reductions.
#
# Usage:
#   sbatch sbatch_rtp.sh [iterations] [max_targets]
#   e.g.: sbatch sbatch_rtp.sh 6400 10
# ============================================================

# Activate conda environment
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate augmentum
export PATH=${HOME}/miniconda3/envs/augmentum/bin:$PATH

# ============================================================
# Configuration
# ============================================================
: "${HEURAKA_ROOT:?set HEURAKA_ROOT to the absolute path of the heuraka checkout (the directory containing augmentum-main/ and llvm10/)}"

AUGMENTUM_HOME=${HEURAKA_ROOT}/augmentum-main
WORKING_DIR=${HEURAKA_ROOT}/working_dir_rtp
CFG_DIR=${AUGMENTUM_HOME}/driver/config
CONFIG=${AUGMENTUM_HOME}/evaluation_config.json

CPUS=${SLURM_CPUS_PER_TASK:-8}

# RTP parameters
ITERATIONS=${1:-6400}       # Number of RTP iterations (paper uses ~6400)
MAX_TARGETS=${2:-10}        # Max tuning targets per profile
MEM_LIMIT=1024              # MB memory limit for compilation

# Set USE_BEST_VALUE=1 in env to use composite-discovered best values
# (deterministic) instead of per-call random sampling from prior range.
USE_BEST_VALUE_FLAG=""
if [ "${USE_BEST_VALUE:-0}" = "1" ]; then
    USE_BEST_VALUE_FLAG="--use_best_value"
fi

# Composite DB with discovered tuning targets
COMPOSITE_DB="sqlite:///${HEURAKA_ROOT}/working_dir_composite/subset_composite.sqlite"

# ============================================================
# Create working directory
# ============================================================
mkdir -p ${WORKING_DIR}

# ============================================================
# Cleanup: remove temporary probe dirs from previous job in chain
# (keeps rtp_results.sqlite for resume, only deletes intermediate files)
# ============================================================
echo "Cleaning up old probe dirs in ${WORKING_DIR} ..."
rm -rf ${WORKING_DIR}/rtp_probe_* 2>/dev/null
rm -rf ${WORKING_DIR}/benchmarks 2>/dev/null
echo "Cleanup done."

# ============================================================
# Run RTP
# ============================================================
CMD="${AUGMENTUM_HOME}/driver/rtp_tuning.py
        --config ${CONFIG}
        --composite_db ${COMPOSITE_DB}
        --working_dir ${WORKING_DIR}
        --function_cache ${CFG_DIR}/function_cache.pickle
        --baseline_cache ${HEURAKA_ROOT}/backup_5bench_20260410/baseline_cache_5bench.pickle
        --iterations ${ITERATIONS}
        --max_targets ${MAX_TARGETS}
        --cpus ${CPUS}
        --probe_mem_limit ${MEM_LIMIT}
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
echo "Use best value:  ${USE_BEST_VALUE:-0}"
echo "Composite DB:    ${COMPOSITE_DB}"
echo "================================================"
echo "Running command:"
echo ${CMD}
echo "================================================"

${HOME}/miniconda3/envs/augmentum/bin/python ${CMD}

EXIT_CODE=$?

echo "================================================"
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "================================================"

exit ${EXIT_CODE}
