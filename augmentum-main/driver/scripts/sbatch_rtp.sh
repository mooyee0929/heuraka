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
AUGMENTUM_HOME=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/augmentum-main
WORKING_DIR=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/working_dir_rtp
CFG_DIR=${AUGMENTUM_HOME}/driver/config
CONFIG=${AUGMENTUM_HOME}/evaluation_config.json

CPUS=${SLURM_CPUS_PER_TASK:-8}

# RTP parameters
ITERATIONS=${1:-6400}       # Number of RTP iterations (paper uses ~6400)
MAX_TARGETS=${2:-10}        # Max tuning targets per profile
MEM_LIMIT=1024              # MB memory limit for compilation

# Composite DB with discovered tuning targets
COMPOSITE_DB="sqlite:////scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/working_dir_composite/subset_composite.sqlite"

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
        --baseline_cache ${CFG_DIR}/baseline_cache.pickle
        --iterations ${ITERATIONS}
        --max_targets ${MAX_TARGETS}
        --cpus ${CPUS}
        --probe_mem_limit ${MEM_LIMIT}
        --verbose"

echo "================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Node:            ${SLURM_NODELIST}"
echo "CPUs:            ${CPUS}"
echo "Start time:      $(date)"
echo "Working dir:     ${WORKING_DIR}"
echo "Iterations:      ${ITERATIONS}"
echo "Max targets:     ${MAX_TARGETS}"
echo "Composite DB:    ${COMPOSITE_DB}"
echo "================================================"
echo "Running command:"
echo ${CMD}
echo "================================================"

/home/fsyang/miniconda3/envs/augmentum/bin/python ${CMD}

EXIT_CODE=$?

echo "================================================"
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "================================================"

exit ${EXIT_CODE}
