#!/bin/bash
#SBATCH --job-name=augmentum_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=standard

# Copyright (c) 2021, Björn Franke
# Modified for SLURM sbatch submission

# ============================================================
# Activate conda environment
# ============================================================
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate augmentum
export PATH=${HOME}/miniconda3/envs/augmentum/bin:$PATH

# ============================================================
# Ensure no other analyzer is running to avoid resource contention
# ============================================================
# Disabled: allow concurrent runs with separate working directories
# if pgrep -f "function_analyser.py" > /dev/null; then
#     echo "Error: Another function_analyser.py is already running. Please terminate it first."
#     exit 1
# fi

# ============================================================
# Configuration
# ============================================================
AUGMENTUM_HOME=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/augmentum-main
WORKING_DIR_BASE=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/working_dir
CFG_DIR=${AUGMENTUM_HOME}/driver/config
CONFIG=${AUGMENTUM_HOME}/evaluation_config.json

# Use SLURM allocated CPUs
CPUS=${SLURM_CPUS_PER_TASK:-8}

# ============================================================
# Driver options
# ============================================================
VERBOSE="--verbose --loglevel DEBUG"
KEEP_PROBES="--keep_probes"
SKIP_IMM="--skip_immutables"
#INDY_TESTS="--independent_test_cases"
#RECORD_EXEC_LOG="--record_exec_log"
#DRY_RUN="--dry_run"
#NO_INSTR="--no_instr"

# Search strategy: composite only (random/bayesian removed — use rtp_tuning.py
# or rtp_bo_tuning.py for multi-target combination search)
SEARCH_STRATEGY="--search_strategy composite"
SEARCH_BUDGET="--search_budget 50"

WORKING_DIR=${WORKING_DIR_BASE}_composite
SQL_DB="--heuristicDB sqlite:///${WORKING_DIR}/subset_composite.sqlite"

#BFILTER="--bmark_filter SNU_NPB#bt"

FUN_CACHE="--function_cache ${CFG_DIR}/function_cache.pickle"
BASE_CACHE="--baseline_cache ${CFG_DIR}/baseline_cache.pickle"
TARGET_FUNS="--target_function ${CFG_DIR}/target_functions.csv"
#TARGET_FILTER="--target_filter ${CFG_DIR}/target_filter.csv"

CHUNK_SIZE="--fn_chunk_size 100000"
CHUNK_OFFSET="--fn_chunk_offset 0"

CPU_COUNT="--cpus ${CPUS}"
#EXACT_MAP="--exact_cpu_map"
MEM_LIMIT="--probe_mem_limit 1024"

RUN_ID=eval_run_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S")

# ============================================================
# Create working directory
# ============================================================
mkdir -p ${WORKING_DIR}

# ============================================================
# Cleanup: remove eval_run dirs from previous job in this chain
# (keeps subset_*.sqlite results, only deletes intermediate logs/worker_output)
# ============================================================
echo "Cleaning up old eval_run dirs in ${WORKING_DIR} ..."
rm -rf ${WORKING_DIR}/eval_run_* 2>/dev/null
echo "Cleanup done."

# ============================================================
# Run evaluation
# ============================================================
CMD="${AUGMENTUM_HOME}/driver/function_analyser.py
        --run_id ${RUN_ID}
        --working_dir ${WORKING_DIR}
        --config ${CONFIG}
        --instr_scope FUNCTION
        --instr_chunk 50
        ${SKIP_IMM}
        ${MEM_LIMIT}
        ${SQL_DB}
        ${TARGET_FUNS}
        ${TARGET_FILTER}
        ${CHUNK_SIZE}
        ${CHUNK_OFFSET}
        ${FUN_CACHE}
        ${BASE_CACHE}
        ${CPU_COUNT}
        ${EXACT_MAP}
        ${DRY_RUN}
        ${NO_INSTR}
        ${RECORD_EXEC_LOG}
        ${VERBOSE}
        ${KEEP_PROBES}
        ${INDY_TESTS}
        ${BFILTER}
        ${SEARCH_STRATEGY}
        ${SEARCH_BUDGET}"

echo "================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Run ID:          ${RUN_ID}"
echo "Node:            ${SLURM_NODELIST}"
echo "CPUs:            ${CPUS}"
echo "Start time:      $(date)"
echo "Working dir:     ${WORKING_DIR}"
echo "Config:          ${CONFIG}"
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