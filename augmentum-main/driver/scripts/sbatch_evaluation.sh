#!/bin/bash
#SBATCH --job-name=augmentum_eval
#SBATCH --output=augmentum_eval_%j.out
#SBATCH --error=augmentum_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=standard

# Copyright (c) 2021, Björn Franke
# Modified for SLURM sbatch submission

# ============================================================
# Activate conda environment
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate augmentum

# ============================================================
# Configuration
# ============================================================
AUGMENTUM_HOME=/n/eecs583a/home/fsyang/Heureka/augmentum-main
WORKING_DIR=/n/eecs583a/home/fsyang/Heureka/working_dir
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
#SQL_DB="--heuristicDB sqlite:///${WORKING_DIR}/heuristic_data.sqlite"

#BFILTER="--bmark_filter SNU_NPB#bt"

#FUN_CACHE="--function_cache ${CFG_DIR}/function_cache.pickle"
#BASE_CACHE="--baseline_cache ${CFG_DIR}/baseline_cache.pickle"
#TARGET_FUNS="--target_function ${CFG_DIR}/target_functions.csv"
#TARGET_FILTER="--target_filter ${CFG_DIR}/target_filter.csv"

CHUNK_SIZE="--fn_chunk_size 5000000"
CHUNK_OFFSET="--fn_chunk_offset 0"

CPU_COUNT="--cpus ${CPUS}"
#EXACT_MAP="--exact_cpu_map"
MEM_LIMIT="--probe_mem_limit 2048"

RUN_ID=eval_run_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S")

# ============================================================
# Create working directory
# ============================================================
mkdir -p ${WORKING_DIR}

# ============================================================
# Print job info
# ============================================================
echo "================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Run ID:          ${RUN_ID}"
echo "Node:            ${SLURM_NODELIST}"
echo "CPUs:            ${CPUS}"
echo "Start time:      $(date)"
echo "Working dir:     ${WORKING_DIR}"
echo "Config:          ${CONFIG}"
echo "================================================"

# ============================================================
# Run evaluation
# ============================================================
CMD="${AUGMENTUM_HOME}/driver/function_analyser.py
        --run_id ${RUN_ID}
        --working_dir ${WORKING_DIR}
        --config ${CONFIG}
        --instr_scope ALL
        --instr_chunk 1
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
        ${BFILTER}"

echo "Running command:"
echo ${CMD}
echo "================================================"

${CMD}

EXIT_CODE=$?

echo "================================================"
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "================================================"

exit ${EXIT_CODE}