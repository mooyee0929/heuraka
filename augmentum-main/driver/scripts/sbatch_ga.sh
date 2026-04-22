#!/bin/bash
#SBATCH --job-name=augmentum_ga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=standard

# ============================================================
# Genetic Algorithm baseline (binary subset over composite-discovered
# tuning targets). Third leg of the random / evolutionary / Bayesian
# comparison alongside RTP and BO. Same search space and value-sampling
# modes as bo_tuning.
#
# Usage:
#   sbatch sbatch_ga.sh [iterations] [top_k] [pop_size]
#   e.g.: sbatch sbatch_ga.sh 2000 10 20
#   e.g.: USE_BEST_VALUE=1 sbatch sbatch_ga.sh 6400 14 24
#
# This script is user-agnostic. It auto-detects:
#   - AUGMENTUM_HOME from ${SLURM_SUBMIT_DIR}/../.. (or override via env)
#   - conda location from ${CONDA_BASE} env, then ~/miniconda3,
#     ~/anaconda3, then the GreatLakes system anaconda at
#     /sw/pkgs/arc/python3.9-anaconda/2021.11 (or override via env)
#   - WORKING_DIR uses $USER so each teammate gets their own (override)
# ============================================================

set -uo pipefail

# ----- Conda activation (portable across teammates) -------------------
# Honour an explicit CONDA_BASE override; otherwise probe common locations.
CONDA_INIT=""
if [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh" ]; then
    CONDA_INIT="/sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh"
fi

if [ -z "${CONDA_INIT}" ]; then
    echo "ERROR: cannot locate conda. Set CONDA_BASE=/path/to/conda root in env." >&2
    exit 1
fi

echo "Using conda init: ${CONDA_INIT}"
source "${CONDA_INIT}"

if ! conda activate augmentum 2>/dev/null; then
    echo "ERROR: 'augmentum' conda env not found. Create it with:" >&2
    echo "  cd <augmentum-main>; conda env create --file environment.yml" >&2
    echo "  conda activate augmentum && pip install scikit-optimize numpy" >&2
    exit 1
fi

PYTHON_BIN="$(command -v python)"
echo "Using python:     ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import skopt, numpy" 2>/dev/null \
    || { echo "ERROR: skopt/numpy missing in 'augmentum' env. Run:  pip install scikit-optimize numpy" >&2; exit 1; }

# ----- Paths ----------------------------------------------------------
# AUGMENTUM_HOME defaults to two levels up from the submit dir
# (scripts/ -> driver/ -> augmentum-main/). Override with env if needed.
DEFAULT_AUGMENTUM_HOME="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}/../.." && pwd)"
AUGMENTUM_HOME="${AUGMENTUM_HOME:-${DEFAULT_AUGMENTUM_HOME}}"

WORKING_DIR=${WORKING_DIR:-/scratch/cse583w26_class_root/cse583w26_class/${USER}/heuraka/working_dir_ga}
CFG_DIR=${AUGMENTUM_HOME}/driver/config
CONFIG=${AUGMENTUM_HOME}/evaluation_config.json

CPUS=${SLURM_CPUS_PER_TASK:-8}

ITERATIONS=${1:-2000}
TOP_K=${2:-10}
POP_SIZE=${3:-20}
MEM_LIMIT=1024

# Set USE_BEST_VALUE=1 in env to use composite-discovered best values
# (deterministic) instead of per-call random sampling from prior range.
USE_BEST_VALUE_FLAG=""
if [ "${USE_BEST_VALUE:-0}" = "1" ]; then
    USE_BEST_VALUE_FLAG="--use_best_value"
fi

# Composite DB: using fsyang's group-readable backup (verified -rw-r--r-- with
# group cse583w26_class_root). Target discovery is path-agnostic so this
# can be shared across users.
COMPOSITE_DB=${COMPOSITE_DB:-"sqlite:////scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/backup_5bench_20260410/working_dir_composite/subset_composite.sqlite"}

# Baseline cache: MUST be measured on the running user's tree — absolute paths
# to benchmark source files get embedded via __FILE__/assert macros so
# baselines taken on a different directory layout produce systematically
# biased rel_objective (~350 bytes / 7-8% inflation on polybench-SMALL
# between fsyang's tree and shresth's tree — confirmed empirically).
# Regenerate with:
#   python driver/generate_baseline.py --config evaluation_config.json \
#       --output driver/config/baseline_cache_<user>_5bench.pickle \
#       --working_dir /scratch/.../heuraka/working_dir_baseline
BASELINE_CACHE=${BASELINE_CACHE:-"${AUGMENTUM_HOME}/driver/config/baseline_cache_shresth_5bench.pickle"}

if [ ! -f "${BASELINE_CACHE}" ]; then
    echo "ERROR: baseline cache not found: ${BASELINE_CACHE}" >&2
    echo "Generate one with:" >&2
    echo "  python ${AUGMENTUM_HOME}/driver/generate_baseline.py \\" >&2
    echo "      --config ${CONFIG} \\" >&2
    echo "      --output ${BASELINE_CACHE} \\" >&2
    echo "      --working_dir /scratch/.../\${USER}/heuraka/working_dir_baseline" >&2
    exit 1
fi

mkdir -p ${WORKING_DIR}

# ============================================================
# Cleanup: remove temporary probe dirs from previous job in chain
# (keeps ga_results.sqlite for resume, only deletes intermediate files)
# ============================================================
echo "Cleaning up old probe dirs in ${WORKING_DIR} ..."
rm -rf ${WORKING_DIR}/ga_probe_* 2>/dev/null
rm -rf ${WORKING_DIR}/benchmarks 2>/dev/null
echo "Cleanup done."

CMD="${AUGMENTUM_HOME}/driver/ga_tuning.py
        --config ${CONFIG}
        --composite_db ${COMPOSITE_DB}
        --working_dir ${WORKING_DIR}
        --function_cache ${CFG_DIR}/function_cache.pickle
        --baseline_cache ${BASELINE_CACHE}
        --iterations ${ITERATIONS}
        --top_k ${TOP_K}
        --pop_size ${POP_SIZE}
        --cpus ${CPUS}
        --probe_mem_limit ${MEM_LIMIT}
        ${USE_BEST_VALUE_FLAG}
        --verbose"

echo "================================================"
echo "SLURM Job ID:    ${SLURM_JOB_ID}"
echo "Node:            ${SLURM_NODELIST}"
echo "User:            ${USER}"
echo "CPUs:            ${CPUS}"
echo "Start time:      $(date)"
echo "AUGMENTUM_HOME:  ${AUGMENTUM_HOME}"
echo "Working dir:     ${WORKING_DIR}"
echo "Composite DB:    ${COMPOSITE_DB}"
echo "Baseline cache:  ${BASELINE_CACHE}"
echo "Iterations:      ${ITERATIONS}"
echo "Top-k targets:   ${TOP_K}"
echo "Pop size:        ${POP_SIZE}"
echo "Use best value:  ${USE_BEST_VALUE:-0}"
echo "================================================"
echo "Running command:"
echo ${CMD}
echo "================================================"

"${PYTHON_BIN}" ${CMD}

EXIT_CODE=$?

echo "================================================"
echo "Finished at:     $(date)"
echo "Exit code:       ${EXIT_CODE}"
echo "================================================"

exit ${EXIT_CODE}
