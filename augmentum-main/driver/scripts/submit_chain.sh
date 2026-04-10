#!/bin/bash
# ============================================================
# Submit a chain of dependent SLURM jobs
#
# Usage:
#   ./submit_chain.sh <sbatch_script> <num_jobs> [script_args...]
#
# Examples:
#   ./submit_chain.sh sbatch_rtp.sh 5 6400 10
#   ./submit_chain.sh sbatch_rtp_bo.sh 3 2000 10
#   ./submit_chain.sh sbatch_evaluation.sh 10
#
# Each subsequent job starts after the previous one finishes
# (regardless of exit code, so resume logic can pick up).
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/${1:?Usage: $0 <sbatch_script> <num_jobs> [script_args...]}"
NUM_JOBS=${2:?Usage: $0 <sbatch_script> <num_jobs> [script_args...]}
shift 2
SCRIPT_ARGS="$@"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "Error: Script not found: ${SBATCH_SCRIPT}"
    exit 1
fi

echo "Submitting chain of ${NUM_JOBS} jobs: ${SBATCH_SCRIPT} ${SCRIPT_ARGS}"
echo "================================================"

# Submit first job
JOB_ID=$(sbatch --parsable "${SBATCH_SCRIPT}" ${SCRIPT_ARGS})
echo "Job 1/${NUM_JOBS}: ${JOB_ID} (no dependency)"

# Submit remaining jobs with dependency on previous
for i in $(seq 2 ${NUM_JOBS}); do
    PREV_ID=${JOB_ID}
    JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_ID} "${SBATCH_SCRIPT}" ${SCRIPT_ARGS})
    echo "Job ${i}/${NUM_JOBS}: ${JOB_ID} (after ${PREV_ID})"
done

echo "================================================"
echo "All ${NUM_JOBS} jobs submitted. Monitor with: squeue -u \$USER"
