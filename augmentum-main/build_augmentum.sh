#!/bin/bash
#SBATCH --job-name=build_augmentum
#SBATCH --output=build_augmentum_%j.out
#SBATCH --error=build_augmentum_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=standard

# Submit from augmentum-main/ so the log files above land next to this script.
: "${HEURAKA_ROOT:?set HEURAKA_ROOT to the absolute path of the heuraka checkout (the directory containing augmentum-main/ and llvm10/)}"

AUGMENTUM_DIR=${HEURAKA_ROOT}/augmentum-main
LLVM_CMAKE_DIR=${HEURAKA_ROOT}/llvm10/install/lib/cmake/llvm
BUILD_DIR=${AUGMENTUM_DIR}/build

echo "Start: $(date)"
echo "Node: ${SLURM_NODELIST}"

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate augmentum || { echo "conda activate failed!"; exit 1; }

cd ${AUGMENTUM_DIR}

make all LLVM_DIR=${LLVM_CMAKE_DIR} BUILD_DIR=${BUILD_DIR}

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Done: $(date)"
ls ${BUILD_DIR}/extensions/augmentum_llvmpass/ 2>/dev/null
