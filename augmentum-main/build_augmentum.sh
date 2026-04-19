#!/bin/bash
#SBATCH --job-name=build_augmentum
#SBATCH --output=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/augmentum-main/build_augmentum_%j.out
#SBATCH --error=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/augmentum-main/build_augmentum_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=standard

AUGMENTUM_DIR=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/augmentum-main
LLVM_CMAKE_DIR=/scratch/cse583w26_class_root/cse583w26_class/fsyang/heuraka/llvm10/install/lib/cmake/llvm
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
