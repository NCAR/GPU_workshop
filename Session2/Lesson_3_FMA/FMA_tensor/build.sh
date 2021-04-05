#!/bin/bash

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11
module load cuda/11.0.3
module list

# Export variables for use in the Makefile
export CUDA_ROOT_PATH="${NCAR_ROOT_CUDA}"

# Remove any previous build attempts
nvcc -O3 -std=c++11 -o tensor fma_tensor.cu -arch sm_75
