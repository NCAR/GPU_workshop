#!/bin/bash
# Changed some lines for running on UD CRPL machines
# Load the necessary modules (software)
module purge
#module load ncarenv/1.2
#module load nvhpc/20.11
module load nvhpc/20.7
#module load cuda/11.0.3
module load cuda/10.2 # Changed to run on UD CRPL machines
module list

# Export variables for use in the Makefile
#export CUDA_ROOT_PATH="${NCAR_ROOT_CUDA}"
export CUDA_ROOT_PATH=/opt/cuda/10.2

# Remove any previous build attempts
make clean
# Do a build
make
