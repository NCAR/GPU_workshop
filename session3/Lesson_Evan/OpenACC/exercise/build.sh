#!/bin/bash
# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11
module list

# Export variables for use in the Makefile
export NVHPC_ROOT_PATH="${NCAR_ROOT_NVHPC}/Linux_x86_64/20.11/compilers"

# Remove any previous build attempts
make clean
# Do a build
# Set OPENACC=true to enable OpenACC, otherwise set as false
make OPENACC=true
