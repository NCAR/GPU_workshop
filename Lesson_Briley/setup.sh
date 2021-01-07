#!/bin/bash
module purge
module load ncarenv/1.2
module load nvhpc/20.11 
#module load openmpi/4.0.5
module list
which nvcc

#export MPI_PATH=${NCAR_ROOT_OPENMPI}
export CUDA_PATH="${NCAR_ROOT_NVHPC}/Linux_x86_64/20.11/"
