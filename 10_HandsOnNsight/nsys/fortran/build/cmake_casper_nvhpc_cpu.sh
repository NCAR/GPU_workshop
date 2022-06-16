#!/bin/bash

module reset >& /dev/null
module load pnetcdf cmake nvhpc/22.2 cuda/11.4.0 openmpi >& /dev/null

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

export OMPI_FC=nvfortran

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${NCAR_ROOT_PNETCDF}   \
      -DFFLAGS="-O3 -Mvect -DNO_INFORM"                                \
      -DNX=200 \
      -DNZ=100 \
      -DSIM_TIME=1000 \
      -DOUT_FREQ=2000 \
      ..

