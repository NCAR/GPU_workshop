#!/bin/bash

module reset >& /dev/null
module load pnetcdf cmake nvhpc/22.2 cuda/11.4.0 openmpi >& /dev/null

export TEST_MPI_COMMAND="mpiexec -n 1"

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90               \
      -DPNETCDF_PATH=${NCAR_ROOT_PNETCDF}   \
      -DOPENMP_FLAGS=-mp                            \
      -DOPENACC_FLAGS="-acc -gpu=cc60,cc70"     \
      -DDO_CONCURRENT_FLAGS="-stdpar=gpu -Minfo=stdpar -gpu=cc60,cc70"     \
      -DFFLAGS="-O3 -DNO_INFORM"                                \
      -DLDFLAGS=""                                  \
      -DNX=200 \
      -DNZ=100 \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES" \
      -DSIM_TIME=600 \
      -DOUT_FREQ=100 \
      ..

# make 
# Use below command instead to make only desired executables
make mpi mpi_test openacc openacc_test