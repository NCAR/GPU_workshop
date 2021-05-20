#!/bin/bash

export UCX_TLS=sm,cuda_copy,cuda_ipc
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader,tcp,smcuda
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=2

#export NV_ACC_TIME=1

# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"

mpirun -n 4 ./mpi_acc_stencil.exe 1024

echo -e "\nEnd code output:\n-------------\n"
