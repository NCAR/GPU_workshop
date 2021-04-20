#!/bin/bash -l
# Batch directives
#PBS -N stncl_c
#PBS -l select=1:ncpus=4:mpiprocs=4:ngpus=4
#PBS -l gpu_type=v100
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -A NTDD0002
#PBS -q R123790

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11 
module load cuda/11.0.3
module load openmpi/4.1.0
module list

echo -e "nvidia-smi output follows:"
nvidia-smi
# Update LD_LIBRARY_PATH so that cuda libraries can be found
export LD_LIBRARY_PATH=${NCAR_ROOT_CUDA}/lib64:${LD_LIBRARY_PATH}
echo -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

#export NV_ACC_TIME=1

export UCX_TLS=rc,sm,cuda_copy,cuda_ipc
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader,tcp,smcuda
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=2


# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"
mpirun -n 4 ./mpi_acc_stencil.exe 2048

# nvprof command line profiling
#mpirun  -n 4 nvprof --devices 0 --print-gpu-trace --profile-api-trace none --kernels "::LaplaceJacobi_MPIACC:2"  --metrics "achieved_occupancy,gld_transactions,gst_transactions,flop_count_sp"  ./mpi_acc_stencil.exe 128

## nsys GUI profiling
#mpirun -n 4 nsys profile -t cuda,openacc,mpi,nvtx --mpi-impl openmpi -o profileData."%q{OMPI_COMM_WORLD_RANK}" ./mpi_acc_stencil.exe 8
echo -e "\nEnd of code output:\n-------------\n"
