#!/bin/bash -l
# Batch directives
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:v100:4
#SBATCH --account NTDD0002
#SBATCH --partition=dav
#SBATCH --reservation=GPU_workshop_2
#SBATCH --time=00:15:00
#SBATCH --output=log.stncl_%j.out
#SBATCH --job-name=GPU_stncl

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11 
module load cuda/11.0.3
module load openmpi/4.0.5x
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
#export UCX_RNDV_SCHEME=get_zcopy

# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"
mpirun -n 16 ./mpi_acc_stencil.exe 64 4
echo -e "\nEnd of code output:\n-------------\n"
