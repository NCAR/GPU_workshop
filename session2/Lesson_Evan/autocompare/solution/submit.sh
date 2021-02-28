#!/bin/bash -l
# Batch directives
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account NTDD0002
#SBATCH --partition=dav
#SBATCH --reservation=casper_8xV100
#SBATCH --time=00:15:00
#SBATCH --output=log.matmul_%j.out
#SBATCH --job-name=GPU_matmul

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11
module list

# Update LD_LIBRARY_PATH so that cuda libraries can be found
export LD_LIBRARY_PATH=${NCAR_ROOT_CUDA}/lib64:${LD_LIBRARY_PATH}
echo ${LD_LIBRARY_PATH}
nvidia-smi

export PCAST_COMPARE=abs=6,summary

# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"
srun ./matmul.exe 
