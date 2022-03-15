#!/bin/bash -l
# Batch directives
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account NTDD0002
#SBATCH --partition=dav
#SBATCH --reservation=casper_8xV100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=log.stncl_%j.out
#SBATCH --job-name=GPU_stncl

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11
module load cuda/11.0.3
module list

echo -e "nvidia-smi output follows:"
nvidia-smi
# Update LD_LIBRARY_PATH so that cuda libraries can be found
export LD_LIBRARY_PATH=${NCAR_ROOT_CUDA}/lib64:${LD_LIBRARY_PATH}
echo -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

export NV_ACC_TIME=1

# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"
srun ./acc_stencil.exe
echo -e "\nEnd of code output:\n-------------\n"
