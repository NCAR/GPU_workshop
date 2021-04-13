#!/bin/bash -l
# Batch directives
#PBS -N matmul
#PBS -A NTDD0002
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=50GB:ngpus=1
#PBS -l gpu_type=v100
#PBS -l walltime=00:05:00
##PBS --reservation=TDD_4xV100 
#PBS -e matmul.err
#PBS -o matmul.log

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11
module load cuda/11.0.3
module list

# Update LD_LIBRARY_PATH so that cuda libraries can be found
export LD_LIBRARY_PATH=${NCAR_ROOT_CUDA}/lib64:${LD_LIBRARY_PATH}
echo ${LD_LIBRARY_PATH}
nvidia-smi

# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"
./matMult.exe 
