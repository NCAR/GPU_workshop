#!/bin/bash -l
# Batch directives
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account NTDD0002
#SBATCH --partition=dav
#SBATCH --reservation=casper_8xV100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=log.matrix_mul_%j.out
#SBATCH --job-name=GPU_matrix_mul

module purge
module load ncarenv/1.2
module load nvhpc/20.11
module load cuda/11.0.3
module list


# Update LD_LIBRARY_PATH so that cuda libraries can be found
#export LD_LIBRARY_PATH=${NCAR_ROOT_CUDA}/lib64:${LD_LIBRARY_PATH}
#echo ${LD_LIBRARY_PATH} 

export CUDA_ROOT_PATH="${NCAR_ROOT_CUDA}"

#Remove any previous build attemps
make clean
#Build code
make
#Run code
srun ./output.exe
