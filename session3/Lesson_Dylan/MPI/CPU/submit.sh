#!/bin/bash -l
# Batch directives
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --account NTDD0002
#SBATCH --partition=dav
#SBATCH --time=00:15:00
#SBATCH --output=log.stncl_%j.out
#SBATCH --job-name=CPU_stncl

# Load the necessary modules (software)
module purge
module load ncarenv/1.2
module load nvhpc/20.11 
module load cuda/11.0.3
module load openmpi/4.0.5x
module list

# Update LD_LIBRARY_PATH so that cuda libraries can be found
export LD_LIBRARY_PATH=${NCAR_ROOT_CUDA}/lib64:${LD_LIBRARY_PATH}
echo -e "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Move to the correct directory and run the executable
echo -e "\nBeginning code output:\n-------------\n"
mpirun -n 16 ./mpi_cpu_stencil.exe 64 4 
echo -e "\nEnd of code output:\n-------------\n"
