### Job Name
#PBS -N GPU_workshop11
### Charging account
### Insert your own project code here when working on self-paced content
#PBS -A UCIS0004
### Specifiy queue, use gpudev queue for faster access to GPUs for debug scale work from 8am to 6:30pm MT. Otherwise, Casper routing queue is fine
##PBS -q gpudev
##PBS -q casper
### Request appropriate number of resources
##PBS -l select=1:ncpus=1:ngpus=1
### Set the GPU type to run on
#PBS -l gpu_type=v100
### Specify walltime limit, ie one minute
#PBS -l walltime=01:00
### Join standard output and error streams into single file
#PBS -j oe
### Specifiy output file name
##PBS -o nsys_MW-MPI%q{EXEC}.out

ml nvhpc/22.5 openmpi/4.1.4 &> /dev/null

# Add NCCL library to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/glade/u/apps/opt/nvhpc/22.5/Linux_x86_64/22.5/comm_libs/nccl/lib

mpiexec -n $N nsys profile -t mpi,openacc,ucx \
-f true -o nsys_MW-mpi_%q{EXEC}_%q{OMPI_COMM_WORLD_RANK}of%q{N} \
./${EXEC} &> nsys_MW-MPI${EXEC}.out