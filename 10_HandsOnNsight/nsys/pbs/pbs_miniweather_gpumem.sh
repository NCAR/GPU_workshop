#!/bin/bash -l
#PBS -A UCIS0004
#PBS -N nsight_prof
#PBS -j oe
#PBS -k oed
#PBS -q casper
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=36:ngpus=1
#PBS -l gpu_type=gp100

export TMPDIR=/glade/scratch/${USER}/temp
mkdir -p $TMPDIR
module load nvhpc cuda

### Run
nsys profile -o miniweather_gpumem2 fortran/build/openacc -t cuda,openacc,mpi --stats=true --force-overwrite=true --cuda-memory-usage=true

### Label

### Store job stats
#qstat -f $PBS_JOBID
