#!/bin/bash -l
#PBS -A UCIS0004
#PBS -N nsight_prof
#PBS -j oe
#PBS -k oed
#PBS -q casper
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l gpu_type=gp100

export TMPDIR=/glade/scratch/${USER}/temp
mkdir -p $TMPDIR
module load nvhpc cuda

### Run
#nsys profile -t 'openacc' --stats true -f true -o vecAdd_profile_${PBS_JOBID} ./vecAdd
#nsys profile vecAdd -t cuda --stats true -f true -o vecAdd_profile_jh

nsys profile openacc_orig -t openacc --stats true -f true -o miniweather_profile
 
### Store job stats
#qstat -f $PBS_JOBID
