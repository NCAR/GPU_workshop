#!/bin/bash -l
#PBS -A SCSG0001
#PBS -N python_cfd_perf
#PBS -j oe
#PBS -k oed
#PBS -q casper
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=18:ompthreads=18:mem=16GB:ngpus=1
#PBS -l gpu_type=v100

export TMPDIR=/glade/scratch/${USER}/temp
mkdir -p $TMPDIR
module load nvhpc cuda
module load conda/latest
### Use your own virtual environment
conda activate pgpu

### Run
python step11_perf_datamovement.py > results.txt
conda deactivate

### Store job stats
#qstat -f $PBS_JOBID