### Job Name
#PBS -N GPU_workshop14
### Charging account
### Insert your own project code here when working on self-paced content
#PBS -A UCIS0004
### Specifiy queue, use gpudev queue for faster access to GPUs for debug scale work from 8am to 6:30pm MT. Otherwise, Casper routing queue is fine
#PBS -q gpudev
##PBS -q casper
### Request appropriate number of resources
#PBS -l select=1:ncpus=1:ngpus=1
### Set the GPU type to run on
#PBS -l gpu_type=v100
### Specify walltime limit, ie one minute
#PBS -l walltime=02:00
### Join standard output and error streams into single file
#PBS -j oe
### Specifiy output file name
#PBS -o LSTM.out

export TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/11.7 conda &> /dev/null
conda activate /glade/work/dhoward/conda/envs/GPU_Workshop/

python magnet_lstm_tutorial.py