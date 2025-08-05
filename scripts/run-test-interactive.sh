#!/bin/sh
#SBATCH --job-name=multi_gpu_job              # Descriptive job name
#SBATCH --time=04:00:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=4                             # Number of nodes to use
#SBATCH --ntasks-per-node=4                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:4                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod           # GPU-enabled partition
#SBATCH --qos=boost_qos_dbg                      # Quality of Service
#SBATCH --output=multiGPUJob.out              # File for standard output
#SBATCH --error=multiGPUJob.err               # File for standard error
#SBATCH --account=<project_account>           # Project account number

module load profile/deeplrn
module load cuda nvhpc python
export CUDA=$CUDA_HOME

cd $WORK/VojoLe-LM

source $WORK/.envs/vojollm/bin/activate


python3 -m generation --run
