#!/bin/bash

#SBATCH --job-name=test_generation
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --account=try25_sgroi

module load python/3.11.7

cd $WORK/VojoLe-LM

python3 -m venv .venv_vllm
source .venv_vllm/bin/activate

pip install --force-reinstall --no-cache -U vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly
