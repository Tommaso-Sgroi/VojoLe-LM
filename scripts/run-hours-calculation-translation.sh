#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=try25_sgroi ### NOME DEL PROGETTO QUA
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00

module load python/3.11.7
module load profile/deeplrn
module load cuda nvhpc
module load gcc/12.2.0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/VojoLe-LM
source $WORK/VojoLe-LM/.venv_vllm/bin/activate


export CUDA=$CUDA_HOME
export GOLD_DICT=data2/commons/gold_dictionary.jsonl
export PROMPT_PATH=$WORK/VojoLe-LM/generation_prompt4.txt
export TORCHDYNAMO_VERBOSE=1
export DB_ITA=$WORK/VojoLe-LM/er-italiano.db
export DB_SOR=$WORK/VojoLe-LM/er-sorianese.db
export BATCH_SIZE=10

nvcc --version

python3 -m dataset_maker.dataset_converter --run
# accelerate launch generation1.py --num_processes 4 --num_machines 1 --run 
