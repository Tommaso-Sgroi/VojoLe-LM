#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=try25_sgroi ### NOME DEL PROGETTO QUA
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

module load python/3.11.7
module load profile/deeplrn
module load cuda/12.1
module load nvhpc
module load gcc/12.2.0

export CUDA=$CUDA_HOME
export SAVE_MODEL_PATH=$WORK/VojoLe-LM/finetune_outputs
export LOAD_IN_4_BITS=0
export LOAD_IN_8_BITS=0
export UNSLOTH_RETURN_LOGITS=1

source $WORK/VojoLe-LM/.venv_unsloth/bin/activate
cd $WORK/VojoLe-LM

python3 -m finetune.text_completion_finetune --tiny_dataset --run --load_in_4bit --model_name Meta-Llama-31-8B --model_path $FAST/models/Meta-Llama-31-8B
# python3 -m finetune.run_multiple_finetunes --run

# sbatch -J prod_generation --output=$WORK/VojoLe-LM/logs/ds_to_db.out --error=$WORK/VojoLe-LM/logs/ds_to_db.err ./tmp.sh