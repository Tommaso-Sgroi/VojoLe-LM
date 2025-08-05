#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=try25_sgroi ### NOME DEL PROGETTO QUA
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

module load python/3.11.7
module load profile/deeplrn
module load cuda nvhpc
module load gcc/12.2.0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $WORK/VojoLe-LM

export CUDA=$CUDA_HOME
export GOLD_DICT=data2/commons/gold_dictionary.jsonl
export PROMPT_PATH=$WORK/VojoLe-LM/generation_prompt4.txt
export TORCHDYNAMO_VERBOSE=1
export BATCH_SIZE=2

nvcc --version

# huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token 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
source $WORK/VojoLe-LM/.venv_vllm/bin/activate
# python3 -c "from accelerate import write_basic_config; write_basic_config()"
# python3 -c "from transformers import AutoTokenizer; at = ; tokenizer.save_pretrained('tokenizer')"

# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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

python3 -m generation1 --run
# accelerate launch generation1.py --num_processes 4 --num_machines 1 --run 
