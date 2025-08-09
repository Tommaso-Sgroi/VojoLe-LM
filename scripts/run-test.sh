#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=try25_sgroi ### NOME DEL PROGETTO QUA
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

module load python/3.11.7
module load profile/deeplrn
module load cuda nvhpc
module load gcc/12.2.0

cd $WORK/VojoLe-LM

export CUDA=$CUDA_HOME
export GOLD_DICT=data2/commons/gold_dictionary.jsonl
export PROMPT_PATH=$WORK/VojoLe-LM/generation_prompt4.txt
# export DB_ITA=$FAST/er-italiano-fp8.db
# export DB_SOR=$FAST/er-sorianese-fp8.db
export DB_ITA=$WORK/VojoLe-LM/er-italiano.db
export DB_SOR=$WORK/VojoLe-LM/er-sorianese.db
export BATCH_SIZE=5
export NUM_GENERATIONS=3
export TEMPERATURE=0.8
export MAX_CONTEXT=31800
# export QUANTIZATION=fp8  # Set to 'fp8' or 'None' as needed

nvcc --version

source $WORK/VojoLe-LM/.venv_vllm/bin/activate
# huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token hf_XguuLcefAFZBhBYTikpAQjZPbufNHewNd
# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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

python3 -m dataset_maker.dataset_converter --run
