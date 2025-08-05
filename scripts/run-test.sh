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
# export TORCHDYNAMO_VERBOSE=1
export DB_ITA=$WORK/VojoLe-LM/er-italiano.db
export DB_SOR=$WORK/VojoLe-LM/er-sorianese.db
export BATCH_SIZE=5


nvcc --version

source $WORK/VojoLe-LM/.venv_vllm/bin/activate
# huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token hf_XguuLcefAFZBhBYTikpAQjZPbufNHewNd
# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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

python3 -c "import torch; print(torch.cuda.get_device_capability())"
python3 -m dataset_maker.dataset_converter --run
