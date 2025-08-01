#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=try25_sgroi ### NOME DEL PROGETTO QUA
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

module load profile/deeplrn
module load cuda nvhpc
module load gcc/12.2.0

export CUDA=$CUDA_HOME

cd $WORK/VojoLe-LM
# huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token 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
source $WORK/VojoLe-LM/.venv_unsloth/bin/activate

# python3 -c "from transformers import AutoTokenizer; at = ; tokenizer.save_pretrained('tokenizer')"

# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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
accelerate launch generation.py --run 
