#!/bin/bash

#SBATCH --job-name=test_generation
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --account=try25_sgroi

cd $WORK/VojoLe-LM
# huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token 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
source $WORK/VojoLe-LM/.venv/bin/activate
huggingface-cli download --include tokenizer --token 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  unsloth/Meta-Llama-3.1-8B-Instruct 

# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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
# python3 generation.py
