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
source $WORK/VojoLe-LM/.venv/bin/activate
huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token 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

# huggingface-cli download --token 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  CohereLabs/c4ai-command-a-03-2025


# python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('CohereLabs/c4ai-command-a-03-2025', token='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').save_pretrained('/leonardo_scratch/fast/try25_sgroi/command_a-tokenizer')" --run
# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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
python3 generation.py
