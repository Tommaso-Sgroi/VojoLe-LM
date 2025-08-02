#!/bin/bash

#SBATCH --job-name=dataset_to_db
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=try25_sgroi

cd $WORK/VojoLe-LM
# source $WORK/VojoLe-LM/.venv_vllm/bin/activate
# huggingface-cli download CohereLabs/c4ai-command-a-03-2025 --local-dir $FAST/models/c4ai --repo-type=model --token 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
# huggingface-cli download --token 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  CohereLabs/c4ai-command-a-03-2025

export SINGULARITY_CACHEDIR=$WORK/VojoLe-LM/singularity/cache
export SINGULARITY_TMPDIR=$WORK/VojoLe-LM/singularity/tmp
export CUDA=$CUDA_HOME
export GOLD_DICT=data2/commons/gold_dictionary.jsonl
export PROMPT_PATH=$WORK/VojoLe-LM/generation_prompt4.txt
export TORCHDYNAMO_VERBOSE=1
export BATCH_SIZE=5

source $WORK/VojoLe-LM/.venv_vllm/bin/activate
python3 -m dataset_maker.download_dataset



# singularity exec --bind $WORK/VojoLe-LM:/code vllm.sif "ls /code && python3 -c \"import torch; torch.cuda.is_available(); print(\"CUDA devices\", torch.cuda.device_count())\""


# singularity remote login -y -p eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2F1dGguc3lsYWJzLmlvL3Rva2VuIiwic3ViIjoiNjg4YTRjNDI2N2Q3NGI1NjI2MDhhMTRlIiwiZXhwIjoxNzU2NDg1OTc2LCJpYXQiOjE3NTM4OTM5NzYsImp0aSI6IjY4OGE0YzU4MDJmMTU5ODI1NzU3ZmVkNCJ9.BTDURmGFQEGRt2iXkprNzLHYBJ48Ve2jSTZl-sPgzxilBeQicghqQyhZv3rAJ_qXfWKbNmgIBXqgHYQkMR0tLdtNsp7Zj9E7RsE_0-qvizQX96lnvS0p4MCAWp-gv-pEoJSFrqk0a3FUH9KqE5ZUn7rQT5sqaE0lvyPQBH2XyQOjPv_cPHrczAf69eOozaZF3W6Qy5kbrA0rpWzMbgo9ZNdsQaKiNMJKpkWVDnb_g64W48ezibLOMyruTFXN3jpOnOhuZiaiof_kegOke7JD2ar8f5TkCg8W5SmN4fDh_GI93U1QESuA7P4NXObRf5keKGoDlc7zmESUYkyaG-yYGQ


 # singularity build $WORK/VojoLe-LM/singularity/cow.sif docker://godlovedc/lolcow # docker://vastai/vllm:v0.9.1-cuda-12.8-pytorch-2.7.0-py312

# python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('CohereLabs/c4ai-command-a-03-2025', token='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').save_pretrained('/leonardo_scratch/fast/try25_sgroi/command_a-tokenizer')" --run
# huggingface-cli download unsloth/Meta-Llama-3.1-8B-Instruct --token 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
# pip install --upgrade pip
# pip install --force-reinstall --no-cache --upgrade pip setuptools wheel
# pip install --force-reinstall --no-cache "vllm==0.9.1" "transformers<4.54.0" accelerate
