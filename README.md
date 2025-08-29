# Finetune Script Guide

This README documents only how to run the finetune script in this repository. It is not comprehended of the script to translate
the dataset, since it get 4 A100 GPUS with 64GB of VRAM and almost 6 days of run.<br/>The best checkpoint adapters are available at `models/best`.

## Organization
Dialect dataset is available at `datasets/dataset_sorianese.zip`.<br/>
Databases used for translation are in  `databases/databases.zip`.<br/>
Under `finetune` package there are the plot and finetune script. <br/> 
Under `evaluate_translation` there is the streamlit script for hosting a service to evaluate the dialect sentences.<br/>
Under `dataset_maker` there are all the scripts and module used to generate the dataset and calculate some statistics.<br/>
`dialect_resources` contains the generation prompt and the gold dictionary dialect-Italian.<br/>
## 1. Prerequisites

- Python 3.11+
- CUDA-capable GPU(s) with matching NVIDIA drivers
- Sufficient VRAM (>=24GB recommended for full finetune; LoRA/QLoRA can use less)
- Training and validation dataset files (JSONL)
- Any Linux OS

## 2. Environment Setup

```bash
python -m venv .unsloth_venv
source .unsloth_venv/bin/activate

pip install -U pip
pip install unsloth tqdm huggingface-hub
unzip -j datasets/dataset_sorianese.zip -d datasets/
```
## 3. Install Models 

```
hf download unsloth/Meta-Llama-3.1-8B --repo-type model --local-dir models/Meta-Llama-3.1-8B
hf download unsloth/mistral-7b-v0.3-bnb-4bit --repo-type model --local-dir models/Mistral-7B 
```

## 4. Train Models
Run train, remove `--tiny_dataset` to run a full sized dataset finetune, otherwise it will just use 1/3 of the size (basically remove all augmented samples). 
```bash
python3 -m finetune.text_completion_finetune --tiny_dataset --run --load_in_4bit --model_name Meta-Llama-31-8B --model_path ./models/Meta-Llama-31-8B --save_model_path models/
python3 -m finetune.text_completion_finetune --tiny_dataset --run --load_in_4bit --model_name Mistral-7B --model_path ./models/Mistral-7B --save_model_path models/

```
## 5. Plot Graphs
Plot graphs from logged stats at ``
```bash
python3 -m finetune.plot
ls plot/
```

## 6.  Report
Read the [report](https://github.com/Tommaso-Sgroi/VojoLe-LM/blob/main/DLAI_25_VojoLe_LM.pdf) too!

