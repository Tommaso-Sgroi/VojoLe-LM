# -*- coding: utf-8 -*-
import unsloth
import os, json, argparse, numpy as np, torch
# from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback, EvalPrediction, TrainerCallback
from unsloth import UnslothTrainer, UnslothTrainingArguments, FastLanguageModel
import numpy as np
import torch.nn.functional as F
from multiprocessing import cpu_count

parser = argparse.ArgumentParser(description="Unsloth model training paths and options")

# Quantization flags
parser.add_argument("--load_in_4bit", default=False, required=False, action="store_true", help="Use 4bit quantization to reduce memory usage")
parser.add_argument("--load_in_8bit", default=False, required=False, action="store_true", help="Use 8bit quantization to reduce memory usage")
parser.add_argument("--tiny_dataset", default=False, required=False, action="store_true", help="Use a tiny dataset for faster training, of size /3 of the original")
parser.add_argument("--run", required=False, action="store_true", help="Leonardo Booster required")

# Model paths
parser.add_argument("--model_path", type=str, required=False, default=None, help="Path to the pretrained model")
parser.add_argument("--save_model_path", type=str, default=None, required=False, help="Directory to save fine-tuned models and logs")
parser.add_argument("--model_name", type=str, default=None, help="Optional model name (defaults to basename of model_path)")

args = parser.parse_args()

# Set quantization options
load_in_4bit = args.load_in_4bit
load_in_8bit = args.load_in_8bit

# Model paths
MODEL_PATH = os.path.join(args.model_path or os.getenv("MODEL_PATH"))
MODEL_NAME = args.model_name or os.path.basename(os.path.normpath(MODEL_PATH)) 
MODEL_OUTPUT_PATH = os.path.join(args.save_model_path or os.getenv("SAVE_MODEL_PATH"), "best", MODEL_NAME)
TMP_MODEL_PATH = os.path.join(args.save_model_path or os.getenv("SAVE_MODEL_PATH"), "tmp", MODEL_NAME)
LOGGED_STATS_PATH = os.path.join(args.save_model_path or os.getenv("SAVE_MODEL_PATH"), "log", MODEL_NAME)
TINY_DATASET = bool(args.tiny_dataset or os.getenv("TINY_DATASET"))

os.makedirs(os.path.join(args.save_model_path or os.getenv("SAVE_MODEL_PATH"), "log"), exist_ok=True)

# Optional: print for verification
print(f"""
{'='*100}
MODEL_PATH = {MODEL_PATH}
MODEL_NAME = {MODEL_NAME}
MODEL_OUTPUT_PATH = {MODEL_OUTPUT_PATH}
TMP_MODEL_PATH = {TMP_MODEL_PATH}
LOGGED_STATS_PATH = {LOGGED_STATS_PATH}
load_in_4bit = {load_in_4bit}, load_in_8bit = {load_in_8bit}
{'='*100}
""")

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+



logged_stats = {
    'raw_perplexity': -1,
    'train_logs': {},
    'fine_tuned_perplexity': -1,
}

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH, #
    local_files_only=True,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    load_in_8bit=load_in_8bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Irrelevant since use_rslora = True; Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 69420,
    use_rslora = True,  # rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



with open('datasets/dataset_sorianese.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train_dataset = data['train']
test_dataset = data['test']
validation_dataset = data['validation']

if TINY_DATASET:
    train_dataset = [d for d in train_dataset if d['sentence_gen_id'] == 0]
    test_dataset = [d for d in test_dataset if d['sentence_gen_id'] == 0]
    validation_dataset = [d for d in validation_dataset if d['sentence_gen_id'] == 0]

del data

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(example):
    return { "text" : example["sentence_text"] + EOS_TOKEN }
train_dataset = list(map(formatting_prompts_func, train_dataset))
test_dataset = list(map(formatting_prompts_func, test_dataset))
validation_dataset = list(map(formatting_prompts_func, validation_dataset))
print('Dataset loaded')

"""Print out 5 sentences."""

for row in train_dataset[:5]:
    print("=========================")
    print(row["text"])

## Start train

from datasets import Dataset

train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)
validation_dataset = Dataset.from_list(validation_dataset)

# You can display the structure of the datasets to verify
print("Train dataset structure:")
print(train_dataset)
print("\nValidation dataset structure:")
print(validation_dataset)
print("\nTest dataset structure:")
print(test_dataset)

"""#### Add Callback to calculate Perplexity

## Prepare Trainer
We prepare the train/validation with Hyperparameters.
"""


trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(

        per_device_train_batch_size = 32, # 32,
        gradient_accumulation_steps = 1,

        warmup_ratio = 0.1,
        num_train_epochs = 10, # thanks to the early stopping

        learning_rate = 2e-5, #5e-5,
        embedding_learning_rate = 1e-6, # 5e-6,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 42069,
        
        output_dir = TMP_MODEL_PATH,
        logging_dir=os.path.join(TMP_MODEL_PATH, "logs"),
        report_to = "none", # Use this for WandB etc
        
        metric_for_best_model="eval_loss",
        eval_strategy="epoch",
        save_strategy = "best",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit = 2,

        logging_steps = 50,

    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.1)],
)


"""### Raw Perplexity on Untrained Model"""

print('-'*200,'\n[PRE-TRAIN EVALUATION]\n',  '-'*200)
# Evaluate the currently loaded and trained model on the test dataset
test_results = trainer.evaluate()
print(test_results)
# The test loss will be in the results
test_loss = test_results["eval_loss"]
print(f"Test Loss: {test_loss}")

# Calculate perplexity from the test loss
test_perplexity = np.exp(test_loss)
logged_stats["raw_perplexity"] = test_perplexity
print(f"Test Perplexity: {test_perplexity}")

"""## Train

By now, we are going to train our model's on the Dataset, at the end of the train we save the best model found during training.
"""

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print('-'*200,'\n[TRAIN]\n',  '-'*200)
trainer_stats = trainer.train()
trainer.save_model(MODEL_OUTPUT_PATH)

# Access the training log history
log_history = trainer.state.log_history
logged_stats["train_logs"] = log_history
# print(log_history)
# Print the full log history
print(json.dumps(log_history, indent=2))

"""## Just Evaluation
We are going to evaluate various models on the completion task using the test dataset. We calculate the Perplexity upon the validation dataset.
"""

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Evaluate the currently loaded and trained model on the test dataset
test_results = trainer.evaluate() # (eval_dataset=validation_dataset)
print(test_results)
# The test loss will be in the results
test_loss = test_results["eval_loss"]
print(f"Test Loss: {test_loss}")

# Calculate perplexity from the test loss
test_perplexity = np.exp(test_loss)
logged_stats["fine_tuned_perplexity"] = test_perplexity
print(f"Test Perplexity: {test_perplexity}")


## save the logged stats
log_path = LOGGED_STATS_PATH + '.log'

with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(logged_stats, f, indent=2)
    print(f"Stats logged at `{log_path}`")
