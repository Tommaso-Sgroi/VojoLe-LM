

from vllm import LLM, SamplingParams
import os
from datasets import load_dataset, load_from_disk
from time import time




model_path = os.path.join(os.getenv('FAST'), 'models', 'c4ai')
dataset_path = os.path.join(os.getenv('FAST'), 'datasets', 'mc4_it')
prompt_path = os.path.join(os.getenv('PROMPT_PATH'))
batch_size = 5
max_context = 80_000

mc4_it_tiny = load_from_disk(dataset_path=dataset_path)
# mc4_it_tiny.save_to_disk(dataset_path)

train = mc4_it_tiny['train']
validation = mc4_it_tiny['validation']

with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()

sampling_params = SamplingParams(
        temperature=0.8, 
        top_p=0.95,
        max_tokens=max_context, 
        logprobs=True
    )
llm = LLM(model_path, 
          max_model_len=max_context, # context length 
          tensor_parallel_size=4, # number of GPUs
          gpu_memory_utilization=0.9,
          dtype="bfloat16",
          enforce_eager=False, # we optimize model inference using CUDA graphs which take up extra memory in the GPU
        )
tokenizer = llm.get_tokenizer()
print('\nGenerating!! EURECAAA\n')

def run_batch(entries):
    global prompt, max_context
    prompts = [
        tokenizer.encode(
                prompt + entry['text']
            )[-max_context:] # truncate the input
        for entry in entries
    ]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {i}, Generated text: {generated_text}")
        print('-'*250)

prompts, outputs = [], []
start = time()
for i, entry in enumerate(train, start=1):
    prompts.append(entry)

    if (i % batch_size) == 0:
        run_batch(prompts)
        print(f'\n\n\nELAPSED: Done {i} prompts in {time() - start}s\n\n\n')
        prompts.clear()

