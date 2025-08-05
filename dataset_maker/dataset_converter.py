from vllm import LLM, SamplingParams
import os
from datasets import load_dataset, load_from_disk
from time import time




model_path = os.path.join(os.getenv('FAST'), 'models', 'c4ai')
dataset_path = os.path.join(os.getenv('FAST'), 'datasets', 'mc4_it')
prompt_path = os.path.join(os.getenv('PROMPT_PATH'))
batch_size = int(os.getenv('BATCH_SIZE')) if os.getenv('BATCH_SIZE') != '' else 1
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
        max_tokens=int(max_context * 1.5), 
        truncate_prompt_tokens=max_context,
        stop=["}"]
        # logprobs=True,
        # prompt_logprobs=-1,
    )
llm = LLM(model_path, 
          max_model_len=max_context, # context length 
          tensor_parallel_size=4, # number of GPUs
          gpu_memory_utilization=0.8,
          dtype="bfloat16",
          enforce_eager=False, # we optimize model inference using CUDA graphs which take up extra memory in the GPU
        )
tokenizer = llm.get_tokenizer()
print('\nGenerating!! EURECAAA\n')


def run_batch(entries):
    global prompt, max_context, sampling_params, llm, tokenizer

    # Construct full prompt strings and truncate with tokenizer if needed
    prompts = [prompt + entry['text'] for entry in entries]
    """
    prompts = []
    for entry in entries:
        full_prompt = prompt + entry['text']
        tokenized = tokenizer.encode(full_prompt)
        truncated = tokenizer.decode(tokenized[-max_context:])
        prompts.append(truncated)
    """
    print('Tokens lengths: ', [len(tokenizer.encode(e)) for e in prompt])
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt_used = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i}:\n{prompt_used}\nGenerated:\n{generated_text}")
        print('-' * 100)


prompts, outputs = [], []
start = time()
for i, entry in enumerate(train, start=1):
    prompts.append(entry)

    if (i % batch_size) == 0:
        run_batch(prompts)
        print(f'\n\n\nELAPSED: Done {i} prompts in {time() - start}s\n\n\n')
        prompts.clear()

