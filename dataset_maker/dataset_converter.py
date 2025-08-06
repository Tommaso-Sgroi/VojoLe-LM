from vllm import LLM, SamplingParams
import os
from datasets import load_dataset, load_from_disk
from time import time
from dataset_maker.database import DatabaseIta, DatabaseSor
from torch.cuda import device_count
from tqdm import tqdm
from math import ceil
from threading import Thread

model_path = os.path.join(os.getenv('FAST'), 'models', 'c4ai')
dataset_path = os.path.join(os.getenv('FAST'), 'datasets', 'mc4_it')
prompt_path = os.path.join(os.getenv('PROMPT_PATH'))
database_ita_path = os.path.join(os.getenv('DB_ITA'))
database_sor_path = os.path.join(os.getenv('DB_SOR'))
batch_size = int(os.getenv('BATCH_SIZE') or 1) 
max_context = 60_000

print(f"""
==================== Job Configuration ====================
Model path         : {model_path}
Dataset path       : {dataset_path}
Prompt path        : {prompt_path}
ITA DB path        : {database_ita_path}
SOR DB path        : {database_sor_path}
Batch size         : {batch_size}
Max context tokens : {max_context}
GPUs               : {device_count()}
============================================================
""")


def run_batch(llm, entries, *, prompt, sampling_params):
    # global prompt, max_context, sampling_params, llm, tokenizer

    # Construct full prompt strings and truncate with tokenizer if needed
    prompts = [prompt + entry['text'] for entry in entries]

    # print('Tokens lengths: ', [len(tokenizer.encode(e)) for e in prompt])
    outputs = llm.generate(prompts, sampling_params)

    return outputs

    
def run(llm, db_ita: DatabaseIta, db_sor: DatabaseSor, *, batch_size, prompt, sampling_params):

    db_ita.reset_working_status().commit()
    
    items_quantity  = db_ita.count_entries()
    batch_quantity = int(ceil(items_quantity / batch_size))

    start = time()
    with open(os.path.join(os.getenv('WORK'), 'VojoLe-LM', 'logs', 'tqdm.log'), 'w', encoding='utf-8') as file:
        for index in tqdm(range(batch_quantity), mininterval=60, file=file):
            batch_sentences = db_ita.get_next_batch_items(batch_size) # [(sentence_id, sentence_text, train), ... , (sentence_id_n, sentence_text_n, train_n)]
            batch_sentences = [{'sentence_id': sentence_id, 'text': sentence_text, 'is_training': train} for sentence_id, sentence_text, train in batch_sentences ]
            
            outputs = run_batch(llm, batch_sentences, prompt=prompt, sampling_params=sampling_params)

            for index, output in enumerate(outputs):
                bs = batch_sentences[index]
                bs['text'] = output.outputs[0].text
                db_sor.add_translation(**bs)
            else:
                db_sor.commit()             # ;)

            for i, output in enumerate(outputs):
                prompt_used = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt {i}:\n{prompt_used}\nGenerated:\n{generated_text}")
                print('-' * 100)
            if (index % batch_size) == 0:
                print(f'\n\n\nELAPSED: Done {index} prompts in {int(ceil((time() - start)/60))}h\n\n\n')

if __name__ == '__main__':

    db_ita = DatabaseIta(database_ita_path)
    db_sor = DatabaseSor(database_sor_path)


    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    sampling_params = SamplingParams(
            temperature=0.8, 
            top_p=0.95,
            max_tokens=int(max_context * 1.2), 
            truncate_prompt_tokens=max_context,
        )
    
    llm = LLM(model_path, 
            max_model_len=max_context, # context length 
            tensor_parallel_size=device_count(), # number of GPUs
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            max_num_seqs = batch_size,
            enforce_eager=False, # we optimize model inference using CUDA graphs which take up extra memory in the GPU
            )
    tokenizer = llm.get_tokenizer()
    
    
    run(llm, db_ita, db_sor, batch_size=batch_size, sampling_params=sampling_params, prompt=prompt)

