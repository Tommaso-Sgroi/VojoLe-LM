from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import os
from time import time
from dataset_maker.database import DatabaseIta, DatabaseSor
from torch.cuda import device_count
from tqdm import tqdm
from math import ceil

hf_token = os.getenv('HF_TOKEN')
model_path = os.path.join(os.getenv('FAST'), 'models', 'c4ai')
dataset_path = os.path.join(os.getenv('FAST'), 'datasets', 'mc4_it')
prompt_path = os.path.join(os.getenv('PROMPT_PATH'))
database_ita_path = os.path.join(os.getenv('DB_ITA'))
database_sor_path = os.path.join(os.getenv('DB_SOR'))
batch_size = int(os.getenv('BATCH_SIZE') or 1)
max_context = int(os.getenv('MAX_CONTEXT'))

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


def run_batch(llm: LLM, entries, *, prompt, sampling_params: SamplingParams, tokenizer):
    # global prompt, max_context, sampling_params, llm, tokenizer

    # Construct full prompt strings and truncate with tokenizer if needed
    prompts = [prompt + entry['text'] for entry in entries]

    # calculate max and min tokens
    tokens_length = [len(tokenizer.encode(entry['text'])) for entry in entries]
    max_tokens = max([ceil(i * 2) for i in tokens_length])
    min_tokens = min(tokens_length)

    # print('Tokens lengths: ', [len(tokenizer.encode(e)) for e in prompt])
    sampling_params.max_tokens = max_tokens
    sampling_params.min_tokens = min_tokens
    outputs = llm.generate(prompts, sampling_params)

    return outputs


def run(llm, db_ita: DatabaseIta, db_sor: DatabaseSor, *, batch_size, prompt, sampling_params, tokenizer):
    db_ita.reset_working_status().commit()

    items_quantity = db_ita.count_entries()
    batch_quantity = int(ceil(items_quantity / batch_size))

    start = time()
    with open(os.path.join(os.getenv('WORK'), 'VojoLe-LM', 'logs', 'tqdm.log'), 'w', encoding='utf-8') as file:
        for index in tqdm(range(batch_quantity), mininterval=60, file=file):
            batch_sentences = db_ita.get_next_batch_items(batch_size)  # [(sentence_id, sentence_text, train), ... , (sentence_id_n, sentence_text_n, train_n)]
            batch_sentences = [{'sentence_id': sentence_id, 'text': sentence_text, 'is_training': train} for
                               sentence_id, sentence_text, train in batch_sentences]

            outputs = run_batch(llm, batch_sentences, prompt=prompt, sampling_params=sampling_params, tokenizer=tokenizer)

            for index, output in enumerate(outputs):
                bs = batch_sentences[index]
                bs['text'] = output.outputs[0].text
                db_sor.add_translation(**bs)
            else:
                db_sor.commit()  # ;)

            for i, output in enumerate(outputs):
                prompt_used = output.prompt
                generated_text = output.outputs[0].text
                print(f"Generated:\n{generated_text}\n{'='*1000}")
            if (index % batch_size) == 0:
                print(f'\n\n\nELAPSED: Done {index} prompts in {int(ceil((time() - start) / 60))}h\n\n\n')


def get_translation_schema():
    from pydantic import BaseModel
    class SorianeseTranslation(BaseModel):
        translation: str

    translation_json_schema = SorianeseTranslation.model_json_schema()
    return translation_json_schema

def get_tokens_number_statistics(tokenizer_path, db_ita: DatabaseIta, prompt, hf_token:str = None):
    from transformers import AutoTokenizer
    import statistics

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)

    batch_sentences_ = db_ita.get_all_items()
    batch_sentences_ = [{'text': sentence_text} for
                       _, sentence_text, _ in batch_sentences_]
    batch_sentences = [len(tokenizer.encode(prompt + sentence['text'])) for sentence in tqdm(batch_sentences)]
    sentence_without_prompts = [len(tokenizer.encode(sentence['text'])) for sentence in tqdm(batch_sentences_)]

    print('Calculating statistics')
    return {
        'Total sentences + system prompt': len(batch_sentences),
        'Mean + system prompt': statistics.mean(batch_sentences),
        'Standard deviation + system prompt': statistics.stdev(batch_sentences),
        'Max length + system prompt': max(batch_sentences),
        'Min length + system prompt': min(batch_sentences),
        'Mean no system prompt': statistics.mean(sentence_without_prompts),
        'Standard deviation no system prompt': statistics.stdev(sentence_without_prompts),
        'Max length no system prompt': max(sentence_without_prompts),
        'Min length no system prompt': min(sentence_without_prompts),
    }


if __name__ == '__main__':
    db_ita = DatabaseIta(database_ita_path)
    db_sor = DatabaseSor(database_sor_path)

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    """
    stats = get_tokens_number_statistics(model_path, db_ita, prompt, hf_token)
    print(stats)
    quit()
    """
    guided_decoding_json = GuidedDecodingParams(json=get_translation_schema())
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        guided_decoding=guided_decoding_json,
        # max_tokens=int(max_context * 1.2),
        # truncate_prompt_tokens=max_context,
    )

    llm = LLM(model_path,
            max_model_len=max_context,  # context length
            tensor_parallel_size=device_count(),  # number of GPUs
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            max_num_seqs=batch_size,
            enforce_eager=False,
            quantization="fp8_e4m3", # they load the model at original precision before quantizing down to 8-bits, so you need enough memory to load the whole model
        )
    tokenizer = llm.get_tokenizer()

    run(llm, db_ita, db_sor, batch_size=batch_size, sampling_params=sampling_params, prompt=prompt, tokenizer=tokenizer)

