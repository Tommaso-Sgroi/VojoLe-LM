import multiprocessing
from functools import partial

import tiktoken
from tqdm import tqdm
import json

def read_fulldataset():
    text = []
    with(open('./data/clean_mc4_it/clean_mc4_it.jsonl', 'r')) as f:
        tl = f.readlines()
        for l in tl:
            text.append(json.loads(l)['text'])

    # with(open('./data/fineweb-2/data/ita_Latn/train.jsonl', 'r')) as f:
    #     tl = f.readlines()
    #     for l in tl:
    #         text.append(json.loads(l)['text'])
    return text

def read_dataset_and_prompt():
    with(open('./data/raw_resources/generation_prompt2.txt', 'r')) as f:
        it = f.read()
    return [it + l for l in read_fulldataset()]


from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


gpt_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

def calculate_length_tokens(text):
    global gpt_tokenizer
    return len(gpt_tokenizer.encode(text))

from time import time
#
# gpt_tokens = [tokenizer.encode(text) for text in tqdm(text)]
#
from deepseek_tokenizer import deepseek_tokenizer

# deepseek_tokens = [ds_token.encode(text) for text in tqdm(text)]
with multiprocessing.Pool(8) as pool:
    results = pool.map(calculate_length_tokens, read_fulldataset())


sum_tokens = sum(results)


print('#expected-out GPT-tokens')
print('average: ', sum_tokens / len(results))
print('Total tokens: ', sum_tokens)

del results
with multiprocessing.Pool(8) as pool:
    results = pool.map(calculate_length_tokens, read_dataset_and_prompt())

sum_tokens = sum(results)
print('#expected-input GPT-tokens')
print('average: ', sum_tokens / len(results))
print('Total tokens: ', sum_tokens)
