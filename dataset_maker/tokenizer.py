import multiprocessing
from functools import partial

from tqdm import tqdm
import json

def read_fulldataset():
    text = []
    with(open('./data/fineweb-2/data/ita_Latn/test.jsonl', 'r')) as f:
        tl = f.readlines()
        for l in tl:
            text.append(json.loads(l)['text'])

    with(open('./data/fineweb-2/data/ita_Latn/train.jsonl', 'r')) as f:
        tl = f.readlines()
        for l in tl:
            text.append(json.loads(l)['text'])
    return text

def read_dataset_and_prompt():
    with(open('./data/generation_prompt.txt', 'r')) as f:
        it = f.read()
    return [it + l for l in read_fulldataset()]

from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer

# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
from time import time
#
# gpt_tokens = [tokenizer.encode(text) for text in tqdm(text)]
#
from deepseek_tokenizer import deepseek_tokenizer

# deepseek_tokens = [ds_token.encode(text) for text in tqdm(text)]
with multiprocessing.Pool(8) as pool:
    results = pool.map(partial(gpt_tokenizer.encode, truncation=True), read_fulldataset())


sum_tokens = 0
for result in results:
    sum_tokens += len(result)

print('#expected-out GPT-tokens')
print('average: ', sum_tokens / len(results))
print('Total tokens: ', sum_tokens)

del results
with multiprocessing.Pool(8) as pool:
    results = pool.map(partial(gpt_tokenizer.encode, truncation=True), read_dataset_and_prompt())

sum_tokens = 0
for result in results:
    sum_tokens += len(result)

print('#expected-input GPT-tokens')
print('average: ', sum_tokens / len(results))
print('Total tokens: ', sum_tokens)

del results
with multiprocessing.Pool(8) as pool:
    results = pool.map(partial(deepseek_tokenizer.encode, truncation=True), read_fulldataset())

# Process the results
sum_tokens = 0
for result in results:
    sum_tokens += len(result)
print('#expected-output deepseek-tokens')
print('average: ', sum_tokens / len(results))
print('Total tokens: ', sum_tokens)

print('#input tokens')

del results
with multiprocessing.Pool(8) as pool:
    results = pool.map(partial(deepseek_tokenizer.encode, truncation=True), read_dataset_and_prompt())

# Process the results
sum_tokens = 0
for result in results:
    sum_tokens += len(result)
print('#expected-input deepseek-tokens')
print('average: ', sum_tokens / len(results))
print('Total tokens: ', sum_tokens)


print('si caga addosso e muore')
