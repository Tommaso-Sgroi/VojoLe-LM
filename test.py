from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from time import time

tk = AutoTokenizer.from_pretrained('CohereLabs/c4ai-command-a-03-2025', token='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')

with open('generation_prompt.txt') as f:
    prompt = f.read()

mc4_it_full_stream = load_dataset("gsarti/clean_mc4_it", "tiny", streaming=False, cache_dir='.cache')
train = mc4_it_full_stream['train']
validation = mc4_it_full_stream['validation']

train = train.add_column("processed", [False] * len(train))
start = time()
train.save_to_disk('data/clean_mc4_it')
print(f'elapsed {time() - start}')




























quit()
max_train_len, max_val_len = 0, 0
train_exceed_60k, val_exceed_60k  = 0, 0
for entry in tqdm(train):
    m = len(entry['text'])
    max_train_len = max(max_train_len, m)
    train_exceed_60k += (m > 60_000)

for entry in tqdm(validation):
    m = len(entry['text'])
    max_val_len = max(max_val_len, m)
    val_exceed_60k += (m > 60_000)

print('max train token len:', max_train_len)
print('max valid token len:', max_val_len)

print(f'Values that exeed 60k :\n\ttrain: {train_exceed_60k}\n\tvalidation: {val_exceed_60k}')