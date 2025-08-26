import datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import json, os, statistics
from transformers import AutoTokenizer


def calculate_length_tokens(text, local_tokenizer):
    return len(local_tokenizer.encode(text))


ilpost = datasets.load_dataset(path=os.path.join(os.getenv('FAST'), 'datasets', 'ilpost'))
fanpage = datasets.load_dataset(path=os.path.join(os.getenv('FAST'), 'datasets', 'fanpage'))

ilpost = datasets.concatenate_datasets([ilpost['train'], ilpost['test'], ilpost['validation']])
fanpage = datasets.concatenate_datasets([fanpage['train'], fanpage['test'], fanpage['validation']])

italian_dataset = datasets.concatenate_datasets([ilpost, fanpage])
del ilpost, fanpage

base_path = os.path.join(os.getenv('FAST'),  'models')
tokenizer_paths = [os.path.join(base_path, 'Mistral-7B'), os.path.join(base_path, 'Meta-Llama-31-8B')]
with open('datasets/dataset_sorianese.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    data = data['train'] + data['test'] + data['validation']
    data = [d['sentence_text'] for d in data]

for i in range(5):
    print(data[i])

for i in range(5):
    print(italian_dataset[i]['target'])

for tokenizer_path in tokenizer_paths:
    local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    results = []

    for text in tqdm(data, f'{os.path.basename(tokenizer_path)} - Processing Sorianese dataset'):
        results.append(calculate_length_tokens(text, local_tokenizer))

    print(f"""DIALECT
    #{os.path.basename(tokenizer_path)}-tokens
    average: {statistics.mean(results):.2f}
    Standard deviation: {statistics.stdev(results):.2f}
    Total tokens: {sum(results)}
    """)
    results = []

    for entry in tqdm(italian_dataset, desc=f'{os.path.basename(tokenizer_path)} -Processing Italian dataset'):
        results.append(calculate_length_tokens(entry['target'], local_tokenizer))


    print(f"""ITALIAN
    #{os.path.basename(tokenizer_path)}-tokens
    average: {statistics.mean(results):.2f}
    Standard deviation: {statistics.stdev(results):.2f}
    Total tokens: {sum(results)}
    {'#'*200}"""
    )