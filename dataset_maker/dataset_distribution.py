"""In this module we parse the full dataset and discover the top K words distribution
the objective is about rebuild a dataset with the most uniform token distribution"""
import string

import nltk
# from bs4 import BeautifulSoup
from nltk import *
from tqdm import tqdm
from string import punctuation
import re

from tqdm.contrib.concurrent import process_map

from dataset_maker.download_dataset import get_dataset_stream, TRAIN_SPLIT
from dataset_maker.dataset import load_dataset

italian_stopwords = nltk.corpus.stopwords.words('italian')

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in italian_stopwords])
    return text

def stemming_distribution(dataset: list[str], use_tqdm=False) -> FreqDist:
    frequency_distribution = FreqDist()

    # tokenized_dataset = tokenizer.tokenize_sents(dataset)
    stemmer = nltk.stem.SnowballStemmer(language='italian')
    for sentence in tqdm(dataset) if use_tqdm else dataset:
        # sentence = id_sentence
        sentence = clean_text(sentence)
        word_tokens = word_tokenize(sentence, language='italian')
        for i in range(len(word_tokens)):
            token_stem = stemmer.stem(word_tokens[i])
            word_tokens[i] = token_stem
            # increment the found token of 1
        frequency_distribution.update(word_tokens)
    return frequency_distribution

# def lemmas_distribution(dataset: list[str], use_tqdm=False):
#     distribution = dict()
#     for sentence in tqdm(dataset) if use_tqdm else dataset:
#         # sentence = id_sentence
#         sentence = clean_text(sentence)
#         tokenized_text = word_tokenize(sentence, language='italian')
#         tokens_tag = pos_tag(tokenized_text, lang='ita')
#         pass
#         for i in range(len(word_tokens)):
#             token_stem = stemmer.stem(word_tokens[i])
#             word_tokens[i] = token_stem
#
#             distribution[token_stem] = 1 + distribution.get(token_stem, 0)
#             # increment the found token of 1
#
#     return distribution

def parse_distribution(path: str = './data/fineweb2-test-distribution.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(
        prog='db-server',
        )
    parser.add_argument('--setup', action='store_true', help='Run nltk.download(\'all\' and exit.')
    parser.add_argument('--delete_data', action='store_true', help='Remove all NLTK data previously downloaded and exit.')
    parser.add_argument('--skip_train_set', action='store_true', default=True, help='Skip the train set for the words frequency count.')
    parser.add_argument('--parallel', '-p', nargs='?', type=int, help='Calculate distributions in parallel.', default=0)
    parser.add_argument('--show_n_common', '-n', nargs='?', type=int, help='Show the n most common words.', default=0)
    args = parser.parse_args()

    if args.show_n_common:
        fd = FreqDist(parse_distribution())
        for n in fd.most_common(args.show_n_common):
            print(n)
        exit(0)

    if args.setup:
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        print('Additional resources downloaded')
        exit(0)
    if args.delete_data:
        import os, shutil
        paths = nltk.data.path
        for p in nltk.data.path:
            if os.path.exists(p):
                shutil.rmtree(p)
        exit(0)

    print('loading dataset\n')
    d = load_dataset()
    dataset = d[1] + d[0]
    dataset = [e['text'] for e in dataset]

    # del d
    print('processing')
    frequency_dist = FreqDist()
    if args.parallel >= 1:
        # for some reason is slower than single core
        m_results = process_map(stemming_distribution, dataset, max_workers=args.parallel, miniters=1000)

        # merge results
        print('Merging test results')

        for d in tqdm(m_results):
            frequency_dist.update(d)
    else:
        frequency_dist = stemming_distribution(dataset, True)

        if not args.skip_train_set:
            stream = get_dataset_stream(TRAIN_SPLIT)

            dataset.clear()
            chunk_size = 50_000
            i = 1
            for entry in tqdm(stream, desc='Preparing chunk...', miniters=1):
                dataset.append(entry['text'])
                if not (i % chunk_size): # chunk size reached
                    f = stemming_distribution(dataset, True)
                    frequency_dist.update(f)
                    dataset.clear()
                i += 1
            else:
                # if for loop ends without reaching the chunk size
                f = stemming_distribution(dataset, True)
                frequency_dist.update(f)
                dataset.clear()


    with open('./data/fineweb2-test-distribution.json', 'w', encoding='utf-8') as f:
        json.dump({k:v for k, v in frequency_dist.items()}, f) # convert in raw dict and store results

    # print top 10 tokens
    print(frequency_dist.most_common(10))






