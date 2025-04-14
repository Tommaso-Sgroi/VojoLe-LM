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
    for sentence in tqdm(dataset, desc='Calculating stems distribution') if use_tqdm else dataset:
        # sentence = id_sentence
        sentence = clean_text(sentence)
        word_tokens = word_tokenize(sentence, language='italian')

        word_tokens = [stemmer.stem(wt) for wt in word_tokens]
        frequency_distribution.update(word_tokens)
    return frequency_distribution

def lemmas_distribution(dataset: list[str], use_tqdm=False, n_process=1) -> FreqDist:
    import spacy
    frequency_distribution = FreqDist()
    lemmatizer = spacy.load("it_core_news_lg")

    iterator = lemmatizer.pipe(dataset, batch_size=64, n_process=n_process)

    if use_tqdm:
        iterator = tqdm(iterator, desc='Calculating lemmas distribution', total=len(dataset))

    for doc in iterator:
        tokens = [clean_text(token.lemma_) for token in doc
                  if token.pos_ != 'PUNCT'
                  and token.text not in italian_stopwords
                  and ' ' not in token.lemma_
                  and clean_text(token.lemma_) != ''
                  ]
        frequency_distribution.update(tokens)

    return frequency_distribution


def parse_distribution(path: str = './data/fineweb2-test-distribution.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _calculate_distribution(distribution_function, dataset, args) -> FreqDist:
    frequency_dist = FreqDist()
    if args.parallel >= 1:
        # for some reason is slower than single core
        m_results = process_map(distribution_function, dataset, max_workers=args.parallel, miniters=1000)

        # merge results
        print('Merging test results')

        for d in tqdm(m_results):
            frequency_dist.update(d)
    else:
        frequency_dist = distribution_function(dataset, True, n_process=args.parallel)

        if not args.skip_train_set:
            stream = get_dataset_stream(TRAIN_SPLIT)

            dataset.clear()
            chunk_size = 50_000
            i = 1
            for entry in tqdm(stream, desc='Preparing chunk...', miniters=1):
                dataset.append(entry['text'])
                if not (i % chunk_size):  # chunk size reached
                    f = distribution_function(dataset, True)
                    frequency_dist.update(f)
                i += 1
            else:
                # if for loop ends without reaching the chunk size
                f = distribution_function(dataset, True)
                frequency_dist.update(f)
    return frequency_dist


def stemming(dataset, args):
    return _calculate_distribution(stemming_distribution, dataset, args)

def lemmatize(dataset, args):
    return _calculate_distribution(lemmas_distribution, dataset, args)


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(
        prog='db-server',
        )
    parser.add_argument('--setup', action='store_true', help='Run nltk.download(\'all\' and exit.')
    parser.add_argument('--delete_data', action='store_true', help='Remove all NLTK data previously downloaded and exit.')
    parser.add_argument('--skip_train_set', action='store_true', default=True, help='Skip the train set for the words frequency count.')
    parser.add_argument('--stem_distribution', action='store_true', default=False, help='Calculate the stems distribution and save it to disk.')
    parser.add_argument('--lemma_distribution', action='store_true', default=False, help='Calculate the lemmas distribution and save it to disk.')
    parser.add_argument('--parallel', '-p', nargs='?', type=int, help='Calculate distributions in parallel.', default=0)
    parser.add_argument('--show_n_common', '-n', nargs='?', type=int, help='Show the n most common words.', default=0)
    args = parser.parse_args()

    if args.show_n_common:
        fd = FreqDist(parse_distribution())
        for n in fd.most_common(args.show_n_common):
            print(n)
        exit(0)

    if args.setup:
        # run python -m spacy download it_core_news_lg # TODO
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
    if args.stem_distribution:
        stemming_frequency = stemming(dataset, args)
        with open('./data/stemming-fineweb2-test-distribution.json', 'w', encoding='utf-8') as f:
            json.dump({k:v for k, v in stemming_frequency.items()}, f) # convert in raw dict and store results
        # print top 10 tokens
        print('_'*10, 'TOP 10 STEMS', '_'*10)
        print(stemming_frequency.most_common(10))

    if args.lemma_distribution:
        lemmatize_frequency = lemmatize(dataset, args)
        with open('./data/lemmatize-fineweb2-test-distribution.json', 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in lemmatize_frequency.items()}, f)  # convert in raw dict and store results
        # print top 10 tokens
        print('_'*10, 'TOP 10 LEMMAS', '_'*10)
        print(lemmatize_frequency.most_common(10))