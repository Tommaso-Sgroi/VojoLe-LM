"""In this module we parse the full dataset and discover the top K words distribution
the objective is about rebuild a dataset with the most uniform token distribution"""
import string

import nltk
from bs4 import BeautifulSoup
from nltk import *
from tqdm import tqdm
from string import punctuation
import re
from dataset_maker.download_dataset import get_dataset_stream
from dataset_maker.dataset import load_dataset

italian_stopwords = nltk.corpus.stopwords.words('italian')


# the following functions are taken from https://github.com/nikhiljsk/preprocess_nlp/blob/master/preprocess/preprocess_nlp.py
def remove_tags(text):
    """
    Helper function for preprocess_nlp which is used to remove HTML Tags, Accented chars, Non-Ascii Characters, Emails and URLs

    :param text: A string to be cleaned

    <Returns the cleaned text>
    """
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)

    text = unicodedata.normalize('NFKD', stripped_text).encode('ascii', 'ignore').decode('utf-8',
                                                                                         'ignore')  # Remove Accented characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove Non-Ascii characters
    text = re.sub("[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", '', text)  # Remove Emails
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    return text

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in italian_stopwords])
    return text

def stemming_distribution(dataset: list[str]):
    distribution = dict()
    # tokenized_dataset = tokenizer.tokenize_sents(dataset)
    stemmer = nltk.stem.SnowballStemmer(language='italian')
    for id_sentence in tqdm(dataset):
        sentence = id_sentence['text']
        sentence = clean_text(sentence)
        word_tokens = word_tokenize(sentence, language='italian')
        for token in word_tokens:
            pass


    return distribution


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='db-server',
        )
    parser.add_argument('--setup', action='store_true', help='Run nltk.download(\'all\' and exit.')
    parser.add_argument('--delete_data', action='store_true', help='Remove all NLTK data previously downloaded and exit.')
    args = parser.parse_args()

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

    d = load_dataset()
    dataset = d[0] + d[1]
    del d

    distribution = stemming_distribution(dataset)
    pass





