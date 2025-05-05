from flair.data import Sentence
from fuzzywuzzy import process, fuzz
from dataset_maker.dataset import load_tatoeba_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from flair.embeddings import TransformerWordEmbeddings


def load_dictionary(path:str = './data2/commons/silver_dictionary.jsonl'):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return [json.loads(js) for js in lines]


dictionary = load_dictionary()


collection = [d['it'] for d in dictionary]
print(collection)

# init embedding
embedding = TransformerWordEmbeddings('roberta-base')

# create a sentence
sentence = Sentence('The grass is green .')

# embed words in sentence
e = embedding.embed(sentence)
# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)

quit()





for sentence in load_tatoeba_dataset():
    for word in word_tokenize(sentence):
        out = process.extract(word, collection, scorer=fuzz.ratio)
        print(word, out, sep='\n')
        break
    break

out = process.extract("barcelona", collection, scorer=fuzz.ratio)
# Returns: [('barcelona fc', 86), ('AFC Barcelona', 82), ('Barcelona AFC', 82), ('afc barcalona', 73)]
