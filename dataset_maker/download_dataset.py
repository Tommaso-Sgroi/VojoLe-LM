# from huggingface_hub import snapshot_download
#
# def download_full_dataset():
#     folder = snapshot_download(
#                     "HuggingFaceFW/fineweb-2",
#                     repo_type="dataset",
#                     local_dir="./data/fineweb2/",
#                     # download the Ita filtered + test data
#                     allow_patterns=["data/ita_Latn/train/*", "data/ita_Latn/test/*"],)
#     return folder
#
# TRAIN_SPLIT, TEST_SPLIT = 'train', 'test'
# def get_dataset_stream(split:str):
#     from datasets import load_dataset
#     # get Croatian data
#     fw = load_dataset("HuggingFaceFW/fineweb-2", name="ita_Latn", split=split, streaming=True)
#     return fw

import os.path
from datasets import load_from_disk
from tqdm import tqdm
from time import time
from dataset_maker.database import *

mc4_it_full_stream = load_from_disk(dataset_path=os.path.join(os.getenv('FAST'), 'datasets', 'mc4_it'))
train = mc4_it_full_stream['train']
validation = mc4_it_full_stream['validation']

base_uri = os.path.join('.', 'data', 'clean_mc4_it', 'tiny')

start = time()
sqlite_db = DatabaseIta(os.path.join(os.getenv('FAST'), 'er-italiano1.db'))
sqlite_sr_db = DatabaseSor(os.path.join(os.getenv('FAST'), 'er-sorianese1.db'))
sqlite_db.create_tables()
sqlite_sr_db.create_tables()

# quit()
batch_phrases = []
for i, entry in tqdm(enumerate(train)):
    sqlite_db.add_entry(i, entry['text'], 1)
    if (i % 1_000_000) == 0:
        sqlite_db.conn.commit()
        batch_phrases = []
else:
    end = i + 1
sqlite_db.conn.commit()

batch_phrases.clear()
for i, entry in tqdm(enumerate(validation, start=end)):
    sqlite_db.add_entry(i, entry['text'], 0)
    if (i % 1_000) == 0:
        sqlite_db.conn.commit()
        batch_phrases = []
sqlite_db.conn.commit()

sqlite_db.conn.close()
sqlite_sr_db.conn.close()
print('Done in', time() - start, "seconds")
