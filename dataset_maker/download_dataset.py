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
from datasets import load_dataset
from tqdm import tqdm

from dataset_maker.job_server import Database
mc4_it_full_stream = load_dataset("gsarti/clean_mc4_it", "tiny", streaming=False, cache_dir='.cache')

train = mc4_it_full_stream['train']
validation = mc4_it_full_stream['validation']

base_uri = os.path.join('.', 'data', 'clean_mc4_it', 'tiny')
sqlite_db = Database()
# sqlite_db.create_database()
# quit()
batch_phrases = []
for i, entry in tqdm(enumerate(train)):
    batch_phrases.append((i, entry['text'], 0, 1))
    if (i % 1_000_000) == 0:
        sqlite_db.add_phrase(batch_phrases, None, None)
        sqlite_db.conn.commit()
        batch_phrases = []
else:
    end = i + 1

if len(batch_phrases) > 0:
    sqlite_db.add_phrase(batch_phrases, None, None)
sqlite_db.conn.commit()

batch_phrases.clear()

for i, entry in tqdm(enumerate(validation, start=end)):
    batch_phrases.append((i, entry['text'], 0, 0))
    if (i % 1_000) == 0:
        sqlite_db.add_phrase(batch_phrases, None, None)
        sqlite_db.conn.commit()
        batch_phrases = []

if len(batch_phrases) > 0:
    sqlite_db.add_phrase(batch_phrases, None, None)
sqlite_db.conn.commit()