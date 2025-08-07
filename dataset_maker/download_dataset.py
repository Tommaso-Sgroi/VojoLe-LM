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
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from time import time
from dataset_maker.database import *
ilpost = load_dataset(path=os.path.join(os.getenv('FAST'), 'datasets', 'ilpost'))
fanpage = load_dataset(path=os.path.join(os.getenv('FAST'), 'datasets', 'fanpage'))




base_uri = os.path.join('.', 'data', 'clean_mc4_it', 'tiny')

start = time()
sqlite_db = DatabaseIta(os.path.join(os.getenv('FAST'), 'er-italiano.db'))
sqlite_sr_db = DatabaseSor(os.path.join(os.getenv('FAST'), 'er-sorianese.db'))
sqlite_db.create_tables()
sqlite_sr_db.create_tables()

# quit()
def add_samples_to_dbs(train, test, validation, start_from_index=0):
    batch_phrases = []
    for i, entry in tqdm(enumerate(train, start=start_from_index)):
        sqlite_db.add_entry(i, entry['target'], TRAIN)
        if (i % 1_000_000) == 0:
            sqlite_db.conn.commit()
            batch_phrases = []
    else:
        end = i + 1
    sqlite_db.conn.commit()

    batch_phrases.clear()
    for i, entry in tqdm(enumerate(validation, start=end)):
        sqlite_db.add_entry(i, entry['target'], VALIDATION)
        if (i % 1_000) == 0:
            sqlite_db.conn.commit()
            batch_phrases = []
    else:
        end = i + 1
    sqlite_db.conn.commit()

    for i, entry in tqdm(enumerate(test, start=end)):
        sqlite_db.add_entry(i, entry['target'], TEST)
        if (i % 1_000) == 0:
            sqlite_db.conn.commit()
            batch_phrases = []
    sqlite_db.conn.commit()
    print('Done in', time() - start, "seconds")
    return i + 1

start_from = 0
for ds in [fanpage, ilpost]:
    start_from = add_samples_to_dbs(ds['train'], ds['test'], ds['validation'], start_from_index=start_from)
    print('Dataset done')
sqlite_db.conn.close()
sqlite_sr_db.conn.close()

