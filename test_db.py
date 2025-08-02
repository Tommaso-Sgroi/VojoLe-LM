from tqdm import tqdm
from dataset_maker.database import *
import os

db_ita = DatabaseIta(os.path.join(os.getenv('FAST'),'er-italiano.db'))
db_sor = DatabaseSor(os.path.join(os.getenv('FAST'),'er-sorianese.db'))
print('resetting status')
db_ita.reset_working_status()
print('i cant count :(')
count = db_ita.get_cursor().execute('SELECT COUNT(sentence_id) FROM ItaSentence ').fetchone()[0]
count = int(count)
batch_size = 25

for _ in tqdm(range(0, count, batch_size)):
    batch = db_ita.get_next_batch_items(batch_size)
    for entry in batch:
        db_sor.add_translation(*entry)
    db_sor.conn.commit()

    db_ita.remove_entries([ids[0] for ids in batch]).commit()


"""
It get avg this time:
[01:28<2:02:36, 53.44it/s]
So, it is not a bottleneck in the system
It can be even more speedy if threads are used 
to handle the insertion and deletions
"""
