
from dataset_maker.job_server import *


db_ita = DatabaseIta('er-sorianese.db')
db_sor = DatabaseSor('real-sorianese.db')
db_ita.reset_working_status()

while True:
    batch = db_ita.get_next_batch_items(25)

    for entry in batch:
        db_sor.add_translation(*entry)
    db_sor.conn.commit()

    db_ita.remove_entries([ids[0] for ids in batch]).commit()


