"""
This module simply serve multiple processes of the next text to translate.
It simply keeps track of the next sentence.
"""
import sqlite3
from sqlite3 import IntegrityError

from tqdm import tqdm

from dataset_maker.dataset import load_clean_mc4_dataset


NOT_DONE = 0
WORK_IN_PROGRESS = 0
DONE = 1



class Database:
    def __init__(self):
        self.conn = self.open_connection()


    def open_connection(self):
        return sqlite3.connect(
            "er-sorianese.db",
        )

    def get_cursor(self):
        return self.conn.cursor()

    def create_database(self):
        """
        Crea un database sqlite3 a partire da uno schema di tabelle,
        eseguendo i comandi SQL necessari.
        """
        schema_sqls = [
            # """CREATE TABLE IF NOT EXISTS BatchJob (
            #     batch_id VARCHAR(47) PRIMARY KEY,
            #     status INT DEFAULT 0
            # );""",
            """CREATE TABLE IF NOT EXISTS ItaSentence (
                sentence_id BIGINT PRIMARY KEY,
                sentence_text MEDIUMTEXT,
                status INT DEFAULT 0,
                train INT DEFAULT 1
            );""",

            """CREATE TABLE IF NOT EXISTS SorSentence (
                sentence_id BIGINT PRIMARY KEY,
                sentence_text MEDIUMTEXT
            )"""
        ]

        cursor = self.get_cursor()
        try:
            for stmt in schema_sqls:
                cursor.execute(stmt)
            self.conn.commit()
        finally:
            cursor.close()
            self.conn.close()

        self.conn = self.open_connection()
        return self.conn

    def add_phrase(self, phrase: list[str], sentence_id: list[int], train: list[int]):
        # self.conn = self.open_connection()
        cursor = self.get_cursor()
        try:
            cursor.executemany(f"INSERT INTO ItaSentence (sentence_id, sentence_text, status, train) VALUES (?, ?, ?, ?);",phrase)
        except IntegrityError as e:
            pass
        # self.conn.commit()

    def get_next_item(self):
        cursor = self.get_cursor()
        try:
            cursor.execute(f"SELECT sentence_id, sentence_text FROM ItaSentence WHERE status={NOT_DONE} LIMIT 1 FOR UPDATE")
            results = cursor.fetchall()

            if len(results) == 0:
                return None
            results = results.pop()

            cursor.execute("UPDATE ItaSentence SET status=%s, time_created=CURRENT_TIMESTAMP() WHERE sentence_id=%s", (WORK_IN_PROGRESS, results[0]) )
            self.conn.commit()

            return results

        except Exception as e:
            print("Error:", e)
            self.conn.rollback()  # rollback on error
        finally:
            cursor.close()

    def get_next_batch_items(self, batch_size: int):
        cursor = self.get_cursor()
        try:
            cursor.execute(f"SELECT sentence_id, sentence_text FROM ItaSentence WHERE status={NOT_DONE} LIMIT ? FOR UPDATE", (batch_size,))
            results = cursor.fetchall()

            if len(results) == 0:
                return None
            results = results.pop()

            return results

        except Exception as e:
            print("Error:", e)
            self.conn.rollback()  # rollback on error
        finally:
            cursor.close()

    def cancel_single_sentence_job(self, sentence_id):
        cursor = self.get_cursor()
        try:
            cursor.execute("UPDATE ItaSentence SET status=-1, time_created=CURRENT_TIMESTAMP() WHERE sentence_id=%s", (sentence_id,))
            self.conn.commit()
        except Exception as e:
            print("Error inserting new translation: ", e)
        finally:
            cursor.close()

    def insert_translation(self, id:int, sorianese_translation:str):
        cursor = self.get_cursor()
        try:
            cursor.execute("INSERT INTO SorSentence (sentence_id, sentence_text) VALUES (%s, %s)",
                           (id, sorianese_translation)
                           )
            cursor.execute(f"UPDATE ItaSentence SET status={DONE} WHERE sentence_id=%s", (id,))
            self.conn.commit()
        except Exception as e:
            print("Error inserting new translation: ", e)
        finally:
            cursor.close()


    def update_batch_job(self, sentences, status):
        cursor = self.get_cursor()
        try:
            cursor.execute("Update ItaSentence SET status = %s WHERE sentence_id = %s", (status, sentences))
            self.conn.commit()
        except Exception as e:
            print("Error inserting new translation: ", e)
        finally:
            cursor.close()


    def completion_status(self):
        query = """SELECT
                  (SELECT COUNT(*) FROM ItaSentence WHERE status = 1)
                  /
                  (SELECT COUNT(*) FROM ItaSentence)
                  AS fraction_not_done;
              """
        cursor = self.get_cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall().pop()[0]
            print(results)
            return results
        except Exception as e:
            print("Error inserting new translation: ", e)
        finally:
            cursor.close()

def schedule_item_status_reset(i: int = 1):
    """
    Every i minutes set all sentences that has not reached the
    :param i:
    :return:
    """
    from time import sleep
    db = Database()
    while True:
        # sleep(1)
        sleep(60*5)
        print('5 minutes elapsed')
        c = db.get_cursor()
        c.execute(f"""
                    UPDATE ItaSentence
                    SET status={NOT_DONE}, time_created=CURRENT_TIMESTAMP()
                    WHERE status={WORK_IN_PROGRESS} AND time_created < DATE_SUB(NOW(), INTERVAL {i} MINUTE);
                    """)
        db.conn.commit()



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        prog='db-server',
        )
    parser.add_argument('--create', action='store_true', help='Create the database and populate it with the dataset.')
    parser.add_argument('--schedule', action='store_true', help='Run the module in schedule mod, every 5 minute set to NOT_DONE every job who is in WIP by 5 1 minute.')
    parser.add_argument('--db_path', nargs='?', type=str, help='Search file at given path.')

    args = parser.parse_args()

    if args.schedule:
        schedule_item_status_reset()

    if args.create:
        phrases_train, phrases_test = load_clean_mc4_dataset()

        phrases = phrases_test + phrases_train
        del phrases_train, phrases_test

        db = Database()
        print("Populating DB")

        # print(db.has_pending_jobs())
        db.create_database()
        db.populate_database(phrases)







