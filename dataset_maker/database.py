"""
This module simply serve multiple processes of the next text to translate.
It simply keeps track of the next sentence.
"""
import sqlite3
from typing import Union, Optional, List, Tuple

NOT_DONE = 0; WORK_IN_PROGRESS = 1; DONE = 2
TRAIN = 0; TEST = 1; VALIDATION = 2

class Database(object):
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self.open_connection()

    def get_cursor(self):
        return self.conn.cursor()

    def open_connection(self):
        return sqlite3.connect(
            self.db_path,
        )

    def commit(self):
        self.conn.commit()

    def get_create_tables_stms(self) -> list[str]:
        pass

    def create_tables(self):
        cursor = self.get_cursor()
        try:
            for stmt in self.get_create_tables_stms():
                cursor.execute(stmt)
            self.conn.commit()
        finally:
            cursor.close()
            self.conn.close()

        self.conn = self.open_connection()
        return self.conn

class DatabaseSor(Database):

    def get_create_tables_stms(self):
        """
        Crea un database sqlite3 a partire da uno schema di tabelle,
        eseguendo i comandi SQL necessari.
        """
        return [
            """CREATE TABLE IF NOT EXISTS SorSentence (
                sentence_id BIGINT,
                sentence_gen_id INT, -- for each sentence, we generate more than one translation, this is the id of the generated sentence
                sentence_text MEDIUMTEXT,
                train INT DEFAULT 1,
                evaluated INT DEFAULT 0, -- 0 is not evaluated, then from 1 to 3 is the value of correctness 
                PRIMARY KEY (sentence_id, sentence_gen_id)
            );""",
            "CREATE INDEX idx_train ON SorSentence(train);",
            "CREATE INDEX SorSentence_IsLocked_index ON SorSentence (IsLocked);",
        ]

    def add_translation(self, sentence_id, text, is_training, sentence_gen_id=0):
        cursor = self.get_cursor()
        try:
            cursor.execute(
                "INSERT INTO SorSentence (sentence_id, sentence_text, train, sentence_gen_id) VALUES (?, ?, ?, ?);",
                       (sentence_id, text, is_training, sentence_gen_id)
           )
        finally:
            cursor.close()
        return self.conn

    def get_translations_for_evaluation(self,
                                        sentence_type: Union[int, str] = None
                                        ) -> Optional[List[Tuple]]:
        """
        Returns *all* rows (different sentence_gen_id) for a single
        sentence that:
          • has not been evaluated (evaluated = 0)
          • is not locked, or the lock is older than 1hour
        The chosen sentence is locked for the next hour so that no other
        worker can fetch it at the same time.
        """
        # ---------- Normalise the optional «train» filter ----------
        if sentence_type is None:
            select_type = ""
        elif isinstance(sentence_type, int) and 0 <= sentence_type < 3:
            select_type = f"AND train = {sentence_type}"
        elif isinstance(sentence_type, str):
            # you already have VALIDATION, TRAIN, TEST constants somewhere
            mapping = {"validation": VALIDATION,
                       "train": TRAIN,
                       "test": TEST}
            try:
                select_type = f"AND train = {mapping[sentence_type.lower()]}"
            except KeyError:
                raise ValueError(f"Unknown sentence type: {sentence_type}")
        else:
            raise ValueError(f"Invalid sentence_type: {sentence_type!r}")

        # ---------- Transaction – atomically pick & lock ----------
        with self.conn:
            cur = self.conn.cursor()

            # Find a candidate sentence_id
            cur.execute(f"""
                SELECT sentence_id
                FROM SorSentence
                WHERE evaluated = 0 {select_type} AND (IsLocked IS NULL OR unixepoch() - IsLocked > 3600)
                GROUP BY sentence_id
                ORDER BY sentence_id            -- deterministic
                LIMIT 1;
                """)
            row = cur.fetchone()
            if row is None:                 # nothing available
                return None

            chosen_id = row[0]               # sentence_id column

            # Lock **all** rows of that sentence (the lock is per‑sentence)
            cur.execute("""
                UPDATE SorSentence
                SET IsLocked = unixepoch()
                WHERE sentence_id = ?;
                """, (chosen_id,))

            # Retrieve every translation for the locked sentence
            cur.execute("""
                SELECT *
                FROM SorSentence
                WHERE sentence_id = ?
                ORDER BY sentence_gen_id;
                """, (chosen_id,))

            results = cur.fetchall()
            return results

    def add_evaluation(self, sentence_id, sentence_gen_id, evaluation: int):
        cursor = self.get_cursor()
        try:
            cursor.execute("UPDATE SorSentence SET evaluated=? WHERE sentence_id=? AND sentence_gen_id=?;", (evaluation, sentence_id, sentence_gen_id))
        finally:
            cursor.close()
        return self.conn

    def reset_timestamps(self):
        cursor = self.get_cursor()
        try:
            cursor.execute("""
            UPDATE SorSentence
            SET IsLocked = 0
            WHERE ((unixepoch() - IsLocked) > 3600); 
            """)
        finally:
            cursor.close()
        return self.conn



class DatabaseIta(Database):

    def get_create_tables_stms(self):
        """
        Crea un database sqlite3 a partire da uno schema di tabelle,
        eseguendo i comandi SQL necessari.
        """
        return [
            f"""CREATE TABLE IF NOT EXISTS ItaSentence (
                sentence_id BIGINT PRIMARY KEY,
                sentence_text MEDIUMTEXT,
                status INT DEFAULT {NOT_DONE},
                train INT DEFAULT {TRAIN},
                length INT 
            );""",
            "CREATE INDEX idx_status ON ItaSentence(status);",
            "CREATE INDEX idx_train ON ItaSentence(train);",
            "CREATE INDEX idx_length ON ItaSentence(length);",

        ]

    def reset_working_status(self):
        cursor = self.get_cursor()
        try:
            cursor.execute("UPDATE ItaSentence SET status = ? WHERE status=?;", (NOT_DONE, WORK_IN_PROGRESS))
        finally:
            cursor.close()
        return self.conn


    def add_entry(self, sentence_id, text, type_train_test_val):
        cursor = self.get_cursor()
        try:
            cursor.execute(
                "INSERT INTO ItaSentence (sentence_id, sentence_text, train, length) VALUES (?, ?, ?, ?);",
                    (sentence_id, text, type_train_test_val, len(text))
           )
        finally:
            cursor.close()
        return self.conn


    def remove_entries(self, sentence_ids):
        cursor = self.get_cursor()
        try:
            placeholders = ','.join(['?'] * len(sentence_ids))
            query = f"DELETE FROM ItaSentence WHERE sentence_id IN ({placeholders});"
            cursor.execute(query, sentence_ids)
            return self.conn
        except Exception as e:
            print("Error deleting batch sequence:", e)
            self.conn.rollback()  # rollback on error
        finally:
            cursor.close()


    def get_next_batch_items(self, batch_size: int, is_train: bool = None, order_by_length = True):
        cursor = self.get_cursor()
        try:
            order_by = 'ORDER BY length' if order_by_length else ''
            if is_train is None:
                cursor.execute(f"SELECT sentence_id, sentence_text, train FROM ItaSentence WHERE status=? {order_by} LIMIT ?;", (NOT_DONE, batch_size))
            else:
                cursor.execute(f"SELECT sentence_id, sentence_text, train FROM ItaSentence WHERE train=? AND status=? {order_by} LIMIT ?;", (is_train, NOT_DONE, batch_size))
                
            results = cursor.fetchall()
            my_results = results.copy()
            if len(results) == 0:
                return None
            # results = results.pop()
            sentence_ids = [sid[0] for sid in results]
            self.update_batch_job(sentence_ids, WORK_IN_PROGRESS).commit()
            return my_results

        except Exception as e:
            print("Error getting next batch sequence:", e)
            self.conn.rollback()  # rollback on error
        finally:
            cursor.close()

    def get_all_items(self):
        cursor = self.get_cursor()
        try:
            cursor.execute("SELECT sentence_id, sentence_text, train FROM ItaSentence")
            results = cursor.fetchall()
            if len(results) == 0:
                return None
            return results

        except Exception as e:
            print("Error getting all sentences:", e)
            self.conn.rollback()  # rollback on error
        finally:
            cursor.close()

    def get_sentence_by_id(self, sentence_id):
        cursor = self.get_cursor()
        try:
            cursor.execute('SELECT sentence_text FROM ItaSentence WHERE sentence_id=?', (sentence_id,))
            results = cursor.fetchone()
            if results is None:
                return None
            return results
        except Exception as e:
            print("Error getting sentence by id:", e)
            self.conn.rollback()
        finally:
            cursor.close()


    def update_batch_job(self, sentence_ids, status):
        cursor = self.get_cursor()
        try:
            placeholders = ','.join(['?'] * len(sentence_ids))
            cursor.execute(f"UPDATE ItaSentence SET status = ? WHERE sentence_id IN ({placeholders})", (status, *sentence_ids))
            return self.conn
        except Exception as e:
            print("Error updating sentence status: ", e)
            raise e
        finally:
            cursor.close()

    def count_entries(self):
        cursor = self.get_cursor()
        try:
            cursor.execute('SELECT COUNT(sentence_id) FROM ItaSentence')
            count = cursor.fetchall().pop()[0]

            return int(count)
        except Exception as e:
            print("Error counting sentences: ", e)
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



if __name__ == "__main__":

    db_ita = DatabaseIta(db_path='/media/tommy/Volume/Uni/Deep Learning/sorianese-llm/er-sorianese')

    results = db_ita.get_next_batch_items(10)
    print(results)





