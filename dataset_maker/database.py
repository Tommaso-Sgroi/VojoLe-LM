"""
This module simply serve multiple processes of the next text to translate.
It simply keeps track of the next sentence.
"""
import sqlite3


NOT_DONE = 0
WORK_IN_PROGRESS = 1
DONE = 1

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
                sentence_id BIGINT PRIMARY KEY,
                sentence_text MEDIUMTEXT,
                train BOOL DEFAULT 1
            );""",
            "CREATE INDEX idx_train ON SorSentence(train);",
        ]

    def add_translation(self, sentence_id, text, is_training):
        cursor = self.get_cursor()
        try:
            cursor.execute(
                "INSERT INTO SorSentence (sentence_id, sentence_text, train) VALUES (?, ?, ?);",
                       (sentence_id, text, is_training)
           )
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
            """CREATE TABLE IF NOT EXISTS ItaSentence (
                sentence_id BIGINT PRIMARY KEY,
                sentence_text MEDIUMTEXT,
                status INT DEFAULT 0,
                train INT DEFAULT 1
            );""",
            "CREATE INDEX idx_status ON ItaSentence(status);",
            "CREATE INDEX idx_train ON ItaSentence(train);",
        ]

    def reset_working_status(self):
        cursor = self.get_cursor()
        try:
            cursor.execute("UPDATE ItaSentence SET status = 0")
        finally:
            cursor.close()
        return self.conn


    def add_entry(self, sentence_id, text, is_training):
        cursor = self.get_cursor()
        try:
            cursor.execute(
                "INSERT INTO ItaSentence (sentence_id, sentence_text, train) VALUES (?, ?, ?);",
                    (sentence_id, text, is_training)
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


    def get_next_batch_items(self, batch_size: int):
        cursor = self.get_cursor()
        try:
            cursor.execute("SELECT sentence_id, sentence_text, train FROM ItaSentence WHERE status=? LIMIT ?;", (0, batch_size))
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





