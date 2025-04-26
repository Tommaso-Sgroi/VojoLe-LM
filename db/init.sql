CREATE DATABASE IF NOT EXISTS sentence_db;
USE sentence_db;
# SET GLOBAL event_scheduler = ON;


CREATE TABLE IF NOT EXISTS ItaSentence (
    sentence_id INT(32) PRIMARY KEY,
    sentence_text MEDIUMTEXT,
    status int(4) DEFAULT -1,
    time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    batch_id VARCHAR(47)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS SorSentence (
    sentence_id INT(32) PRIMARY KEY,
    sentence_text MEDIUMTEXT
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS BatchJob (
    batch_id VARCHAR(47) PRIMARY KEY,
    status int(4) DEFAULT 0 -- -1 means error, 0 means pending, 1 means done
);
