CREATE DATABASE IF NOT EXISTS sentence_db;
USE sentence_db;


CREATE TABLE IF NOT EXISTS ItaSentence (
    sentence_id CHAR(255) PRIMARY KEY,
    sentence_text MEDIUMTEXT,
    status int(4) DEFAULT -1
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE TABLE IF NOT EXISTS SorSentence (
    sentence_id CHAR(47) PRIMARY KEY,
    sentence_text MEDIUMTEXT
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;