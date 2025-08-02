CREATE TABLE IF NOT EXISTS ItaSentence (
    sentence_id BIGINT PRIMARY KEY,
    sentence_text MEDIUMTEXT,
    status INT DEFAULT 0,
    train INT DEFAULT 1
);
CREATE INDEX idx_status ON ItaSentence(status);
CREATE INDEX idx_train ON ItaSentence(train);
