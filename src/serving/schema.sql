DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS signature;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  username TEXT
);

CREATE TABLE signature (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL,
  signature_path TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES user (id)
);