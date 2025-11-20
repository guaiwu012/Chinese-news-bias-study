PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT,
  title TEXT,
  topic TEXT,
  time_place TEXT,
  stance TEXT,
  frame TEXT,
  strength REAL,
  text TEXT,
  auto_label TEXT,
  created_at TEXT,
  source_file TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic);
CREATE INDEX IF NOT EXISTS idx_articles_created ON articles(created_at);

CREATE TABLE IF NOT EXISTS judgments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id INTEGER NOT NULL,
  bias_yes INTEGER NOT NULL,
  bias_side TEXT,
  bias_strength INTEGER,
  bias_types TEXT,
  reasons TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  user_agent TEXT,
  FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_judgments_article ON judgments(article_id);
