# News Bias Lab — AI-Assisted Bias Detection in Chinese News
*A research-oriented interactive system for practicing bias recognition and evaluating AI-generated news.*

This project provides a full pipeline for exploring **media bias in Chinese news**:
✔ dataset ingestion →  
✔ article sampling →  
✔ human annotation UI →  
✔ statistical analysis →  
✔ BERT multi-task inference (bias + frame classification)

It supports both **user study experiments** and **teaching/learning tools** for identifying news bias.

## 🌟 Features

### 🔍 1. Interactive News Annotation Interface
- Random or topic-filtered article sampling
- Users judge:
  - Whether the article has bias
  - Which side / frame it aligns with
  - Strength (0–2)
  - Bias type(s)
  - Textual reasoning
- Submissions stored in SQLite (`judgments` table)

### 📊 2. Real-time Statistics Dashboard
- Global bias rate
- Bias rate grouped by topic
- Distribution of bias types
- Topic list with counts
- Designed for experiments & perception studies

### 🤖 3. Offline AI Bias Scoring
Two AI scoring modes:

**Heuristic model** (keyword-based, no GPU required)  
**BERT Multi-Task Model** (bias + frame prediction + strength)

### 🗄 4. CSV → SQLite Auto Importer
`data_loader.py` adapts to various CSV column names and imports to `articles` table.

### 🧱 5. Clean, Minimal Flask Backend
All API routes defined in `app.py`.

## 📁 Project Structure
```
project/
│ app.py
│ data_loader.py
│ bert_infer_multitask.py
│ new_train_bert.py
│ schema.sql
│ requirements.txt
│ README.md
│ app.db / bias_news.db
│ models/
│   └─ bert-mt/
│       └─ best_bias_model.pt
└ static/ & templates/
```

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize the database
```bash
python app.py
```

Or import manually:

```bash
python data_loader.py --csv output.csv --db app.db
```

### 3. Run the system
```bash
python app.py
```

Open:
```
http://127.0.0.1:8011
```

### 4. Use the BERT scorer
POST:
```
/api/score-user-news
```
Payload:
```json
{ "text": "..." }
```

## 📊 Database Schema
Defined in `schema.sql`:
- `articles`
- `judgments`

## 📘 License
Released under the **MIT License**.

## ✨ About
Created by **Yufei Zhang**  
Hong Kong Baptist University — AIDM  
2024–2025
