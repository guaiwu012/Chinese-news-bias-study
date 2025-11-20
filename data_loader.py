#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load /mnt/data/output.csv (or a specified CSV) into SQLite articles table.
Tries to adapt to column names present in your file.
"""
import argparse, sqlite3, sys, json
from pathlib import Path
import pandas as pd

POSSIBLE_COLS = {
    "event_id": ["event_id","id","eventID","eventId"],
    "title": ["title","headline","Title"],
    "topic": ["topic","Theme","subject","category"],
    "time_place": ["time_place","timePlace","time","date","place","location"],
    "stance": ["stance","position","lean","side"],
    "frame": ["frame","framing"],
    "strength": ["strength","bias_strength","intensity"],
    "text": ["text","content","article","body"],
    "auto_label": ["auto_label","label","bias_label","tag"],
    "created_at": ["created_at","created","time","date"],
}

def best_match(colset, options):
    for c in options:
        if c in colset:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/output.csv", help="Path to the csv with generated news")
    ap.add_argument("--db", default="app.db", help="SQLite DB path (created if not exists)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded CSV with shape: {df.shape}")
    cols = set(df.columns)

    # Map columns
    mapping = {}
    for key, opts in POSSIBLE_COLS.items():
        col = best_match(cols, opts)
        mapping[key] = col
    print("[INFO] Column mapping:", mapping)

    if mapping["text"] is None:
        print("[ERROR] No text/content column found. Available:", df.columns.tolist())
        sys.exit(2)

    out = pd.DataFrame()
    out["event_id"] = df[mapping["event_id"]] if mapping["event_id"] else ""
    out["title"] = df[mapping["title"]] if mapping["title"] else ""
    out["topic"] = df[mapping["topic"]] if mapping["topic"] else ""

    if mapping["time_place"]:
        out["time_place"] = df[mapping["time_place"]]
    else:
        date_col = best_match(cols, ["date","time","created_at","created"])
        loc_col = best_match(cols, ["place","location","region","country","city"])
        if date_col or loc_col:
            out["time_place"] = (df[date_col].astype(str) if date_col else "") + " " + (df[loc_col].astype(str) if loc_col else "")
        else:
            out["time_place"] = ""

    out["stance"] = df[mapping["stance"]] if mapping["stance"] else ""
    out["frame"] = df[mapping["frame"]] if mapping["frame"] else ""
    out["strength"] = df[mapping["strength"]] if mapping["strength"] else None
    out["text"] = df[mapping["text"]]
    out["auto_label"] = df[mapping["auto_label"]] if mapping["auto_label"] else ""
    out["created_at"] = df[mapping["created_at"]] if mapping["created_at"] else ""
    out["source_file"] = str(csv_path)

    conn = sqlite3.connect(args.db)
    with open("schema.sql","r",encoding="utf-8") as f:
        conn.executescript(f.read())

    cur = conn.cursor()
    cur.execute("DELETE FROM articles;")
    conn.commit()

    records = out.to_dict(orient="records")
    cur.executemany(
        """INSERT INTO articles(event_id,title,topic,time_place,stance,frame,strength,text,auto_label,created_at,source_file)
           VALUES(:event_id,:title,:topic,:time_place,:stance,:frame,:strength,:text,:auto_label,:created_at,:source_file)""",
        records
    )
    conn.commit()
    print(f"[OK] Inserted {len(records)} articles into {args.db}")

if __name__ == "__main__":
    main()
