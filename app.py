#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, sqlite3, subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, render_template, g, Response

# ==== 新增：BERT 相关依赖 ====
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 全局路径 & Flask
# -----------------------------
APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "app.db"

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates"
)

# -----------------------------
# DB helpers
# -----------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

# -----------------------------
# Pages
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# 简易写作试用页（可选保留）
@app.get("/compose")
def compose_page():
    html = """
<!doctype html><meta charset="utf-8">
<title>写新闻并评分</title>
<style>body{font-family:system-ui,Arial;padding:16px;max-width:960px;margin:0 auto}textarea{width:100%;padding:10px;box-sizing:border-box}</style>
<h2>📝 对照事实卡写新闻 & 一键评分</h2>
<textarea id="t" rows="10" placeholder="写 100–300 字新闻报道…"></textarea><br>
<button id="b">评分（BERT）</button> <span id="h"></span>
<pre id="o" style="background:#f7f7f7;padding:10px;white-space:pre-wrap"></pre>
<script>
document.getElementById('b').onclick=async()=>{
  const text=(document.getElementById('t').value||'').trim();
  if(!text){alert('先写点内容');return;}
  document.getElementById('h').textContent='评分中…';
  const r=await fetch('/api/score-user-news',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const j=await r.json(); document.getElementById('h').textContent='';
  document.getElementById('o').textContent=JSON.stringify(j,null,2);
};
</script>
"""
    return Response(html, mimetype="text/html")

# -----------------------------
# Data APIs
# -----------------------------
def query_one_article(topic: Optional[str]=None, article_id: Optional[int]=None) -> Optional[Dict[str,Any]]:
    db = get_db()
    if article_id:
        row = db.execute("SELECT * FROM articles WHERE id=?;", (article_id,)).fetchone()
        return dict(row) if row else None
    if topic:
        row = db.execute("SELECT * FROM articles WHERE topic=? ORDER BY RANDOM() LIMIT 1;", (topic,)).fetchone()
    else:
        row = db.execute("SELECT * FROM articles ORDER BY RANDOM() LIMIT 1;").fetchone()
    return dict(row) if row else None

@app.route("/api/sample")
def api_sample():
    topic = request.args.get("topic") or None
    article_id = request.args.get("id", type=int)
    art = query_one_article(topic, article_id)
    if not art:
        return jsonify({"ok": False, "error": "No article found"}), 404
    payload = {
        "id": art["id"],
        "title": art["title"],
        "topic": art["topic"],
        "time_place": art["time_place"],
        "stance": art["stance"],
        "frame": art["frame"],
        "strength": art["strength"],
        "text": art["text"],
        "created_at": art["created_at"],
    }
    return jsonify({"ok": True, "article": payload})

@app.route("/api/topics")
def api_topics():
    db = get_db()
    rows = db.execute("""
        SELECT COALESCE(NULLIF(TRIM(topic),''),'(Unlabeled)') AS t, COUNT(*) AS n
        FROM articles
        GROUP BY t
        ORDER BY t ASC;
    """).fetchall()
    # 前端只显示名字，这里仍返回 n 但不会被使用
    return jsonify({"ok": True, "topics": [{"topic": r["t"], "count": r["n"]} for r in rows]})

@app.route("/api/submit", methods=["POST"])
def api_submit():
    data = request.get_json(force=True, silent=True) or {}
    required = ["article_id","bias_yes"]
    if not all(k in data for k in required):
        return jsonify({"ok": False, "error": "Missing fields"}), 400

    bias_yes = 1 if str(data["bias_yes"]).lower() in ("1","true","yes","y") else 0
    bias_side = data.get("bias_side","")
    try:
        bias_strength = int(data.get("bias_strength", 0))
    except Exception:
        bias_strength = None

    # 这里仍接收 bias_types(JSON)，前端会把 frame 单选作为单元素列表传入
    bias_types = data.get("bias_types", [])
    if isinstance(bias_types, list):
        bias_types_json = json.dumps(bias_types, ensure_ascii=False)
    else:
        bias_types_json = json.dumps([s.strip() for s in str(bias_types).split(",") if s.strip()], ensure_ascii=False)

    reasons = (data.get("reasons","") or "")[:2000]

    db = get_db()
    db.execute(
        """INSERT INTO judgments(article_id, bias_yes, bias_side, bias_strength, bias_types, reasons, user_agent)
           VALUES(?,?,?,?,?,?,?);""",
        (data["article_id"], bias_yes, bias_side, bias_strength, bias_types_json, reasons, request.headers.get("User-Agent",""))
    )
    db.commit()
    return jsonify({"ok": True})

@app.route("/api/stats")
def api_stats():
    db = get_db()
    rows = db.execute("""
        SELECT COALESCE(NULLIF(TRIM(a.topic),''),'(Unlabeled)') AS topic,
               COUNT(j.id) AS votes,
               SUM(CASE WHEN j.bias_yes=1 THEN 1 ELSE 0 END) AS yes_cnt
        FROM articles a
        LEFT JOIN judgments j ON a.id=j.article_id
        GROUP BY topic
        ORDER BY yes_cnt DESC;
    """).fetchall()
    by_topic = []
    for r in rows:
        votes = r["votes"]
        yes_cnt = r["yes_cnt"] if r["yes_cnt"] is not None else 0
        rate = (yes_cnt / votes) if votes else 0.0
        by_topic.append({"topic": r["topic"], "votes": votes, "bias_yes": yes_cnt, "bias_rate": rate})

    r = db.execute("SELECT COUNT(*) AS votes, SUM(CASE WHEN bias_yes=1 THEN 1 ELSE 0 END) AS yes_cnt FROM judgments;").fetchone()
    votes = r["votes"]
    yes_cnt = r["yes_cnt"] if r["yes_cnt"] is not None else 0
    global_stats = {"votes": votes, "bias_yes": yes_cnt, "bias_rate": (yes_cnt / votes) if votes else 0.0}

    rows2 = db.execute("SELECT bias_types FROM judgments WHERE bias_types IS NOT NULL AND bias_types != '' ;").fetchall()
    type_counts = {}
    for row in rows2:
        try:
            arr = json.loads(row["bias_types"])
            for t in arr:
                type_counts[t] = type_counts.get(t, 0) + 1
        except Exception:
            pass

    return jsonify({"ok": True, "global": global_stats, "by_topic": by_topic, "bias_types": type_counts})

# -----------------------------
# 简单启发式（保留，不在 UI 调用）
# -----------------------------
KEYWORDS = {
    "wording": ["clearly","undeniably","must","always","never","当然","显然","必须","绝不","肯定"],
    "framing": ["despite","although","however","yet","但","然而","尽管"],
    "selection": ["study shows","experts say","according to","据称","据报道","有人认为"],
}
def simple_bias_score(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    score = 0
    hits = []
    for typ, keys in KEYWORDS.items():
        for k in keys:
            if k.lower() in t:
                score += 1
                hits.append([typ, k])
    bias_yes = int(score >= 2)
    return {"bias_yes": bias_yes, "score": score, "hits": hits}

@app.post("/api/judge")
def api_judge():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text","")
    result = simple_bias_score(text)
    return jsonify({"ok": True, "result": result})

# -----------------------------
# 仅查 DB 的参考评分
# -----------------------------
@app.post("/api/ref-score")
def api_ref_score():
    data = request.get_json(silent=True) or {}
    aid = data.get("article_id", None)
    if not isinstance(aid, int):
        return jsonify({"ok": False, "error": "missing/invalid article_id"}), 400

    db = get_db()
    row = db.execute("SELECT frame, strength FROM articles WHERE id=?;", (aid,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": f"article {aid} not found"}), 404

    frame = (row["frame"] or "").strip().lower()
    try:
        strength = int(row["strength"])
    except Exception:
        strength = 0
    strength = max(0, min(2, strength))
    bias_yes = 1 if strength >= 1 else 0

    return jsonify({"ok": True, "result": {"bias_yes": bias_yes, "side": frame, "strength_cls": strength}})

# =====================================================
# BERT 多任务推理（写作评分）—— 从 bert_infer_multitask 导入
# =====================================================
from bert_infer_multitask import BertBiasJudgeMT

_BERT_MT = {"loaded": False, "obj": None, "err": None}

def _ensure_bert_mt():
    """懒加载模型，只初始化一次"""
    if _BERT_MT["loaded"]:
        return
    try:
        model_dir = APP_DIR / "models" / "bert-mt"   # 与原路径一致
        _BERT_MT["obj"] = BertBiasJudgeMT(str(model_dir))
        _BERT_MT["loaded"] = True
        print("[BERT] Loaded external bert_infer_multitask model.")
    except Exception as e:
        _BERT_MT["err"] = str(e)
        _BERT_MT["loaded"] = True
        print(f"[BERT] Failed to load model: {e}")


@app.post("/api/score-user-news")
def api_score_user_news():
    _ensure_bert_mt()
    if _BERT_MT["obj"] is None:
        return jsonify({"ok": False, "error": f"BERT model not available: {_BERT_MT['err']}"}), 500
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Empty text"}), 400
    res = _BERT_MT["obj"].score(text)
    return jsonify({"ok": True, "result": res})

# -----------------------------
# Bootstrap DB & run
# -----------------------------
if __name__ == "__main__":
    if not DB_PATH.exists():
        print("[BOOT] Creating DB and loading data from output.csv (if exists).")
        conn = sqlite3.connect(DB_PATH)
        with open(APP_DIR / "schema.sql","r",encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.close()
        if (APP_DIR.parent / "output.csv").exists():
            subprocess.call(["python", "data_loader.py", "--csv", str(APP_DIR.parent / "output.csv"), "--db", str(DB_PATH)])
        else:
            print("[WARN] output.csv not found; run data_loader.py manually.")
    app.run(host="0.0.0.0", port=8011, debug=True)
