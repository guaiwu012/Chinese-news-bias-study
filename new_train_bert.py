# train_bert_bias_multitask.py
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from transformers import AutoTokenizer, AutoModel

# ========= 配置区 =========
CSV_PATH = "bias_5x3_selflabel.csv"   # 改成你的真实路径
PRETRAINED_MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
SEED = 42
# =======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ========= 数据准备 =========
df = pd.read_csv(CSV_PATH)

# 构造 “是否有 bias” 标签：strength > 0 视为有偏见
df["bias_label"] = (df["strength"] > 0).astype(int)

# Frame 映射到 id
frame_list = sorted(df["frame"].unique().tolist())
frame2id: Dict[str, int] = {f: i for i, f in enumerate(frame_list)}
id2frame: Dict[int, str] = {i: f for f, i in frame2id.items()}
df["frame_id"] = df["frame"].map(frame2id)

print("Frame mapping:", frame2id)
print(df[["text", "frame", "strength", "bias_label", "frame_id"]].head())

# 8:2 划分训练集和测试集（按 frame 分层保持分布）
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["frame_id"]
)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


# ========= Dataset 定义 =========
class BiasFrameDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["text"])

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels_bias": torch.tensor(int(row["bias_label"]), dtype=torch.long),
            "labels_frame": torch.tensor(int(row["frame_id"]), dtype=torch.long),
        }
        return item


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

train_dataset = BiasFrameDataset(train_df, tokenizer, MAX_LEN)
test_dataset = BiasFrameDataset(test_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ========= 模型定义（多任务 BERT） =========
class BertForBiasAndFrame(nn.Module):
    def __init__(self, pretrained_name: str, num_bias_labels: int, num_frame_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # 两个分类头：一个预测 bias(0/1)，一个预测 frame(5 类)
        self.bias_classifier = nn.Linear(hidden_size, num_bias_labels)
        self.frame_classifier = nn.Linear(hidden_size, num_frame_labels)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels_bias=None,
        labels_frame=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS 向量
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)

        bias_logits = self.bias_classifier(pooled_output)      # [B, 2]
        frame_logits = self.frame_classifier(pooled_output)    # [B, num_frame]

        loss = None
        if labels_bias is not None and labels_frame is not None:
            loss_bias = self.loss_fct(bias_logits, labels_bias)
            loss_frame = self.loss_fct(frame_logits, labels_frame)
            # 可以视情况加权，这里简单相加
            loss = loss_bias + loss_frame

        return {
            "loss": loss,
            "bias_logits": bias_logits,
            "frame_logits": frame_logits
        }


num_bias_labels = 2          # 有/无偏见
num_frame_labels = len(frame2id)

model = BertForBiasAndFrame(
    PRETRAINED_MODEL_NAME,
    num_bias_labels=num_bias_labels,
    num_frame_labels=num_frame_labels
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ========= 最优模型配置 =========
BEST_MODEL_PATH = "best_bias_model.pt"  # 训练过程中会自动保存



# ========= 训练与评估 =========
def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_bias = batch["labels_bias"].to(device)
        labels_frame = batch["labels_frame"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels_bias=labels_bias,
            labels_frame=labels_frame
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()

    all_bias_true, all_bias_pred = [], []
    all_frame_true, all_frame_pred = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_bias = batch["labels_bias"].to(device)
            labels_frame = batch["labels_frame"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            bias_logits = outputs["bias_logits"]
            frame_logits = outputs["frame_logits"]

            bias_pred = bias_logits.argmax(dim=-1)
            frame_pred = frame_logits.argmax(dim=-1)

            all_bias_true.extend(labels_bias.cpu().numpy().tolist())
            all_bias_pred.extend(bias_pred.cpu().numpy().tolist())
            all_frame_true.extend(labels_frame.cpu().numpy().tolist())
            all_frame_pred.extend(frame_pred.cpu().numpy().tolist())

    # macro F1
    bias_f1 = f1_score(all_bias_true, all_bias_pred, average="macro")
    frame_f1 = f1_score(all_frame_true, all_frame_pred, average="macro")

    print("=== Bias (有/无偏见) Classification Report ===")
    print(classification_report(all_bias_true, all_bias_pred, digits=4))

    print("=== Frame (框架类型) Classification Report ===")
    print(classification_report(all_frame_true, all_frame_pred, digits=4, target_names=frame_list))

    return bias_f1, frame_f1

global best_bias_f1, best_epoch
best_bias_f1 = 0.0
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)

    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    bias_f1, frame_f1 = evaluate(model, test_loader, device)
    print(f"[Epoch {epoch}] Test Bias F1 (macro): {bias_f1:.4f}")
    print(f"[Epoch {epoch}] Test Frame F1 (macro): {frame_f1:.4f}")
    print("-" * 50)

    # 以 Bias F1 作为主指标，保存最好的模型
  
    if bias_f1 > best_bias_f1:
        best_bias_f1 = bias_f1
        best_epoch = epoch
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"*** New best model saved at epoch {epoch}! Bias F1={bias_f1:.4f} ***")

print(f"Best epoch (by Bias F1): {best_epoch}, F1={best_bias_f1:.4f}")



# ========= 推理函数示例：给出预测和置信度 =========
def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

def predict_texts(text_list, model, tokenizer, device, max_len=256):
    """
    输入: 一组中文新闻文本
    输出: 每条的
      - bias_pred: 0/1
      - bias_conf: 对预测类别的置信度
      - frame_pred: frame 名称
      - frame_conf: 对预测 frame 的置信度
    """
    model.eval()
    results = []
    with torch.no_grad():
        for text in text_list:
            encoded = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            bias_logits = outputs["bias_logits"]  # [1,2]
            frame_logits = outputs["frame_logits"]  # [1,5]

            bias_probs = softmax(bias_logits, dim=-1).cpu().numpy()[0]
            frame_probs = softmax(frame_logits, dim=-1).cpu().numpy()[0]

            bias_pred_id = int(bias_probs.argmax())
            frame_pred_id = int(frame_probs.argmax())

            results.append({
                "text": text,
                "bias_pred": bias_pred_id,
                "bias_conf": float(bias_probs[bias_pred_id]),
                "frame_pred": id2frame[frame_pred_id],
                "frame_conf": float(frame_probs[frame_pred_id]),
                "frame_probs_full": {id2frame[i]: float(p) for i, p in enumerate(frame_probs)},
                "bias_probs_full": {str(i): float(p) for i, p in enumerate(bias_probs)},
            })
    return results

if __name__ == "__main__":
    # 训练结束后，加载最佳模型权重
    if os.path.exists(BEST_MODEL_PATH):
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded best model from {BEST_MODEL_PATH}")
    else:
        print("Warning: best model file not found, using last epoch weights.")

    # 简单测一下推理
    demo_texts = [
        "这项新政策引发了各方激烈争论，支持者和反对者在媒体上互相指责。",
        "该项目稳步推进，为当地居民提供了更多就业机会。"
    ]
    preds = predict_texts(demo_texts, model, tokenizer, device)
    for r in preds:
        print("=" * 80)
        print("Text:", r["text"])
        print("Bias predicted:", r["bias_pred"], "(1=有偏见, 0=无偏见), confidence:", r["bias_conf"])
        print("Frame predicted:", r["frame_pred"], ", confidence:", r["frame_conf"])
