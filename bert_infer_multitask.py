# bert_infer_multitask.py
# -*- coding: utf-8 -*-
"""
BERT 多任务推理包装器，给 Flask 用的。

要求：
- 提供 class BertBiasJudgeMT:
    - __init__(model_dir: str)
    - score(text: str) -> Dict[str, Any]

- 返回字段要满足前端的预期：
    bias_yes: 0/1
    bias_conf: float
    side: str（这里我们用 frame 标签）
    side_probs: list
    strength_cls: 0/1/2
    strength_probs: list
"""

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 和训练时保持一致
PRETRAINED_MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256
FRAME_LABELS = ["conflict", "economic", "human_interest", "morality", "responsibility"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertForBiasAndFrame(nn.Module):
    """
    与 new_train_bert.py 中的结构保持一致：
    - BERT backbone
    - bias classifier (2 类：无偏见 / 有偏见)
    - frame classifier (5 类：conflict / economic / human_interest / morality / responsibility)
    """
    def __init__(self, pretrained_name: str, num_bias_labels: int, num_frame_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.bias_classifier = nn.Linear(hidden_size, num_bias_labels)
        self.frame_classifier = nn.Linear(hidden_size, num_frame_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS
        pooled_output = self.dropout(pooled_output)
        bias_logits = self.bias_classifier(pooled_output)
        frame_logits = self.frame_classifier(pooled_output)
        return bias_logits, frame_logits


class BertBiasJudgeMT:
    """
    被 app.py 调用：
        from bert_infer_multitask import BertBiasJudgeMT
        model = BertBiasJudgeMT(model_dir)
        model.score(text)
    """
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        # 尝试几种可能位置找到 best_bias_model.pt
        candidates = [
            model_dir / "best_bias_model.pt",          # 传进来的是目录（比如 models/bert-mt）
            model_dir.parent / "best_bias_model.pt",   # 目录的上一层（比如 news-bias-eval）
            Path(__file__).parent / "best_bias_model.pt",  # 当前文件所在目录
        ]
        weights_path = None
        for c in candidates:
            if c.exists():
                weights_path = c
                break
        if weights_path is None:
            raise FileNotFoundError(
                f"Could not find best_bias_model.pt under {model_dir} "
                f"or its parent. Checked: {', '.join(str(c) for c in candidates)}"
            )

        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

        self.model = BertForBiasAndFrame(
            PRETRAINED_MODEL_NAME,
            num_bias_labels=2,
            num_frame_labels=len(FRAME_LABELS)
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"[BERT MT] Loaded weights from {weights_path} on device {self.device}.")

    def score(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            # 返回一个结构完整但“空”的结果，避免前端 undefined
            return {
                "bias_yes": 0,
                "bias_conf": 0.0,
                "side": "",
                "side_probs": [],
                "strength_cls": 0,
                "strength_probs": [],
            }

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            bias_logits, frame_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            bias_probs = torch.softmax(bias_logits, dim=-1)[0]   # [2]
            frame_probs = torch.softmax(frame_logits, dim=-1)[0] # [5]

        bias_pred_id = int(torch.argmax(bias_probs).item())  # 0/1
        frame_pred_id = int(torch.argmax(frame_probs).item())

        bias_conf = float(bias_probs[bias_pred_id].item())
        frame_conf = float(frame_probs[frame_pred_id].item())

        # 是否有偏见：直接用 0/1
        bias_yes = bias_pred_id

        # 偏见方向：用 frame 标签，当成 side
        side = FRAME_LABELS[frame_pred_id]

        # frame 概率列表：side_probs
        side_probs = [
            {"side": FRAME_LABELS[i], "prob": float(frame_probs[i].item())}
            for i in range(len(FRAME_LABELS))
        ]

        # 偏见强度：我们没有专门训练 strength，但前端需要字段，就做一个简单启发式：
        # - 如果预测无偏见：strength_cls = 0（无偏见）
        # - 如果预测有偏见且置信度 < 0.75：strength_cls = 1（中等）
        # - 如果预测有偏见且置信度 >= 0.75：strength_cls = 2（强）
        if bias_yes == 0:
            strength_cls = 0
        else:
            strength_cls = 2 if bias_conf >= 0.75 else 1

        # strength_probs 简单 one-hot（保证前端不再是 [] 或 undefined）
        strength_probs = [
            {"cls": 0, "prob": 1.0 if strength_cls == 0 else 0.0},
            {"cls": 1, "prob": 1.0 if strength_cls == 1 else 0.0},
            {"cls": 2, "prob": 1.0 if strength_cls == 2 else 0.0},
        ]

        # 返回前端期望的字段 + 一些额外信息（方便以后调试）
        result: Dict[str, Any] = {
            # 前端实际会用到的：
            "bias_yes": bias_yes,
            "bias_conf": bias_conf,
            "side": side,
            "side_probs": side_probs,
            "strength_cls": strength_cls,
            "strength_probs": strength_probs,

            # 额外调试字段（不影响前端现有逻辑）：
            "frame_pred": side,
            "frame_conf": frame_conf,
            "bias_probs_raw": {
                "no_bias": float(bias_probs[0].item()),
                "has_bias": float(bias_probs[1].item()),
            },
            "frame_probs_raw": {
                FRAME_LABELS[i]: float(frame_probs[i].item())
                for i in range(len(FRAME_LABELS))
            },
        }
        return result
