#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A 簡易推論テスト - train_bigfive.pyと同じ構造
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import numpy as np

# train_bigfive.pyと同じモデルクラス
class BigFiveRegressionModel(nn.Module):
    def __init__(self, base_model, num_traits=5):
        super().__init__()
        self.encoder = base_model
        self.hidden_dim = base_model.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_traits)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        predictions = self.regressor(cls_embedding)
        return torch.sigmoid(predictions)

print("=" * 80)
print("Model A 簡易推論テスト")
print("=" * 80)

# モデルロード
print("\n[1/3] Model Aをロード中...")
adapter_path = 'model_a_extracted/lora_weights'
base_model = AutoModel.from_pretrained('xlm-roberta-large')
base_model = PeftModel.from_pretrained(base_model, adapter_path)
base_model = base_model.merge_and_unload()

model = BigFiveRegressionModel(base_model)
checkpoint = torch.load('model_a_extracted/best_model.pt', map_location='cpu')

# regressorの重みのみロード（encoderは既にLoRA適用済み）
state_dict = checkpoint['model_state_dict']
regressor_state_dict = {k.replace('regressor.', ''): v for k, v in state_dict.items() if 'regressor' in k}
model.regressor.load_state_dict(regressor_state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
print(f"✅ Model Aロード完了（RMSE: {checkpoint['best_val_rmse']:.4f}）")

# テストテキスト
test_texts = [
    "I love meeting new people and going to social events. I'm very outgoing and energetic!",
    "I prefer to stay at home and read books quietly. Social gatherings make me uncomfortable.",
    "I'm organized and always plan ahead. I make detailed to-do lists every day."
]

print("\n[2/3] 3つのテキストで推論中...")
predictions_list = []

for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])

    pred_np = predictions[0].numpy() * 99
    predictions_list.append(pred_np)

    print(f"\n  テキスト {i}: \"{text[:50]}...\"")
    print(f"    Openness:          {pred_np[0]:.2f}")
    print(f"    Conscientiousness: {pred_np[1]:.2f}")
    print(f"    Extraversion:      {pred_np[2]:.2f}")
    print(f"    Agreeableness:     {pred_np[3]:.2f}")
    print(f"    Neuroticism:       {pred_np[4]:.2f}")

# 多様性確認
print("\n[3/3] 予測の多様性を検証中...")
all_stds = [np.std([pred[i] for pred in predictions_list]) for i in range(5)]
avg_std = np.mean(all_stds)

print(f"\n  各特性の標準偏差:")
print(f"    Openness:          {all_stds[0]:.2f}")
print(f"    Conscientiousness: {all_stds[1]:.2f}")
print(f"    Extraversion:      {all_stds[2]:.2f}")
print(f"    Agreeableness:     {all_stds[3]:.2f}")
print(f"    Neuroticism:       {all_stds[4]:.2f}")
print(f"  平均標準偏差: {avg_std:.2f}")

print("\n" + "=" * 80)
if avg_std < 0.1:
    print("❌ 失敗: 予測が似すぎています")
else:
    print("✅ 成功: Model Aは正常に動作しています！")
print("=" * 80)
