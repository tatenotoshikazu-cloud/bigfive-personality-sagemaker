#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A ダウンロード・検証スクリプト
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
import tarfile
import os
import torch
import numpy as np
from transformers import AutoTokenizer

# 設定
REGION = 'ap-northeast-1'
BUCKET = 'bigfive-personality-sagemaker-1761305156'
MODEL_A_S3_PATH = 'output/stage2_bigfive_full/pytorch-training-2025-10-25-08-23-19-001/output/model.tar.gz'
LOCAL_TAR = 'model_a.tar.gz'
LOCAL_DIR = 'model_a_extracted'

print("=" * 80)
print("Model A ダウンロード・検証")
print("=" * 80)

# S3からダウンロード
print(f"\n[1/6] S3からダウンロード中...")
s3 = boto3.client('s3', region_name=REGION)
s3.download_file(BUCKET, MODEL_A_S3_PATH, LOCAL_TAR)
print(f"✅ ダウンロード完了: {LOCAL_TAR}")

# 解凍
print(f"\n[2/6] モデルを解凍中...")
os.makedirs(LOCAL_DIR, exist_ok=True)
with tarfile.open(LOCAL_TAR, 'r:gz') as tar:
    tar.extractall(LOCAL_DIR)
print(f"✅ 解凍完了: {LOCAL_DIR}/")

# モデルロード
print(f"\n[3/6] モデルをロード中...")
checkpoint_path = os.path.join(LOCAL_DIR, 'best_model.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
print(f"✅ ロード完了")

# 重み検証
print(f"\n[4/6] 重みを検証中...")
regressor_weight = checkpoint['model_state_dict']['regressor.0.weight'].numpy()
regressor_weight_std = regressor_weight.std()
regressor_weight_mean = regressor_weight.mean()

print(f"  Regressor weight std: {regressor_weight_std:.6f}")
print(f"  Regressor weight mean: {regressor_weight_mean:.6f}")

if regressor_weight_std < 0.01:
    print("  ⚠️ 警告: 重みのstdが小さい（初期値の可能性）")
else:
    print("  ✅ 重みは学習されています")

# 推論テスト
print(f"\n[5/6] 推論テスト（3つの異なるテキスト）...")

from transformers import AutoModel
from peft import PeftModel
import torch.nn as nn

# モデル構築（train_bigfive.pyと同じ構造）
class BigFiveModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 5),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regressor(cls_output)
        return predictions

# LoRAモデルをロード
print("  LoRAモデルをロード中...")
adapter_path = os.path.join(LOCAL_DIR, 'lora_weights')
base_model = AutoModel.from_pretrained('xlm-roberta-large')
base_model = PeftModel.from_pretrained(base_model, adapter_path)
base_model = base_model.merge_and_unload()

model = BigFiveModel(base_model)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

# テストテキスト
test_texts = [
    "I love meeting new people and going to social events. I'm very outgoing and energetic!",
    "I prefer to stay at home and read books quietly. Social gatherings make me uncomfortable.",
    "I'm organized and always plan ahead. I make detailed to-do lists every day."
]

print(f"\n  推論結果:")
predictions_list = []

for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])

    pred_np = predictions[0].numpy() * 99  # 0-99スケール
    predictions_list.append(pred_np)

    print(f"\n  テキスト {i}: \"{text[:50]}...\"")
    print(f"    Openness:          {pred_np[0]:.2f}")
    print(f"    Conscientiousness: {pred_np[1]:.2f}")
    print(f"    Extraversion:      {pred_np[2]:.2f}")
    print(f"    Agreeableness:     {pred_np[3]:.2f}")
    print(f"    Neuroticism:       {pred_np[4]:.2f}")

# 予測の多様性を確認
print(f"\n[6/6] 予測の多様性を検証中...")
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
    print("❌ 失敗: 予測が似すぎています（平均std < 0.1）")
    print("→ モデルは学習されていない可能性があります")
    sys.exit(1)
else:
    print("✅ 成功: 予測は多様です！")
    print("→ Model Aは正常に学習されています")
print("=" * 80)
