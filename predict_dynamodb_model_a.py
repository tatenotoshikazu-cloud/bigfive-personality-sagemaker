#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A でDynamoDB 28件の推論を実行
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import numpy as np
import json
from decimal import Decimal

print("=" * 80)
print("Model A - DynamoDB 28件 Big Five推論")
print("=" * 80)

# DynamoDBからデータ取得
print("\n[1/5] DynamoDBからデータ取得中...")
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table = dynamodb.Table('recording-poc-records')

response = table.scan()
items = response['Items']
print(f"✅ {len(items)}件取得")

# モデル構築
print("\n[2/5] Model Aをロード中...")

class BigFiveModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # train_bigfive.pyのRegressor構造に合わせる
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),       # Layer 0
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),        # Layer 3
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5),          # Layer 6 (最終層)
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regressor(cls_output)
        return predictions

# LoRAモデルをロード
adapter_path = 'model_a_extracted/lora_weights'
base_model = AutoModel.from_pretrained('xlm-roberta-large')
base_model = PeftModel.from_pretrained(base_model, adapter_path)
base_model = base_model.merge_and_unload()

# BigFiveModelを構築
model = BigFiveModel(base_model)

# Checkpointをロード
checkpoint = torch.load('model_a_extracted/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
print(f"✅ Model Aロード完了（RMSE: {checkpoint['best_val_rmse']:.4f}）")

# 推論実行
print("\n[3/5] 28件の推論実行中...")
predictions_all = []

for i, item in enumerate(items, 1):
    transcript_id = item['uuid']
    text = item.get('transcript_text', '')

    if not text:
        print(f"  [{i}] {transcript_id}: テキストなし - スキップ")
        continue

    # トークン化
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )

    # 推論
    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])

    # 0-99スケールに変換
    pred_np = predictions[0].numpy() * 99

    predictions_all.append({
        'transcriptId': transcript_id,
        'predictions': pred_np,
        'text_length': len(text)
    })

    print(f"  [{i:2d}] {transcript_id}: "
          f"O={pred_np[0]:.1f} C={pred_np[1]:.1f} E={pred_np[2]:.1f} "
          f"A={pred_np[3]:.1f} N={pred_np[4]:.1f}")

# 多様性確認
print("\n[4/5] 予測の多様性を検証中...")
all_stds = [
    np.std([p['predictions'][i] for p in predictions_all])
    for i in range(5)
]
avg_std = np.mean(all_stds)

print(f"\n  各特性の標準偏差:")
print(f"    Openness:          {all_stds[0]:.2f}")
print(f"    Conscientiousness: {all_stds[1]:.2f}")
print(f"    Extraversion:      {all_stds[2]:.2f}")
print(f"    Agreeableness:     {all_stds[3]:.2f}")
print(f"    Neuroticism:       {all_stds[4]:.2f}")
print(f"  平均標準偏差: {avg_std:.2f}")

# DynamoDBに保存
print("\n[5/5] DynamoDBに結果を保存中...")
saved_count = 0

for pred in predictions_all:
    try:
        # Decimal型に変換（DynamoDB用）
        bigfive = {
            'openness': Decimal(str(round(float(pred['predictions'][0]), 2))),
            'conscientiousness': Decimal(str(round(float(pred['predictions'][1]), 2))),
            'extraversion': Decimal(str(round(float(pred['predictions'][2]), 2))),
            'agreeableness': Decimal(str(round(float(pred['predictions'][3]), 2))),
            'neuroticism': Decimal(str(round(float(pred['predictions'][4]), 2)))
        }

        table.update_item(
            Key={'uuid': pred['transcriptId']},
            UpdateExpression='SET bigFive = :bigfive',
            ExpressionAttributeValues={':bigfive': bigfive}
        )
        saved_count += 1

    except Exception as e:
        print(f"  ⚠️ {pred['transcriptId']}: 保存失敗 - {e}")

print(f"✅ {saved_count}/{len(predictions_all)}件保存完了")

print("\n" + "=" * 80)
if avg_std < 0.1:
    print("⚠️ 警告: 予測が似すぎています（平均std < 0.1）")
    print("→ モデルは学習されていない可能性があります")
else:
    print("✅ 成功: 予測は多様です！")
    print(f"→ Model Aは正常に動作しています（平均std: {avg_std:.2f}）")
print("=" * 80)
