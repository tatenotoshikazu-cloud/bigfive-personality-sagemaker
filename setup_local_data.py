# -*- coding: utf-8 -*-
"""
ローカルデータセットセットアップ
Nemotron-Personas-Japanから小規模データを作成
"""
from datasets import load_dataset, Dataset
import os
import json
import pandas as pd

print("=" * 60)
print("ローカル用データセット作成")
print("=" * 60)

# Nemotronデータ取得（小規模）
num_samples = 500
print(f"\nNemotron-Personas-Japanから{num_samples}件ダウンロード中...")

dataset = load_dataset("nvidia/Nemotron-Personas-Japan", split="train", streaming=True)

samples = []
for i, sample in enumerate(dataset):
    samples.append(sample)
    if (i + 1) % 100 == 0:
        print(f"  進捗: {i + 1}件")
    if i >= num_samples - 1:
        break

print(f"\nOK: {len(samples)}件取得完了")

# Dataset形式に変換
print("\nDataset形式に変換中...")
df = pd.DataFrame(samples)
small_dataset = Dataset.from_pandas(df)

# 保存
os.makedirs("data/local/nemotron", exist_ok=True)
small_dataset.save_to_disk("data/local/nemotron/personas")

print(f"OK: 保存完了 data/local/nemotron/personas/")

# データ分割（train/val）
print("\ntrain/val分割中...")
split = small_dataset.train_test_split(test_size=0.2, seed=42)

os.makedirs("data/local/processed", exist_ok=True)
split['train'].save_to_disk("data/local/processed/train")
split['test'].save_to_disk("data/local/processed/val")

print(f"OK: Train {len(split['train'])}件")
print(f"OK: Val {len(split['test'])}件")

# サンプル表示
print("\n" + "=" * 60)
print("サンプルデータ")
print("=" * 60)
sample = split['train'][0]

print("\n主要フィールド:")
display_fields = ['persona', 'age', 'sex', 'occupation', 'region']
for field in display_fields:
    if field in sample:
        value = sample[field]
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        print(f"\n{field}:")
        print(f"  {value}")

print("\n" + "=" * 60)
print("セットアップ完了！")
print("=" * 60)
print("\nデータ構造:")
print(f"  data/local/nemotron/personas/  - 元データ")
print(f"  data/local/processed/train/    - 訓練用")
print(f"  data/local/processed/val/      - 検証用")
print("\n次のステップ:")
print("  1. ローカルテスト用トレーニングスクリプト実行")
print("  2. Nemotronデータで言語理解モデル訓練")
print("  3. Big Fiveデータが入手できたら追加学習")
