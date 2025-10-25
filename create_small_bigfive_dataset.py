#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小規模Big Fiveデータセット作成（100サンプル）
ローカルでの完全な動作確認用
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from datasets import load_dataset
import os

print("=" * 80)
print("Creating Small Big Five Test Dataset")
print("=" * 80)

# 元のデータセットをロード
print("\nLoading full dataset: Fatima0923/Automated-Personality-Prediction")
dataset = load_dataset("Fatima0923/Automated-Personality-Prediction")

print(f"\nDataset structure: {list(dataset.keys())}")

# Train/Validationから小規模サンプルを抽出
if 'train' in dataset and 'validation' in dataset:
    # Train: 100サンプル
    # Val: 50サンプル
    small_train = dataset['train'].select(range(min(100, len(dataset['train']))))
    small_val = dataset['validation'].select(range(min(50, len(dataset['validation']))))

    print(f"\nExtracted:")
    print(f"  Train: {len(small_train)} samples (from {len(dataset['train'])})")
    print(f"  Val: {len(small_val)} samples (from {len(dataset['validation'])})")

    # サンプルデータ確認
    print("\nSample data check:")
    sample = small_train[0]
    print(f"  Fields: {list(sample.keys())}")
    print(f"  Text length: {len(sample['text'])} chars")
    print(f"  Text preview: {sample['text'][:100]}...")
    print(f"  Openness: {sample.get('openness', 'N/A')}")
    print(f"  Conscientiousness: {sample.get('conscientiousness', 'N/A')}")
    print(f"  Extraversion: {sample.get('extraversion', 'N/A')}")
    print(f"  Agreeableness: {sample.get('agreeableness', 'N/A')}")
    print(f"  Neuroticism: {sample.get('neuroticism', 'N/A')}")

    # フィールド存在確認
    required_fields = ['text', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    missing_fields = [f for f in required_fields if f not in sample]

    if missing_fields:
        print(f"\nWARNING: Missing required fields: {missing_fields}")
    else:
        print("\nAll required fields present")

    # ローカルに保存
    save_dir = 'data/small_bigfive'
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving to: {save_dir}")
    small_train.save_to_disk(f"{save_dir}/train")
    small_val.save_to_disk(f"{save_dir}/val")

    print("\nDataset saved successfully!")
    print(f"  Location: {save_dir}")
    print(f"  Train: {len(small_train)} samples")
    print(f"  Val: {len(small_val)} samples")

    print("\n" + "=" * 80)
    print("Small dataset creation completed!")
    print("=" * 80)
    print("\nNext step: Run local training test with this small dataset")

else:
    print("\nERROR: Unexpected dataset structure")
    print(f"Available splits: {list(dataset.keys())}")
    print("\nExpected: 'train' and 'validation' splits")
