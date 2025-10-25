#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重みの比較：checkpoint vs 新規初期化
"""

import torch
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

MODEL_FILE = "models/stage2_bigfive/best_model.pt"

print("Loading checkpoint...")
checkpoint = torch.load(MODEL_FILE, map_location='cpu')

# Checkpoint内の重みサンプル
print("\n" + "=" * 80)
print("Checkpoint encoder weights sample:")
print("=" * 80)

sample_key = 'encoder.base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight'
if sample_key in checkpoint['model_state_dict']:
    weight = checkpoint['model_state_dict'][sample_key]
    print(f"{sample_key}:")
    print(f"  Shape: {weight.shape}")
    print(f"  Mean: {weight.mean().item():.6f}")
    print(f"  Std: {weight.std().item():.6f}")
    print(f"  First 5 values: {weight.flatten()[:5].tolist()}")

# 新規初期化したモデルの重み
print("\n" + "=" * 80)
print("Newly initialized encoder weights:")
print("=" * 80)

base_model = AutoModel.from_pretrained("xlm-roberta-large")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
encoder_new = get_peft_model(base_model, lora_config)

# 対応する重みを取得
new_state_dict = encoder_new.state_dict()

# encoder.をbase_model.model.に変換
new_key = sample_key.replace('encoder.', '')
if new_key in new_state_dict:
    weight_new = new_state_dict[new_key]
    print(f"{new_key}:")
    print(f"  Shape: {weight_new.shape}")
    print(f"  Mean: {weight_new.mean().item():.6f}")
    print(f"  Std: {weight_new.std().item():.6f}")
    print(f"  First 5 values: {weight_new.flatten()[:5].tolist()}")

# 比較
print("\n" + "=" * 80)
print("Comparison:")
print("=" * 80)

if sample_key in checkpoint['model_state_dict'] and new_key in new_state_dict:
    checkpoint_weight = checkpoint['model_state_dict'][sample_key]
    new_weight = new_state_dict[new_key]

    diff = (checkpoint_weight - new_weight).abs().mean().item()
    print(f"Absolute difference (mean): {diff:.6f}")

    if diff < 0.0001:
        print("⚠️  WARNING: Weights are almost identical!")
        print("This suggests the checkpoint weights were NOT loaded correctly.")
    else:
        print("✓ Weights are different (expected for trained model)")

# regressorの重みも確認
print("\n" + "=" * 80)
print("Regressor weights:")
print("=" * 80)

regressor_key = 'regressor.0.weight'
if regressor_key in checkpoint['model_state_dict']:
    weight = checkpoint['model_state_dict'][regressor_key]
    print(f"{regressor_key}:")
    print(f"  Shape: {weight.shape}")
    print(f"  Mean: {weight.mean().item():.6f}")
    print(f"  Std: {weight.std().item():.6f}")
    print(f"  Max abs value: {weight.abs().max().item():.6f}")

print("\nDone.")
