#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify if checkpoint contains trained weights
"""

import torch
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import numpy as np

MODEL_FILE = "models/stage2_bigfive/best_model.pt"

print("Loading checkpoint...")
checkpoint = torch.load(MODEL_FILE, map_location='cpu')
print(f"Checkpoint epoch: {checkpoint['epoch']}, Best RMSE: {checkpoint['best_val_rmse']:.4f}")

# Checkpointからエンコーダーの重みを取得
sample_key_base = 'encoder.base_model.model.encoder.layer.0.attention.self.query.base_layer.weight'
sample_key_lora_A = 'encoder.base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight'

if sample_key_base in checkpoint['model_state_dict']:
    ckpt_base_weight = checkpoint['model_state_dict'][sample_key_base].numpy()
    print(f"\nCheckpoint base_layer weight ({sample_key_base}):")
    print(f"  Shape: {ckpt_base_weight.shape}")
    print(f"  Mean: {ckpt_base_weight.mean():.6f}")
    print(f"  Std: {ckpt_base_weight.std():.6f}")
    print(f"  Min: {ckpt_base_weight.min():.6f}")
    print(f"  Max: {ckpt_base_weight.max():.6f}")

if sample_key_lora_A in checkpoint['model_state_dict']:
    ckpt_lora_weight = checkpoint['model_state_dict'][sample_key_lora_A].numpy()
    print(f"\nCheckpoint LoRA_A weight ({sample_key_lora_A}):")
    print(f"  Shape: {ckpt_lora_weight.shape}")
    print(f"  Mean: {ckpt_lora_weight.mean():.6f}")
    print(f"  Std: {ckpt_lora_weight.std():.6f}")
    print(f"  Min: {ckpt_lora_weight.min():.6f}")
    print(f"  Max: {ckpt_lora_weight.max():.6f}")

# 新規初期化したモデルの重みと比較
print("\n" + "="*80)
print("Creating newly initialized model for comparison...")
print("="*80)

# 新規モデル作成
base_model = AutoModel.from_pretrained("xlm-roberta-large")

# LoRA適用（学習時と同じ設定）
lora_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'dense']
)
encoder_new = get_peft_model(base_model, lora_config)

# 新規モデルの重みを取得
new_state_dict = encoder_new.state_dict()

new_key_base = sample_key_base.replace('encoder.', '')
new_key_lora_A = sample_key_lora_A.replace('encoder.', '')

if new_key_base in new_state_dict:
    new_base_weight = new_state_dict[new_key_base].numpy()
    print(f"\nNew model base_layer weight:")
    print(f"  Shape: {new_base_weight.shape}")
    print(f"  Mean: {new_base_weight.mean():.6f}")
    print(f"  Std: {new_base_weight.std():.6f}")
    print(f"  Min: {new_base_weight.min():.6f}")
    print(f"  Max: {new_base_weight.max():.6f}")

if new_key_lora_A in new_state_dict:
    new_lora_weight = new_state_dict[new_key_lora_A].numpy()
    print(f"\nNew model LoRA_A weight:")
    print(f"  Shape: {new_lora_weight.shape}")
    print(f"  Mean: {new_lora_weight.mean():.6f}")
    print(f"  Std: {new_lora_weight.std():.6f}")
    print(f"  Min: {new_lora_weight.min():.6f}")
    print(f"  Max: {new_lora_weight.max():.6f}")

# 比較
print("\n" + "="*80)
print("Comparison:")
print("="*80)

if sample_key_base in checkpoint['model_state_dict'] and new_key_base in new_state_dict:
    base_diff = np.abs(ckpt_base_weight - new_base_weight).mean()
    print(f"\nBase layer weight difference (mean absolute): {base_diff:.6f}")

    if base_diff < 0.0001:
        print("  ⚠️  WARNING: Checkpoint and new model base weights are almost IDENTICAL!")
        print("  This suggests the base model weights were NOT trained (still at pretrained values)")
    else:
        print("  ✓ Base weights are different (good - suggests training occurred)")

if sample_key_lora_A in checkpoint['model_state_dict'] and new_key_lora_A in new_state_dict:
    lora_diff = np.abs(ckpt_lora_weight - new_lora_weight).mean()
    print(f"\nLoRA weight difference (mean absolute): {lora_diff:.6f}")

    if lora_diff < 0.001:
        print("  ⚠️  WARNING: Checkpoint and new model LoRA weights are almost IDENTICAL!")
        print("  This suggests LoRA weights were NOT trained!")
    else:
        print("  ✓ LoRA weights are different (expected - they should have been trained)")

# Regressorの重みも確認
regressor_key = 'regressor.0.weight'
if regressor_key in checkpoint['model_state_dict']:
    regressor_weight = checkpoint['model_state_dict'][regressor_key].numpy()
    print(f"\n" + "="*80)
    print("Regressor weights:")
    print("="*80)
    print(f"  Shape: {regressor_weight.shape}")
    print(f"  Mean: {regressor_weight.mean():.6f}")
    print(f"  Std: {regressor_weight.std():.6f}")
    print(f"  Min: {regressor_weight.min():.6f}")
    print(f"  Max: {regressor_weight.max():.6f}")

    # 初期化パターンと比較
    # Kaiming uniform初期化の場合、stdは約0.018前後
    if 0.017 < regressor_weight.std() < 0.020 and abs(regressor_weight.mean()) < 0.001:
        print("  ⚠️  WARNING: Regressor weights look like they are at INITIALIZATION values!")
        print("  Expected std for trained weights would be significantly different.")
    else:
        print("  ✓ Regressor weights appear to have been trained")

print("\n" + "="*80)
print("Conclusion:")
print("="*80)

# 最終判定
issues = []

if base_diff < 0.0001:
    issues.append("Base model weights NOT trained")
if lora_diff < 0.001:
    issues.append("LoRA weights NOT trained")
if 0.017 < regressor_weight.std() < 0.020:
    issues.append("Regressor weights NOT trained")

if issues:
    print("\n❌ CRITICAL ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nThis explains why all predictions are identical!")
    print("The model was saved BEFORE training completed, or training didn't update weights.")
else:
    print("\n✓ All weights appear to have been trained correctly")
    print("The issue must be elsewhere (e.g., in the inference code)")

print("\nDone.")
