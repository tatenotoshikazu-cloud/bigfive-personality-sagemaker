#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model A 簡易検証スクリプト
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import numpy as np

print("=" * 80)
print("Model A 簡易検証")
print("=" * 80)

# モデルロード
print(f"\n[1/2] モデルをロード中...")
checkpoint = torch.load('model_a_extracted/best_model.pt', map_location='cpu')
print(f"✅ ロード完了")

# チェックポイント内容確認
print(f"\n[2/2] チェックポイント内容:")
print(f"  Keys: {list(checkpoint.keys())}")

# Regressor の重みを確認
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']

    # Regressorの層を確認
    regressor_keys = [k for k in state_dict.keys() if 'regressor' in k]
    print(f"\n  Regressor layers ({len(regressor_keys)} layers):")
    for key in sorted(regressor_keys):
        print(f"    {key}: {state_dict[key].shape}")

    # 最終層の重み確認（Big Five出力層）
    final_layer_keys = [k for k in regressor_keys if '.weight' in k]
    if final_layer_keys:
        last_layer = sorted(final_layer_keys)[-1]
        weight = state_dict[last_layer].numpy()
        print(f"\n  最終層 ({last_layer}):")
        print(f"    Shape: {weight.shape}")
        print(f"    Std: {weight.std():.6f}")
        print(f"    Mean: {weight.mean():.6f}")

        if weight.std() < 0.01:
            print("    ⚠️ 警告: 重みのstdが小さい（初期値の可能性）")
        else:
            print("    ✅ 重みは学習されています")

print("\n" + "=" * 80)
print("✅ 検証完了")
print("=" * 80)
