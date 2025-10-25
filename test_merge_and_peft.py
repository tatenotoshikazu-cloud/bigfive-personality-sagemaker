#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: Does applying LoRA after merge_and_unload() cause issues?
Hypothesis: get_peft_model() might reset base weights
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig
import numpy as np

print("=" * 80)
print("Testing: merge_and_unload() → get_peft_model() weight preservation")
print("=" * 80)

# Step 1: Load base model
print("\nStep 1: Loading base xlm-roberta-large...")
base_model_original = AutoModel.from_pretrained("xlm-roberta-large")

# Get initial weight reference
initial_weight = base_model_original.encoder.layer[0].attention.self.query.weight.detach().clone().cpu().numpy()
print(f"Initial weight mean: {initial_weight.mean():.6f}")
print(f"Initial weight std: {initial_weight.std():.6f}")

# Step 2: Simulate Stage 1 - apply LoRA
print("\nStep 2: Applying Stage 1 LoRA...")
lora_config_stage1 = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'dense']
)
model_stage1 = get_peft_model(base_model_original, lora_config_stage1)
print("Stage 1 LoRA applied")

# Step 3: Merge and unload (simulate Stage 1 completion)
print("\nStep 3: Merging Stage 1 LoRA...")
merged_model = model_stage1.merge_and_unload()
print("Merged successfully")

merged_weight = merged_model.encoder.layer[0].attention.self.query.weight.detach().clone().cpu().numpy()
print(f"Merged weight mean: {merged_weight.mean():.6f}")
print(f"Merged weight std: {merged_weight.std():.6f}")

diff_after_merge = np.abs(merged_weight - initial_weight).mean()
print(f"Difference from initial: {diff_after_merge:.6f}")

# Step 4: Apply Stage 2 LoRA (THIS IS WHERE THE PROBLEM MIGHT OCCUR)
print("\nStep 4: Applying Stage 2 LoRA on merged model...")
print("⚠️  CRITICAL: Does get_peft_model() preserve the merged weights?")

lora_config_stage2 = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'dense']
)
model_stage2 = get_peft_model(merged_model, lora_config_stage2)
print("Stage 2 LoRA applied")

# Step 5: Check if base weights are preserved
print("\nStep 5: Checking if merged weights were preserved...")

# In PEFT model, base weights are at base_model.model.encoder...
stage2_base_weight = model_stage2.base_model.model.encoder.layer[0].attention.self.query.base_layer.weight.detach().clone().cpu().numpy()

print(f"Stage 2 base_layer weight mean: {stage2_base_weight.mean():.6f}")
print(f"Stage 2 base_layer weight std: {stage2_base_weight.std():.6f}")

diff_vs_merged = np.abs(stage2_base_weight - merged_weight).mean()
diff_vs_initial = np.abs(stage2_base_weight - initial_weight).mean()

print(f"\nDifference vs merged weights: {diff_vs_merged:.6f}")
print(f"Difference vs initial weights: {diff_vs_initial:.6f}")

print("\n" + "=" * 80)
print("RESULT:")
print("=" * 80)

if diff_vs_merged < 0.0001:
    print("✅ SUCCESS: Merged weights PRESERVED in Stage 2 base_layer")
    print(f"   (diff from merged: {diff_vs_merged:.6f})")
else:
    print(f"❌ ERROR: Merged weights NOT preserved!")
    print(f"   Difference from merged: {diff_vs_merged:.6f}")
    print(f"   Difference from initial: {diff_vs_initial:.6f}")

    if diff_vs_initial < 0.0001:
        print("\n🚨 CRITICAL: Weights were RESET to initial pretrained values!")
        print("   This explains why predictions were identical!")

print("\n" + "=" * 80)
