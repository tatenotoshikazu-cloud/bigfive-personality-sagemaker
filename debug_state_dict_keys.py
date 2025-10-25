#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State dict keys comparison
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_FILE = "models/stage2_bigfive/best_model.pt"

# Checkpointロード
checkpoint = torch.load(MODEL_FILE, map_location='cpu')
checkpoint_keys = set(checkpoint['model_state_dict'].keys())

print(f"Checkpoint has {len(checkpoint_keys)} keys")
print("\nFirst 20 checkpoint keys:")
for i, key in enumerate(sorted(list(checkpoint_keys))[:20], 1):
    print(f"  {i}. {key}")

# モデル構築
base_model = AutoModel.from_pretrained("xlm-roberta-large")

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

model = BigFiveRegressionModel(base_model, num_traits=5)

peft_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'dense']
)
model.encoder = get_peft_model(model.encoder, peft_config)

model_keys = set(model.state_dict().keys())

print(f"\n\nModel has {len(model_keys)} keys")
print("\nFirst 20 model keys:")
for i, key in enumerate(sorted(list(model_keys))[:20], 1):
    print(f"  {i}. {key}")

# キーの比較
only_in_checkpoint = checkpoint_keys - model_keys
only_in_model = model_keys - checkpoint_keys

print(f"\n\nKeys only in checkpoint: {len(only_in_checkpoint)}")
if len(only_in_checkpoint) > 0 and len(only_in_checkpoint) <= 20:
    for key in sorted(list(only_in_checkpoint)):
        print(f"  - {key}")

print(f"\nKeys only in model: {len(only_in_model)}")
if len(only_in_model) > 0 and len(only_in_model) <= 20:
    for key in sorted(list(only_in_model)):
        print(f"  - {key}")

common_keys = checkpoint_keys & model_keys
print(f"\nCommon keys: {len(common_keys)}")

if len(only_in_checkpoint) > 0 or len(only_in_model) > 0:
    print("\n⚠️  WARNING: Key mismatch detected!")
    print("This will cause load_state_dict to fail or load incorrectly.")
else:
    print("\n✓ All keys match!")

print("\nDone.")
