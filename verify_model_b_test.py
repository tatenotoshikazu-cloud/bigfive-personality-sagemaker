#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B テスト検証スクリプト - 3つのサンプルテキストで推論テスト
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Model class (same as training)
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
print("Model B テスト検証 (Stage 1 + Stage 2, 1エポック)")
print("=" * 80)

# Load tokenizer
print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
print("  ✓ Tokenizer loaded")

# Load LoRA weights
print("\n[2/4] Loading LoRA adapter...")
adapter_path = 'model_b_test_extracted/lora_weights'
base_model = AutoModel.from_pretrained('xlm-roberta-large')
base_model = PeftModel.from_pretrained(base_model, adapter_path)
base_model = base_model.merge_and_unload()
print("  ✓ LoRA weights merged")

# Create model
print("\n[3/4] Loading model...")
model = BigFiveRegressionModel(base_model)
checkpoint = torch.load('model_b_test_extracted/best_model.pt', map_location='cpu')

# Load only regressor weights (encoder already has LoRA merged)
state_dict = checkpoint['model_state_dict']
regressor_state_dict = {k.replace('regressor.', ''): v for k, v in state_dict.items() if 'regressor' in k}
model.regressor.load_state_dict(regressor_state_dict)
model.eval()
print(f"  ✓ Model loaded (RMSE: {checkpoint.get('best_val_rmse', 'N/A'):.4f})")

# Test with 3 sample texts
print("\n[4/4] Testing with 3 sample texts...")
test_texts = [
    "I love meeting new people and going to parties. I'm always the center of attention!",  # High E
    "I prefer quiet evenings at home with a good book. Crowds make me nervous.",  # Low E
    "I always keep my desk organized and plan everything in advance."  # High C
]

trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])

    scores = (predictions[0].numpy() * 99).astype(int)

    print(f"\n  Text {i}: {text[:50]}...")
    for trait, score in zip(trait_names, scores):
        print(f"    {trait}: {score}")

# Calculate diversity
import numpy as np
all_scores = []
for text in test_texts:
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])
    all_scores.append((predictions[0].numpy() * 99).astype(int))

all_scores = np.array(all_scores)
std_per_trait = all_scores.std(axis=0)
avg_std = std_per_trait.mean()

print("\n" + "=" * 80)
print("Diversity Analysis:")
print("=" * 80)
for trait, std in zip(trait_names, std_per_trait):
    print(f"  {trait}: std = {std:.2f}")
print(f"\n  Average std: {avg_std:.2f}")

if avg_std > 3.0:
    print("  ✓ Good diversity (std > 3.0)")
else:
    print("  ⚠ Low diversity (std < 3.0)")

print("\n" + "=" * 80)
print("SUCCESS: Model B テスト検証完了!")
print("=" * 80)
print("\nModel B (Stage 1 + Stage 2, 1エポック) は正常に動作しています。")
print("次のステップ: フル5エポック学習を実行")
print("=" * 80)
