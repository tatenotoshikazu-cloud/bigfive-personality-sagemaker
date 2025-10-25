#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル出力のデバッグ
各レイヤーの出力を確認
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, LoraConfig, get_peft_model

MODEL_DIR = "models/stage2_bigfive"
MODEL_FILE = f"{MODEL_DIR}/best_model.pt"
TOKENIZER_DIR = f"{MODEL_DIR}/lora_weights"

device = torch.device('cpu')

print("Loading checkpoint...")
checkpoint = torch.load(MODEL_FILE, map_location=device)
print(f"Checkpoint epoch: {checkpoint['epoch']}, RMSE: {checkpoint['best_val_rmse']:.4f}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# Model
base_model = AutoModel.from_pretrained("xlm-roberta-large")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
encoder = get_peft_model(base_model, lora_config)

hidden_dim = encoder.config.hidden_size
regressor = nn.Sequential(
    nn.Linear(hidden_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 5)
)

class FullModel(nn.Module):
    def __init__(self, encoder, regressor):
        super().__init__()
        self.encoder = encoder
        self.regressor = regressor

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        predictions = self.regressor(cls_embedding)
        return torch.sigmoid(predictions)

model = FullModel(encoder, regressor)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\n" + "=" * 80)
print("Testing different texts:")
print("=" * 80)

test_texts = [
    "こんにちは、私は太郎です。",
    "I am very happy and excited today!",
    "The weather is terrible and I feel sad.",
    "私は非常に几帳面で、計画的に物事を進めるのが好きです。"
]

for i, text in enumerate(test_texts, 1):
    print(f"\n[Test {i}] Text: {text}")

    inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        # Encoder output
        encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]

        print(f"  CLS embedding shape: {cls_embedding.shape}")
        print(f"  CLS embedding mean: {cls_embedding.mean().item():.6f}")
        print(f"  CLS embedding std: {cls_embedding.std().item():.6f}")
        print(f"  CLS embedding min: {cls_embedding.min().item():.6f}")
        print(f"  CLS embedding max: {cls_embedding.max().item():.6f}")

        # Regressor output (before sigmoid)
        regressor_output = model.regressor(cls_embedding)
        print(f"  Regressor output (raw): {regressor_output.numpy()[0]}")

        # Final prediction (after sigmoid)
        final_output = torch.sigmoid(regressor_output)
        print(f"  Final prediction (0-1): {final_output.numpy()[0]}")
        print(f"  Final prediction (0-99): {(final_output * 99).numpy()[0]}")

print("\n" + "=" * 80)
print("Regressor weight statistics:")
print("=" * 80)

with torch.no_grad():
    first_layer_weight = model.regressor[0].weight
    print(f"First layer weight shape: {first_layer_weight.shape}")
    print(f"First layer weight mean: {first_layer_weight.mean().item():.6f}")
    print(f"First layer weight std: {first_layer_weight.std().item():.6f}")

print("\nDone.")
