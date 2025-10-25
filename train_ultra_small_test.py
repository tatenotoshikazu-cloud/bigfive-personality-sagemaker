#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-small training test (10 samples, CPU-friendly)
Goal: Verify training→save→load→inference pipeline in minimal time
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_from_disk
import numpy as np
import os

print("=" * 80)
print("ULTRA-SMALL TRAINING TEST (10 samples)")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Dataset
class BigFiveDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):  # 短くする
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trait_columns = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example.get('text', '')

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        labels = torch.tensor([
            float(example.get(trait, 50.0)) / 100.0
            for trait in self.trait_columns
        ], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

# Model
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

# Load data
print("\n" + "=" * 80)
print("STEP 1: Loading ultra-small dataset (10 samples)")
print("=" * 80)

train_dataset_raw = load_from_disk('data/small_bigfive/train')
# 最初の10サンプルのみ使用
train_dataset_raw = train_dataset_raw.select(range(10))

print(f"Train: {len(train_dataset_raw)} samples")

# Tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

train_dataset = BigFiveDataset(train_dataset_raw, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Small batch

# Model
print("\n" + "=" * 80)
print("STEP 2: Creating model")
print("=" * 80)

print("\nLoading base model...")
base_model = AutoModel.from_pretrained("xlm-roberta-large")

print("Creating regression model...")
model = BigFiveRegressionModel(base_model, num_traits=5)

# LoRA
print("Applying LoRA...")
lora_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'dense']
)
model.encoder = get_peft_model(model.encoder, lora_config)

model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# Save initial weights
print("\n" + "=" * 80)
print("STEP 3: Saving BEFORE-TRAINING weights")
print("=" * 80)

initial_regressor_weight = model.regressor[0].weight.detach().clone().cpu().numpy()
print(f"Initial regressor weight mean: {initial_regressor_weight.mean():.6f}")
print(f"Initial regressor weight std: {initial_regressor_weight.std():.6f}")

# Training (1 epoch)
print("\n" + "=" * 80)
print("STEP 4: Training (1 epoch, 10 samples)")
print("=" * 80)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

model.train()
total_loss = 0

for i, batch in enumerate(train_loader, 1):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()
    predictions = model(input_ids, attention_mask)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    print(f"  Batch {i}/{len(train_loader)}: loss={loss.item():.4f}")

avg_loss = total_loss / len(train_loader)
print(f"\nTraining completed!")
print(f"Average loss: {avg_loss:.4f}")

# Verify weights changed
print("\n" + "=" * 80)
print("STEP 5: Verifying weights CHANGED")
print("=" * 80)

trained_regressor_weight = model.regressor[0].weight.detach().clone().cpu().numpy()
print(f"Trained regressor weight mean: {trained_regressor_weight.mean():.6f}")
print(f"Trained regressor weight std: {trained_regressor_weight.std():.6f}")

weight_diff = np.abs(trained_regressor_weight - initial_regressor_weight).mean()
print(f"\nWeight difference (mean abs): {weight_diff:.6f}")

if weight_diff < 0.0001:
    print("\n❌ ERROR: Weights did NOT change! Training failed!")
    sys.exit(1)
else:
    print(f"\n✅ SUCCESS: Weights changed by {weight_diff:.6f}")

# Save model
print("\n" + "=" * 80)
print("STEP 6: Saving trained model")
print("=" * 80)

save_dir = 'models/ultra_test'
os.makedirs(save_dir, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': 1,
    'train_loss': avg_loss,
}, f"{save_dir}/test_model.pt")

tokenizer.save_pretrained(f"{save_dir}/tokenizer")

print(f"Model saved to: {save_dir}")

# Verify saved weights
print("\n" + "=" * 80)
print("STEP 7: Verifying SAVED weights are trained")
print("=" * 80)

saved_checkpoint = torch.load(f"{save_dir}/test_model.pt", map_location='cpu')
saved_weight = saved_checkpoint['model_state_dict']['regressor.0.weight'].numpy()
print(f"Saved checkpoint regressor weight mean: {saved_weight.mean():.6f}")
print(f"Saved checkpoint regressor weight std: {saved_weight.std():.6f}")

saved_diff = np.abs(saved_weight - initial_regressor_weight).mean()
print(f"Saved weight vs initial: {saved_diff:.6f}")

if saved_diff < 0.0001:
    print("\n❌ ERROR: Saved checkpoint contains INITIAL weights!")
    sys.exit(1)
else:
    print(f"\n✅ SUCCESS: Saved checkpoint contains trained weights!")

# Load into new model
print("\n" + "=" * 80)
print("STEP 8: Loading into NEW model instance")
print("=" * 80)

base_model_new = AutoModel.from_pretrained("xlm-roberta-large")
model_new = BigFiveRegressionModel(base_model_new, num_traits=5)

lora_config_new = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['query', 'key', 'value', 'dense']
)
model_new.encoder = get_peft_model(model_new.encoder, lora_config_new)

model_new.load_state_dict(saved_checkpoint['model_state_dict'])
model_new = model_new.to(device)
model_new.eval()

print("Model loaded successfully!")

# Test inference
print("\n" + "=" * 80)
print("STEP 9: Testing inference with DIFFERENT texts")
print("=" * 80)

test_texts = [
    "I am very happy and excited today!",
    "I always organize my tasks carefully.",
    "I prefer to stay quiet and work alone."
]

BIG5_LABELS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

predictions_list = []

for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        predictions = model_new(input_ids, attention_mask)
        predictions = predictions.cpu().numpy()[0]

    predictions_list.append(predictions)

    print(f"\nTest {i}: {text[:40]}...")
    for j, label in enumerate(BIG5_LABELS):
        score = predictions[j] * 99
        print(f"  {label:20s}: {score:5.2f}")

# Verify diversity
print("\n" + "=" * 80)
print("FINAL VERIFICATION: Predictions differ?")
print("=" * 80)

for i in range(len(BIG5_LABELS)):
    trait = BIG5_LABELS[i]
    scores = [pred[i] * 99 for pred in predictions_list]
    std = np.std(scores)
    range_val = max(scores) - min(scores)

    print(f"\n{trait}:")
    print(f"  Scores: {[f'{s:.2f}' for s in scores]}")
    print(f"  Std: {std:.2f}")
    print(f"  Range: {range_val:.2f}")

all_stds = [np.std([pred[i] * 99 for pred in predictions_list]) for i in range(5)]
avg_std = np.mean(all_stds)

print("\n" + "=" * 80)
print("OVERALL RESULT")
print("=" * 80)

if avg_std < 0.1:
    print(f"\n❌ ERROR: Predictions too similar (avg std: {avg_std:.2f})")
    print("Model is not using input text!")
    sys.exit(1)
else:
    print(f"\n✅ SUCCESS: Predictions vary (avg std: {avg_std:.2f})")

print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED!")
print("=" * 80)
print("\nThe training→save→load→inference pipeline works correctly!")
print("Ready to proceed with full training.")
