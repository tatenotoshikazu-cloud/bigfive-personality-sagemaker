#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B テスト - 実際の文字起こしデータで予測（28件）
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
import boto3
import numpy as np

# Model class
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
print("Model B テスト - 実際の文字起こしデータで予測")
print("=" * 80)

# Load model
print("\n[1/4] Loading Model B (Stage 1 + Stage 2, 1エポック)...")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

adapter_path = 'model_b_test_extracted/lora_weights'
base_model = AutoModel.from_pretrained('xlm-roberta-large')
base_model = PeftModel.from_pretrained(base_model, adapter_path)
base_model = base_model.merge_and_unload()

model = BigFiveRegressionModel(base_model)
checkpoint = torch.load('model_b_test_extracted/best_model.pt', map_location='cpu')

state_dict = checkpoint['model_state_dict']
regressor_state_dict = {k.replace('regressor.', ''): v for k, v in state_dict.items() if 'regressor' in k}
model.regressor.load_state_dict(regressor_state_dict)
model.eval()
print(f"  ✓ Model loaded (RMSE: {checkpoint.get('best_val_rmse', 'N/A'):.4f})")

# Get DynamoDB records with transcripts
print("\n[2/4] Fetching DynamoDB records with transcripts...")
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table = dynamodb.Table('recording-poc-records')

response = table.scan()
all_records = response.get('Items', [])
records_with_transcript = [r for r in all_records if r.get('transcript_text')]
print(f"  ✓ Found {len(records_with_transcript)} records with transcripts")

# Predict
print("\n[3/4] Making predictions on all transcripts...")
trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

predictions_list = []
for i, record in enumerate(records_with_transcript, 1):
    uuid = record.get('uuid', 'Unknown')
    transcript = record.get('transcript_text', '')

    # Predict
    inputs = tokenizer(transcript, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    with torch.no_grad():
        predictions = model(inputs['input_ids'], inputs['attention_mask'])

    scores = (predictions[0].numpy() * 99).astype(int)
    predictions_list.append(scores)

    # Show first 3 predictions in detail
    if i <= 3:
        print(f"\n  Record {i}/{len(records_with_transcript)} (uuid: {uuid[:15]}...):")
        print(f"    Transcript length: {len(transcript)} chars")
        print(f"    Transcript preview: {transcript[:80]}...")
        for trait, score in zip(trait_names, scores):
            print(f"    {trait}: {score}")

# Diversity analysis
print("\n[4/4] Diversity analysis...")
all_scores = np.array(predictions_list)
std_per_trait = all_scores.std(axis=0)
avg_std = std_per_trait.mean()

print("\n" + "=" * 80)
print("Prediction Summary:")
print("=" * 80)
print(f"Total predictions: {len(predictions_list)}")
print(f"\nScore statistics per trait:")
for trait, scores_col in zip(trait_names, all_scores.T):
    print(f"  {trait}:")
    print(f"    Mean: {scores_col.mean():.1f}, Std: {scores_col.std():.2f}, Min: {scores_col.min()}, Max: {scores_col.max()}")

print("\n" + "=" * 80)
print("Diversity Analysis:")
print("=" * 80)
for trait, std in zip(trait_names, std_per_trait):
    print(f"  {trait}: std = {std:.2f}")
print(f"\n  Average std: {avg_std:.2f}")

if avg_std > 3.0:
    print("  ✓ Good diversity (std > 3.0)")
    status = "PASS"
else:
    print("  ⚠ Low diversity (std < 3.0)")
    status = "WARNING"

print("\n" + "=" * 80)
print(f"Result: {status}")
print("=" * 80)
print(f"\nModel B (Stage 1 + Stage 2, 1エポック) で{len(predictions_list)}件の予測を完了しました。")
print("=" * 80)
