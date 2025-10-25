#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big Five Personality Prediction - Simple Direct Load
best_model.ptから直接重みをロード
"""

import torch
import torch.nn as nn
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, LoraConfig, get_peft_model

def main():
    print("=" * 80)
    print("Big Five Personality Prediction - Simple Direct Load")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    MODEL_DIR = "models/stage2_bigfive"
    MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pt")
    TOKENIZER_DIR = os.path.join(MODEL_DIR, "lora_weights")

    print(f"\nLoading checkpoint from: {MODEL_FILE}")
    checkpoint = torch.load(MODEL_FILE, map_location=device)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation RMSE: {checkpoint['best_val_rmse']:.4f}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"\nTokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # モデル構造を手動で再構築（学習時と完全に同じ）
    print("\nRebuilding model architecture...")

    # Base model with LoRA
    base_model = AutoModel.from_pretrained("xlm-roberta-large")

    # LoRA config (学習時と同じ設定)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    encoder = get_peft_model(base_model, lora_config)

    # Regression head (学習時と完全に同じ構造)
    hidden_dim = encoder.config.hidden_size  # 1024
    regressor = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 5)
    )

    # 完全なモデル
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

    # 重みをロード
    print("\nLoading weights from checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Big Fiveラベル
    BIG5_LABELS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

    def predict_big5(text):
        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
            predictions = predictions.cpu().numpy()[0]

        result = {}
        for i, label in enumerate(BIG5_LABELS):
            score_01 = float(predictions[i])
            score_100 = score_01 * 99

            result[label] = {
                'score_01': round(score_01, 4),
                'score_100': round(score_100, 2)
            }

        return result

    # Transcripts読み込み
    transcript_file = 'transcripts_data.json'
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    print(f"\n{len(transcripts)} transcripts loaded")
    print("\nRunning predictions...")
    print("-" * 80)

    results = []

    for i, item in enumerate(transcripts, 1):
        transcript_text = item['transcript_text']
        text_length = len(transcript_text)

        print(f"\n[{i}/{len(transcripts)}] {item['uuid']}")
        print(f"  Student: {item['student_number']}, Length: {text_length} chars")

        big5_scores = predict_big5(transcript_text)

        print("  Big Five Scores (0-99):")
        for trait, scores in big5_scores.items():
            print(f"    {trait:20s}: {scores['score_100']:5.2f}")

        results.append({
            'uuid': item['uuid'],
            'student_number': item['student_number'],
            'recorded_at': item['recorded_at'],
            'duration_sec': item.get('duration_sec'),
            'text_length': text_length,
            'big5_scores': big5_scores
        })

    # 結果保存
    output_file = 'bigfive_predictions_simple.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'prediction_date': datetime.now().isoformat(),
                'model_file': MODEL_FILE,
                'checkpoint_epoch': checkpoint['epoch'],
                'checkpoint_val_rmse': checkpoint['best_val_rmse'],
                'total_transcripts': len(results)
            },
            'predictions': results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Predictions saved to: {output_file}")
    print("=" * 80)

    # 統計
    import numpy as np

    print("\nStatistics Summary:")
    print("-" * 80)

    for trait in BIG5_LABELS:
        scores = [r['big5_scores'][trait]['score_100'] for r in results]

        print(f"\n{trait}:")
        print(f"  Mean:   {np.mean(scores):5.2f}")
        print(f"  Std:    {np.std(scores):5.2f}")
        print(f"  Min:    {np.min(scores):5.2f}")
        print(f"  Max:    {np.max(scores):5.2f}")
        print(f"  Median: {np.median(scores):5.2f}")

    print("\n" + "=" * 80)
    print("Prediction completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
