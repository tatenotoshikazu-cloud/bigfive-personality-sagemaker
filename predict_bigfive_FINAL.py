#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big Five Personality Prediction - FINAL VERSION
checkpoint全体をロード（Stage 1マージ済み + Stage 2 LoRA + Regressor）
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

def main():
    print("=" * 80)
    print("Big Five Personality Prediction - FINAL VERSION")
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
    print(f"\nTokenizer loaded")

    # モデル構造を学習時と完全に同じに再構築
    print("\nRebuilding model structure (matching training code)...")

    # 1. Base model (これは新規ダウンロードでOK - 後でcheckpointの重みで上書きする)
    base_model = AutoModel.from_pretrained("xlm-roberta-large")

    # 2. BigFiveRegressionModel (学習時と同じ)
    class BigFiveRegressionModel(nn.Module):
        def __init__(self, base_model, num_traits=5):
            super().__init__()
            self.encoder = base_model
            self.hidden_dim = base_model.config.hidden_size

            # Regression head (学習時と完全一致)
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

    # 3. Stage 2用LoRAを再適用（学習時と同じ設定）
    peft_config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['query', 'key', 'value', 'dense']
    )
    model.encoder = get_peft_model(model.encoder, peft_config)

    print("Model structure created")

    # 4. checkpoint重みをロード（全ての学習済み重みを復元）
    print("\nLoading trained weights from checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Trained weights loaded successfully!")

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

    # テスト：異なるテキストで異なる予測が出るか確認
    print("\n" + "=" * 80)
    print("Testing with sample texts (verifying model works):")
    print("=" * 80)

    test_samples = [
        transcripts[0]['transcript_text'][:100],
        transcripts[1]['transcript_text'][:100] if len(transcripts) > 1 else "Test text",
    ]

    for i, text in enumerate(test_samples, 1):
        scores = predict_big5(text)
        print(f"\nTest {i}: {text[:50]}...")
        print(f"  Openness: {scores['Openness']['score_100']:.2f}")
        print(f"  Conscientiousness: {scores['Conscientiousness']['score_100']:.2f}")

    # 実際の予測実行
    print("\n" + "=" * 80)
    print("Running predictions on all 28 transcripts...")
    print("=" * 80)

    results = []

    for i, item in enumerate(transcripts, 1):
        transcript_text = item['transcript_text']
        text_length = len(transcript_text)

        print(f"\n[{i}/{len(transcripts)}] {item['student_number']} ({text_length} chars)")

        big5_scores = predict_big5(transcript_text)

        # 簡潔な表示
        print(f"  O:{big5_scores['Openness']['score_100']:5.1f} C:{big5_scores['Conscientiousness']['score_100']:5.1f} E:{big5_scores['Extraversion']['score_100']:5.1f} A:{big5_scores['Agreeableness']['score_100']:5.1f} N:{big5_scores['Neuroticism']['score_100']:5.1f}")

        results.append({
            'uuid': item['uuid'],
            'student_number': item['student_number'],
            'recorded_at': item['recorded_at'],
            'duration_sec': item.get('duration_sec'),
            'text_length': text_length,
            'big5_scores': big5_scores
        })

    # 結果保存
    output_file = 'bigfive_predictions.json'
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
    print(f"✓ Predictions saved to: {output_file}")
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
        print(f"  Range:  {np.max(scores) - np.min(scores):5.2f}")

    print("\n" + "=" * 80)
    print("✓ Prediction completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
