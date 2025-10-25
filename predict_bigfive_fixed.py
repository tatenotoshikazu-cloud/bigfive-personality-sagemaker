#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Big Five Personality Prediction Script (Fixed)
best_model.ptから完全なモデル（LoRA + Regression Head）をロード
"""

import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer

def main():
    print("=" * 80)
    print("Big Five Personality Prediction - Real Transcript Data (Fixed)")
    print("=" * 80)

    # Device設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # モデルパス
    MODEL_DIR = "models/stage2_bigfive"
    MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pt")
    TOKENIZER_DIR = os.path.join(MODEL_DIR, "lora_weights")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    print(f"\nLoading model from: {MODEL_FILE}")
    print(f"Model file size: {os.path.getsize(MODEL_FILE) / 1024 / 1024:.2f} MB")

    # Tokenizer読み込み
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # モデル全体をロード（LoRA + Regression Head）
    checkpoint = torch.load(MODEL_FILE, map_location=device)

    # チェックポイントの内容確認
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")

    # モデルの再構築
    from transformers import AutoModel
    import torch.nn as nn
    from peft import PeftModel

    class BigFiveRegressionModel(nn.Module):
        def __init__(self, base_model_name, lora_weights_dir):
            super().__init__()
            # Base model + LoRA
            base_model = AutoModel.from_pretrained(base_model_name)
            self.encoder = PeftModel.from_pretrained(base_model, lora_weights_dir)
            self.hidden_dim = self.encoder.config.hidden_size

            # Regression head (3-layer: 1024 → 512 → 256 → 5)
            self.regressor = nn.Sequential(
                nn.Linear(self.hidden_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 5)
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            predictions = self.regressor(cls_embedding)
            return torch.sigmoid(predictions)  # 0-1に正規化

    # モデル初期化
    model = BigFiveRegressionModel("xlm-roberta-large", TOKENIZER_DIR)

    # 重みをロード
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\nLoaded model_state_dict from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("\nLoaded checkpoint directly as state_dict")

    model = model.to(device)
    model.eval()
    print("Model loaded successfully with regression head")

    # Big Fiveラベル
    BIG5_LABELS = [
        'Openness',
        'Conscientiousness',
        'Extraversion',
        'Agreeableness',
        'Neuroticism'
    ]

    def predict_big5(text):
        """
        テキストからBig Five予測

        Args:
            text: 入力テキスト

        Returns:
            Big Five scores (0-1スケールと0-99スケール)
        """
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 推論
        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
            predictions = predictions.cpu().numpy()[0]

        # スコア辞書作成
        result = {}
        for i, label in enumerate(BIG5_LABELS):
            score_01 = float(predictions[i])
            score_100 = score_01 * 99  # 0-99スケール

            result[label] = {
                'score_01': round(score_01, 4),
                'score_100': round(score_100, 2)
            }

        return result

    # Transcriptsデータ読み込み
    transcript_file = 'transcripts_data.json'
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Transcript data not found: {transcript_file}")

    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    print(f"\n{len(transcripts)} transcripts loaded from {transcript_file}")

    # 予測実行
    print("\nRunning Big Five predictions...")
    print("-" * 80)

    results = []

    for i, item in enumerate(transcripts, 1):
        transcript_text = item['transcript_text']
        text_length = len(transcript_text)

        print(f"\n[{i}/{len(transcripts)}] Processing: {item['uuid']}")
        print(f"  Student: {item['student_number']}")
        print(f"  Text length: {text_length} chars")

        # 予測実行
        big5_scores = predict_big5(transcript_text)

        # 結果表示
        print("  Big Five Scores (0-99 scale):")
        for trait, scores in big5_scores.items():
            print(f"    {trait:20s}: {scores['score_100']:5.2f}")

        # 結果保存
        results.append({
            'uuid': item['uuid'],
            'student_number': item['student_number'],
            'recorded_at': item['recorded_at'],
            'duration_sec': item.get('duration_sec'),
            'text_length': text_length,
            'big5_scores': big5_scores
        })

    # 結果保存
    output_file = 'bigfive_predictions_fixed.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'prediction_date': datetime.now().isoformat(),
                'model_file': MODEL_FILE,
                'tokenizer': TOKENIZER_DIR,
                'total_transcripts': len(results)
            },
            'predictions': results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Predictions saved to: {output_file}")
    print("=" * 80)

    # 統計情報計算
    print("\nStatistics Summary:")
    print("-" * 80)

    import numpy as np

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
