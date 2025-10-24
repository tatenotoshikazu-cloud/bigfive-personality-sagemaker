#!/usr/bin/env python3
"""
Big Five Personality Prediction - Inference Script
トレーニング済みモデルでテキストからBig Five推定

使い方:
1. チーキューデータ: DynamoDBから取得したペルソナテキスト
2. 文字起こしデータ: 会話ログから抽出したテキスト
3. 任意のテキスト: どんなテキストでもBig Five推定可能
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import pandas as pd


class BigFiveRegressionModel(nn.Module):
    """Big Five予測モデル（train_bigfive.pyと同じ構造）"""

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
        predictions = torch.sigmoid(self.regressor(cls_embedding))
        return predictions


class BigFivePredictor:
    """Big Five性格予測器"""

    def __init__(self, model_path: str, model_name: str = 'xlm-roberta-large', device: str = 'auto'):
        """
        Args:
            model_path: トレーニング済みモデルのパス
            model_name: ベースモデル名
            device: 'cuda', 'cpu', または 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Loading model from: {model_path}")

        # トークナイザーロード
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ベースモデルロード
        base_model = AutoModel.from_pretrained(model_name)

        # LoRA weightロード
        lora_path = Path(model_path) / 'lora_weights'
        if lora_path.exists():
            print(f"Loading LoRA weights from {lora_path}")
            base_model = PeftModel.from_pretrained(base_model, str(lora_path))
            base_model = base_model.merge_and_unload()

        # Big Fiveモデル構築
        self.model = BigFiveRegressionModel(base_model, num_traits=5)

        # モデルステート読み込み
        checkpoint_path = Path(model_path) / 'best_model.pt'
        if checkpoint_path.exists():
            print(f"Loading model checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            print(f"Best validation RMSE: {checkpoint.get('best_val_rmse', 'unknown'):.4f}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Big Five特性名
        self.trait_names = [
            'openness',           # 開放性
            'conscientiousness',  # 誠実性
            'extraversion',       # 外向性
            'agreeableness',      # 協調性
            'neuroticism'         # 神経症傾向
        ]

        print("[OK] Model loaded successfully!")

    def predict_text(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        単一テキストからBig Five推定

        Args:
            text: ペルソナテキスト（チーキューデータ、文字起こしなど）
            max_length: 最大トークン長

        Returns:
            Big Five スコア辞書 (0-100スケール)
        """
        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 推論
        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)

        # 0-1 → 0-100に変換
        scores = predictions.squeeze(0).cpu().numpy() * 100

        return {
            trait: float(score)
            for trait, score in zip(self.trait_names, scores)
        }

    def predict_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 16) -> List[Dict[str, float]]:
        """
        複数テキストを一括でBig Five推定

        Args:
            texts: ペルソナテキストのリスト
            max_length: 最大トークン長
            batch_size: バッチサイズ

        Returns:
            Big Fiveスコアのリスト
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # トークン化
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # 推論
            with torch.no_grad():
                predictions = self.model(input_ids, attention_mask)

            # 0-1 → 0-100に変換
            scores = predictions.cpu().numpy() * 100

            # 結果を辞書形式に変換
            for score in scores:
                results.append({
                    trait: float(s)
                    for trait, s in zip(self.trait_names, score)
                })

        return results

    def predict_from_dynamodb_data(self, dynamodb_record: Dict) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        DynamoDB（チーキューデータ）からBig Five推定

        Args:
            dynamodb_record: DynamoDBレコード
                例: {'user_id': '123', 'persona_text': '...', 'conversation_log': '...'}

        Returns:
            推定結果（user_id + Big Five スコア）
        """
        # ペルソナテキスト取得（複数フィールドを結合）
        persona_fields = ['persona_text', 'conversation_log', 'profile_description', 'bio']
        persona_text = ' '.join([
            str(dynamodb_record.get(field, ''))
            for field in persona_fields
            if field in dynamodb_record and dynamodb_record[field]
        ])

        if not persona_text or len(persona_text.strip()) < 10:
            persona_text = "No persona information available."

        # Big Five推定
        big_five = self.predict_text(persona_text)

        return {
            'user_id': dynamodb_record.get('user_id', 'unknown'),
            'big_five': big_five,
            'input_text_length': len(persona_text)
        }

    def predict_from_transcript(self, transcript_text: str, speaker_id: str = None) -> Dict:
        """
        文字起こしデータからBig Five推定

        Args:
            transcript_text: 文字起こしテキスト
            speaker_id: 話者ID（オプション）

        Returns:
            推定結果
        """
        big_five = self.predict_text(transcript_text)

        return {
            'speaker_id': speaker_id or 'unknown',
            'big_five': big_five,
            'transcript_length': len(transcript_text)
        }


def main():
    parser = argparse.ArgumentParser(description='Big Five Personality Prediction')
    parser.add_argument('--model-path', type=str, required=True,
                        help='トレーニング済みモデルのパス')
    parser.add_argument('--model-name', type=str, default='xlm-roberta-large',
                        help='ベースモデル名')
    parser.add_argument('--input-text', type=str,
                        help='推定するテキスト')
    parser.add_argument('--input-file', type=str,
                        help='推定するテキストファイル（1行1テキスト）')
    parser.add_argument('--input-json', type=str,
                        help='DynamoDBデータJSON（チーキューデータ）')
    parser.add_argument('--output-file', type=str,
                        help='結果を保存するファイル')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])

    args = parser.parse_args()

    # 予測器初期化
    predictor = BigFivePredictor(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device
    )

    results = []

    # 単一テキスト推定
    if args.input_text:
        print("\n" + "=" * 80)
        print("Predicting from text...")
        print("=" * 80)
        result = predictor.predict_text(args.input_text)
        print("\nBig Five Scores:")
        for trait, score in result.items():
            print(f"  {trait:18s}: {score:5.1f}")
        results.append({'text': args.input_text, 'big_five': result})

    # ファイルから一括推定
    elif args.input_file:
        print("\n" + "=" * 80)
        print(f"Predicting from file: {args.input_file}")
        print("=" * 80)
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(texts)} texts...")
        predictions = predictor.predict_batch(texts)

        for text, pred in zip(texts, predictions):
            results.append({'text': text, 'big_five': pred})
            print(f"\nText: {text[:50]}...")
            for trait, score in pred.items():
                print(f"  {trait:18s}: {score:5.1f}")

    # DynamoDBデータから推定
    elif args.input_json:
        print("\n" + "=" * 80)
        print(f"Predicting from DynamoDB data: {args.input_json}")
        print("=" * 80)
        with open(args.input_json, 'r', encoding='utf-8') as f:
            records = json.load(f)

        if not isinstance(records, list):
            records = [records]

        print(f"Processing {len(records)} records...")
        for record in records:
            result = predictor.predict_from_dynamodb_data(record)
            results.append(result)
            print(f"\nUser ID: {result['user_id']}")
            for trait, score in result['big_five'].items():
                print(f"  {trait:18s}: {score:5.1f}")

    # 結果保存
    if args.output_file and results:
        print(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("[OK] Results saved!")


if __name__ == '__main__':
    main()
