#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Big Five Personality Prediction
Stage 1で学習したContrastive Learningモデルを基盤に
RealPersonaChatのBig Fiveラベル付きデータでファインチューニング
"""

import sys
import io
# Set UTF-8 encoding for stdout/stderr
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class BigFivePersonaDataset(Dataset):
    """
    Big Five予測用データセット
    Fatima0923/Automated-Personality-Predictionデータ（Big Fiveラベル付き）
    """

    def __init__(self, dataset, tokenizer, max_length=512, min_text_length=10):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_text_length = min_text_length

        # Big Five特性（データセットのフィールド名に合わせる）
        self.trait_columns = [
            'openness', 'conscientiousness', 'extraversion',
            'agreeableness', 'neuroticism'
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # テキスト取得（'text'フィールドを使用）
        text_content = example.get('text', '')
        if not text_content or len(text_content.strip()) < self.min_text_length:
            text_content = "No personality information available."

        # トークン化
        encoding = self.tokenizer(
            text_content,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Big Five スコア取得 (0-1に正規化)
        # データセットは0-99スケールなので、デフォルト値は50.0
        labels = torch.tensor([
            float(example.get(trait, 50.0)) / 99.0  # 0-99 → 0-1
            for trait in self.trait_columns
        ], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


class BigFiveRegressionModel(nn.Module):
    """
    Big Five予測モデル
    Stage 1のContrastive Learningエンコーダーを基盤に使用
    """

    def __init__(self, base_model, num_traits=5):
        super().__init__()
        self.encoder = base_model
        self.hidden_dim = base_model.config.hidden_size

        # Big Five回帰ヘッド
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
        # エンコーダーで表現抽出
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]トークン

        # Big Five スコア予測
        predictions = self.regressor(cls_embedding)
        return torch.sigmoid(predictions)  # 0-1の範囲に正規化


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """1エポックのトレーニング"""
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 予測
        predictions = model(input_ids, attention_mask)

        # 損失計算
        loss = criterion(predictions, labels)

        # バックプロパゲーション
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """検証"""
    model.eval()
    all_predictions = []
    all_labels = []
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # メトリクス計算
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)

    return {
        'loss': total_loss / len(dataloader),
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Big Five Personality Prediction')

    # モデル設定
    parser.add_argument('--stage1-model-path', type=str, required=True,
                        help='Stage 1で学習したモデルのパス')
    parser.add_argument('--model-name', type=str, default='xlm-roberta-large',
                        help='ベースモデル名')

    # データ設定
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/train'),
                        help='Training data path')
    parser.add_argument('--val-data', type=str, default=os.environ.get('SM_CHANNEL_VAL', 'data/val'),
                        help='Validation data path')

    # トレーニング設定
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (Stage 2は小さめ)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')

    # LoRA設定
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout')

    # 出力設定
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'output'),
                        help='Output directory')

    args = parser.parse_args()

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("=" * 80)
    print("Stage 2: Big Five Personality Prediction Fine-tuning")
    print("=" * 80)
    print(f"Base Model: {args.model_name}")
    print(f"Stage 1 Model: {args.stage1_model_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {device}")
    print("=" * 80)

    # トークナイザーロード
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # データロード
    print("\n[2/6] Loading datasets...")
    # Automated Personality Predictionデータロード（Hugging Faceから直接）
    print("Loading Automated-Personality-Prediction dataset from Hugging Face...")
    print("Dataset: Fatima0923/Automated-Personality-Prediction")
    print("Source: Reddit comments with Big Five labels (0-99 scale)")

    dataset = load_dataset("Fatima0923/Automated-Personality-Prediction", split='train')

    # データセット構造を検証
    print("\n[Validation] Checking dataset structure...")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    required_fields = ['text', 'openness', 'conscientiousness', 'extraversion',
                       'agreeableness', 'neuroticism']
    missing_fields = [f for f in required_fields if f not in sample]

    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")

    print(f"✓ All required fields present")
    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Sample text length: {len(sample['text'])} chars")

    # Train/Val分割
    print("\nSplitting dataset into train/validation (80/20)...")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_raw = dataset['train']
    val_raw = dataset['test']

    print(f"Train samples: {len(train_raw)}")
    print(f"Val samples: {len(val_raw)}")

    # データセット作成
    train_dataset = BigFivePersonaDataset(train_raw, tokenizer, args.max_length)
    val_dataset = BigFivePersonaDataset(val_raw, tokenizer, args.max_length)

    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Stage 1モデルをロード
    print("\n[3/6] Loading Stage 1 model...")

    if os.path.exists(args.stage1_model_path):
        # Check if it's a PEFT adapter or full model
        adapter_config_path = os.path.join(args.stage1_model_path, 'adapter_config.json')

        if os.path.exists(adapter_config_path):
            # Case 1: LoRA adapter format (expected in future)
            print(f"✓ Loading Stage 1 LoRA adapter from {args.stage1_model_path}")
            base_model = AutoModel.from_pretrained(args.model_name)
            base_model = PeftModel.from_pretrained(base_model, args.stage1_model_path)
            base_model = base_model.merge_and_unload()
            print("✓ LoRA weights merged successfully")
        else:
            # Case 2: Full model format (current Stage 1 output)
            print(f"✓ Loading Stage 1 full model from {args.stage1_model_path}")
            try:
                base_model = AutoModel.from_pretrained(args.stage1_model_path)
                print("✓ Stage 1 model loaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Failed to load Stage 1 model: {e}")
                print("→ Falling back to base model")
                base_model = AutoModel.from_pretrained(args.model_name)
    else:
        print(f"⚠ Warning: Stage 1 model not found at {args.stage1_model_path}")
        print("→ Using base XLM-RoBERTa without Stage 1 weights")
        base_model = AutoModel.from_pretrained(args.model_name)

    # Big Five予測モデル作成
    print("\n[4/6] Creating Big Five regression model...")
    model = BigFiveRegressionModel(base_model, num_traits=5)
    model = model.to(device)

    # LoRA再適用（Stage 2用）
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['query', 'key', 'value', 'dense']
    )
    model.encoder = get_peft_model(model.encoder, peft_config)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # オプティマイザーとスケジューラー
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # トレーニングループ
    print("\n[5/6] Starting training...")
    best_val_rmse = float('inf')
    results = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # トレーニング
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        # 検証
        val_metrics = validate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")

        # ベストモデル保存
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            print(f"New best RMSE: {best_val_rmse:.4f} - Saving model...")

            # モデル保存
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # LoRA weights保存
            model.encoder.save_pretrained(output_path / 'lora_weights')
            tokenizer.save_pretrained(output_path / 'lora_weights')

            # Full model保存
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_val_rmse': float(best_val_rmse),
                'epoch': epoch + 1,
            }, output_path / 'best_model.pt')

        results.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_metrics['loss']),
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
        })

    # 結果保存
    print("\n[6/6] Saving results...")
    output_path = Path(args.output_dir)
    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("[SUCCESS] Stage 2 Training Complete!")
    print("=" * 80)
    print(f"Best Validation RMSE: {best_val_rmse:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
