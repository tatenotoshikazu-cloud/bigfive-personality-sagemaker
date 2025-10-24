"""
Big Five Personality Estimation - SageMaker Training Script
xlm-roberta-large + LoRA (Low-Rank Adaptation)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BigFiveDataset(Dataset):
    """Big Five性格特性データセット"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # トークン化
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


def compute_metrics(eval_pred):
    """評価メトリクス計算"""
    predictions, labels = eval_pred

    # MSE（平均二乗誤差）
    mse = mean_squared_error(labels, predictions)

    # RMSE（二乗平均平方根誤差）
    rmse = np.sqrt(mse)

    # 相関係数
    correlation = np.corrcoef(labels.flatten(), predictions.flatten())[0, 1]

    return {
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation
    }


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser()

    # SageMakerの環境変数
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    # ハイパーパラメータ
    parser.add_argument('--model-name', type=str, default='xlm-roberta-large')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.1)

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Big Five Personality Fine-tuning with LoRA")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"LoRA r: {args.lora_r}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info("=" * 80)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # トークナイザー読み込み
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ベースモデル読み込み（Big Fiveは5つの特性 → num_labels=5）
    logger.info(f"Loading base model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=5,  # Big Five: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
        problem_type="regression"  # 回帰問題
    )

    # LoRA設定
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],  # RoBERTaのattention層
        bias="none"
    )

    # LoRAモデル作成
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)

    # データセット読み込み
    logger.info(f"Loading training data from: {args.train}")
    train_dataset = load_from_disk(args.train)

    logger.info(f"Loading validation data from: {args.val}")
    val_dataset = load_from_disk(args.val)

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f'{args.output_data_dir}/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True if torch.cuda.is_available() else False,  # 混合精度学習
        report_to="tensorboard",
        save_total_limit=2,
    )

    # Trainer作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # トレーニング実行
    logger.info("Starting training...")
    train_result = trainer.train()

    # モデル保存
    logger.info(f"Saving model to: {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # 評価
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()

    # 結果保存
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Train Loss: {train_result.training_loss:.4f}")
    logger.info(f"Eval Loss: {eval_result['eval_loss']:.4f}")
    logger.info(f"Eval MSE: {eval_result.get('eval_mse', 'N/A')}")
    logger.info(f"Eval RMSE: {eval_result.get('eval_rmse', 'N/A')}")
    logger.info(f"Eval Correlation: {eval_result.get('eval_correlation', 'N/A')}")
    logger.info("=" * 80)

    # メトリクス保存
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_loss': float(train_result.training_loss),
            'eval_loss': float(eval_result['eval_loss']),
            'eval_mse': float(eval_result.get('eval_mse', 0)),
            'eval_rmse': float(eval_result.get('eval_rmse', 0)),
            'eval_correlation': float(eval_result.get('eval_correlation', 0)),
        }, f, indent=2)

    logger.info(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
