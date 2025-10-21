"""
AWS SageMaker対応トレーニングスクリプト
xlm-roberta-large + LoRA による Big Five性格特性推定モデル学習
"""
import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class BigFiveRegressionModel(nn.Module):
    """Big Five回帰モデル（xlm-roberta-large + LoRA + 回帰ヘッド）"""

    def __init__(self, base_model_name: str, lora_config: LoraConfig):
        super().__init__()

        # ベースモデル読み込み
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # LoRA適用
        self.base_model = get_peft_model(self.base_model, lora_config)

        # 回帰ヘッド（Big Five 5次元出力）
        hidden_size = self.base_model.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)  # Big Five: 5次元
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # エンコーダ出力
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # [CLS]トークンの出力を使用
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # 回帰予測
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            # MSE損失
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }


class BigFiveDataset(torch.utils.data.Dataset):
    """Big Five用データセット"""

    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # テキストのトークナイズ
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Big Fiveラベル
        labels = torch.tensor([
            item['openness'],
            item['conscientiousness'],
            item['extraversion'],
            item['agreeableness'],
            item['neuroticism']
        ], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }


def compute_metrics(eval_pred):
    """評価メトリクス計算"""
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)

    # MSE
    mse = mean_squared_error(labels, predictions)

    # MAE
    mae = mean_absolute_error(labels, predictions)

    # 各Big Five次元ごとのMAE
    trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    trait_maes = {}
    for i, trait in enumerate(trait_names):
        trait_mae = mean_absolute_error(labels[:, i], predictions[:, i])
        trait_maes[f'{trait}_mae'] = trait_mae

    return {
        'mse': mse,
        'mae': mae,
        **trait_maes
    }


def train_model(args):
    """モデル学習メイン処理"""
    print("=" * 60)
    print("Big Five性格特性推定モデル学習開始")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"Base Model: {args.model_name}")
    print(f"Epoch: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 60)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # トークナイザー読み込み
    print("\nトークナイザー読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # データセット読み込み
    print(f"\nデータセット読み込み中... (Stage {args.stage})")
    train_data = load_from_disk(os.path.join(args.data_dir, f'stage{args.stage}_train'))
    val_data = load_from_disk(os.path.join(args.data_dir, f'stage{args.stage}_val'))

    print(f"Train: {len(train_data)}件, Validation: {len(val_data)}件")

    # Dataset作成
    train_dataset = BigFiveDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = BigFiveDataset(val_data, tokenizer, max_length=args.max_length)

    # LoRA設定
    print("\nLoRA設定...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],  # xlm-robertaのAttentionレイヤー
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # モデル構築
    print("\nモデル構築中...")
    if args.stage == 2 and args.stage1_model_path:
        # Stage 2: Stage 1の重みをロード
        print(f"Stage 1モデルをロード: {args.stage1_model_path}")
        model = BigFiveRegressionModel(args.model_name, lora_config)
        # Stage1の重みをロード（実装例）
        # model.load_state_dict(torch.load(args.stage1_model_path))
    else:
        # Stage 1: 新規モデル
        model = BigFiveRegressionModel(args.model_name, lora_config)

    model.to(device)

    # 学習可能パラメータ数確認
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n学習可能パラメータ: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='mae',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to='tensorboard'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 学習開始
    print("\n" + "=" * 60)
    print("学習開始")
    print("=" * 60)
    trainer.train()

    # 評価
    print("\n" + "=" * 60)
    print("評価")
    print("=" * 60)
    eval_results = trainer.evaluate()
    print(json.dumps(eval_results, indent=2))

    # モデル保存
    print("\nモデル保存中...")
    model.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))

    print("\n✓ 学習完了！")
    print(f"モデル保存先: {os.path.join(args.output_dir, 'final_model')}")


def parse_args():
    """コマンドライン引数パース（SageMaker対応）"""
    parser = argparse.ArgumentParser()

    # モデル設定
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large')
    parser.add_argument('--max_length', type=int, default=512)

    # LoRA設定
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # 学習設定
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # パス設定（SageMaker用）
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/processed'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'output'))
    parser.add_argument('--stage1_model_path', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
