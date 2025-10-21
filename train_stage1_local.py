# -*- coding: utf-8 -*-
"""
Stage 1: Nemotronデータで補助タスク学習
ペルソナテキストから年齢・性別・職業を予測
"""
import os
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

print("=" * 60)
print("Stage 1: Nemotron 補助タスク学習")
print("=" * 60)

# データ読み込み
print("\nNemotronデータ読み込み中...")
train_data = load_from_disk("data/local/processed/train")
val_data = load_from_disk("data/local/processed/val")

print(f"Train: {len(train_data)}件")
print(f"Val: {len(val_data)}件")

# ラベル情報確認
sample = train_data[0]
print(f"\nサンプル確認:")
print(f"  age: {sample.get('age')}")
print(f"  sex: {sample.get('sex')}")
print(f"  occupation: {sample.get('occupation')}")

# 性別をラベルエンコーディング
print("\n性別ラベル作成中...")
sex_labels = list(set([s['sex'] for s in train_data]))
sex_to_id = {sex: i for i, sex in enumerate(sex_labels)}
print(f"性別カテゴリ: {sex_labels}")

# 職業をラベルエンコーディング
print("\n職業ラベル作成中...")
all_occupations = list(set([s['occupation'] for s in train_data] + [s['occupation'] for s in val_data]))
occupation_to_id = {occ: i for i, occ in enumerate(all_occupations)}
print(f"職業カテゴリ数: {len(all_occupations)}")


# Multi-task モデル定義
class MultiTaskPersonaModel(nn.Module):
    """
    ペルソナテキストから年齢・性別・職業を予測するマルチタスクモデル
    """
    def __init__(self, base_model_name, lora_config, num_sex_classes, num_occupation_classes):
        super().__init__()

        # ベースモデル + LoRA
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.base_model = get_peft_model(self.base_model, lora_config)

        hidden_size = self.base_model.config.hidden_size

        # タスク別ヘッド
        # 1. 年齢予測（回帰）
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 2. 性別予測（分類）
        self.sex_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_sex_classes)
        )

        # 3. 職業予測（分類）
        self.occupation_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_occupation_classes)
        )

    def forward(self, input_ids, attention_mask, age_labels=None, sex_labels=None, occupation_labels=None):
        # エンコーダ出力
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # [CLS]トークンの出力
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # 各タスクの予測
        age_pred = self.age_head(pooled_output).squeeze(-1)
        sex_logits = self.sex_head(pooled_output)
        occupation_logits = self.occupation_head(pooled_output)

        # 損失計算
        loss = None
        if age_labels is not None and sex_labels is not None and occupation_labels is not None:
            # 年齢: MSE Loss
            age_loss = nn.MSELoss()(age_pred, age_labels)

            # 性別: Cross Entropy Loss
            sex_loss = nn.CrossEntropyLoss()(sex_logits, sex_labels)

            # 職業: Cross Entropy Loss
            occupation_loss = nn.CrossEntropyLoss()(occupation_logits, occupation_labels)

            # 合計損失（重み付け平均）
            loss = age_loss + sex_loss + occupation_loss

        return {
            'loss': loss,
            'age_pred': age_pred,
            'sex_logits': sex_logits,
            'occupation_logits': occupation_logits
        }


# Dataset作成
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, sex_to_id, occupation_to_id, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sex_to_id = sex_to_id
        self.occupation_to_id = occupation_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # ペルソナテキスト取得
        text = item.get('persona', '')
        if not text:
            # personaがない場合は他のフィールドを使用
            text = f"{item.get('professional_persona', '')} {item.get('persona', '')}"

        # トークナイズ
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # ラベル
        age = float(item['age']) if isinstance(item['age'], (int, float)) else 40.0  # デフォルト値
        sex_id = self.sex_to_id.get(item['sex'], 0)
        occupation_id = self.occupation_to_id.get(item['occupation'], 0)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'age_labels': torch.tensor(age, dtype=torch.float32),
            'sex_labels': torch.tensor(sex_id, dtype=torch.long),
            'occupation_labels': torch.tensor(occupation_id, dtype=torch.long)
        }


# 評価関数
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # predictions: (age_pred, sex_logits, occupation_logits)
    # labels: (age_labels, sex_labels, occupation_labels)

    # 簡易的な評価
    return {
        'eval_loss': 0.0  # Trainerが自動計算
    }


print("\nモデル初期化中...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

model = MultiTaskPersonaModel(
    base_model_name="xlm-roberta-base",
    lora_config=lora_config,
    num_sex_classes=len(sex_labels),
    num_occupation_classes=len(all_occupations)
)

print(f"OK: モデル初期化完了")
print(f"  性別クラス数: {len(sex_labels)}")
print(f"  職業クラス数: {len(all_occupations)}")

# Dataset作成
print("\nDataset作成中...")
train_dataset = MultiTaskDataset(train_data, tokenizer, sex_to_id, occupation_to_id)
val_dataset = MultiTaskDataset(val_data, tokenizer, sex_to_id, occupation_to_id)

print(f"OK: Train Dataset {len(train_dataset)}件")
print(f"OK: Val Dataset {len(val_dataset)}件")

# トレーニング設定
print("\nトレーニング設定...")
training_args = TrainingArguments(
    output_dir="output/stage1_nemotron",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to='none'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 学習実行
print("\n" + "=" * 60)
print("Stage 1 学習開始")
print("=" * 60)
print("タスク: 年齢予測 + 性別分類 + 職業分類")
print("データ: Nemotron-Personas-Japan")
print("エポック: 3")
print("=" * 60 + "\n")

trainer.train()

# モデル保存
print("\n" + "=" * 60)
print("Stage 1 学習完了！")
print("=" * 60)

save_path = "output/stage1_nemotron/final_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nモデル保存: {save_path}")
print("\n次のステップ:")
print("  Stage 2でこのモデルの重みをロードして")
print("  RealPersonaChatでBig Five推定にファインチューニング")
