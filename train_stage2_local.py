# -*- coding: utf-8 -*-
"""
Stage 2: RealPersonaChatでBig Five推定学習
Stage 1の重みをロードしてファインチューニング
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=" * 60)
print("Stage 2: RealPersonaChat Big Five学習")
print("=" * 60)

# データ読み込み
print("\nRealPersonaChatデータ読み込み中...")
train_data = load_from_disk("data/local/realpersonachat_split/train")
val_data = load_from_disk("data/local/realpersonachat_split/val")

print(f"Train: {len(train_data)}件")
print(f"Val: {len(val_data)}件")

# サンプル確認
sample = train_data[0]
print(f"\nサンプル確認:")
print(f"  text: {sample['text'][:100]}...")
print(f"  openness: {sample['openness']:.2f}")
print(f"  conscientiousness: {sample['conscientiousness']:.2f}")
print(f"  extraversion: {sample['extraversion']:.2f}")
print(f"  agreeableness: {sample['agreeableness']:.2f}")
print(f"  neuroticism: {sample['neuroticism']:.2f}")


# Big Five回帰モデル
class BigFiveRegressionModel(nn.Module):
    """
    Big Five性格特性推定モデル
    Stage 1で学習したエンコーダを使用
    """
    def __init__(self, base_model_name, lora_config, load_stage1=False, stage1_path=None):
        super().__init__()

        # ベースモデル + LoRA
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.base_model = get_peft_model(self.base_model, lora_config)

        # Stage 1の重みをロード（オプション）
        if load_stage1 and stage1_path:
            print(f"\nStage 1の重みをロード: {stage1_path}")
            try:
                # Stage 1のPEFTモデルから重みをロード
                self.base_model = PeftModel.from_pretrained(
                    AutoModel.from_pretrained(base_model_name),
                    stage1_path,
                    is_trainable=True
                )
                print("OK: Stage 1の重みロード成功")
            except Exception as e:
                print(f"警告: Stage 1の重みロードに失敗: {e}")
                print("新規モデルで学習を開始します")

        hidden_size = self.base_model.config.hidden_size

        # Big Five回帰ヘッド
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

        # [CLS]トークンの出力
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Big Five予測
        logits = self.regressor(pooled_output)

        # 損失計算
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }


# Dataset作成
class BigFiveDataset(torch.utils.data.Dataset):
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


# 評価関数
def compute_metrics(eval_pred):
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

# Stage 1の重みをロードするか確認
stage1_model_path = "output/stage1_nemotron/final_model"
load_stage1 = os.path.exists(stage1_model_path)

if load_stage1:
    print(f"\nStage 1のモデルが見つかりました: {stage1_model_path}")
    print("Stage 1の重みをロードしてファインチューニングします")
else:
    print("\nStage 1のモデルが見つかりません")
    print("新規モデルで学習を開始します")

model = BigFiveRegressionModel(
    base_model_name="xlm-roberta-base",
    lora_config=lora_config,
    load_stage1=load_stage1,
    stage1_path=stage1_model_path if load_stage1 else None
)

print("OK: モデル初期化完了")

# Dataset作成
print("\nDataset作成中...")
train_dataset = BigFiveDataset(train_data, tokenizer, max_length=256)
val_dataset = BigFiveDataset(val_data, tokenizer, max_length=256)

print(f"OK: Train Dataset {len(train_dataset)}件")
print(f"OK: Val Dataset {len(val_dataset)}件")

# トレーニング設定
print("\nトレーニング設定...")
training_args = TrainingArguments(
    output_dir="output/stage2_realpersonachat",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-4,  # Stage 1より低め
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='mae',
    greater_is_better=False,
    report_to='none'
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 学習実行
print("\n" + "=" * 60)
print("Stage 2 学習開始")
print("=" * 60)
print("タスク: Big Five性格特性推定（5次元回帰）")
print("データ: RealPersonaChat")
print("エポック: 5")
if load_stage1:
    print("転移学習: Stage 1の重みを使用")
print("=" * 60 + "\n")

trainer.train()

# 評価
print("\n" + "=" * 60)
print("最終評価")
print("=" * 60)
eval_results = trainer.evaluate()

print("\n評価結果:")
print(f"  MSE: {eval_results['eval_mse']:.4f}")
print(f"  MAE: {eval_results['eval_mae']:.4f}")
print("\n各Big Five次元のMAE:")
trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
for trait in trait_names:
    mae = eval_results.get(f'eval_{trait}_mae', 0)
    print(f"  {trait}: {mae:.4f}")

# モデル保存
print("\n" + "=" * 60)
print("Stage 2 学習完了！")
print("=" * 60)

save_path = "output/stage2_realpersonachat/final_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nモデル保存: {save_path}")
print("\n二段階学習が完了しました！")
print("Stage 1 (Nemotron補助タスク) → Stage 2 (Big Five推定)")
