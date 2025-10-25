#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【1回目検証】テスト学習モデルのダウンロードと検証
- S3からモデルをダウンロード
- 重みが学習済みか検証（初期値でないこと）
- 3つの異なるサンプルテキストで推論テスト
- 異なるBig Five値が出力されることを確認
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
import os
import tarfile
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch.nn as nn

# 設定
REGION = 'ap-northeast-1'
BUCKET_NAME = 'bigfive-personality-sagemaker-1761305156'
JOB_NAME = 'pytorch-training-2025-10-25-07-49-36-999'
DOWNLOAD_DIR = 'models/stage2_test'
MODEL_TAR_FILE = f"{DOWNLOAD_DIR}/model.tar.gz"

print("=" * 80)
print("【1回目検証】テスト学習モデルのダウンロードと検証")
print("=" * 80)

# S3クライアント作成
s3 = boto3.client('s3', region_name=REGION)
sagemaker = boto3.client('sagemaker', region_name=REGION)

# STEP 1: S3パスを取得
print("\n[STEP 1/5] S3モデルパスを取得中...")
try:
    response = sagemaker.describe_training_job(TrainingJobName=JOB_NAME)
    s3_model_path = response['ModelArtifacts']['S3ModelArtifacts']
    print(f"✅ S3パス: {s3_model_path}")

    # S3パスをバケット名とキーに分割
    s3_parts = s3_model_path.replace('s3://', '').split('/', 1)
    bucket = s3_parts[0]
    key = s3_parts[1]

except Exception as e:
    print(f"❌ エラー: {e}")
    sys.exit(1)

# STEP 2: S3からモデルをダウンロード
print("\n[STEP 2/5] S3からモデルをダウンロード中...")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

try:
    s3.download_file(bucket, key, MODEL_TAR_FILE)
    print(f"✅ ダウンロード完了: {MODEL_TAR_FILE}")

    # tar.gzを解凍
    with tarfile.open(MODEL_TAR_FILE, 'r:gz') as tar:
        tar.extractall(path=DOWNLOAD_DIR)
    print(f"✅ 解凍完了: {DOWNLOAD_DIR}")

except Exception as e:
    print(f"❌ エラー: {e}")
    sys.exit(1)

# STEP 3: 重みを検証（初期値でないこと）
print("\n[STEP 3/5] モデルの重みを検証中...")

MODEL_FILE = f"{DOWNLOAD_DIR}/best_model.pt"

try:
    # チェックポイントをロード
    checkpoint = torch.load(MODEL_FILE, map_location='cpu')

    # regressor層の重みを確認
    regressor_weight = checkpoint['model_state_dict']['regressor.0.weight'].numpy()
    regressor_weight_mean = regressor_weight.mean()
    regressor_weight_std = regressor_weight.std()

    print(f"Regressor weight mean: {regressor_weight_mean:.6f}")
    print(f"Regressor weight std: {regressor_weight_std:.6f}")

    # 初期値チェック（stdが0.018程度なら初期値の可能性）
    if regressor_weight_std < 0.02:
        print("\n⚠️  警告: 重みのstdが小さすぎます（初期値の可能性）")
        print("   初期値の典型的なstd: ~0.018")
        print("   学習済みなら通常stdは変化しているはず")
    else:
        print(f"\n✅ 重みは学習済みと思われます（std: {regressor_weight_std:.6f}）")

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 4: モデルを読み込んで推論テスト
print("\n[STEP 4/5] 3つの異なるサンプルテキストで推論テスト...")

# モデル定義
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

try:
    # トークナイザーとモデルをロード
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    base_model = AutoModel.from_pretrained("xlm-roberta-large")

    model = BigFiveRegressionModel(base_model, num_traits=5)

    # LoRAを適用
    lora_config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['query', 'key', 'value', 'dense']
    )
    model.encoder = get_peft_model(model.encoder, lora_config)

    # チェックポイントをロード
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✅ モデル読み込み成功")

    # テストテキスト
    test_texts = [
        "営業経験10年。積極的に顧客と関わり、新しい提案を考えるのが好きです。",
        "静かな環境で計画的に仕事をするのが好きです。ルーチンワークが得意です。",
        "新しいことに挑戦するのが好きで、創造的なアイデアを出すことに興奮します。"
    ]

    BIG5_LABELS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

    predictions_list = []

    print("\n推論結果:")
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
            predictions = predictions.cpu().numpy()[0]

        predictions_list.append(predictions)

        print(f"\nテスト {i}: {text[:40]}...")
        for j, label in enumerate(BIG5_LABELS):
            score = predictions[j] * 99
            print(f"  {label:20s}: {score:5.2f}")

except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 5: 予測の多様性を検証
print("\n" + "=" * 80)
print("[STEP 5/5] 予測の多様性を検証中...")
print("=" * 80)

all_stds = []

for i in range(5):
    trait = BIG5_LABELS[i]
    scores = [pred[i] * 99 for pred in predictions_list]
    std = np.std(scores)
    range_val = max(scores) - min(scores)
    all_stds.append(std)

    print(f"\n{trait}:")
    print(f"  スコア: {[f'{s:.2f}' for s in scores]}")
    print(f"  標準偏差: {std:.2f}")
    print(f"  範囲: {range_val:.2f}")

avg_std = np.mean(all_stds)

print("\n" + "=" * 80)
print("最終判定")
print("=" * 80)

if avg_std < 0.1:
    print(f"\n❌ 失敗: 予測が似すぎています（平均std: {avg_std:.2f}）")
    print("   モデルは入力テキストを使用していない可能性があります")
    print("   前回と同じ問題が再発しています")
    print("\n対処方法:")
    print("   1. train_bigfive.pyのチェックポイント保存ロジックを確認")
    print("   2. Stage 1ロード処理を確認")
    print("   3. ローカルテストと同じコードパターンを使用")
    sys.exit(1)
else:
    print(f"\n✅ 成功: 予測は多様です（平均std: {avg_std:.2f}）")
    print("   モデルは正しく学習され、入力テキストを使用しています")
    print("\n次のステップ:")
    print("   1. 【2回目】5エポックのフル学習を実行")
    print("   2. フル学習モデルで同様の検証を実行")
    print("   3. 検証成功後、DynamoDBの28件で本番推論")
    print("=" * 80)
