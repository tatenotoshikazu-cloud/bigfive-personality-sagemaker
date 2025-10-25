#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B 小規模テスト（1エポック、Stage 1 + Stage 2）
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime

# 設定
REGION = 'ap-northeast-1'
BUCKET = 'bigfive-personality-sagemaker-1761305156'
ROLE = 'arn:aws:iam::531390799864:role/service-role/AmazonSageMaker-ExecutionRole-20250522T011848'
GITHUB_REPO = 'https://github.com/tatenotoshikazu-cloud/bigfive-personality-sagemaker.git'

# 既存のStage 1モデル（以前学習済み）
STAGE1_MODEL_S3 = 's3://bigfive-personality-sagemaker-1761305156/output/stage1_contrastive/pytorch-training-2025-10-24-12-26-47-276/output/model.tar.gz'

print("=" * 80)
print("Model B 小規模テスト - 2段階ファインチューニング（1エポック）")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET}")
print(f"Stage 1 Model: {STAGE1_MODEL_S3}")
print("=" * 80)
print()

# Estimator作成
print("[1/2] Creating PyTorch Estimator for small test (1 epoch)...")
estimator = PyTorch(
    entry_point='train_bigfive.py',
    source_dir='./',
    role=ROLE,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    framework_version='2.0.0',
    py_version='py310',
    git_config={
        'repo': GITHUB_REPO,
        'branch': 'main'
    },
    hyperparameters={
        'model-name': 'xlm-roberta-large',
        'stage1-model-path': '/opt/ml/input/data/stage1_model',  # Stage 1使用
        'epochs': 1,  # 1エポックのみ
        'batch-size': 16,
        'learning-rate': 1e-4,
        'max-length': 512,
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9.]+)'},
        {'Name': 'val:rmse', 'Regex': 'val_rmse: ([0-9.]+)'},
        {'Name': 'val:mae', 'Regex': 'val_mae: ([0-9.]+)'},
    ],
    keep_alive_period_in_seconds=1800,
    output_path=f's3://{BUCKET}/output/model_b_test/',
    code_location=f's3://{BUCKET}/code',
)
print("[OK] Estimator created")
print()

# 学習開始
print("[2/2] Starting Model B small test (1 epoch with Stage 1)...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    estimator.fit({
        'stage1_model': STAGE1_MODEL_S3,  # Stage 1モデルを渡す
    })

    print()
    print("=" * 80)
    print("SUCCESS: Model B small test completed!")
    print("=" * 80)
    print(f"Model saved to: {estimator.model_data}")
    print()
    print("Next: Download and verify the model, then test with DynamoDB data")
    print("=" * 80)

except Exception as e:
    print()
    print("=" * 80)
    print("ERROR: Model B small test failed")
    print("=" * 80)
    print(f"Error: {e}")
    print()
    print("Check CloudWatch Logs for details")
    print("=" * 80)
    sys.exit(1)
