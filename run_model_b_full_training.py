#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B フル学習 - 2段階ファインチューニング（5エポック）
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime

# Settings
REGION = 'ap-northeast-1'
ROLE = 'arn:aws:iam::590183888241:role/SageMakerFullAccessRole'
BUCKET = 'bigfive-personality-sagemaker-1761305156'

# Use existing Stage 1 model
STAGE1_MODEL_S3 = 's3://bigfive-personality-sagemaker-1761305156/output/stage1_contrastive/pytorch-training-2025-10-24-12-26-47-276/output/model.tar.gz'

print("=" * 80)
print("Model B フル学習 - 2段階ファインチューニング（5エポック）")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET}")
print(f"Stage 1 Model: {STAGE1_MODEL_S3}")
print("=" * 80)

# Create PyTorch estimator
print("\n[1/2] Creating PyTorch Estimator for full training (5 epochs)...")
estimator = PyTorch(
    entry_point='train_bigfive.py',
    source_dir='./',
    role=ROLE,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'model-name': 'xlm-roberta-large',
        'stage1-model-path': '/opt/ml/input/data/stage1_model',  # Use Stage 1 model
        'epochs': 5,  # Full 5 epochs
        'batch-size': 16,
        'learning-rate': 1e-4,
        'max-length': 512,
    },
    output_path=f's3://{BUCKET}/output/model_b_full/',
)
print("[OK] Estimator created")

# Start training
print("\n[2/2] Starting Model B full training (5 epochs with Stage 1)...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

estimator.fit({
    'stage1_model': STAGE1_MODEL_S3,  # Pass Stage 1 model as input
})

print("\n" + "=" * 80)
print("SUCCESS: Model B full training completed!")
print("=" * 80)
print(f"Model location: {estimator.model_data}")
print("=" * 80)
