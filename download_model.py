# -*- coding: utf-8 -*-
"""
Stage 2モデルをS3からダウンロード
"""
import boto3
import os
import tarfile

# S3パス
S3_MODEL_PATH = "s3://bigfive-personality-sagemaker-1761305156/output/stage2_bigfive/pytorch-training-2025-10-25-05-23-17-454/output/model.tar.gz"
LOCAL_MODEL_DIR = "models/stage2_bigfive"
LOCAL_TAR_FILE = "models/stage2_model.tar.gz"

# ディレクトリ作成
os.makedirs("models", exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# S3からダウンロード
s3 = boto3.client('s3', region_name='ap-northeast-1')
bucket = "bigfive-personality-sagemaker-1761305156"
key = "output/stage2_bigfive/pytorch-training-2025-10-25-05-23-17-454/output/model.tar.gz"

print(f"Downloading model from S3...")
print(f"  Bucket: {bucket}")
print(f"  Key: {key}")
print(f"  Local: {LOCAL_TAR_FILE}")

s3.download_file(bucket, key, LOCAL_TAR_FILE)
print(f"✓ Download completed: {os.path.getsize(LOCAL_TAR_FILE) / 1024 / 1024:.2f} MB")

# 解凍
print(f"\nExtracting model files to: {LOCAL_MODEL_DIR}")
with tarfile.open(LOCAL_TAR_FILE, 'r:gz') as tar:
    tar.extractall(LOCAL_MODEL_DIR)

print(f"✓ Extraction completed")

# ファイル確認
print(f"\nExtracted files:")
for root, dirs, files in os.walk(LOCAL_MODEL_DIR):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {filepath} ({size_mb:.2f} MB)")

print(f"\n✓ Model ready at: {LOCAL_MODEL_DIR}")
