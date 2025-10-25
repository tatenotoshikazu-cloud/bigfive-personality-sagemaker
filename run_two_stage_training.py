"""
【モデルB】2段階ファインチューニング
Stage 1 (Contrastive Learning) + Stage 2 (Big Five Regression)
"""
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime
import time

# 設定
BUCKET_NAME = 'bigfive-personality-sagemaker-1761305156'
ROLE_ARN = 'arn:aws:iam::531390799864:role/service-role/AmazonSageMaker-ExecutionRole-20250522T011848'
REGION = 'ap-northeast-1'
GITHUB_REPO = 'https://github.com/tatenotoshikazu-cloud/bigfive-personality-sagemaker.git'
GITHUB_BRANCH = 'main'

# Stage 1は既に完了しているのでスキップ
STAGE1_MODEL_S3 = 's3://bigfive-personality-sagemaker-1761305156/output/stage1_contrastive/pytorch-training-2025-10-24-12-26-47-276/output/model.tar.gz'

print("=" * 80)
print("【モデルB】2段階ファインチューニング")
print("Stage 1: Contrastive Learning (既存モデル使用)")
print("Stage 2: Big Five Regression (これから実行)")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET_NAME}")
print(f"Stage 1 Model: {STAGE1_MODEL_S3}")
print("=" * 80)

# SageMakerセッション作成
session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# Stage 2: Big Five Regression
print("\n[1/2] Creating PyTorch Estimator for Stage 2 (with Stage 1)...")
estimator_stage2 = PyTorch(
    entry_point='train_bigfive.py',
    source_dir='.',
    git_config={
        'repo': GITHUB_REPO,
        'branch': GITHUB_BRANCH,
    },
    role=ROLE_ARN,
    instance_type='ml.g5.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'model-name': 'xlm-roberta-large',
        'stage1-model-path': '/opt/ml/input/data/stage1_model',  # Stage 1を使用
        'epochs': 5,
        'batch-size': 16,
        'learning-rate': 1e-4,
        'max-length': 512,
        'lora-r': 8,
        'lora-alpha': 16,
        'lora-dropout': 0.1,
    },
    output_path=f's3://{BUCKET_NAME}/output/stage2_two_stage',
    sagemaker_session=session,
    keep_alive_period_in_seconds=1800,
    use_spot_instances=False,
    max_run=7200,
)

print("[OK] Estimator created")

# トレーニング実行
print("\n[2/2] Starting Stage 2 with Stage 1 model...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNote: Training will take 45-60 minutes")
print("Progress can be monitored at AWS SageMaker Console:")
print(f"https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
print()

try:
    # Stage 1モデルを入力として渡す
    estimator_stage2.fit({
        'stage1_model': STAGE1_MODEL_S3,
    })

    print("\n" + "=" * 80)
    print("【モデルB】2段階ファインチューニング Complete!")
    print("=" * 80)
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved at: {estimator_stage2.model_data}")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("[ERROR] Training failed")
    print("=" * 80)
    print(f"Error: {e}")
    raise
