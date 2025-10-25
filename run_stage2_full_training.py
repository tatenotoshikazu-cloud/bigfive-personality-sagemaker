"""
【2回目】Big Five Personality Prediction - 5エポックフル学習
Stage 1をスキップして、fresh xlm-roberta-largeから5エポック学習
"""
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime

# 設定
BUCKET_NAME = 'bigfive-personality-sagemaker-1761305156'
ROLE_ARN = 'arn:aws:iam::531390799864:role/service-role/AmazonSageMaker-ExecutionRole-20250522T011848'
REGION = 'ap-northeast-1'
GITHUB_REPO = 'https://github.com/tatenotoshikazu-cloud/bigfive-personality-sagemaker.git'
GITHUB_BRANCH = 'main'

print("=" * 80)
print("【2回目】Big Five Personality Prediction - 5エポックフル学習")
print("Stage 1をスキップして、fresh xlm-roberta-largeから学習")
print("Data: Fatima0923/Automated-Personality-Prediction")
print("Instance: ml.g5.2xlarge (NVIDIA A10G GPU)")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET_NAME}")
print(f"Role: {ROLE_ARN}")
print(f"GitHub Repo: {GITHUB_REPO}")
print("=" * 80)

# SageMakerセッション作成
session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# PyTorch Estimator作成
print("\n[1/3] Creating PyTorch Estimator for Full Training (5 Epochs)...")
estimator = PyTorch(
    entry_point='train_bigfive.py',
    source_dir='.',
    git_config={
        'repo': GITHUB_REPO,
        'branch': GITHUB_BRANCH,
    },
    role=ROLE_ARN,
    instance_type='ml.g5.2xlarge',  # GPU: NVIDIA A10G (24GB VRAM)
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'model-name': 'xlm-roberta-large',
        'stage1-model-path': 'NONE',  # Skip Stage 1
        'epochs': 5,            # Full training: 5 epochs
        'batch-size': 16,
        'learning-rate': 1e-4,
        'max-length': 512,
        'lora-r': 8,
        'lora-alpha': 16,
        'lora-dropout': 0.1,
    },
    output_path=f's3://{BUCKET_NAME}/output/stage2_bigfive_full',
    sagemaker_session=session,
    keep_alive_period_in_seconds=1800,
    use_spot_instances=False,
    max_run=7200,
)

print("[OK] Estimator created")

# トレーニング実行
print("\n[2/3] Starting Full Training (5 Epochs)...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNote: Training will take 45-60 minutes")
print("Progress can be monitored at AWS SageMaker Console:")
print(f"https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
print()

try:
    # No inputs needed - dataset auto-downloaded from HF
    estimator.fit()

    print("\n" + "=" * 80)
    print("【2回目】Full Training Complete!")
    print("=" * 80)
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved at: {estimator.model_data}")
    print(f"\nS3 bucket: s3://{BUCKET_NAME}/output/stage2_bigfive_full")
    print("\n[3/3] Next Steps:")
    print("1. Download and verify full training model")
    print("2. Test predictions on 3 sample texts")
    print("3. If successful, run final predictions on DynamoDB 28 records")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("[ERROR] Training failed")
    print("=" * 80)
    print(f"Error: {e}")
    raise
