"""
Stage 1: Contrastive Learning on AWS SageMaker
Nemotron-Personas-Japanデータでペルソナ表現を学習
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

# S3データパス（Nemotron）
S3_TRAIN_PATH = f's3://{BUCKET_NAME}/data/nemotron/train'
S3_VAL_PATH = f's3://{BUCKET_NAME}/data/nemotron/val'

print("=" * 80)
print("Stage 1: Contrastive Learning for Persona Representation")
print("Model: xlm-roberta-large + LoRA")
print("Method: SimCLR (NT-Xent Loss)")
print("Data: Nemotron-Personas-Japan (No Big Five labels needed)")
print("Instance: ml.g5.2xlarge (NVIDIA A10G GPU)")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET_NAME}")
print(f"Role: {ROLE_ARN}")
print(f"GitHub Repo: {GITHUB_REPO}")
print(f"Train Data: {S3_TRAIN_PATH}")
print(f"Val Data: {S3_VAL_PATH}")
print("=" * 80)

# SageMakerセッション作成
session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# PyTorch Estimator作成
print("\n[1/3] Creating PyTorch Estimator for Stage 1...")
estimator = PyTorch(
    entry_point='train_contrastive.py',  # Contrastive Learning script
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
        'projection-dim': 128,  # Contrastive projection dimension
        'temperature': 0.5,     # NT-Xent temperature
        'epochs': 3,
        'batch-size': 8,        # Conservative for 24GB GPU
        'learning-rate': 2e-4,
        'max-length': 512,
        'lora-r': 8,
        'lora-alpha': 16,
        'lora-dropout': 0.1,
    },
    output_path=f's3://{BUCKET_NAME}/output/stage1_contrastive',
    sagemaker_session=session,
    keep_alive_period_in_seconds=1800,  # Warm pool有効化（30分）
    use_spot_instances=False,  # G5インスタンスは通常インスタンス推奨
    max_run=7200,   # 最大実行時間（2時間）
)

print("[OK] Estimator created")

# トレーニング実行
print("\n[2/3] Starting Stage 1 Contrastive Learning...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNote: Training will take 30-60 minutes")
print("Progress can be monitored at AWS SageMaker Console:")
print(f"https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
print()

try:
    estimator.fit({
        'train': S3_TRAIN_PATH,
        'val': S3_VAL_PATH,
    })

    print("\n" + "=" * 80)
    print("[SUCCESS] Stage 1 Contrastive Learning Complete!")
    print("=" * 80)
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved at: {estimator.model_data}")
    print(f"\nS3 bucket: s3://{BUCKET_NAME}")
    print("\n[3/3] Next Steps:")
    print("1. Verify contrastive learning results in AWS SageMaker Console")
    print("2. Check training logs and loss curves")
    print("3. Prepare Stage 2: Big Five Regression Fine-tuning")
    print("   - Use this model as base encoder")
    print("   - Fine-tune with RealPersonaChat (with Big Five labels)")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("[ERROR] Training failed")
    print("=" * 80)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check IAM role permissions")
    print("2. Verify S3 data paths")
    print("3. Check GitHub repository accessibility")
    print("4. Verify SageMaker service limits")
    raise
