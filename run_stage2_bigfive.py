"""
Stage 2: Big Five Personality Prediction on AWS SageMaker
Stage 1のContrastive Learningモデルをベースに
RealPersonaChatのBig Fiveラベル付きデータでファインチューニング
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

# Stage 1モデルパス（S3）
STAGE1_MODEL_S3 = 's3://bigfive-personality-sagemaker-1761305156/output/stage1_contrastive/pytorch-training-2025-10-24-12-26-47-276/output/model.tar.gz'

print("=" * 80)
print("Stage 2: Big Five Personality Prediction Fine-tuning")
print("Base: Stage 1 Contrastive Learning Model")
print("Data: RealPersonaChat (Big Five Labels)")
print("Instance: ml.g5.2xlarge (NVIDIA A10G GPU)")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET_NAME}")
print(f"Role: {ROLE_ARN}")
print(f"GitHub Repo: {GITHUB_REPO}")
print(f"Stage 1 Model: {STAGE1_MODEL_S3}")
print("=" * 80)

# SageMakerセッション作成
session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# PyTorch Estimator作成
print("\n[1/3] Creating PyTorch Estimator for Stage 2...")
estimator = PyTorch(
    entry_point='train_bigfive.py',  # Stage 2スクリプト
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
        'stage1-model-path': '/opt/ml/input/data/stage1_model',  # SageMaker内パス
        'epochs': 5,           # Stage 2は多めに
        'batch-size': 16,      # Big Fiveデータ用
        'learning-rate': 1e-4, # Stage 2は小さめ（微調整）
        'max-length': 512,
        'lora-r': 8,
        'lora-alpha': 16,
        'lora-dropout': 0.1,
    },
    output_path=f's3://{BUCKET_NAME}/output/stage2_bigfive',
    sagemaker_session=session,
    keep_alive_period_in_seconds=1800,  # Warm pool有効化
    use_spot_instances=False,
    max_run=7200,  # 最大2時間
)

print("[OK] Estimator created")

# トレーニング実行
print("\n[2/3] Starting Stage 2 Big Five Fine-tuning...")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNote: Training will take 40-90 minutes")
print("Progress can be monitored at AWS SageMaker Console:")
print(f"https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
print()

try:
    # RealPersonaChatデータは自動ロード（train_bigfive.py内で）
    # Stage 1モデルをS3から渡す
    estimator.fit({
        'stage1_model': STAGE1_MODEL_S3,  # Stage 1モデル
    })

    print("\n" + "=" * 80)
    print("[SUCCESS] Stage 2 Big Five Fine-tuning Complete!")
    print("=" * 80)
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved at: {estimator.model_data}")
    print(f"\nS3 bucket: s3://{BUCKET_NAME}")
    print("\n[3/3] Next Steps:")
    print("1. Download model from S3")
    print("2. Test Big Five prediction with sample data")
    print("3. Deploy for production use:")
    print("   - DynamoDB (チーキューデータ)")
    print("   - 文字起こしデータ")
    print("   - 任意のテキスト")
    print()
    print("使用方法:")
    print("  python predict_bigfive.py \\")
    print(f"    --model-path ./stage2_model \\")
    print("    --input-text '営業経験10年...'")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("[ERROR] Training failed")
    print("=" * 80)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check IAM role permissions")
    print("2. Verify Stage 1 model exists at S3")
    print("3. Check RealPersonaChat dataset accessibility")
    print("4. Verify SageMaker service limits")
    raise
