"""
Big Five Personality Fine-tuning on AWS SageMaker
xlm-roberta-large + LoRA - トレーニング実行スクリプト
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

# S3データパス
S3_DATA_PATH = f's3://{BUCKET_NAME}/data'

print("=" * 80)
print("Big Five Personality Fine-tuning - SageMaker Training")
print("Model: xlm-roberta-large + LoRA")
print("Instance: ml.g5.2xlarge (NVIDIA A10G GPU)")
print("=" * 80)
print(f"Region: {REGION}")
print(f"Bucket: {BUCKET_NAME}")
print(f"Role: {ROLE_ARN}")
print(f"GitHub Repo: {GITHUB_REPO}")
print(f"Data Path: {S3_DATA_PATH}")
print("=" * 80)

# SageMakerセッション作成
session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

# PyTorch Estimator作成
print("\n[1/3] PyTorch Estimatorを作成中...")
estimator = PyTorch(
    entry_point='train_sagemaker.py',  # トレーニングスクリプト
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
        'epochs': 3,
        'batch-size': 8,
        'learning-rate': 2e-4,
        'max-length': 512,
        'lora-r': 8,
        'lora-alpha': 16,
        'lora-dropout': 0.1,
    },
    output_path=f's3://{BUCKET_NAME}/output',
    sagemaker_session=session,
    keep_alive_period_in_seconds=1800,  # Warm pool有効化（30分）
    use_spot_instances=False,  # G5インスタンスは通常インスタンス推奨
    max_run=7200,   # 最大実行時間（2時間）
)

print("[OK] Estimator作成完了")

# トレーニング実行
print("\n[2/3] トレーニングを開始します...")
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n注意: トレーニングには30分〜1時間程度かかります")
print("進行状況はAWS SageMaker Consoleで確認できます:")
print(f"https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
print()

try:
    estimator.fit({
        'train': f'{S3_DATA_PATH}/train',
        'val': f'{S3_DATA_PATH}/val',
    })

    print("\n" + "=" * 80)
    print("[SUCCESS] トレーニング完了！")
    print("=" * 80)
    print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"モデル保存先: {estimator.model_data}")
    print(f"\nS3バケット: s3://{BUCKET_NAME}")
    print("\n[3/3] 次のステップ:")
    print("1. AWS SageMaker Consoleでログ・メトリクスを確認")
    print("2. モデルをダウンロードしてローカルで推論テスト")
    print("3. SageMakerエンドポイントを作成してデプロイ")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("[ERROR] エラーが発生しました")
    print("=" * 80)
    print(f"エラー内容: {e}")
    print("\n確認事項:")
    print("1. IAMロールに適切な権限があるか")
    print("2. S3バケットにデータがアップロードされているか")
    print("3. GitHubリポジトリが公開されているか")
    print("4. SageMakerのサービス制限を超えていないか")
    raise
