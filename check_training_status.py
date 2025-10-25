"""
SageMaker学習ジョブのステータスを確認
"""
import boto3
from datetime import datetime

# 設定
REGION = 'ap-northeast-1'
JOB_NAME = 'pytorch-training-2025-10-25-07-49-36-999'

# SageMakerクライアント作成
client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print(f"SageMaker Training Job Status: {JOB_NAME}")
print("=" * 80)

try:
    response = client.describe_training_job(TrainingJobName=JOB_NAME)

    status = response['TrainingJobStatus']
    creation_time = response['CreationTime']

    print(f"\nステータス: {status}")
    print(f"作成時刻: {creation_time}")

    if 'TrainingStartTime' in response:
        print(f"学習開始時刻: {response['TrainingStartTime']}")

    if 'TrainingEndTime' in response:
        print(f"学習終了時刻: {response['TrainingEndTime']}")

    if 'SecondaryStatus' in response:
        print(f"詳細ステータス: {response['SecondaryStatus']}")

    if 'FailureReason' in response:
        print(f"\n失敗理由: {response['FailureReason']}")

    if 'FinalMetricDataList' in response and response['FinalMetricDataList']:
        print("\n最終メトリクス:")
        for metric in response['FinalMetricDataList']:
            print(f"  {metric['MetricName']}: {metric['Value']}")

    if 'ModelArtifacts' in response:
        print(f"\nモデル保存先: {response['ModelArtifacts']['S3ModelArtifacts']}")

    print("\n" + "=" * 80)
    print("結論:")
    print("=" * 80)

    if status == 'Completed':
        print("✅ 学習は正常に完了しました！")
    elif status == 'Failed':
        print("❌ 学習は失敗しました")
    elif status == 'InProgress':
        print("🏃 学習は現在実行中です")
    elif status == 'Stopping':
        print("⏹️ 学習は停止中です")
    elif status == 'Stopped':
        print("⏹️ 学習は停止されました")
    else:
        print(f"ℹ️ ステータス: {status}")

    print("=" * 80)

except Exception as e:
    print(f"\nエラー: {e}")
    print("\n考えられる原因:")
    print("1. ジョブ名が間違っている")
    print("2. ジョブがまだ作成されていない")
    print("3. AWS認証情報に問題がある")
