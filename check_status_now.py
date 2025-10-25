#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
即座にステータスを確認
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
from datetime import datetime

REGION = 'ap-northeast-1'
MODEL_A_JOB_NAME = 'pytorch-training-2025-10-25-08-23-19-001'

client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print(f"現在の状況確認 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    response = client.describe_training_job(TrainingJobName=MODEL_A_JOB_NAME)
    status = response['TrainingJobStatus']
    secondary = response.get('SecondaryStatus', 'N/A')

    print(f"\n【Model A - Stage 2のみ、5エポック】")
    print(f"ジョブ名: {MODEL_A_JOB_NAME}")
    print(f"ステータス: {status}")
    print(f"詳細: {secondary}")

    if 'TrainingStartTime' in response:
        start_time = response['TrainingStartTime']
        elapsed = datetime.now(start_time.tzinfo) - start_time
        elapsed_min = int(elapsed.total_seconds() // 60)
        print(f"経過時間: {elapsed_min}分")

        # 予想残り時間
        if elapsed_min < 45:
            remaining = 45 - elapsed_min
            print(f"予想残り: 約{remaining}分以上")
        else:
            print(f"予想残り: まもなく完了")

    if status == 'Completed':
        print("\n✅ 学習完了！")

        if 'FinalMetricDataList' in response and response['FinalMetricDataList']:
            print("\n最終メトリクス:")
            for metric in response['FinalMetricDataList']:
                print(f"  {metric['MetricName']}: {metric['Value']:.4f}")

        if 'ModelArtifacts' in response:
            print(f"\n保存先:")
            print(f"  {response['ModelArtifacts']['S3ModelArtifacts']}")

    elif status == 'Failed':
        print(f"\n❌ 学習失敗")
        if 'FailureReason' in response:
            print(f"理由: {response['FailureReason']}")

    print("\n" + "=" * 80)

except Exception as e:
    print(f"エラー: {e}")
