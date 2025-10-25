#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B（2段階ファインチューニング）監視スクリプト
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
import time
from datetime import datetime

REGION = 'ap-northeast-1'
MODEL_B_JOB_NAME = 'pytorch-training-2025-10-25-12-39-31-353'
CHECK_INTERVAL = 60

client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print("Model B（2段階ファインチューニング）監視")
print("=" * 80)
print(f"ジョブ名: {MODEL_B_JOB_NAME}")
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

iteration = 0
start_time = time.time()

while True:
    iteration += 1

    try:
        response = client.describe_training_job(TrainingJobName=MODEL_B_JOB_NAME)
        status = response['TrainingJobStatus']
        secondary = response.get('SecondaryStatus', 'N/A')

        print(f"[監視 #{iteration}] {datetime.now().strftime('%H:%M:%S')}")
        print(f"  ステータス: {status}")
        print(f"  詳細: {secondary}")

        if 'TrainingStartTime' in response:
            training_start = response['TrainingStartTime']
            elapsed = datetime.now(training_start.tzinfo) - training_start
            elapsed_min = int(elapsed.total_seconds() // 60)
            print(f"  経過時間: {elapsed_min}分")

            if elapsed_min < 45:
                print(f"  予想残り: 約{45 - elapsed_min}分以上")
            else:
                print(f"  予想残り: まもなく完了")

        if status == 'Completed':
            print("\n" + "=" * 80)
            print("✅ Model B 学習完了！")
            print("=" * 80)

            if 'FinalMetricDataList' in response and response['FinalMetricDataList']:
                print("\n最終メトリクス:")
                for metric in response['FinalMetricDataList']:
                    print(f"  {metric['MetricName']}: {metric['Value']:.4f}")

            if 'ModelArtifacts' in response:
                print(f"\n保存先:")
                print(f"  {response['ModelArtifacts']['S3ModelArtifacts']}")

            print("\n次: Model AとModel Bを比較します")
            print("=" * 80)
            sys.exit(0)

        elif status == 'Failed':
            print("\n" + "=" * 80)
            print("❌ Model B 学習失敗")
            print("=" * 80)
            if 'FailureReason' in response:
                print(f"\n失敗理由: {response['FailureReason']}")
            sys.exit(1)

        print(f"  次のチェック: {CHECK_INTERVAL}秒後\n")
        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n監視を中断しました")
        sys.exit(0)
    except Exception as e:
        print(f"\nエラー: {e}")
        print(f"{CHECK_INTERVAL}秒後に再試行...")
        time.sleep(CHECK_INTERVAL)
