#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model B 小規模テスト監視スクリプト (1エポック)
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
import time
from datetime import datetime

# 設定
REGION = 'ap-northeast-1'
JOB_NAME = 'pytorch-training-2025-10-25-12-44-47-995'
CHECK_INTERVAL = 60  # 60秒ごと

client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print("Model B 小規模テスト監視 (1エポック、Stage 1 + Stage 2)")
print("=" * 80)
print(f"ジョブ名: {JOB_NAME}")
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

last_status = None
start_time = time.time()

while True:
    try:
        response = client.describe_training_job(TrainingJobName=JOB_NAME)
        status = response['TrainingJobStatus']
        secondary_status = response.get('SecondaryStatus', 'N/A')

        # ステータスが変わった場合のみ表示
        if status != last_status:
            elapsed = time.time() - start_time
            elapsed_min = int(elapsed // 60)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed_min}分経過)")
            print(f"  ステータス: {status}")
            print(f"  詳細: {secondary_status}")

            if status == 'Completed':
                print("\n" + "=" * 80)
                print("SUCCESS: Model B 小規模テスト完了!")
                print("=" * 80)

                # メトリクス表示
                if 'FinalMetricDataList' in response and response['FinalMetricDataList']:
                    print("\n最終メトリクス:")
                    for metric in response['FinalMetricDataList']:
                        print(f"  {metric['MetricName']}: {metric['Value']:.4f}")

                # モデル保存先
                if 'ModelArtifacts' in response:
                    model_path = response['ModelArtifacts']['S3ModelArtifacts']
                    print(f"\n保存先: {model_path}")

                print("\n次のステップ:")
                print("  1. モデルをダウンロードして検証")
                print("  2. DynamoDBデータで推論テスト")
                print("  3. 成功したらフル学習(5エポック)実行")
                print("=" * 80)

                sys.exit(0)

            elif status == 'Failed':
                print("\n" + "=" * 80)
                print("ERROR: Model B 小規模テスト失敗")
                print("=" * 80)

                if 'FailureReason' in response:
                    print(f"\n失敗理由: {response['FailureReason']}")

                print("\nCloudWatch Logsで詳細を確認してください")
                print("=" * 80)

                sys.exit(1)

            elif status == 'Stopped':
                print("\n" + "=" * 80)
                print("STOPPED: テストが停止されました")
                print("=" * 80)
                sys.exit(1)

            last_status = status

        # 60秒待機
        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n監視を中断しました")
        sys.exit(0)

    except Exception as e:
        print(f"\nエラー: {e}")
        print("60秒後に再試行します...")
        time.sleep(CHECK_INTERVAL)
