#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【2回目】5エポックフル学習の自動監視スクリプト
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
JOB_NAME = 'pytorch-training-2025-10-25-08-23-19-001'
CHECK_INTERVAL = 60  # 60秒ごとにチェック

# SageMakerクライアント作成
client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print("【2回目】5エポックフル学習 自動監視開始")
print("=" * 80)
print(f"ジョブ名: {JOB_NAME}")
print(f"チェック間隔: {CHECK_INTERVAL}秒")
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
                print("✅ 【2回目】フル学習が完了しました！")
                print("=" * 80)

                # メトリクス表示
                if 'FinalMetricDataList' in response and response['FinalMetricDataList']:
                    print("\n最終メトリクス:")
                    for metric in response['FinalMetricDataList']:
                        print(f"  {metric['MetricName']}: {metric['Value']:.4f}")

                # モデル保存先表示
                if 'ModelArtifacts' in response:
                    model_path = response['ModelArtifacts']['S3ModelArtifacts']
                    print(f"\nモデル保存先:")
                    print(f"  {model_path}")

                print("\n次のステップ:")
                print("  1. フル学習モデルをS3からダウンロード")
                print("  2. 重みを検証")
                print("  3. 3つのサンプルテキストで推論テスト")
                print("  4. 異なるBig Five値が出力されることを確認")
                print("  5. 検証成功後、DynamoDBの28件で本番推論")
                print("=" * 80)

                sys.exit(0)  # 正常終了

            elif status == 'Failed':
                print("\n" + "=" * 80)
                print("❌ 【2回目】フル学習が失敗しました")
                print("=" * 80)

                if 'FailureReason' in response:
                    print(f"\n失敗理由:")
                    print(f"  {response['FailureReason']}")

                print("\n対処方法:")
                print("  1. CloudWatch Logsでエラー詳細を確認")
                print("  2. train_bigfive.pyのコードを修正")
                print("  3. 再度学習ジョブを実行")
                print("=" * 80)

                sys.exit(1)  # エラー終了

            elif status == 'Stopped':
                print("\n" + "=" * 80)
                print("⏹️  【2回目】フル学習が停止されました")
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
