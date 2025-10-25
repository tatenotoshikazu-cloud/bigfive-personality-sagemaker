#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動順次学習スクリプト
Model A完了後、自動的にModel Bを起動
"""

import sys
import io
if sys.version_info[0] >= 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import boto3
import time
from datetime import datetime
import subprocess

# 設定
REGION = 'ap-northeast-1'
MODEL_A_JOB_NAME = 'pytorch-training-2025-10-25-08-23-19-001'  # Stage 2 only, 5 epochs
CHECK_INTERVAL = 60  # 60秒ごとにチェック

# SageMakerクライアント作成
client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print("自動順次学習システム")
print("=" * 80)
print(f"Model A (Stage 2のみ): {MODEL_A_JOB_NAME}")
print(f"チェック間隔: {CHECK_INTERVAL}秒")
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()
print("【ステップ1】Model A (Stage 2のみ、5エポック) の完了を待機中...")
print()

last_status = None
start_time = time.time()

while True:
    try:
        response = client.describe_training_job(TrainingJobName=MODEL_A_JOB_NAME)
        status = response['TrainingJobStatus']
        secondary_status = response.get('SecondaryStatus', 'N/A')

        # ステータスが変わった場合のみ表示
        if status != last_status:
            elapsed = time.time() - start_time
            elapsed_min = int(elapsed // 60)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed_min}分経過)")
            print(f"  Model A ステータス: {status}")
            print(f"  詳細: {secondary_status}")

            if status == 'Completed':
                print("\n" + "=" * 80)
                print("✅ Model A (Stage 2のみ) の学習が完了しました！")
                print("=" * 80)

                # メトリクス表示
                if 'FinalMetricDataList' in response and response['FinalMetricDataList']:
                    print("\nModel A 最終メトリクス:")
                    for metric in response['FinalMetricDataList']:
                        print(f"  {metric['MetricName']}: {metric['Value']:.4f}")

                # モデル保存先表示
                if 'ModelArtifacts' in response:
                    model_path = response['ModelArtifacts']['S3ModelArtifacts']
                    print(f"\nModel A 保存先:")
                    print(f"  {model_path}")

                print("\n" + "=" * 80)
                print("【ステップ2】Model B (2段階ファインチューニング) を起動中...")
                print("=" * 80)

                # Model B起動
                try:
                    result = subprocess.run(
                        ['py', 'run_two_stage_training.py'],
                        cwd='c:\\Users\\musta\\Documents\\IE\\DL',
                        capture_output=True,
                        text=True,
                        timeout=600,
                        encoding='utf-8'
                    )

                    print("\nModel B 起動結果:")
                    if result.returncode == 0:
                        print("✅ Model B 起動成功！")
                        print("\n標準出力:")
                        print(result.stdout)
                    else:
                        print("❌ Model B 起動失敗")
                        print("\n標準出力:")
                        print(result.stdout)
                        print("\n標準エラー:")
                        print(result.stderr)

                except subprocess.TimeoutExpired:
                    print("⚠️  Model B 起動がタイムアウト（10分）")
                except Exception as e:
                    print(f"❌ Model B 起動エラー: {e}")

                print("\n" + "=" * 80)
                print("次のステップ:")
                print("  1. Model B の学習が完了するまで待機（auto_monitor_two_stage.py で監視）")
                print("  2. 両モデルをダウンロード・検証")
                print("  3. 3つのサンプルテキストで推論比較")
                print("  4. 優れたモデルを選択してDynamoDB 28件推論")
                print("=" * 80)

                sys.exit(0)  # 正常終了

            elif status == 'Failed':
                print("\n" + "=" * 80)
                print("❌ Model A の学習が失敗しました")
                print("=" * 80)

                if 'FailureReason' in response:
                    print(f"\n失敗理由:")
                    print(f"  {response['FailureReason']}")

                print("\n対処方法:")
                print("  1. CloudWatch Logsでエラー詳細を確認")
                print("  2. train_bigfive.pyのコードを確認")
                print("  3. 再度学習ジョブを実行")
                print("=" * 80)

                sys.exit(1)  # エラー終了

            elif status == 'Stopped':
                print("\n" + "=" * 80)
                print("⏹️  Model A の学習が停止されました")
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
