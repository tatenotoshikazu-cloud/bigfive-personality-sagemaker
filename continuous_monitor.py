#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
継続監視スクリプト - 1分ごとにModel A/Bの状況を確認
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
MODEL_A_JOB_NAME = 'pytorch-training-2025-10-25-08-23-19-001'
CHECK_INTERVAL = 60  # 60秒ごと

# SageMakerクライアント
client = boto3.client('sagemaker', region_name=REGION)

print("=" * 80)
print("🔍 継続監視システム起動")
print("=" * 80)
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"チェック間隔: {CHECK_INTERVAL}秒")
print("=" * 80)
print()

model_a_completed = False
model_b_job_name = None
model_b_launched = False

iteration = 0

while True:
    iteration += 1
    print(f"\n{'='*80}")
    print(f"📊 監視ループ #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")

    try:
        # Model A のステータス確認
        print("\n【Model A - Stage 2のみ、5エポック】")
        response_a = client.describe_training_job(TrainingJobName=MODEL_A_JOB_NAME)
        status_a = response_a['TrainingJobStatus']
        secondary_a = response_a.get('SecondaryStatus', 'N/A')

        print(f"  ジョブ名: {MODEL_A_JOB_NAME}")
        print(f"  ステータス: {status_a}")
        print(f"  詳細: {secondary_a}")

        if status_a == 'InProgress':
            # 学習時間を計算
            if 'TrainingStartTime' in response_a:
                start_time = response_a['TrainingStartTime']
                elapsed = datetime.now(start_time.tzinfo) - start_time
                elapsed_min = int(elapsed.total_seconds() // 60)
                print(f"  経過時間: {elapsed_min}分")

                # 予想残り時間（45-60分想定）
                if elapsed_min < 45:
                    remaining = 45 - elapsed_min
                    print(f"  予想残り: 約{remaining}分以上")
                else:
                    print(f"  予想残り: まもなく完了")

        elif status_a == 'Completed':
            if not model_a_completed:
                print("  ✅ 学習完了！")

                # メトリクス表示
                if 'FinalMetricDataList' in response_a and response_a['FinalMetricDataList']:
                    print("\n  📈 最終メトリクス:")
                    for metric in response_a['FinalMetricDataList']:
                        print(f"    {metric['MetricName']}: {metric['Value']:.4f}")

                # モデル保存先
                if 'ModelArtifacts' in response_a:
                    print(f"\n  💾 保存先: {response_a['ModelArtifacts']['S3ModelArtifacts']}")

                model_a_completed = True

                # Model B を起動
                if not model_b_launched:
                    print("\n" + "="*80)
                    print("🚀 Model B (2段階ファインチューニング) を起動します...")
                    print("="*80)

                    try:
                        result = subprocess.run(
                            ['py', 'run_two_stage_training.py'],
                            cwd='c:\\Users\\musta\\Documents\\IE\\DL',
                            capture_output=True,
                            text=True,
                            timeout=600,
                            encoding='utf-8'
                        )

                        if result.returncode == 0:
                            print("✅ Model B 起動成功")

                            # ジョブ名を抽出（stdout から）
                            for line in result.stdout.split('\n'):
                                if 'pytorch-training-' in line:
                                    parts = line.strip().split()
                                    for part in parts:
                                        if part.startswith('pytorch-training-'):
                                            model_b_job_name = part
                                            print(f"  ジョブ名: {model_b_job_name}")
                                            break

                            model_b_launched = True
                        else:
                            print("❌ Model B 起動失敗")
                            print(f"エラー: {result.stderr}")

                    except Exception as e:
                        print(f"❌ Model B 起動エラー: {e}")

        elif status_a == 'Failed':
            print(f"  ❌ 学習失敗: {response_a.get('FailureReason', '不明')}")
            print("\n⚠️ Model A が失敗したため、監視を終了します")
            sys.exit(1)

        # Model B のステータス確認（起動済みの場合）
        if model_b_job_name:
            print(f"\n【Model B - 2段階ファインチューニング】")
            try:
                response_b = client.describe_training_job(TrainingJobName=model_b_job_name)
                status_b = response_b['TrainingJobStatus']
                secondary_b = response_b.get('SecondaryStatus', 'N/A')

                print(f"  ジョブ名: {model_b_job_name}")
                print(f"  ステータス: {status_b}")
                print(f"  詳細: {secondary_b}")

                if status_b == 'InProgress':
                    if 'TrainingStartTime' in response_b:
                        start_time = response_b['TrainingStartTime']
                        elapsed = datetime.now(start_time.tzinfo) - start_time
                        elapsed_min = int(elapsed.total_seconds() // 60)
                        print(f"  経過時間: {elapsed_min}分")

                elif status_b == 'Completed':
                    print("  ✅ 学習完了！")

                    if 'FinalMetricDataList' in response_b and response_b['FinalMetricDataList']:
                        print("\n  📈 最終メトリクス:")
                        for metric in response_b['FinalMetricDataList']:
                            print(f"    {metric['MetricName']}: {metric['Value']:.4f}")

                    if 'ModelArtifacts' in response_b:
                        print(f"\n  💾 保存先: {response_b['ModelArtifacts']['S3ModelArtifacts']}")

                    print("\n" + "="*80)
                    print("✅ 両モデルの学習が完了しました！")
                    print("="*80)
                    print("\n次のステップ:")
                    print("  1. 両モデルをダウンロード")
                    print("  2. 3つのサンプルテキストで推論")
                    print("  3. 予測の多様性とRMSEで比較")
                    print("  4. 優れたモデルでDynamoDB 28件推論")
                    print("="*80)

                    sys.exit(0)  # 正常終了

                elif status_b == 'Failed':
                    print(f"  ❌ 学習失敗: {response_b.get('FailureReason', '不明')}")

                    # Stage 1 エラーの可能性を確認
                    failure_reason = response_b.get('FailureReason', '')
                    if 'stage1' in failure_reason.lower() or 'contrastive' in failure_reason.lower():
                        print("\n  ⚠️ Stage 1 (Contrastive Learning) でエラーが発生した可能性があります")
                        print("  → CloudWatch Logsで詳細を確認してください")

                    print("\n⚠️ Model B が失敗しました")
                    print("  → Model A のみ使用するか、Model B を再実行してください")
                    sys.exit(1)

            except client.exceptions.ResourceNotFound:
                print(f"  ⏳ ジョブ作成待ち...")

        # 次のチェックまで待機
        print(f"\n⏰ 次のチェックまで {CHECK_INTERVAL}秒待機...")

        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n✋ 監視を手動停止しました")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ エラー: {e}")
        print(f"   {CHECK_INTERVAL}秒後に再試行します...")
        time.sleep(CHECK_INTERVAL)
