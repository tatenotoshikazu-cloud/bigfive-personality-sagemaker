#!/usr/bin/env python3
"""
SageMaker Training Job Auto Monitor & Error Handler
トレーニングジョブを自動監視し、エラーを検知して報告
"""

import boto3
import time
import json
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    """トレーニングジョブの自動監視"""

    def __init__(self, region='ap-northeast-1', check_interval=30):
        self.sm = boto3.client('sagemaker', region_name=region)
        self.check_interval = check_interval
        self.region = region

    def get_latest_job(self):
        """最新のトレーニングジョブを取得"""
        response = self.sm.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )

        if response['TrainingJobSummaries']:
            return response['TrainingJobSummaries'][0]['TrainingJobName']
        return None

    def get_job_status(self, job_name):
        """ジョブの詳細情報を取得"""
        return self.sm.describe_training_job(TrainingJobName=job_name)

    def analyze_error(self, job_info):
        """エラーを分析して解決策を提案"""
        failure_reason = job_info.get('FailureReason', '')

        # 既知のエラーパターンと解決策
        error_patterns = {
            'Directory data/train not found': {
                'cause': 'データディレクトリが見つからない',
                'solution': 'train_bigfive.pyでRealPersonaChatを直接ロードするように修正',
                'fix_applied': True,
                'file': 'train_bigfive.py',
                'line': '233-246'
            },
            'No columns in the dataset match': {
                'cause': 'データセットのカラムがモデルの入力と一致しない',
                'solution': 'データをトークン化してinput_ids, attention_maskを作成',
                'fix_applied': False,
                'recommendation': 'データ前処理スクリプトを確認'
            },
            'CUDA out of memory': {
                'cause': 'GPUメモリ不足',
                'solution': 'batch_sizeを減らす（現在16 → 8に変更推奨）',
                'fix_applied': False,
                'recommendation': 'run_stage2_bigfive.py の batch-size を変更'
            },
            'trust_remote_code': {
                'cause': 'Hugging Face datasetsの非推奨パラメータ',
                'solution': 'trust_remote_code パラメータを削除',
                'fix_applied': False,
                'recommendation': 'データロード部分を修正'
            },
        }

        # エラーパターンをマッチング
        for pattern, info in error_patterns.items():
            if pattern in failure_reason:
                return {
                    'pattern': pattern,
                    'error': failure_reason,
                    **info
                }

        # 未知のエラー
        return {
            'pattern': 'Unknown',
            'error': failure_reason,
            'cause': '未知のエラー',
            'solution': 'ログを詳細に確認してください',
            'fix_applied': False
        }

    def monitor(self, job_name=None, auto_report=True):
        """
        トレーニングジョブを監視

        Args:
            job_name: 監視するジョブ名（Noneなら最新ジョブ）
            auto_report: エラー時に自動でレポート生成
        """
        if job_name is None:
            job_name = self.get_latest_job()
            if not job_name:
                print('[ERROR] No training jobs found')
                return

        print('=' * 80)
        print(f'Training Monitor Started')
        print('=' * 80)
        print(f'Job Name: {job_name}')
        print(f'Region: {self.region}')
        print(f'Check Interval: {self.check_interval}s')
        print(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 80)
        print()

        last_status = None
        last_secondary = None
        start_time = time.time()

        while True:
            try:
                job_info = self.get_job_status(job_name)
                status = job_info['TrainingJobStatus']
                secondary = job_info.get('SecondaryStatus', 'N/A')

                # ステータス変更時に表示
                if status != last_status or secondary != last_secondary:
                    elapsed = int(time.time() - start_time)
                    elapsed_str = f'{elapsed//60}m {elapsed%60}s'

                    print(f'[{elapsed_str}] Status: {status}')
                    if secondary != 'N/A':
                        print(f'         Secondary: {secondary}')

                    last_status = status
                    last_secondary = secondary

                # 完了
                if status == 'Completed':
                    print()
                    print('=' * 80)
                    print('[SUCCESS] Training Completed!')
                    print('=' * 80)
                    print(f'Model Location: {job_info["ModelArtifacts"]["S3ModelArtifacts"]}')
                    print(f'Training Time: {elapsed_str}')

                    # 結果をJSON保存
                    self._save_result(job_name, job_info, 'SUCCESS')
                    break

                # 失敗
                elif status == 'Failed':
                    print()
                    print('=' * 80)
                    print('[FAILED] Training Failed!')
                    print('=' * 80)

                    # エラー分析
                    error_analysis = self.analyze_error(job_info)

                    print(f'Error Pattern: {error_analysis["pattern"]}')
                    print(f'Cause: {error_analysis["cause"]}')
                    print(f'Solution: {error_analysis["solution"]}')

                    if error_analysis.get('fix_applied'):
                        print(f'\n[INFO] Fix already applied in: {error_analysis.get("file")}')
                        print(f'       Line: {error_analysis.get("line")}')
                        print('       Please retry the training job.')
                    else:
                        print(f'\n[ACTION REQUIRED] {error_analysis.get("recommendation")}')

                    print('\nFull Error Message:')
                    print(error_analysis['error'])

                    # エラーレポート保存
                    if auto_report:
                        self._save_error_report(job_name, job_info, error_analysis)

                    break

                # 停止
                elif status == 'Stopped':
                    print()
                    print('=' * 80)
                    print('[STOPPED] Training was stopped')
                    print('=' * 80)
                    break

                # 待機
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print()
                print('=' * 80)
                print('[INTERRUPTED] Monitoring stopped by user')
                print('=' * 80)
                break

            except Exception as e:
                print(f'\n[ERROR] Monitor exception: {e}')
                time.sleep(self.check_interval)

    def _save_result(self, job_name, job_info, status):
        """結果を保存"""
        result = {
            'job_name': job_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'model_location': job_info.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
            'training_time_seconds': job_info.get('TrainingTimeInSeconds'),
        }

        output_file = Path('training_results') / f'{job_name}_result.json'
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f'\n[INFO] Result saved to: {output_file}')

    def _save_error_report(self, job_name, job_info, error_analysis):
        """エラーレポートを保存"""
        report = {
            'job_name': job_name,
            'timestamp': datetime.now().isoformat(),
            'error_pattern': error_analysis['pattern'],
            'cause': error_analysis['cause'],
            'solution': error_analysis['solution'],
            'fix_applied': error_analysis.get('fix_applied', False),
            'recommendation': error_analysis.get('recommendation'),
            'full_error': error_analysis['error'],
            'job_info': {
                'creation_time': job_info.get('CreationTime').isoformat() if job_info.get('CreationTime') else None,
                'training_start_time': job_info.get('TrainingStartTime').isoformat() if job_info.get('TrainingStartTime') else None,
                'failure_time': job_info.get('TrainingEndTime').isoformat() if job_info.get('TrainingEndTime') else None,
            }
        }

        output_file = Path('training_results') / f'{job_name}_error_report.json'
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f'\n[INFO] Error report saved to: {output_file}')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SageMaker Training Monitor')
    parser.add_argument('--job-name', type=str, help='Job name to monitor (latest if not specified)')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--region', type=str, default='ap-northeast-1', help='AWS region')
    parser.add_argument('--no-report', action='store_true', help='Disable auto error report')

    args = parser.parse_args()

    monitor = TrainingMonitor(region=args.region, check_interval=args.interval)
    monitor.monitor(job_name=args.job_name, auto_report=not args.no_report)


if __name__ == '__main__':
    main()
