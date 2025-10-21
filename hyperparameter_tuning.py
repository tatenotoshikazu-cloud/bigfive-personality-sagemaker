"""
AWS SageMaker ハイパーパラメータチューニング
Bayesian最適化で最適なハイパーパラメータを探索
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    HyperparameterTuner,
    IntegerParameter,
    ContinuousParameter,
    CategoricalParameter
)
import boto3
import yaml


class BigFiveHyperparameterTuner:
    """ハイパーパラメータチューニングクラス"""

    def __init__(self, config_path='config.yaml'):
        """
        Args:
            config_path: 設定ファイルパス
        """
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # SageMaker設定
        sm_config = self.config['sagemaker']
        self.region = sm_config['region']
        self.session = sagemaker.Session(boto_session=boto3.Session(region_name=self.region))
        self.bucket = self.session.default_bucket()

        # 実行ロール（環境に応じて設定）
        try:
            self.role = sagemaker.get_execution_role()
        except:
            print("警告: SageMaker実行ロールを取得できませんでした")
            self.role = None

    def create_estimator(self, stage=1):
        """
        PyTorch Estimator作成

        Args:
            stage: 学習ステージ（1 or 2）

        Returns:
            PyTorch Estimator
        """
        stage_config = self.config[f'stage{stage}']
        sm_config = self.config['sagemaker']

        # 固定ハイパーパラメータ
        fixed_hyperparameters = {
            'stage': stage,
            'model_name': self.config['model']['name'],
            'max_length': self.config['model']['max_length'],
            'epochs': stage_config['epochs'],
        }

        estimator = PyTorch(
            entry_point='train.py',
            source_dir='.',
            role=self.role,
            instance_type=sm_config['instance_type'],
            instance_count=sm_config['instance_count'],
            framework_version='2.0.0',
            py_version='py310',
            hyperparameters=fixed_hyperparameters,
            output_path=f's3://{self.bucket}/bigfive/tuning/stage{stage}',
            code_location=f's3://{self.bucket}/bigfive/code',
            sagemaker_session=self.session,
            keep_alive_period_in_seconds=sm_config['keep_alive_period'],
            use_spot_instances=sm_config.get('use_spot_instances', False),
            max_wait=sm_config.get('max_wait_time', 86400) if sm_config.get('use_spot_instances') else None,
        )

        return estimator

    def define_hyperparameter_ranges(self):
        """
        チューニング対象のハイパーパラメータ範囲定義

        Returns:
            ハイパーパラメータ範囲辞書
        """
        tuning_config = self.config['hyperparameter_tuning']['tunable_parameters']

        hyperparameter_ranges = {}

        # Learning Rate
        if 'learning_rate' in tuning_config:
            lr_config = tuning_config['learning_rate']
            hyperparameter_ranges['learning_rate'] = ContinuousParameter(
                lr_config['min'],
                lr_config['max'],
                scaling_type=lr_config.get('scale', 'Auto')
            )

        # LoRA Rank
        if 'lora_r' in tuning_config:
            r_config = tuning_config['lora_r']
            hyperparameter_ranges['lora_r'] = IntegerParameter(
                r_config['min'],
                r_config['max']
            )

        # LoRA Alpha
        if 'lora_alpha' in tuning_config:
            alpha_config = tuning_config['lora_alpha']
            hyperparameter_ranges['lora_alpha'] = IntegerParameter(
                alpha_config['min'],
                alpha_config['max']
            )

        # Batch Size
        if 'batch_size' in tuning_config:
            bs_config = tuning_config['batch_size']
            hyperparameter_ranges['batch_size'] = CategoricalParameter(
                bs_config['values']
            )

        # Weight Decay
        if 'weight_decay' in tuning_config:
            wd_config = tuning_config['weight_decay']
            hyperparameter_ranges['weight_decay'] = ContinuousParameter(
                wd_config['min'],
                wd_config['max'],
                scaling_type=wd_config.get('scale', 'Auto')
            )

        return hyperparameter_ranges

    def run_tuning(self, s3_data_path, stage=1):
        """
        ハイパーパラメータチューニング実行

        Args:
            s3_data_path: S3データパス
            stage: 学習ステージ

        Returns:
            HyperparameterTuner
        """
        tuning_config = self.config['hyperparameter_tuning']

        if not tuning_config.get('enabled', False):
            print("ハイパーパラメータチューニングが無効です")
            print("config.yamlで enabled: true に設定してください")
            return None

        print("=" * 60)
        print(f"ハイパーパラメータチューニング開始 (Stage {stage})")
        print("=" * 60)

        # Estimator作成
        estimator = self.create_estimator(stage)

        # ハイパーパラメータ範囲定義
        hyperparameter_ranges = self.define_hyperparameter_ranges()

        print("\nチューニング対象パラメータ:")
        for param, range_obj in hyperparameter_ranges.items():
            print(f"  - {param}: {range_obj}")

        # 評価指標設定
        objective_metric = tuning_config['objective_metric']
        objective_metric_name = objective_metric['name']
        objective_type = objective_metric['type']

        # Tuner作成
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=objective_metric_name,
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[
                {'Name': 'validation:mae', 'Regex': 'validation.*mae: ([0-9\\.]+)'},
                {'Name': 'validation:mse', 'Regex': 'validation.*mse: ([0-9\\.]+)'},
                {'Name': 'train:loss', 'Regex': 'train.*loss: ([0-9\\.]+)'},
            ],
            max_jobs=tuning_config['max_jobs'],
            max_parallel_jobs=tuning_config['max_parallel_jobs'],
            strategy=tuning_config['strategy'],
            objective_type=objective_type,
        )

        print(f"\n最適化指標: {objective_metric_name} ({objective_type})")
        print(f"最大ジョブ数: {tuning_config['max_jobs']}")
        print(f"並列ジョブ数: {tuning_config['max_parallel_jobs']}")
        print(f"戦略: {tuning_config['strategy']}\n")

        # チューニング実行
        tuner.fit({'train': s3_data_path})

        print("\n" + "=" * 60)
        print("ハイパーパラメータチューニング完了")
        print("=" * 60)

        return tuner

    def get_best_hyperparameters(self, tuner):
        """
        最良のハイパーパラメータ取得

        Args:
            tuner: HyperparameterTuner

        Returns:
            最良のハイパーパラメータ辞書
        """
        best_training_job = tuner.best_training_job()

        # 結果取得（SageMaker API経由）
        sm_client = boto3.client('sagemaker', region_name=self.region)
        response = sm_client.describe_training_job(TrainingJobName=best_training_job)

        best_hyperparameters = response['HyperParameters']
        final_metric = response['FinalMetricDataList']

        print("\n" + "=" * 60)
        print("最良のハイパーパラメータ")
        print("=" * 60)
        print(f"トレーニングジョブ: {best_training_job}")
        print("\nハイパーパラメータ:")
        for key, value in best_hyperparameters.items():
            print(f"  {key}: {value}")

        print("\n評価指標:")
        for metric in final_metric:
            print(f"  {metric['MetricName']}: {metric['Value']}")

        return best_hyperparameters

    def analyze_tuning_results(self, tuner):
        """
        チューニング結果分析

        Args:
            tuner: HyperparameterTuner
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 結果取得
        analytics = tuner.analytics()
        df = analytics.dataframe()

        print("\n" + "=" * 60)
        print("チューニング結果統計")
        print("=" * 60)
        print(df.describe())

        # トップ5結果表示
        print("\n" + "=" * 60)
        print("トップ5のジョブ")
        print("=" * 60)
        print(df.head())

        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Learning Rate vs MAE
        if 'learning_rate' in df.columns:
            axes[0, 0].scatter(df['learning_rate'], df['FinalObjectiveValue'])
            axes[0, 0].set_xlabel('Learning Rate')
            axes[0, 0].set_ylabel('Validation MAE')
            axes[0, 0].set_title('Learning Rate vs MAE')
            axes[0, 0].set_xscale('log')

        # 2. LoRA Rank vs MAE
        if 'lora_r' in df.columns:
            axes[0, 1].scatter(df['lora_r'], df['FinalObjectiveValue'])
            axes[0, 1].set_xlabel('LoRA Rank (r)')
            axes[0, 1].set_ylabel('Validation MAE')
            axes[0, 1].set_title('LoRA Rank vs MAE')

        # 3. Batch Size vs MAE
        if 'batch_size' in df.columns:
            sns.boxplot(x='batch_size', y='FinalObjectiveValue', data=df, ax=axes[1, 0])
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Validation MAE')
            axes[1, 0].set_title('Batch Size vs MAE')

        # 4. 時系列（ジョブの進行）
        axes[1, 1].plot(range(len(df)), df['FinalObjectiveValue'])
        axes[1, 1].set_xlabel('Tuning Job Index')
        axes[1, 1].set_ylabel('Validation MAE')
        axes[1, 1].set_title('Tuning Progress')

        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ 結果グラフを保存: hyperparameter_tuning_results.png")

        return df


def main():
    """メイン実行"""
    print("=" * 60)
    print("Big Five性格特性推定 - ハイパーパラメータチューニング")
    print("=" * 60)

    # Tuner初期化
    tuner_manager = BigFiveHyperparameterTuner(config_path='config.yaml')

    # データパス設定（実際のS3パスに変更）
    s3_data_path = f's3://{tuner_manager.bucket}/bigfive/data/processed'

    # Stage 1チューニング実行
    print("\n[Stage 1] Nemotron-Personas-Japan")
    tuner_stage1 = tuner_manager.run_tuning(
        s3_data_path=s3_data_path,
        stage=1
    )

    if tuner_stage1:
        # 最良のハイパーパラメータ取得
        best_params_stage1 = tuner_manager.get_best_hyperparameters(tuner_stage1)

        # 結果分析
        df_stage1 = tuner_manager.analyze_tuning_results(tuner_stage1)

        # 結果をYAMLで保存
        import yaml
        with open('best_hyperparameters_stage1.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(best_params_stage1, f, default_flow_style=False)
        print("\n✓ 最良のパラメータを保存: best_hyperparameters_stage1.yaml")

    print("\n" + "=" * 60)
    print("チューニング完了！")
    print("=" * 60)


if __name__ == '__main__':
    # 設定確認後に実行
    print("\n注意:")
    print("- config.yamlで hyperparameter_tuning.enabled を true に設定")
    print("- AWS認証情報を設定")
    print("- データをS3にアップロード済みか確認")
    print("\n準備完了後、main()のコメントを解除して実行\n")

    # main()  # コメントを解除して実行
