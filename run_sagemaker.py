"""
AWS SageMaker実行スクリプト
ローカルで準備したデータをS3にアップロードし、SageMakerでトレーニング実行
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import boto3
import os
from datetime import datetime


class BigFiveSageMakerTrainer:
    """SageMaker学習実行クラス"""

    def __init__(
        self,
        role_arn=None,
        bucket_name=None,
        region='us-west-2'
    ):
        """
        Args:
            role_arn: SageMaker実行ロールARN（省略時は自動取得）
            bucket_name: S3バケット名（省略時は自動生成）
            region: AWSリージョン
        """
        self.region = region
        self.session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

        # 実行ロール取得
        if role_arn:
            self.role = role_arn
        else:
            try:
                self.role = get_execution_role()
            except:
                print("警告: SageMaker実行ロールを自動取得できませんでした")
                print("ローカル実行の場合、IAMロールのARNを明示的に指定してください")
                self.role = None

        # S3バケット設定
        if bucket_name:
            self.bucket = bucket_name
        else:
            self.bucket = self.session.default_bucket()

        print("=" * 60)
        print("SageMaker設定")
        print("=" * 60)
        print(f"Region: {self.region}")
        print(f"Bucket: {self.bucket}")
        print(f"Role: {self.role}")
        print("=" * 60)

    def upload_data_to_s3(self, local_data_dir, s3_prefix='bigfive/data'):
        """
        ローカルデータをS3にアップロード

        Args:
            local_data_dir: ローカルデータディレクトリ
            s3_prefix: S3プレフィックス

        Returns:
            S3データパス
        """
        print(f"\nデータアップロード中: {local_data_dir} -> s3://{self.bucket}/{s3_prefix}")

        s3_data_path = self.session.upload_data(
            path=local_data_dir,
            bucket=self.bucket,
            key_prefix=s3_prefix
        )

        print(f"✓ アップロード完了: {s3_data_path}")
        return s3_data_path

    def train_stage1(
        self,
        s3_data_path,
        instance_type='ml.g4dn.xlarge',
        instance_count=1,
        epochs=3,
        batch_size=8,
        learning_rate=2e-4
    ):
        """
        Stage 1学習実行（Nemotronデータ）

        Args:
            s3_data_path: S3データパス
            instance_type: SageMakerインスタンスタイプ
            instance_count: インスタンス数
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率

        Returns:
            学習済みモデルのS3パス
        """
        print("\n" + "=" * 60)
        print("Stage 1 学習開始（Nemotron-Personas-Japan）")
        print("=" * 60)

        estimator = PyTorch(
            entry_point='train.py',
            source_dir='.',
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='2.0.0',
            py_version='py310',
            hyperparameters={
                'stage': 1,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_name': 'xlm-roberta-large',
                'lora_r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1
            },
            output_path=f's3://{self.bucket}/bigfive/output/stage1',
            code_location=f's3://{self.bucket}/bigfive/code',
            sagemaker_session=self.session,
            keep_alive_period_in_seconds=1800,  # Warm pool有効化
        )

        # 学習実行
        estimator.fit({'train': s3_data_path})

        print(f"\n✓ Stage 1 学習完了")
        print(f"モデル保存先: {estimator.model_data}")

        return estimator.model_data

    def train_stage2(
        self,
        s3_data_path,
        stage1_model_path,
        instance_type='ml.g4dn.xlarge',
        instance_count=1,
        epochs=5,
        batch_size=8,
        learning_rate=1e-4
    ):
        """
        Stage 2学習実行（RealPersonaChatデータ）

        Args:
            s3_data_path: S3データパス
            stage1_model_path: Stage1モデルのS3パス
            instance_type: SageMakerインスタンスタイプ
            instance_count: インスタンス数
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率

        Returns:
            学習済みモデルのS3パス
        """
        print("\n" + "=" * 60)
        print("Stage 2 学習開始（RealPersonaChat）")
        print("=" * 60)

        estimator = PyTorch(
            entry_point='train.py',
            source_dir='.',
            role=self.role,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='2.0.0',
            py_version='py310',
            hyperparameters={
                'stage': 2,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_name': 'xlm-roberta-large',
                'lora_r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'stage1_model_path': stage1_model_path
            },
            output_path=f's3://{self.bucket}/bigfive/output/stage2',
            code_location=f's3://{self.bucket}/bigfive/code',
            sagemaker_session=self.session,
            keep_alive_period_in_seconds=1800,
        )

        # 学習実行
        estimator.fit({'train': s3_data_path})

        print(f"\n✓ Stage 2 学習完了")
        print(f"モデル保存先: {estimator.model_data}")

        return estimator.model_data


def main():
    """メイン実行例"""
    print("=" * 60)
    print("Big Five性格特性推定 - AWS SageMaker実行")
    print("=" * 60)

    # 設定（実際の環境に合わせて変更してください）
    ROLE_ARN = None  # 例: 'arn:aws:iam::123456789012:role/SageMakerRole'
    BUCKET_NAME = None  # 例: 'my-bigfive-bucket'
    REGION = 'us-west-2'

    # SageMakerトレーナー初期化
    trainer = BigFiveSageMakerTrainer(
        role_arn=ROLE_ARN,
        bucket_name=BUCKET_NAME,
        region=REGION
    )

    # データアップロード
    s3_data_path = trainer.upload_data_to_s3(
        local_data_dir='data/processed',
        s3_prefix='bigfive/data/processed'
    )

    # Stage 1学習
    stage1_model = trainer.train_stage1(
        s3_data_path=s3_data_path,
        instance_type='ml.g4dn.xlarge',  # GPU: NVIDIA T4
        epochs=3,
        batch_size=8,
        learning_rate=2e-4
    )

    # Stage 2学習
    stage2_model = trainer.train_stage2(
        s3_data_path=s3_data_path,
        stage1_model_path=stage1_model,
        instance_type='ml.g4dn.xlarge',
        epochs=5,
        batch_size=8,
        learning_rate=1e-4
    )

    print("\n" + "=" * 60)
    print("全学習完了！")
    print("=" * 60)
    print(f"Stage 1モデル: {stage1_model}")
    print(f"Stage 2モデル: {stage2_model}")
    print("\n次のステップ:")
    print("1. SageMaker Studioでログ・メトリクスを確認")
    print("2. モデルをデプロイしてエンドポイント作成")
    print("3. 推論テスト実行")


if __name__ == '__main__':
    # 実行前の確認
    print("\n注意:")
    print("- このスクリプトを実行する前に、AWS認証情報を設定してください")
    print("- SageMaker実行ロールのARNを設定してください")
    print("- S3バケット名を設定してください")
    print("\n設定後、main()のコメントアウトを解除して実行してください\n")

    # main()  # コメントアウトを解除して実行
