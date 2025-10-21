# AWS SageMaker実行ガイド

このドキュメントは、ローカルで動作確認したコードをAWS SageMaker上で実行する手順を説明します。

## ✅ ローカルでの動作確認完了

以下のコードはローカル環境で動作確認済みです：
- ✅ データセット読み込み（RealPersonaChat）
- ✅ Stage 1トレーニング（Nemotron補助タスク学習）
- ✅ Stage 2トレーニング（Big Five予測）
- ✅ LoRA + xlm-roberta-large モデル
- ✅ マルチタスク学習（年齢・性別・職業予測）

## 📋 前提条件

### 1. AWS環境の準備

```bash
# AWS CLIのインストール（未インストールの場合）
pip install awscli

# AWS認証情報の設定
aws configure
# - Access Key ID を入力
# - Secret Access Key を入力
# - Default region を入力（例: us-west-2）
# - Default output format を入力（例: json）
```

### 2. SageMaker実行ロールの作成

AWS IAMコンソールで以下の権限を持つロールを作成：
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`（または必要最小限のS3権限）

ロールARN例: `arn:aws:iam::123456789012:role/SageMakerExecutionRole`

### 3. S3バケットの作成

```bash
# S3バケット作成
aws s3 mb s3://your-bigfive-bucket --region us-west-2
```

## 🚀 SageMaker実行手順

### ステップ1: 設定ファイルの編集

[run_sagemaker.py](run_sagemaker.py) の設定を編集：

```python
# 206-209行目を編集
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE'
BUCKET_NAME = 'your-bigfive-bucket'
REGION = 'us-west-2'
```

### ステップ2: データの準備

ローカルでデータを準備（既に完了している場合はスキップ）：

```bash
# データセットダウンロード
python download_datasets.py

# データ前処理（Nemotronデータ作成）
python setup_local_data.py
```

### ステップ3: SageMaker実行

```bash
# run_sagemaker.pyの262行目のコメントアウトを解除
# main()  # ← このコメントを外す

# 実行
python run_sagemaker.py
```

実行されるフロー:
1. ローカルの `data/local/processed/` をS3にアップロード
2. SageMaker Stage 1学習開始（Nemotron補助タスク）
   - インスタンス: ml.g4dn.xlarge (NVIDIA T4 GPU)
   - エポック: 3
   - バッチサイズ: 8
   - 推定時間: 約3-5時間
3. SageMaker Stage 2学習開始（Big Five予測）
   - Stage 1モデルをロード
   - エポック: 5
   - 推定時間: 約4-6時間

## 📊 学習モニタリング

### SageMaker Studioで確認

1. AWS Management Console → SageMaker Studio
2. 左メニュー「Training」→「Training jobs」
3. 実行中のジョブをクリック
4. 「CloudWatch logs」タブでログ確認
5. 「Metrics」タブでloss/accuracyグラフ確認

### コマンドラインで確認

```bash
# 学習ジョブ一覧
aws sagemaker list-training-jobs --region us-west-2

# 学習ジョブ詳細
aws sagemaker describe-training-job \
    --training-job-name <job-name> \
    --region us-west-2
```

## 💰 コスト見積もり

### ml.g4dn.xlarge（推奨）
- **料金**: $0.736/時間
- **GPU**: NVIDIA T4 (16GB)
- **Stage 1**: 約3-5時間 → **約$2.21-$3.68**
- **Stage 2**: 約4-6時間 → **約$2.94-$4.42**
- **合計**: **約$5-$8**

### コスト削減オプション

1. **Spot Instanceの使用**（最大70%割引）
   ```python
   # config.yamlで設定
   sagemaker:
     use_spot_instances: true
     max_wait_time: 86400  # 24時間
   ```

2. **Warm Pool使用**（既に有効化済み）
   - Stage 1 → Stage 2の切り替え時にインスタンス再利用
   - 起動待機時間を削減

## 🔧 トラブルシューティング

### エラー: 「ResourceLimitExceeded」

**原因**: インスタンスタイプのクォータ不足

**解決策**:
1. Service Quotas コンソールで「SageMaker」→「ml.g4dn.xlarge for training job usage」のクォータ引き上げリクエスト
2. または、より小さいインスタンス（ml.g4dn.xlarge → ml.g4dn.2xlarge）を試す

### エラー: 「AccessDeniedException」

**原因**: IAMロールの権限不足

**解決策**:
1. SageMaker実行ロールに `AmazonSageMakerFullAccess` を付与
2. S3バケットへのアクセス権限を確認

### 学習が途中で止まる

**原因**: メモリ不足 or GPUメモリ不足

**解決策**:
```python
# config.yamlでバッチサイズを削減
stage1:
  batch_size: 4  # 8 → 4に削減
  gradient_accumulation_steps: 8  # 4 → 8に増加（実質バッチサイズは維持）
```

## 📤 学習済みモデルの取得

### S3から手動ダウンロード

```bash
# Stage 1モデル
aws s3 cp s3://your-bigfive-bucket/bigfive/output/stage1/model.tar.gz ./models/

# Stage 2モデル
aws s3 cp s3://your-bigfive-bucket/bigfive/output/stage2/model.tar.gz ./models/

# 解凍
tar -xzf models/model.tar.gz -C models/
```

### Pythonで取得

```python
import boto3
s3 = boto3.client('s3')

# モデルダウンロード
s3.download_file(
    'your-bigfive-bucket',
    'bigfive/output/stage2/model.tar.gz',
    'models/stage2_model.tar.gz'
)
```

## 🎯 次のステップ

### 1. モデルのデプロイ

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data='s3://your-bucket/bigfive/output/stage2/model.tar.gz',
    role=role_arn,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py'  # 推論スクリプト（別途作成）
)

# エンドポイント作成
predictor = model.deploy(
    instance_type='ml.m5.xlarge',
    initial_instance_count=1
)
```

### 2. 推論テスト

```python
# テキストから性格特性を予測
test_text = "こんにちは！私は新しいことに挑戦するのが好きです。"
result = predictor.predict(test_text)

print("Big Five予測:")
print(f"  開放性: {result['openness']:.2f}")
print(f"  誠実性: {result['conscientiousness']:.2f}")
print(f"  外向性: {result['extraversion']:.2f}")
print(f"  協調性: {result['agreeableness']:.2f}")
print(f"  神経症傾向: {result['neuroticism']:.2f}")
```

### 3. ハイパーパラメータチューニング（オプション）

より高精度なモデルを求める場合、SageMaker Automatic Model Tuningを使用：

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

hyperparameter_ranges = {
    'learning_rate': ContinuousParameter(1e-5, 5e-4, scaling_type='Logarithmic'),
    'lora_r': IntegerParameter(8, 64),
    'batch_size': CategoricalParameter([4, 8, 16])
}

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name='validation:mae',
    objective_type='Minimize',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=2,
    strategy='Bayesian'
)

tuner.fit({'train': s3_data_path})
```

## 📚 参考リンク

- [AWS SageMaker公式ドキュメント](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [PyTorch on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)
- [SageMaker料金](https://aws.amazon.com/jp/sagemaker/pricing/)

---

**作成日**: 2025-10-22
**ステータス**: ✅ ローカル動作確認済み、SageMaker実行準備完了
