# Big Five性格特性推定 - クイックスタートガイド

## ローカルテスト → SageMaker本格実行の流れ

### ステップ0: 環境準備

```bash
# 依存パッケージインストール
pip install -r requirements.txt
```

---

## Phase 1: ローカルテスト（小規模データ）

### ステップ1: データ構造確認

```bash
python inspect_datasets.py
```

**目的**: データセットの構造・フォーマットを確認
**出力**:
- `dataset_structure_report.txt` - データ構造レポート
- `DATA_FORMAT_GUIDE.md` - データ形式ガイド

**確認項目**:
- ✅ RealPersonaChatにBig Five特性が含まれている
- ✅ 話者IDで対話データと特性データを紐付けられる
- ✅ Nemotronデータの構造を理解

---

### ステップ2: 小規模データセット作成

```bash
python create_small_dataset.py
```

**目的**: ローカルテスト用の小規模データ作成
**処理内容**:
- RealPersonaChatから100件抽出
- Nemotron-Personas-Japanから1000件抽出
- 前処理・train/val分割

**出力**:
```
data/small/
├── realpersonachat/
├── nemotron/
└── processed/
    ├── stage1_train/
    ├── stage1_val/
    ├── stage2_train/
    └── stage2_val/
```

**実行時間**: 約2-5分

---

### ステップ3: ローカル学習テスト

```bash
python train_local.py
```

**目的**: コードの動作確認（SageMaker実行前）
**テスト項目**:
1. データ読み込み
2. モデル初期化（xlm-roberta-base + LoRA）
3. Dataset作成
4. 簡易学習（1エポック、20サンプル）

**注意**:
- xlm-roberta-**base**を使用（軽量版）
- CPU環境では5-10分程度
- GPU環境では1-2分程度

**成功条件**:
```
✓ 成功 - data_loading
✓ 成功 - model_init
✓ 成功 - dataset_creation
✓ 成功 - quick_training
```

---

## Phase 2: フルデータセット準備

### ステップ4: フルデータセットダウンロード

```bash
python download_datasets.py
```

**処理内容**:
- RealPersonaChat全件（約14,000件）
- Nemotron-Personas-Japan全件（1M-10M件）

**出力**:
```
data/
├── realpersonachat/
│   ├── dialogue/
│   └── interlocutor/
└── nemotron/
    └── personas_japan/
```

**実行時間**: 10-30分（Nemotronが大容量）

**ディスク容量**: 約5-10GB必要

---

### ステップ5: フルデータ前処理

```bash
python preprocess_data.py
```

**処理内容**:
- 会話テキスト抽出
- Big Five特性ラベル統合
- Stage 1/2用にデータ分割

**出力**:
```
data/processed/
├── stage1_train/
├── stage1_val/
├── stage2_train/
└── stage2_val/
```

**実行時間**: 5-15分

---

## Phase 3: AWS SageMaker実行

### ステップ6: AWS認証設定

```bash
aws configure
```

または環境変数:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

---

### ステップ7: SageMaker実行設定

`run_sagemaker.py` を編集:

```python
# 28-30行目あたり
ROLE_ARN = 'arn:aws:iam::123456789012:role/YourSageMakerRole'
BUCKET_NAME = 'your-s3-bucket-name'
REGION = 'us-west-2'
```

**必要なIAMポリシー**:
- AmazonSageMakerFullAccess
- AmazonS3FullAccess

---

### ステップ8: SageMaker学習実行

```python
python run_sagemaker.py
```

**処理フロー**:
1. データをS3にアップロード
2. **Stage 1学習**: Nemotron-Personas-Japan（3エポック）
3. **Stage 2学習**: RealPersonaChat + Stage 1重み（5エポック）

**実行時間**:
- Stage 1: 6-10時間
- Stage 2: 2-3時間
- **合計**: 8-13時間

**コスト見積もり**（ml.g4dn.xlarge使用）:
- Stage 1: $5-8
- Stage 2: $2-3
- **合計**: $7-11

---

## （オプション）Phase 4: ハイパーパラメータチューニング

### config.yaml編集

```yaml
hyperparameter_tuning:
  enabled: true  # falseからtrueに変更
  max_jobs: 20
  max_parallel_jobs: 2
```

### チューニング実行

```python
python hyperparameter_tuning.py
```

**処理内容**:
- Bayesian最適化で最適なハイパーパラメータ探索
- 探索範囲: learning_rate, lora_r, lora_alpha, batch_size, weight_decay

**実行時間**: 1-3日（max_jobs=20の場合）

**コスト**: $100-200（max_jobs=20、並列2の場合）

**出力**:
- `best_hyperparameters_stage1.yaml` - 最良のパラメータ
- `hyperparameter_tuning_results.png` - 結果グラフ

---

## トラブルシューティング

### データダウンロードエラー

**症状**: HuggingFaceからダウンロードできない

**解決策**:
```bash
# HuggingFace CLIでログイン
huggingface-cli login
```

### GPU メモリ不足（SageMaker）

**症状**: CUDA out of memory

**解決策**:
```python
# run_sagemaker.py でバッチサイズを減らす
batch_size=4  # 8から4に
```

または

```python
# インスタンスタイプをアップグレード
instance_type='ml.p3.2xlarge'  # V100 GPU (16GB)
```

### 学習が進まない

**症状**: Loss が下がらない

**解決策**:
1. Learning Rateを調整（1e-5 ～ 5e-4の範囲で実験）
2. LoRA rankを増やす（r=32, 64）
3. Gradient Accumulationを使用

---

## 実行チェックリスト

### ローカルテスト完了
- [ ] `inspect_datasets.py` 実行完了
- [ ] `create_small_dataset.py` 実行完了
- [ ] `train_local.py` で全テスト成功
- [ ] エラーがないことを確認

### SageMaker準備完了
- [ ] AWS認証設定完了
- [ ] SageMaker実行ロール作成
- [ ] S3バケット作成
- [ ] `run_sagemaker.py` のROLE_ARN/BUCKET設定完了
- [ ] フルデータセットダウンロード完了
- [ ] フルデータ前処理完了

### SageMaker実行
- [ ] データS3アップロード完了
- [ ] Stage 1学習開始
- [ ] Stage 1学習完了（モデルS3保存確認）
- [ ] Stage 2学習開始
- [ ] Stage 2学習完了（最終モデル確認）

---

## 推奨ワークフロー

### 初回実行（確実に動作確認）

```bash
# 1. データ構造確認
python inspect_datasets.py

# 2. 小規模データでローカルテスト
python create_small_dataset.py
python train_local.py

# 3. 全テスト成功したらフルデータ準備
python download_datasets.py
python preprocess_data.py

# 4. SageMaker実行
python run_sagemaker.py
```

### 2回目以降（データ準備済み）

```bash
# ハイパーパラメータを変更して再学習
# config.yaml を編集
python run_sagemaker.py
```

---

## 成果物

### 学習済みモデル
- S3: `s3://your-bucket/bigfive/output/stage2/model.tar.gz`
- ローカル: `output/final_model/`

### 評価結果
- TensorBoard ログ: `output/logs/`
- 評価指標: `eval_results.json`

### チューニング結果（オプション）
- 最良パラメータ: `best_hyperparameters_stage1.yaml`
- 結果グラフ: `hyperparameter_tuning_results.png`

---

## サポート・質問

問題が発生した場合:
1. エラーメッセージを確認
2. `dataset_structure_report.txt` でデータ構造を再確認
3. `train_local.py` でローカルテスト実行
4. ログファイルを確認

---

## 次のステップ（学習完了後）

1. **モデル評価**: テストデータでの性能評価
2. **デプロイ**: SageMakerエンドポイント作成
3. **推論API**: Lambda + API Gateway構成
4. **継続的改善**: 新しいデータで再学習
