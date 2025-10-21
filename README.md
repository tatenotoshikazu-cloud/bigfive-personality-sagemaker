# Big Five性格特性推定プロジェクト

日本語会話データからBig Five性格特性（開放性、誠実性、外向性、協調性、神経症傾向）を高精度で推定するAIモデル

## プロジェクト概要

- **目的**: 日本語会話データからBig Five性格特性を推定
- **学習データ**: Nemotron-Personas-Japan + RealPersonaChat（二段階学習）
- **技術スタック**: AWS SageMaker + LoRA + xlm-roberta-large

## ディレクトリ構成

```
DL/
├── download_datasets.py      # データセット取得スクリプト
├── preprocess_data.py         # データ前処理スクリプト
├── train.py                   # SageMaker対応トレーニングスクリプト
├── run_sagemaker.py           # SageMaker実行スクリプト
├── requirements.txt           # 依存パッケージ
├── README.md                  # このファイル
└── data/                      # データディレクトリ（自動生成）
    ├── realpersonachat/       # RealPersonaChatデータ
    ├── nemotron/              # Nemotron-Personas-Japanデータ
    └── processed/             # 前処理済みデータ
```

## セットアップ

### 1. 依存パッケージインストール

```bash
pip install -r requirements.txt
```

### 2. データセット取得

```bash
python download_datasets.py
```

**取得されるデータセット:**
- **RealPersonaChat**: 約14,000件の日本語対話データ + Big Five特性ラベル
- **Nemotron-Personas-Japan**: 1M-10M件の日本語ペルソナ合成データ

### 3. データ前処理

```bash
python preprocess_data.py
```

**処理内容:**
- 会話データをモデル入力形式に変換
- Big Five特性ラベルの抽出
- Stage 1（Nemotron）/ Stage 2（RealPersonaChat）用にデータ分割

## ローカル学習（オプション）

SageMaker実行前にローカルでテストする場合：

```bash
# Stage 1学習
python train.py --stage 1 --epochs 3 --batch_size 8

# Stage 2学習
python train.py --stage 2 --epochs 5 --batch_size 8 --stage1_model_path output/final_model
```

## ✅ ローカル動作確認済み

以下のコードは**ローカル環境で動作確認済み**です：
- ✅ データセット読み込み（RealPersonaChat: 500件）
- ✅ Stage 1トレーニング（Nemotron補助タスク学習）
- ✅ マルチタスク学習（年齢・性別・職業予測）
- ✅ LoRA + xlm-roberta-large モデル

**ローカルテストコマンド:**
```bash
# Stage 1学習（動作確認）
python train_stage1_local.py

# Stage 2学習（動作確認）
python train_stage2_local.py
```

## 🚀 AWS SageMaker実行（本番）

ローカルで動作確認したコードをSageMaker上で実行します。

**詳細は [SAGEMAKER_GUIDE.md](SAGEMAKER_GUIDE.md) を参照してください。**

### クイックスタート

1. **AWS認証設定**
   ```bash
   aws configure
   ```

2. **設定ファイル編集**

   [run_sagemaker.py](run_sagemaker.py) の206-209行目を編集：
   ```python
   ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE'
   BUCKET_NAME = 'your-bigfive-bucket'
   REGION = 'us-west-2'
   ```

3. **SageMaker実行**
   ```python
   # run_sagemaker.py の262行目のコメントアウトを解除
   python run_sagemaker.py
   ```

### 実行フロー

1. ローカルデータ → S3アップロード
2. **Stage 1学習**（Nemotron補助タスク）
   - 推定時間: 3-5時間
   - コスト: 約$2.21-$3.68
3. **Stage 2学習**（Big Five予測）
   - 推定時間: 4-6時間
   - コスト: 約$2.94-$4.42

### 推奨インスタンス

- **ml.g4dn.xlarge** (推奨): NVIDIA T4 GPU、$0.736/時間
- ml.p3.2xlarge: NVIDIA V100 GPU、高速だが高コスト
- ml.g5.xlarge: NVIDIA A10G GPU、最新世代

## 学習パラメータ

### LoRA設定

```python
lora_r = 16              # LoRAのランク
lora_alpha = 32          # LoRAのスケーリング係数
lora_dropout = 0.1       # Dropout率
```

### 学習設定

**Stage 1（Nemotron）:**
- Epochs: 3
- Batch Size: 8
- Learning Rate: 2e-4

**Stage 2（RealPersonaChat）:**
- Epochs: 5
- Batch Size: 8
- Learning Rate: 1e-4（Stage 1より低め）

## Big Five特性

モデルは以下の5次元を推定します：

1. **Openness（開放性）**: 新しい経験への開放度
2. **Conscientiousness（誠実性）**: 計画性・責任感
3. **Extraversion（外向性）**: 社交性・活発性
4. **Agreeableness（協調性）**: 協調性・思いやり
5. **Neuroticism（神経症傾向）**: 情緒不安定性

## 評価指標

- **MSE（Mean Squared Error）**: 平均二乗誤差
- **MAE（Mean Absolute Error）**: 平均絶対誤差
- 各Big Five次元ごとのMAE

## コスト見積もり（AWS SageMaker）

### Stage 1学習（Nemotron）
- インスタンス: ml.g4dn.xlarge
- 学習時間: 約6-10時間（データ量に依存）
- コスト: 約$5-8

### Stage 2学習（RealPersonaChat）
- インスタンス: ml.g4dn.xlarge
- 学習時間: 約2-3時間
- コスト: 約$2-3

**合計見積もり**: $7-11（1回の完全学習）

## トラブルシューティング

### データセット取得エラー

```python
# ストリーミングモードで取得（メモリ節約）
dataset = load_dataset("nvidia/Nemotron-Personas-Japan", streaming=True)
```

### GPU メモリ不足

```python
# バッチサイズを減らす
--batch_size 4

# Gradient Accumulationを使用
--gradient_accumulation_steps 2
```

### SageMaker認証エラー

1. IAMロールに以下のポリシーをアタッチ：
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess

2. 信頼関係に `sagemaker.amazonaws.com` を追加

## 次のステップ

1. **モデル評価**: テストデータでの性能評価
2. **ハイパーパラメータチューニング**: SageMaker Automatic Model Tuning
3. **デプロイ**: SageMakerエンドポイント作成
4. **推論API**: Lambda + API Gateway構成

## 参考文献

- **RealPersonaChat**: Yamashita et al. (2023) "RealPersonaChat: A Realistic Persona Chat Corpus with Interlocutors' Own Personalities"
- **Nemotron-Personas-Japan**: NVIDIA NeMo Data Designer
- **xlm-roberta-large**: Conneau et al. (2020) "Unsupervised Cross-lingual Representation Learning at Scale"
- **LoRA**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"

## ライセンス

- RealPersonaChat: MIT License
- Nemotron-Personas-Japan: CC BY 4.0
- xlm-roberta-large: MIT License

## お問い合わせ

プロジェクトに関する質問や提案は、Issueまでお願いします。
