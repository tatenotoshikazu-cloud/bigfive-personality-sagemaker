# Big Five Personality Traits Datasets on Hugging Face Hub

## Overview
このガイドでは、Hugging Face Hubで利用可能なBig Five性格特性（OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism）のデータセットを紹介します。

---

## 1. jingjietan/essays-big5 (推奨)

### 基本情報
- **サイズ**: 2,470件のエッセイ
- **ライセンス**: Apache 2.0
- **言語**: 英語
- **タイプ**: 個人的なエッセイ（意識の流れスタイル）

### データ構造
| フィールド | 型 | 説明 |
|-----------|-----|------|
| text | string | エッセイテキスト (217-12,900文字) |
| O | binary (0/1) | Openness (開放性) |
| C | binary (0/1) | Conscientiousness (誠実性) |
| E | binary (0/1) | Extraversion (外向性) |
| A | binary (0/1) | Agreeableness (協調性) |
| N | binary (0/1) | Neuroticism (神経症的傾向) |
| ptype | string | 性格タイプ (32カテゴリ) |

### データ分割
- Train: 1,580サンプル
- Validation: 395サンプル
- Test: 494サンプル

### アクセス方法
```python
from datasets import load_dataset

# データセットの読み込み
dataset = load_dataset("jingjietan/essays-big5")

# 例: 訓練データの確認
print(dataset['train'][0])
# Output:
# {
#   'text': 'Well, right now I'm finishing up my senior year...',
#   'O': 1,
#   'C': 0,
#   'E': 1,
#   'A': 1,
#   'N': 0,
#   'ptype': 'OEAC'
# }

# 統計情報
print(f"Train size: {len(dataset['train'])}")
print(f"Val size: {len(dataset['validation'])}")
print(f"Test size: {len(dataset['test'])}")
```

### 特徴
- **長所**:
  - エッセイ形式で自然な文章
  - バイナリラベルでシンプル
  - 適度なサイズで扱いやすい
  - Apache 2.0ライセンスで商用利用可能

- **用途**:
  - テキストから性格予測モデルの訓練
  - 性格と言語パターンの分析
  - 小規模プロジェクトに最適

---

## 2. jingjietan/pandora-big5 (大規模データセット)

### 基本情報
- **サイズ**: 3,006,566件のRedditコメント
- **ライセンス**: Apache 2.0
- **言語**: 英語
- **タイプ**: ソーシャルメディアテキスト（Reddit）

### データ構造
| フィールド | 型 | 説明 |
|-----------|-----|------|
| text | string | Redditコメント (1-48,000文字) |
| O | float64 | Openness (0-100スコア) |
| C | float64 | Conscientiousness (0-100スコア) |
| E | float64 | Extraversion (0-99スコア) |
| A | float64 | Agreeableness (0-99スコア) |
| N | float64 | Neuroticism (0-100スコア) |
| ptype | int64 | 性格タイプ (0-31) |

### データ分割
- Train: 1,923,402サンプル
- Validation: 481,565サンプル
- Test: 601,599サンプル

### アクセス方法
```python
from datasets import load_dataset

# データセットの読み込み
dataset = load_dataset("jingjietan/pandora-big5")

# 例: 訓練データの確認
print(dataset['train'][0])
# Output:
# {
#   'text': 'I think this is a great idea...',
#   'O': 67.5,
#   'C': 45.2,
#   'E': 78.9,
#   'A': 56.3,
#   'N': 34.1,
#   'ptype': 15
# }

# サンプルサイズの確認
print(f"Total samples: {sum([len(dataset[split]) for split in dataset.keys()])}")
```

### 特徴
- **長所**:
  - 非常に大規模（300万件以上）
  - 連続値スコア（0-100）でより詳細
  - Redditの多様な会話データ
  - ディープラーニングに適したサイズ

- **短所**:
  - サイズが大きい（511 MB）
  - ダウンロード・処理に時間がかかる

- **用途**:
  - 大規模言語モデルのファインチューニング
  - ソーシャルメディア上の性格分析
  - 深層学習プロジェクト

---

## 3. Fatima0923/Automated-Personality-Prediction

### 基本情報
- **サイズ**: 20,877件のRedditコメント
- **ライセンス**: 不明（PANDORAのサブセット）
- **言語**: 英語（前処理済み）
- **タイプ**: Redditコメント（1,608ユーザー）

### データ構造
| フィールド | 型 | 説明 |
|-----------|-----|------|
| text | string | Redditコメント (16-8,880文字) |
| agreeableness | float | 協調性 (0-99) |
| openness | float | 開放性 (9-98) |
| conscientiousness | float | 誠実性 (1-98) |
| extraversion | float | 外向性 (0-99) |
| neuroticism | float | 神経症的傾向 (0-99) |

### データ分割
- Train: 16,000サンプル
- Validation: 2,420サンプル
- Test: 2,420サンプル

### アクセス方法
```python
from datasets import load_dataset

# データセットの読み込み
dataset = load_dataset("Fatima0923/Automated-Personality-Prediction")

# 例: 訓練データの確認
print(dataset['train'][0])
# Output:
# {
#   'text': 'This is an interesting discussion...',
#   'agreeableness': 65.3,
#   'openness': 72.1,
#   'conscientiousness': 58.9,
#   'extraversion': 45.6,
#   'neuroticism': 38.2
# }
```

### 特徴
- **長所**:
  - 中規模で扱いやすい
  - Reddit特有の会話スタイル
  - PANDORAデータセットの厳選版

- **用途**:
  - 中規模プロジェクト
  - Redditスタイルのテキスト分析
  - 性格予測の実験

---

## 4. google/Synthetic-Persona-Chat (会話データ)

### 基本情報
- **サイズ**: 20,000件の会話
- **ライセンス**: CC-BY-4.0
- **言語**: 英語
- **タイプ**: 合成会話データ（ペルソナベース）

### データ構造
| フィールド | 型 | 説明 |
|-----------|-----|------|
| user_1_personas | string | ユーザー1のペルソナ記述 (52-391文字) |
| user_2_personas | string | ユーザー2のペルソナ記述 (49-391文字) |
| Best_Generated_Conversation | string | 会話テキスト (388-3,460文字) |

### データ分割
- Train: 8,940サンプル
- Validation: 1,000サンプル
- Test: 968サンプル

### アクセス方法
```python
from datasets import load_dataset

# データセットの読み込み
dataset = load_dataset("google/Synthetic-Persona-Chat")

# 例: 訓練データの確認
print(dataset['train'][0])
# Output:
# {
#   'user_1_personas': 'I love outdoor activities and hiking...',
#   'user_2_personas': 'I enjoy reading books and quiet evenings...',
#   'Best_Generated_Conversation': 'User1: Hi! ...'
# }
```

### 特徴
- **注意**: Big Fiveの明示的なラベルなし
- **長所**:
  - ペルソナベースの会話
  - 対話生成に有用
  - Googleが提供

- **用途**:
  - ペルソナベースの対話生成
  - 会話AI開発
  - Big Fiveラベルは別途付与が必要

---

## 5. その他の関連リソース

### モデル

#### Minej/bert-base-personality
Big Five予測用の事前学習済みBERTモデル

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

text = "I love exploring new ideas and being creative..."
inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.squeeze().detach().numpy()

print(f"Openness: {predictions[0]:.2f}")
print(f"Conscientiousness: {predictions[1]:.2f}")
print(f"Extraversion: {predictions[2]:.2f}")
print(f"Agreeableness: {predictions[3]:.2f}")
print(f"Neuroticism: {predictions[4]:.2f}")
```

#### vladinc/bigfive-regression-model
essays-big5データセットで訓練された回帰モデル

### Hugging Face Spaces
- **thoucentric/Big-Five-Personality-Traits-Detection**: オンラインデモ

---

## データセット比較表

| データセット | サイズ | テキストタイプ | ラベル形式 | 推奨用途 |
|------------|--------|--------------|-----------|----------|
| jingjietan/essays-big5 | 2,470 | エッセイ | Binary (0/1) | 小規模プロジェクト、実験 |
| jingjietan/pandora-big5 | 3.0M | Reddit | Float (0-100) | 大規模DL、ファインチューニング |
| Fatima0923/Automated-Personality-Prediction | 20,877 | Reddit | Float (0-99) | 中規模プロジェクト |
| google/Synthetic-Persona-Chat | 20,000 | 会話 | ラベルなし | 対話生成 |

---

## 推奨データセット選択ガイド

### 用途別推奨

1. **小規模プロジェクト・学習目的**
   - `jingjietan/essays-big5`
   - 理由: 扱いやすいサイズ、シンプルなラベル

2. **大規模モデルのファインチューニング**
   - `jingjietan/pandora-big5`
   - 理由: 300万件のデータ、連続値スコア

3. **ソーシャルメディア分析**
   - `Fatima0923/Automated-Personality-Prediction`
   - 理由: Redditデータ、中規模

4. **会話AI開発**
   - `google/Synthetic-Persona-Chat`
   - 理由: ペルソナベース会話データ

---

## 完全な使用例

### 例1: essays-big5でBig Five予測モデルを訓練

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# データセットの読み込み
dataset = load_dataset("jingjietan/essays-big5")

# 訓練データとテストデータの準備
train_texts = dataset['train']['text']
train_labels = np.array([
    dataset['train']['O'],
    dataset['train']['C'],
    dataset['train']['E'],
    dataset['train']['A'],
    dataset['train']['N']
]).T

test_texts = dataset['test']['text']
test_labels = np.array([
    dataset['test']['O'],
    dataset['test']['C'],
    dataset['test']['E'],
    dataset['test']['A'],
    dataset['test']['N']
]).T

# TF-IDF特徴量抽出
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 各性格特性ごとにモデルを訓練
trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
models = {}

for i, trait in enumerate(trait_names):
    print(f"\nTraining {trait} classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, train_labels[:, i])
    models[trait] = model

    # 評価
    predictions = model.predict(X_test)
    print(f"\n{trait} Results:")
    print(classification_report(test_labels[:, i], predictions))

# 新しいテキストで予測
new_text = ["I love meeting new people and trying new experiences. I'm always open to creative ideas."]
new_features = vectorizer.transform(new_text)

print("\n=== Personality Prediction for New Text ===")
for trait, model in models.items():
    prediction = model.predict(new_features)[0]
    proba = model.predict_proba(new_features)[0]
    print(f"{trait}: {prediction} (confidence: {proba[prediction]:.2%})")
```

### 例2: pandora-big5でディープラーニングモデル

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np

# データセットの読み込み（サンプル版）
dataset = load_dataset("jingjietan/pandora-big5", split="train[:10000]")  # 最初の10,000件

# モデルとトークナイザーの準備
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# データの前処理
def preprocess_function(examples):
    # テキストのトークン化
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    # Big Fiveスコアを正規化（0-1範囲）
    tokenized['labels'] = [
        [
            examples['O'][i] / 100.0,
            examples['C'][i] / 100.0,
            examples['E'][i] / 100.0,
            examples['A'][i] / 100.0,
            examples['N'][i] / 100.0
        ]
        for i in range(len(examples['text']))
    ]
    return tokenized

# データセットの前処理
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 訓練設定
training_args = TrainingArguments(
    output_dir="./big5_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainerの作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 訓練
trainer.train()

# 新しいテキストで予測
def predict_personality(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits).detach().numpy()[0]

    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    results = {trait: float(score * 100) for trait, score in zip(traits, predictions)}
    return results

# テスト
test_text = "I really enjoy spending time alone, reading books and thinking deeply about life."
print(predict_personality(test_text))
```

### 例3: データセットの統計分析

```python
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データセットの読み込み
dataset = load_dataset("jingjietan/essays-big5")

# DataFrameに変換
df = pd.DataFrame(dataset['train'])

# 性格特性の分布
traits = ['O', 'C', 'E', 'A', 'N']
trait_sums = df[traits].sum()

# 可視化
plt.figure(figsize=(12, 5))

# 1. 性格特性の出現頻度
plt.subplot(1, 2, 1)
trait_sums.plot(kind='bar')
plt.title('Big Five Trait Distribution')
plt.xlabel('Personality Traits')
plt.ylabel('Count (High = 1)')
plt.xticks(range(5), ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'], rotation=45)

# 2. 性格タイプの分布（上位10）
plt.subplot(1, 2, 2)
ptype_counts = df['ptype'].value_counts().head(10)
ptype_counts.plot(kind='bar')
plt.title('Top 10 Personality Types')
plt.xlabel('Personality Type')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('big5_distribution.png')
plt.show()

# テキスト長の統計
df['text_length'] = df['text'].apply(len)
print("\n=== Text Length Statistics ===")
print(df['text_length'].describe())

# 性格特性間の相関
print("\n=== Trait Correlations ===")
print(df[traits].corr())
```

---

## トラブルシューティング

### よくあるエラーと解決策

1. **`DatasetNotFoundError`**
   ```python
   # ❌ 間違い
   dataset = load_dataset("DavidIRL/real-persona-chat")  # 存在しない

   # ✅ 正しい
   dataset = load_dataset("jingjietan/essays-big5")
   ```

2. **メモリエラー（pandora-big5が大きすぎる）**
   ```python
   # 一部のみ読み込み
   dataset = load_dataset("jingjietan/pandora-big5", split="train[:10000]")

   # またはストリーミング
   dataset = load_dataset("jingjietan/pandora-big5", streaming=True)
   ```

3. **ラベル形式の違い**
   ```python
   # essays-big5: バイナリ (0/1)
   # pandora-big5: 連続値 (0-100)
   # 必要に応じて正規化

   # バイナリ → 連続値
   score = binary_label * 100  # 0 → 0, 1 → 100

   # 連続値 → バイナリ（閾値50）
   binary = 1 if score >= 50 else 0
   ```

---

## 参考リンク

- **Hugging Face Datasets Documentation**: https://huggingface.co/docs/datasets
- **Big Five Personality Traits Collection**: https://huggingface.co/collections/DmitryRyumin/big-five-personality-traits-661fb545292ab3d12a5a4890
- **PANDORA Dataset Paper**: https://arxiv.org/abs/2004.04460
- **Big5-Chat Paper**: https://arxiv.org/abs/2410.16491

---

## まとめ

Hugging Face Hubで利用可能なBig Fiveデータセットは以下の通りです：

### 最も推奨するデータセット
1. **jingjietan/essays-big5** - 小〜中規模プロジェクトに最適
2. **jingjietan/pandora-big5** - 大規模ディープラーニングに最適
3. **Fatima0923/Automated-Personality-Prediction** - Redditデータ分析に最適

これらはすべて実際にアクセス可能で、Big Fiveのラベルが付いており、テキストデータを含んでいます。

**プロジェクトの規模とリソースに応じて適切なデータセットを選択してください。**
