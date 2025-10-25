# Big Five Personality Datasets - Quick Start Guide

## TL;DR - 最も重要な情報

### ✅ 正しいデータセット名（実際にアクセス可能）

| データセット名 | サイズ | 用途 |
|--------------|--------|------|
| **jingjietan/essays-big5** | 2,470 | 小規模プロジェクト・学習 ✨ おすすめ |
| **jingjietan/pandora-big5** | 3,006,566 | 大規模DL・ファインチューニング |
| **Fatima0923/Automated-Personality-Prediction** | 20,877 | 中規模プロジェクト |

### ❌ 存在しないデータセット（使用不可）
- `DavidIRL/real-persona-chat` ← 存在しません
- その他、検証されていない名前

---

## 🚀 クイックスタート（30秒で始める）

### 最小コード例

```python
from datasets import load_dataset

# 最も使いやすいデータセット（推奨）
dataset = load_dataset("jingjietan/essays-big5")

# データの確認
print(dataset['train'][0])
# {'text': 'Well, right now I\'m finishing up my senior year...',
#  'O': 1, 'C': 0, 'E': 1, 'A': 1, 'N': 0, 'ptype': 'OEAC'}
```

### データセットの構造

```python
# essays-big5
{
    'text': str,           # エッセイテキスト
    'O': int (0/1),       # Openness（開放性）
    'C': int (0/1),       # Conscientiousness（誠実性）
    'E': int (0/1),       # Extraversion（外向性）
    'A': int (0/1),       # Agreeableness（協調性）
    'N': int (0/1),       # Neuroticism（神経症的傾向）
    'ptype': str          # 性格タイプ（例: "OEAC"）
}

# pandora-big5
{
    'text': str,           # Redditコメント
    'O': float (0-100),   # Opennessスコア
    'C': float (0-100),   # Conscientiousnessスコア
    'E': float (0-100),   # Extraversionスコア
    'A': float (0-100),   # Agreeablenessスコア
    'N': float (0-100),   # Neuroticismスコア
    'ptype': int (0-31)   # 性格タイプID
}
```

---

## 📁 このフォルダのファイル

1. **big5_datasets_guide.md** - 完全ガイド（詳細な説明・使用例）
2. **load_big5_datasets.py** - データセット読み込みスクリプト
3. **compare_datasets.py** - データセット比較スクリプト
4. **BIG5_DATASETS_README.md** - このファイル（クイックスタート）

---

## 🎯 用途別おすすめデータセット

### 1. 学習・小規模プロジェクト
```python
# jingjietan/essays-big5 を使用
dataset = load_dataset("jingjietan/essays-big5")

# メリット:
# ✓ ダウンロード速度: 高速（< 10秒）
# ✓ サイズ: 小さい（~5 MB）
# ✓ ラベル: シンプルなバイナリ（0/1）
# ✓ 扱いやすい（2,470サンプル）
```

### 2. 大規模ディープラーニング
```python
# jingjietan/pandora-big5 を使用
dataset = load_dataset("jingjietan/pandora-big5")

# メリット:
# ✓ 大規模データ（3M+サンプル）
# ✓ 連続値スコア（0-100）
# ✓ 多様なRedditコメント

# 注意:
# ⚠️  ダウンロードに時間がかかる（2-5分）
# ⚠️  511 MBのストレージが必要
```

### 3. 中規模プロジェクト
```python
# Fatima0923/Automated-Personality-Prediction を使用
dataset = load_dataset("Fatima0923/Automated-Personality-Prediction")

# メリット:
# ✓ ちょうど良いサイズ（20,877サンプル）
# ✓ Redditコメント
# ✓ 高速ダウンロード（< 10秒）
```

---

## 💻 実行可能なスクリプト

### スクリプト1: データセット読み込み

```bash
# essays-big5を読み込む
python load_big5_datasets.py --dataset essays

# pandora-big5のサンプル1000件を読み込む
python load_big5_datasets.py --dataset pandora --sample 1000

# 統計情報を表示
python load_big5_datasets.py --dataset essays --stats
```

### スクリプト2: データセット比較

```bash
# 全データセットを比較
python compare_datasets.py

# 出力例:
# ┌─────────────────┬────────────┬──────────────┬─────────────┐
# │ Short Name      │ Total Size │ Text Type    │ Label Type  │
# ├─────────────────┼────────────┼──────────────┼─────────────┤
# │ essays-big5     │ 2,470      │ Essays       │ Binary      │
# │ pandora-big5    │ 3,006,566  │ Reddit       │ Float       │
# │ ...             │ ...        │ ...          │ ...         │
# └─────────────────┴────────────┴──────────────┴─────────────┘
```

---

## 🔧 インストール

### 必要なライブラリ

```bash
pip install datasets transformers pandas numpy scikit-learn

# オプション（データセット比較用）
pip install tabulate
```

---

## 📊 データセット詳細比較

| 項目 | essays-big5 | pandora-big5 | automated-personality |
|------|-------------|--------------|---------------------|
| **サイズ** | 2,470 | 3,006,566 | 20,877 |
| **テキストタイプ** | エッセイ | Redditコメント | Redditコメント |
| **ラベル形式** | Binary (0/1) | Float (0-100) | Float (0-99) |
| **ファイルサイズ** | ~5 MB | 511 MB | 6.02 MB |
| **ダウンロード時間** | < 10秒 | 2-5分 | < 10秒 |
| **ライセンス** | Apache 2.0 | Apache 2.0 | 不明 |
| **推奨用途** | 学習・小規模 | 大規模DL | 中規模 |

---

## 📚 使用例

### 例1: データセットの読み込みと確認

```python
from datasets import load_dataset

# データセット読み込み
dataset = load_dataset("jingjietan/essays-big5")

# データ分割の確認
print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")
print(f"Test: {len(dataset['test'])} samples")

# サンプルデータの確認
sample = dataset['train'][0]
print(f"\nText: {sample['text'][:100]}...")
print(f"Openness: {sample['O']}")
print(f"Conscientiousness: {sample['C']}")
print(f"Extraversion: {sample['E']}")
print(f"Agreeableness: {sample['A']}")
print(f"Neuroticism: {sample['N']}")
```

### 例2: 簡単な性格予測モデル

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# データセット読み込み
dataset = load_dataset("jingjietan/essays-big5")

# テキストとラベルの準備
train_texts = dataset['train']['text']
test_texts = dataset['test']['text']

# 特徴量抽出
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Opennessの予測（例）
y_train = dataset['train']['O']
y_test = dataset['test']['O']

# モデル訓練
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 予測と評価
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Openness prediction accuracy: {accuracy:.2%}")

# 新しいテキストで予測
new_text = ["I love trying new things and exploring creative ideas."]
new_features = vectorizer.transform(new_text)
prediction = model.predict(new_features)[0]
print(f"Predicted Openness: {prediction} (1=High, 0=Low)")
```

### 例3: データセットの統計分析

```python
from datasets import load_dataset
import pandas as pd

# データセット読み込み
dataset = load_dataset("jingjietan/essays-big5")

# DataFrameに変換
df = pd.DataFrame(dataset['train'])

# Big Five特性の分布
traits = ['O', 'C', 'E', 'A', 'N']
trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

print("Big Five Trait Distribution:")
for trait, name in zip(traits, trait_names):
    count = df[trait].sum()
    percentage = (count / len(df)) * 100
    print(f"{name}: {count}/{len(df)} ({percentage:.1f}%)")

# テキスト長の統計
df['text_length'] = df['text'].apply(len)
print(f"\nText Length Statistics:")
print(f"Mean: {df['text_length'].mean():.0f} characters")
print(f"Median: {df['text_length'].median():.0f} characters")
print(f"Min: {df['text_length'].min()} characters")
print(f"Max: {df['text_length'].max()} characters")
```

---

## 🚨 よくあるエラーと解決策

### エラー1: データセットが見つからない

```python
# ❌ 間違い
dataset = load_dataset("DavidIRL/real-persona-chat")
# DatasetNotFoundError: Dataset 'DavidIRL/real-persona-chat' doesn't exist

# ✅ 正しい
dataset = load_dataset("jingjietan/essays-big5")
```

### エラー2: メモリ不足（pandora-big5が大きすぎる）

```python
# ❌ 問題: 全データを一度に読み込もうとするとメモリ不足

# ✅ 解決策1: 一部のみ読み込む
dataset = load_dataset("jingjietan/pandora-big5", split="train[:10000]")

# ✅ 解決策2: ストリーミングモード
dataset = load_dataset("jingjietan/pandora-big5", streaming=True)
for sample in dataset['train'].take(100):
    print(sample)
```

### エラー3: ラベル形式の違い

```python
# essays-big5: Binary (0/1)
# pandora-big5: Float (0-100)

# 統一する場合の変換例

# Binary → Float (0-100)
float_score = binary_label * 100  # 0 → 0, 1 → 100

# Float → Binary（閾値50）
binary_label = 1 if float_score >= 50 else 0
```

---

## 🔗 関連リンク

### データセットページ
- [jingjietan/essays-big5](https://huggingface.co/datasets/jingjietan/essays-big5)
- [jingjietan/pandora-big5](https://huggingface.co/datasets/jingjietan/pandora-big5)
- [Fatima0923/Automated-Personality-Prediction](https://huggingface.co/datasets/Fatima0923/Automated-Personality-Prediction)
- [google/Synthetic-Persona-Chat](https://huggingface.co/datasets/google/Synthetic-Persona-Chat)

### 関連モデル
- [Minej/bert-base-personality](https://huggingface.co/Minej/bert-base-personality) - Big Five予測用BERTモデル
- [vladinc/bigfive-regression-model](https://huggingface.co/vladinc/bigfive-regression-model) - 回帰モデル

### コレクション
- [Big Five Personality Traits Collection](https://huggingface.co/collections/DmitryRyumin/big-five-personality-traits-661fb545292ab3d12a5a4890)

### 研究論文
- [PANDORA Dataset Paper](https://arxiv.org/abs/2004.04460)
- [Big5-Chat Paper](https://arxiv.org/abs/2410.16491)

---

## 📝 まとめ

### ✅ 確認済み・利用可能なデータセット

1. **jingjietan/essays-big5** - 最も使いやすい、初心者におすすめ
2. **jingjietan/pandora-big5** - 大規模プロジェクト向け
3. **Fatima0923/Automated-Personality-Prediction** - 中規模プロジェクト向け

### 🎯 推奨フロー

1. **学習段階**: `essays-big5`で実験・プロトタイプ作成
2. **開発段階**: `automated-personality`で中規模テスト
3. **本番段階**: `pandora-big5`で大規模訓練

### 📂 次のステップ

1. `big5_datasets_guide.md`を読んで詳細を理解
2. `load_big5_datasets.py`を実行してデータセットを試す
3. `compare_datasets.py`で各データセットを比較
4. 自分のプロジェクトに最適なデータセットを選択

---

**作成日**: 2025-10-24
**最終更新**: 2025-10-24
