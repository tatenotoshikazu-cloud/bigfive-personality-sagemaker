# -*- coding: utf-8 -*-
"""
代替Big Fiveデータセット探索
"""
from datasets import load_dataset
from huggingface_hub import HfApi
import json

print("=" * 60)
print("代替Big Fiveデータセット探索")
print("=" * 60)

# HuggingFace APIで検索
api = HfApi()

# Big Five関連のデータセット検索
keywords = ["big five", "personality", "ocean", "personality traits"]

print("\nHuggingFaceで Big Five 関連データセットを検索中...")

all_datasets = []

for keyword in keywords:
    print(f"\n検索: '{keyword}'")
    try:
        datasets = api.list_datasets(search=keyword, limit=10)
        for ds in datasets:
            all_datasets.append({
                'id': ds.id,
                'downloads': ds.downloads if hasattr(ds, 'downloads') else 0,
                'tags': ds.tags if hasattr(ds, 'tags') else [],
            })
            print(f"  - {ds.id} (DL: {ds.downloads if hasattr(ds, 'downloads') else 'N/A'})")
    except Exception as e:
        print(f"  エラー: {e}")

# ダウンロード数でソート
all_datasets.sort(key=lambda x: x['downloads'], reverse=True)

print("\n" + "=" * 60)
print("有望なデータセット（ダウンロード数順）")
print("=" * 60)

for i, ds in enumerate(all_datasets[:10]):
    print(f"\n{i+1}. {ds['id']}")
    print(f"   Downloads: {ds['downloads']}")
    print(f"   Tags: {', '.join(ds['tags'][:5])}")

# 既知の代替データセット
print("\n" + "=" * 60)
print("既知のBig Five / Personalityデータセット")
print("=" * 60)

known_datasets = [
    "SetFit/20_newsgroups",
    "mteb/amazon_massive_intent",
    # これらは別タスク用だが、参考として
]

print("\n推奨アプローチ:")
print("1. RealPersonaChatを手動でダウンロード")
print("   → GitHub/論文サイトから直接取得")
print("2. Nemotronデータのみで進める")
print("   → ペルソナテキストを入力とした汎用モデル")
print("3. 合成データ生成")
print("   → GPT-4などでBig Five特性ラベルを生成")

print("\n最も現実的な選択肢:")
print("→ Nemotronのペルソナデータで言語モデルを訓練")
print("→ 後でBig Fiveデータが手に入ったらファインチューニング")
