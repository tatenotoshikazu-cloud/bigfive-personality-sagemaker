# -*- coding: utf-8 -*-
"""
Nemotron-Personas-Japan データ詳細分析
Big Five特性が含まれているか確認
"""
from datasets import load_dataset
import json

print("=" * 60)
print("Nemotron-Personas-Japan 詳細分析")
print("=" * 60)

# データ読み込み（最初の1000件）
print("\nデータ読み込み中（最初の1000件）...")
dataset = load_dataset("nvidia/Nemotron-Personas-Japan", split="train", streaming=True)

samples = []
for i, sample in enumerate(dataset):
    samples.append(sample)
    if i >= 999:
        break
    if (i + 1) % 100 == 0:
        print(f"  進捗: {i + 1}件")

print(f"\n取得: {len(samples)}件")

# データ構造分析
print("\n" + "=" * 60)
print("データ構造")
print("=" * 60)

if samples:
    first_sample = samples[0]
    print(f"\nキー一覧（{len(first_sample.keys())}個）:")
    for key in first_sample.keys():
        value = first_sample[key]
        value_type = type(value).__name__
        if isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
        else:
            preview = str(value)
        print(f"  {key}: {value_type} = {preview}")

# Big Five特性の探索
print("\n" + "=" * 60)
print("Big Five特性の探索")
print("=" * 60)

big_five_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
big_five_japanese = ['開放性', '誠実性', '外向性', '協調性', '神経症']

# キーに含まれているか確認
print("\n[直接キー検索]")
for trait in big_five_traits:
    if trait in first_sample:
        print(f"  ✓ {trait}: {first_sample[trait]}")

# 日本語版
for trait in big_five_japanese:
    if trait in first_sample:
        print(f"  ✓ {trait}: {first_sample[trait]}")

# personality, traits などのネストキーを探索
print("\n[ネストされたキー検索]")
nested_keys = ['personality', 'traits', 'character', 'attributes', 'characteristics']
for nkey in nested_keys:
    if nkey in first_sample and isinstance(first_sample[nkey], dict):
        print(f"\n  {nkey}キー発見:")
        for k, v in first_sample[nkey].items():
            print(f"    {k}: {v}")

# ペルソナテキストから性格情報を抽出できるか確認
print("\n" + "=" * 60)
print("ペルソナテキスト分析")
print("=" * 60)

persona_keys = ['persona', 'professional_persona', 'sports_persona', 'arts_persona', 'travel_persona', 'culinary_persona']

print("\nペルソナ関連のテキストサンプル:")
for pkey in persona_keys:
    if pkey in first_sample:
        text = first_sample[pkey]
        print(f"\n[{pkey}]")
        print(f"{text[:200]}...")

# 統計情報
print("\n" + "=" * 60)
print("データセット統計（1000サンプル）")
print("=" * 60)

# 年齢分布
if 'age' in first_sample:
    ages = [s.get('age') for s in samples if 'age' in s]
    print(f"\n年齢:")
    print(f"  平均: {sum(ages) / len(ages):.1f}歳")
    print(f"  範囲: {min(ages)}-{max(ages)}歳")

# 性別分布
if 'sex' in first_sample:
    sexes = [s.get('sex') for s in samples if 'sex' in s]
    from collections import Counter
    sex_count = Counter(sexes)
    print(f"\n性別:")
    for sex, count in sex_count.items():
        print(f"  {sex}: {count}件")

# 地域分布
if 'region' in first_sample:
    regions = [s.get('region') for s in samples if 'region' in s]
    region_count = Counter(regions)
    print(f"\n地域（トップ5）:")
    for region, count in region_count.most_common(5):
        print(f"  {region}: {count}件")

# 職業分布
if 'occupation' in first_sample:
    occupations = [s.get('occupation') for s in samples if 'occupation' in s]
    occ_count = Counter(occupations)
    print(f"\n職業（トップ10）:")
    for occ, count in occ_count.most_common(10):
        print(f"  {occ}: {count}件")

# 結論
print("\n" + "=" * 60)
print("結論")
print("=" * 60)

has_big_five = False
for trait in big_five_traits + big_five_japanese:
    if trait in first_sample:
        has_big_five = True
        break

if has_big_five:
    print("\n✓ Big Five特性が直接含まれています")
    print("  → そのまま使用可能")
else:
    print("\n✗ Big Five特性は直接含まれていません")
    print("\n対処法:")
    print("  1. ペルソナテキストからBig Five特性を推定するモデルを別途作成")
    print("  2. 別のBig Fiveデータセットを使用")
    print("  3. RealPersonaChatを手動でダウンロード")
    print("\n推奨:")
    print("  → ペルソナテキストと属性情報（年齢、職業など）を入力として")
    print("     合成的にBig Five特性を生成するアプローチを検討")
