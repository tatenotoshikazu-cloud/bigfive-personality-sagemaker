# -*- coding: utf-8 -*-
"""
RealPersonaChat JSONデータ読み込み・処理
GitHubからクローンしたデータをDatasets形式に変換
"""
import json
import os
from datasets import Dataset
import pandas as pd
from glob import glob

print("=" * 60)
print("RealPersonaChat データ読み込み")
print("=" * 60)

# interlocutors.json読み込み
print("\ninterlocutors.json読み込み中...")
with open("data/realpersonachat_raw/real_persona_chat/interlocutors.json", "r", encoding="utf-8") as f:
    interlocutors_data = json.load(f)

print(f"OK: {len(interlocutors_data)}人の話者データ読み込み完了")

# Big Five特性を抽出
print("\nBig Five特性抽出中...")
big_five_data = []
for speaker_id, speaker_info in interlocutors_data.items():
    personality = speaker_info.get("personality", {})

    big_five_data.append({
        "speaker_id": speaker_id,
        "openness": personality.get("BigFive_Openness", 0.0),
        "conscientiousness": personality.get("BigFive_Conscientiousness", 0.0),
        "extraversion": personality.get("BigFive_Extraversion", 0.0),
        "agreeableness": personality.get("BigFive_Agreeableness", 0.0),
        "neuroticism": personality.get("BigFive_Neuroticism", 0.0),
        "persona": speaker_info.get("persona", []),
        "gender": speaker_info.get("demographic_information", {}).get("gender", ""),
        "age": speaker_info.get("demographic_information", {}).get("age", ""),
    })

print(f"OK: {len(big_five_data)}人のBig Five特性抽出完了")

# 統計情報表示
df_big_five = pd.DataFrame(big_five_data)
print("\nBig Five統計:")
for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
    print(f"  {trait}: 平均={df_big_five[trait].mean():.2f}, 範囲=[{df_big_five[trait].min():.2f}, {df_big_five[trait].max():.2f}]")

# dialoguesフォルダから対話データ読み込み
print("\ndialoguesフォルダ読み込み中...")
dialogue_files = glob("data/realpersonachat_raw/real_persona_chat/dialogues/*.json")
print(f"対話ファイル数: {len(dialogue_files)}件")

# 小規模データセット作成（最初の100件）
num_dialogues = 100
print(f"\n最初の{num_dialogues}件の対話を処理中...")

processed_data = []
for i, dialogue_file in enumerate(dialogue_files[:num_dialogues]):
    if (i + 1) % 20 == 0:
        print(f"  進捗: {i + 1}件")

    with open(dialogue_file, "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    # 各話者の発話を結合
    speaker_utterances = {}
    for utt in dialogue.get("utterances", []):
        speaker_id = utt.get("interlocutor_id")
        text = utt.get("text", "")

        if speaker_id not in speaker_utterances:
            speaker_utterances[speaker_id] = []
        speaker_utterances[speaker_id].append(text)

    # 各話者のデータを作成
    for speaker_id, utterances in speaker_utterances.items():
        # Big Five特性を取得
        speaker_big_five = next((item for item in big_five_data if item["speaker_id"] == speaker_id), None)

        if speaker_big_five is None:
            continue

        # 発話を結合
        combined_text = " ".join(utterances)

        processed_data.append({
            "text": combined_text,
            "openness": speaker_big_five["openness"],
            "conscientiousness": speaker_big_five["conscientiousness"],
            "extraversion": speaker_big_five["extraversion"],
            "agreeableness": speaker_big_five["agreeableness"],
            "neuroticism": speaker_big_five["neuroticism"],
            "speaker_id": speaker_id,
            "dialogue_id": dialogue.get("dialogue_id"),
        })

print(f"\nOK: {len(processed_data)}件のサンプル作成完了")

# Dataset形式に変換
print("\nDataset形式に変換中...")
df = pd.DataFrame(processed_data)
dataset = Dataset.from_pandas(df)

# 保存
os.makedirs("data/local/realpersonachat", exist_ok=True)
dataset.save_to_disk("data/local/realpersonachat/processed")

print(f"OK: 保存完了 data/local/realpersonachat/processed/")

# train/val分割
print("\ntrain/val分割中...")
split = dataset.train_test_split(test_size=0.2, seed=42)

os.makedirs("data/local/realpersonachat_split", exist_ok=True)
split['train'].save_to_disk("data/local/realpersonachat_split/train")
split['test'].save_to_disk("data/local/realpersonachat_split/val")

print(f"OK: Train {len(split['train'])}件")
print(f"OK: Val {len(split['test'])}件")

# サンプル表示
print("\n" + "=" * 60)
print("サンプルデータ")
print("=" * 60)

sample = split['train'][0]
print(f"\nテキスト: {sample['text'][:200]}...")
print(f"\nBig Five特性:")
print(f"  開放性: {sample['openness']:.2f}")
print(f"  誠実性: {sample['conscientiousness']:.2f}")
print(f"  外向性: {sample['extraversion']:.2f}")
print(f"  協調性: {sample['agreeableness']:.2f}")
print(f"  神経症傾向: {sample['neuroticism']:.2f}")

print("\n" + "=" * 60)
print("RealPersonaChat データ準備完了！")
print("=" * 60)
print("\nデータ保存先:")
print("  data/local/realpersonachat/processed/")
print("  data/local/realpersonachat_split/train/")
print("  data/local/realpersonachat_split/val/")
