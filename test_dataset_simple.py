# -*- coding: utf-8 -*-
"""
シンプルなデータセット読み込みテスト
"""
from datasets import load_dataset
import json

print("Testing RealPersonaChat...")

try:
    # Parquet形式で直接読み込み
    print("Trying to load via parquet...")
    dataset = load_dataset("nu-dialogue/real-persona-chat", data_files="**/*.parquet")
    print("Success!")
    print(f"Keys: {dataset.keys()}")

    for split_name in dataset.keys():
        print(f"\n{split_name}:")
        print(f"  Count: {len(dataset[split_name])}")
        print(f"  Columns: {dataset[split_name].column_names}")
        print(f"  Sample:")
        print(json.dumps(dataset[split_name][0], indent=2, ensure_ascii=False))

except Exception as e:
    print(f"Failed: {e}")

    # 代替: HuggingFace APIで確認
    print("\nChecking dataset info via API...")
    from huggingface_hub import HfApi
    api = HfApi()

    try:
        dataset_info = api.dataset_info("nu-dialogue/real-persona-chat")
        print(f"Dataset ID: {dataset_info.id}")
        print(f"Downloads: {dataset_info.downloads}")
        print(f"Tags: {dataset_info.tags}")

        # ファイルリスト取得
        files = api.list_repo_files("nu-dialogue/real-persona-chat", repo_type="dataset")
        print(f"\nFiles in repo:")
        for f in files[:20]:  # 最初の20個
            print(f"  {f}")

    except Exception as e2:
        print(f"API failed: {e2}")

print("\n\nTesting Nemotron-Personas-Japan...")

try:
    dataset = load_dataset("nvidia/Nemotron-Personas-Japan", split="train", streaming=True)
    print("Success!")

    sample_count = 0
    for sample in dataset:
        print(f"\nSample {sample_count + 1}:")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        sample_count += 1
        if sample_count >= 2:
            break

except Exception as e:
    print(f"Failed: {e}")
