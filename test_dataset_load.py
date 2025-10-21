"""
データセット読み込みテスト
HuggingFaceから実際に読み込めるか確認
"""
from datasets import load_dataset
import json

print("=" * 60)
print("RealPersonaChat データセット読み込みテスト")
print("=" * 60)

try:
    # trust_remote_codeなしで試す
    print("\n[方法1] trust_remote_code なしで読み込み試行...")
    dataset = load_dataset("nu-dialogue/real-persona-chat")
    print(f"✓ 成功！")
    print(f"Splits: {dataset.keys()}")

    if 'train' in dataset:
        print(f"\nTrain データ:")
        print(f"  件数: {len(dataset['train'])}")
        print(f"  カラム: {dataset['train'].column_names}")
        print(f"  Features: {dataset['train'].features}")

        # サンプル表示
        print(f"\n最初のサンプル:")
        print(json.dumps(dataset['train'][0], indent=2, ensure_ascii=False))

except Exception as e1:
    print(f"✗ 失敗: {e1}")

    try:
        # 別の方法: 全データを一度に読む
        print("\n[方法2] split指定なしで読み込み試行...")
        dataset = load_dataset("nu-dialogue/real-persona-chat", split="train")
        print(f"✓ 成功！")
        print(f"件数: {len(dataset)}")
        print(f"カラム: {dataset.column_names}")

        # サンプル表示
        print(f"\n最初のサンプル:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    except Exception as e2:
        print(f"✗ 失敗: {e2}")

        # 利用可能な設定を確認
        print("\n[方法3] データセット情報を確認...")
        from datasets import get_dataset_config_names
        try:
            configs = get_dataset_config_names("nu-dialogue/real-persona-chat")
            print(f"利用可能な設定: {configs}")

            # 各設定で試す
            for config in configs:
                print(f"\n--- {config} で読み込み試行 ---")
                try:
                    dataset = load_dataset("nu-dialogue/real-persona-chat", config)
                    print(f"✓ {config} 成功！")
                    print(f"  Splits: {dataset.keys()}")
                    if 'train' in dataset:
                        print(f"  Train件数: {len(dataset['train'])}")
                        print(f"  サンプル:")
                        print(json.dumps(dataset['train'][0], indent=2, ensure_ascii=False))
                except Exception as e3:
                    print(f"✗ {config} 失敗: {e3}")

        except Exception as e4:
            print(f"✗ 設定取得失敗: {e4}")

print("\n" + "=" * 60)
print("Nemotron-Personas-Japan データセット読み込みテスト")
print("=" * 60)

try:
    print("\nストリーミングモードで読み込み中...")
    dataset = load_dataset("nvidia/Nemotron-Personas-Japan", split="train", streaming=True)
    print("✓ 成功！")

    # 最初の3件取得
    print("\n最初の3サンプル:")
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"\n[サンプル {i+1}]")
        print(json.dumps(sample, indent=2, ensure_ascii=False))

except Exception as e:
    print(f"✗ 失敗: {e}")
