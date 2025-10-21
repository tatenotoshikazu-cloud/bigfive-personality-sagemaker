"""
データセット取得スクリプト
RealPersonaChatとNemotron-Personas-JapanをHuggingFaceからダウンロード
"""
from datasets import load_dataset
import os
import json

def download_realpersonachat():
    """
    RealPersonaChat データセット取得
    - 日本語会話データ + Big Five性格特性ラベル
    - 約14,000件の対話データ
    """
    print("=" * 60)
    print("RealPersonaChat データセット取得開始...")
    print("=" * 60)

    # dialogue データ取得
    print("\n[1/2] dialogue データセット読み込み中...")
    dialogue_dataset = load_dataset(
        "nu-dialogue/real-persona-chat",
        name="dialogue",
        trust_remote_code=True
    )

    # interlocutor データ取得（話者情報 + Big Five特性）
    print("[2/2] interlocutor データセット読み込み中...")
    interlocutor_dataset = load_dataset(
        "nu-dialogue/real-persona-chat",
        name="interlocutor",
        trust_remote_code=True
    )

    # データセット情報表示
    print("\n" + "=" * 60)
    print("RealPersonaChat 取得完了")
    print("=" * 60)
    print(f"Dialogue データ: {dialogue_dataset}")
    print(f"Interlocutor データ: {interlocutor_dataset}")

    # サンプル表示
    if 'train' in dialogue_dataset:
        print("\n[Dialogue サンプル]")
        print(dialogue_dataset['train'][0])

    if 'train' in interlocutor_dataset:
        print("\n[Interlocutor サンプル (Big Five含む)]")
        sample = interlocutor_dataset['train'][0]
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    # ローカル保存
    os.makedirs("data/realpersonachat", exist_ok=True)
    dialogue_dataset.save_to_disk("data/realpersonachat/dialogue")
    interlocutor_dataset.save_to_disk("data/realpersonachat/interlocutor")
    print("\n✓ データをdata/realpersonachat/に保存しました")

    return dialogue_dataset, interlocutor_dataset


def download_nemotron_personas_japan():
    """
    Nemotron-Personas-Japan データセット取得
    - NVIDIA製の日本語ペルソナ合成データ
    - 1M-10M件のレコード
    """
    print("\n" + "=" * 60)
    print("Nemotron-Personas-Japan データセット取得開始...")
    print("=" * 60)

    print("\n読み込み中... (大容量のため時間がかかる場合があります)")

    # ストリーミングモードで取得（メモリ節約）
    dataset = load_dataset(
        "nvidia/Nemotron-Personas-Japan",
        split="train",
        streaming=True  # 大容量データ対応
    )

    print("\n" + "=" * 60)
    print("Nemotron-Personas-Japan 取得完了")
    print("=" * 60)
    print(f"データセット: {dataset}")

    # 最初の3サンプルを表示
    print("\n[サンプル (最初の3件)]")
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"\n--- サンプル {i+1} ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    # 通常モードで再読み込みして保存（オプション）
    print("\n全データをローカル保存中...")
    dataset_full = load_dataset(
        "nvidia/Nemotron-Personas-Japan",
        split="train"
    )

    os.makedirs("data/nemotron", exist_ok=True)
    dataset_full.save_to_disk("data/nemotron/personas_japan")
    print("✓ データをdata/nemotron/personas_japan/に保存しました")

    return dataset_full


def main():
    """メイン実行"""
    print("\n" + "=" * 60)
    print("Big Five性格特性推定 - データセット取得")
    print("=" * 60)

    # 1. RealPersonaChat 取得
    dialogue_data, interlocutor_data = download_realpersonachat()

    # 2. Nemotron-Personas-Japan 取得
    nemotron_data = download_nemotron_personas_japan()

    print("\n" + "=" * 60)
    print("全データセット取得完了！")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. data/ フォルダにデータが保存されています")
    print("2. train.py でモデル学習を実行できます")
    print("3. AWS SageMaker へのアップロードは別途実行してください")


if __name__ == "__main__":
    main()
