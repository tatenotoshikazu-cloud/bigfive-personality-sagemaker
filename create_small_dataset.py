"""
小規模データセット作成
ローカルテスト用に、元データセットから少数サンプルを抽出
"""
from datasets import load_dataset, Dataset
import os
import json


def create_small_realpersonachat(num_samples=100):
    """
    RealPersonaChatの小規模版作成

    Args:
        num_samples: 抽出するサンプル数（デフォルト100件）
    """
    print("=" * 60)
    print(f"RealPersonaChat 小規模データセット作成（{num_samples}件）")
    print("=" * 60)

    # 元データ読み込み
    print("\n元データセット読み込み中...")
    dialogue_data = load_dataset(
        "nu-dialogue/real-persona-chat",
        name="dialogue",
        trust_remote_code=True
    )

    interlocutor_data = load_dataset(
        "nu-dialogue/real-persona-chat",
        name="interlocutor",
        trust_remote_code=True
    )

    print(f"✓ 元データ: Dialogue={len(dialogue_data['train'])}件, Interlocutor={len(interlocutor_data['train'])}件")

    # 小規模データ抽出
    print(f"\n最初の{num_samples}件を抽出中...")
    small_dialogue = dialogue_data['train'].select(range(min(num_samples, len(dialogue_data['train']))))
    small_interlocutor = interlocutor_data['train'].select(range(min(num_samples, len(interlocutor_data['train']))))

    # 保存
    os.makedirs("data/small/realpersonachat", exist_ok=True)
    small_dialogue.save_to_disk("data/small/realpersonachat/dialogue")
    small_interlocutor.save_to_disk("data/small/realpersonachat/interlocutor")

    print(f"✓ 保存完了: data/small/realpersonachat/")
    print(f"  Dialogue: {len(small_dialogue)}件")
    print(f"  Interlocutor: {len(small_interlocutor)}件")

    # サンプル表示
    print("\n[サンプル]")
    print(json.dumps(small_dialogue[0], indent=2, ensure_ascii=False))

    return small_dialogue, small_interlocutor


def create_small_nemotron(num_samples=1000):
    """
    Nemotron-Personas-Japanの小規模版作成

    Args:
        num_samples: 抽出するサンプル数（デフォルト1000件）
    """
    print("\n" + "=" * 60)
    print(f"Nemotron-Personas-Japan 小規模データセット作成（{num_samples}件）")
    print("=" * 60)

    # ストリーミングモードで読み込み
    print("\n元データセット読み込み中（ストリーミング）...")
    dataset_stream = load_dataset(
        "nvidia/Nemotron-Personas-Japan",
        split="train",
        streaming=True
    )

    # 指定件数を抽出
    print(f"\n最初の{num_samples}件を抽出中...")
    samples = []
    for i, sample in enumerate(dataset_stream):
        samples.append(sample)
        if (i + 1) % 100 == 0:
            print(f"  進捗: {i + 1}件")
        if i >= num_samples - 1:
            break

    # Dataset形式に変換
    import pandas as pd
    small_nemotron = Dataset.from_pandas(pd.DataFrame(samples))

    # 保存
    os.makedirs("data/small/nemotron", exist_ok=True)
    small_nemotron.save_to_disk("data/small/nemotron/personas_japan")

    print(f"\n✓ 保存完了: data/small/nemotron/personas_japan/")
    print(f"  件数: {len(small_nemotron)}件")

    # サンプル表示
    print("\n[サンプル]")
    print(json.dumps(small_nemotron[0], indent=2, ensure_ascii=False))

    return small_nemotron


def preprocess_small_data():
    """小規模データを前処理"""
    print("\n" + "=" * 60)
    print("小規模データセット前処理")
    print("=" * 60)

    # preprocess_data.pyのロジックを使用
    from preprocess_data import BigFivePreprocessor

    preprocessor = BigFivePreprocessor()

    # RealPersonaChat処理
    print("\n[1/2] RealPersonaChat処理中...")
    realpersona_dataset = preprocessor.process_realpersonachat(
        dialogue_path="data/small/realpersonachat/dialogue",
        interlocutor_path="data/small/realpersonachat/interlocutor"
    )

    # Nemotron処理
    print("\n[2/2] Nemotron処理中...")
    nemotron_dataset = preprocessor.process_nemotron_personas(
        nemotron_path="data/small/nemotron/personas_japan"
    )

    # データ分割（train/val）
    print("\nデータ分割中...")
    nemotron_split = preprocessor.create_train_val_split(nemotron_dataset, val_ratio=0.2)
    realpersona_split = preprocessor.create_train_val_split(realpersona_dataset, val_ratio=0.2)

    print(f"Stage 1 (Nemotron) - Train: {len(nemotron_split['train'])}件, Val: {len(nemotron_split['validation'])}件")
    print(f"Stage 2 (RealPersona) - Train: {len(realpersona_split['train'])}件, Val: {len(realpersona_split['validation'])}件")

    # 保存
    os.makedirs("data/small/processed", exist_ok=True)
    nemotron_split['train'].save_to_disk("data/small/processed/stage1_train")
    nemotron_split['validation'].save_to_disk("data/small/processed/stage1_val")
    realpersona_split['train'].save_to_disk("data/small/processed/stage2_train")
    realpersona_split['validation'].save_to_disk("data/small/processed/stage2_val")

    print("\n✓ 処理済みデータ保存: data/small/processed/")

    # サンプル表示
    print("\n[処理済みサンプル - Stage 2]")
    if len(realpersona_split['train']) > 0:
        print(json.dumps(realpersona_split['train'][0], indent=2, ensure_ascii=False))


def main():
    """メイン実行"""
    print("=" * 60)
    print("Big Five性格特性推定 - ローカルテスト用小規模データ作成")
    print("=" * 60)
    print("\nこのスクリプトは以下を実行します:")
    print("1. RealPersonaChatから100件抽出")
    print("2. Nemotron-Personas-Japanから1000件抽出")
    print("3. データ前処理・分割")
    print("4. data/small/ に保存")
    print("\n推定実行時間: 2-5分\n")

    # 1. RealPersonaChat小規模データ作成
    small_dialogue, small_interlocutor = create_small_realpersonachat(num_samples=100)

    # 2. Nemotron小規模データ作成
    small_nemotron = create_small_nemotron(num_samples=1000)

    # 3. 前処理
    preprocess_small_data()

    print("\n" + "=" * 60)
    print("小規模データセット作成完了！")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. train_local.py でローカル学習テスト実行")
    print("2. エラーがなければ本番データで再実行")
    print("3. SageMakerで本格学習")


if __name__ == "__main__":
    main()
