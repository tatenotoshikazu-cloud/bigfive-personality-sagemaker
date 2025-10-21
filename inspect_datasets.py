"""
データセット形式確認スクリプト
RealPersonaChatとNemotron-Personas-Japanのデータ構造を詳細に調査
"""
from datasets import load_dataset
import json
import sys


def inspect_realpersonachat():
    """RealPersonaChatデータセットの詳細調査"""
    print("=" * 80)
    print("RealPersonaChat データセット調査")
    print("=" * 80)

    try:
        # Dialogue データ
        print("\n[1/2] Dialogue データセット読み込み中...")
        dialogue_data = load_dataset(
            "nu-dialogue/real-persona-chat",
            name="dialogue",
            trust_remote_code=True
        )

        print(f"\n✓ 読み込み成功")
        print(f"Splits: {dialogue_data.keys()}")

        if 'train' in dialogue_data:
            train_data = dialogue_data['train']
            print(f"Train件数: {len(train_data)}")
            print(f"\nカラム: {train_data.column_names}")
            print(f"Features: {train_data.features}")

            # サンプル3件を詳細表示
            print("\n" + "-" * 80)
            print("サンプルデータ（最初の3件）")
            print("-" * 80)
            for i in range(min(3, len(train_data))):
                print(f"\n[サンプル {i+1}]")
                sample = train_data[i]
                print(json.dumps(sample, indent=2, ensure_ascii=False))

        # Interlocutor データ
        print("\n" + "=" * 80)
        print("[2/2] Interlocutor データセット読み込み中...")
        print("=" * 80)
        interlocutor_data = load_dataset(
            "nu-dialogue/real-persona-chat",
            name="interlocutor",
            trust_remote_code=True
        )

        print(f"\n✓ 読み込み成功")
        print(f"Splits: {interlocutor_data.keys()}")

        if 'train' in interlocutor_data:
            train_data = interlocutor_data['train']
            print(f"Train件数: {len(train_data)}")
            print(f"\nカラム: {train_data.column_names}")
            print(f"Features: {train_data.features}")

            # Big Five特性の確認
            print("\n" + "-" * 80)
            print("Big Five特性の確認")
            print("-" * 80)

            sample = train_data[0]
            print("\n[最初のサンプル]")
            print(json.dumps(sample, indent=2, ensure_ascii=False))

            # Big Five特性のキー名を特定
            big_five_candidates = [
                'openness', 'conscientiousness', 'extraversion',
                'agreeableness', 'neuroticism', 'personality',
                'big_five', 'traits', 'Big5'
            ]

            found_keys = []
            for key in big_five_candidates:
                if key in sample:
                    found_keys.append(key)
                    print(f"\n✓ 発見: '{key}' = {sample[key]}")

            if not found_keys:
                print("\n⚠ Big Five特性のキーが見つかりませんでした")
                print("全キー一覧:")
                for key in sample.keys():
                    print(f"  - {key}: {type(sample[key])}")

            # 統計情報
            print("\n" + "-" * 80)
            print("Big Five特性の統計（最初の100サンプル）")
            print("-" * 80)

            stats = {
                'openness': [],
                'conscientiousness': [],
                'extraversion': [],
                'agreeableness': [],
                'neuroticism': []
            }

            for i in range(min(100, len(train_data))):
                sample = train_data[i]
                for trait in stats.keys():
                    if trait in sample:
                        stats[trait].append(sample[trait])

            import numpy as np
            for trait, values in stats.items():
                if values:
                    print(f"\n{trait}:")
                    print(f"  平均: {np.mean(values):.3f}")
                    print(f"  標準偏差: {np.std(values):.3f}")
                    print(f"  最小値: {np.min(values):.3f}")
                    print(f"  最大値: {np.max(values):.3f}")

        return dialogue_data, interlocutor_data

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def inspect_nemotron_personas():
    """Nemotron-Personas-Japanデータセットの詳細調査"""
    print("\n\n" + "=" * 80)
    print("Nemotron-Personas-Japan データセット調査")
    print("=" * 80)

    try:
        # ストリーミングモードで最初の10件を取得
        print("\nストリーミングモードで読み込み中...")
        dataset_stream = load_dataset(
            "nvidia/Nemotron-Personas-Japan",
            split="train",
            streaming=True
        )

        print("✓ 読み込み成功（ストリーミング）")

        # 最初の10サンプルを取得
        samples = []
        print("\n最初の10サンプルを取得中...")
        for i, sample in enumerate(dataset_stream):
            samples.append(sample)
            if i >= 9:
                break

        print(f"✓ {len(samples)}件取得")

        # データ構造分析
        print("\n" + "-" * 80)
        print("データ構造")
        print("-" * 80)

        if samples:
            first_sample = samples[0]
            print(f"\nキー数: {len(first_sample.keys())}")
            print("\nキー一覧:")
            for key in first_sample.keys():
                value = first_sample[key]
                value_type = type(value).__name__
                value_preview = str(value)[:100] if value else "None"
                print(f"  - {key} ({value_type}): {value_preview}...")

            # サンプル3件の詳細表示
            print("\n" + "-" * 80)
            print("サンプルデータ（最初の3件）")
            print("-" * 80)
            for i, sample in enumerate(samples[:3]):
                print(f"\n[サンプル {i+1}]")
                print(json.dumps(sample, indent=2, ensure_ascii=False))

            # Big Five特性の確認
            print("\n" + "-" * 80)
            print("Big Five特性の探索")
            print("-" * 80)

            big_five_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            personality_keys = ['personality', 'traits', 'big_five', 'Big5', 'attributes']

            found_traits = {}
            for sample in samples:
                # 直接Big Fiveキーを探索
                for trait in big_five_keys:
                    if trait in sample:
                        if trait not in found_traits:
                            found_traits[trait] = []
                        found_traits[trait].append(sample[trait])

                # ネストされた辞書を探索
                for pkey in personality_keys:
                    if pkey in sample and isinstance(sample[pkey], dict):
                        for trait in big_five_keys:
                            if trait in sample[pkey]:
                                full_key = f"{pkey}.{trait}"
                                if full_key not in found_traits:
                                    found_traits[full_key] = []
                                found_traits[full_key].append(sample[pkey][trait])

            if found_traits:
                print("\n✓ Big Five特性を発見:")
                import numpy as np
                for trait, values in found_traits.items():
                    print(f"\n  {trait}:")
                    print(f"    件数: {len(values)}")
                    if all(isinstance(v, (int, float)) for v in values):
                        print(f"    平均: {np.mean(values):.3f}")
                        print(f"    範囲: [{np.min(values):.3f}, {np.max(values):.3f}]")
            else:
                print("\n⚠ Big Five特性が見つかりませんでした")
                print("このデータセットにはBig Five特性が含まれていない可能性があります")
                print("\n対処法:")
                print("1. Nemotronデータは事前学習用として使用")
                print("2. RealPersonaChatのみでBig Five推定を学習")
                print("3. または、Nemotronデータに人工的にBig Five特性を付与")

        return dataset_stream

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_structure_report():
    """データ構造レポートをファイルに保存"""
    print("\n\n" + "=" * 80)
    print("データ構造レポート生成")
    print("=" * 80)

    # 標準出力をファイルにリダイレクト
    import io
    from contextlib import redirect_stdout

    report_buffer = io.StringIO()

    with redirect_stdout(report_buffer):
        dialogue_data, interlocutor_data = inspect_realpersonachat()
        nemotron_data = inspect_nemotron_personas()

    report = report_buffer.getvalue()

    # ファイル保存
    with open('dataset_structure_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n✓ レポートを保存: dataset_structure_report.txt")


def create_data_format_guide():
    """データ形式ガイド作成"""
    guide = """
# データ形式ガイド

## RealPersonaChat

### Dialogue データ
- **目的**: 会話履歴
- **キー構造**:
  - `speaker_id` or `interlocutor_id`: 話者ID
  - `utterances` or `dialogue`: 発話リスト
  - 各発話: `text`フィールドを含む

### Interlocutor データ
- **目的**: 話者のBig Five特性
- **キー構造**:
  - `speaker_id` or `interlocutor_id`: 話者ID
  - `openness`: 開放性（0-1または1-5スケール）
  - `conscientiousness`: 誠実性
  - `extraversion`: 外向性
  - `agreeableness`: 協調性
  - `neuroticism`: 神経症傾向

## Nemotron-Personas-Japan

### 構造
- **目的**: 日本語ペルソナ合成データ
- **注意**: Big Five特性が含まれていない可能性あり
- **使用法**:
  - Stage 1: 汎用的な言語理解の事前学習
  - Stage 2: RealPersonaChatで性格特性推定をファインチューニング

## 前処理要件

1. **テキスト抽出**: 会話データから連続テキストを生成
2. **ラベル統合**: 話者IDをキーにBig Five特性を結合
3. **正規化**: スケールを0-1に統一（必要に応じて）
4. **欠損値処理**: Big Five特性が欠けているサンプルを除外

## 検証項目

- [ ] RealPersonaChat dialogueデータが読み込める
- [ ] RealPersonaChat interlocutorデータが読み込める
- [ ] Big Five特性（5次元）が全て存在する
- [ ] 話者IDで2つのデータセットを紐付けられる
- [ ] Nemotronデータが読み込める
- [ ] Nemotronデータの構造を理解している
"""

    with open('DATA_FORMAT_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)

    print("✓ データ形式ガイド作成: DATA_FORMAT_GUIDE.md")


def main():
    """メイン実行"""
    print("=" * 80)
    print("Big Five性格特性推定 - データセット詳細調査")
    print("=" * 80)
    print("\nこのスクリプトはデータセットの構造を詳細に調査します")
    print("SageMaker実行前に必ずローカルで実行してください\n")

    # 1. RealPersonaChat調査
    dialogue_data, interlocutor_data = inspect_realpersonachat()

    # 2. Nemotron調査
    nemotron_data = inspect_nemotron_personas()

    # 3. ガイド作成
    create_data_format_guide()

    print("\n" + "=" * 80)
    print("調査完了")
    print("=" * 80)
    print("\n次のステップ:")
    print("1. DATA_FORMAT_GUIDE.md を確認")
    print("2. データ構造に問題がなければ download_datasets.py を実行")
    print("3. preprocess_data.py でデータを前処理")
    print("4. 処理済みデータをS3にアップロード")
    print("5. SageMakerで学習実行")


if __name__ == '__main__':
    main()
