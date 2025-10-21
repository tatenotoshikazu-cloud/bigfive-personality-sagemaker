"""
データ前処理スクリプト
会話データをBig Five推定用のモデル入力形式に変換
"""
import json
import os
from datasets import load_from_disk, Dataset
from typing import Dict, List, Any
import pandas as pd


class BigFivePreprocessor:
    """Big Five性格特性推定用のデータ前処理クラス"""

    def __init__(self):
        self.big_five_keys = [
            'openness',          # 開放性
            'conscientiousness', # 誠実性
            'extraversion',      # 外向性
            'agreeableness',     # 協調性
            'neuroticism'        # 神経症傾向
        ]

    def process_realpersonachat(self, dialogue_path: str, interlocutor_path: str) -> Dataset:
        """
        RealPersonaChatデータを処理

        Args:
            dialogue_path: dialogue データセットのパス
            interlocutor_path: interlocutor データセットのパス

        Returns:
            処理済みデータセット
        """
        print("RealPersonaChat データ処理中...")

        # データ読み込み
        dialogue_data = load_from_disk(dialogue_path)
        interlocutor_data = load_from_disk(interlocutor_path)

        # 話者IDをキーにしたBig Five辞書作成
        personality_dict = {}
        for item in interlocutor_data['train']:
            speaker_id = item.get('speaker_id') or item.get('interlocutor_id')
            if speaker_id:
                personality_dict[speaker_id] = {
                    'openness': item.get('openness', 0.0),
                    'conscientiousness': item.get('conscientiousness', 0.0),
                    'extraversion': item.get('extraversion', 0.0),
                    'agreeableness': item.get('agreeableness', 0.0),
                    'neuroticism': item.get('neuroticism', 0.0),
                }

        processed_data = []

        # 対話データを処理
        for item in dialogue_data['train']:
            speaker_id = item.get('speaker_id') or item.get('interlocutor_id')
            utterances = item.get('utterances') or item.get('dialogue')

            if not speaker_id or not utterances:
                continue

            # 発話を結合してテキスト化
            if isinstance(utterances, list):
                # リスト形式の場合
                text = " ".join([u.get('text', '') if isinstance(u, dict) else str(u) for u in utterances])
            else:
                text = str(utterances)

            # Big Five特性を取得
            personality = personality_dict.get(speaker_id)
            if not personality:
                continue

            processed_data.append({
                'text': text,
                'openness': personality['openness'],
                'conscientiousness': personality['conscientiousness'],
                'extraversion': personality['extraversion'],
                'agreeableness': personality['agreeableness'],
                'neuroticism': personality['neuroticism'],
                'source': 'realpersonachat'
            })

        print(f"✓ RealPersonaChat: {len(processed_data)}件処理")
        return Dataset.from_pandas(pd.DataFrame(processed_data))

    def process_nemotron_personas(self, nemotron_path: str) -> Dataset:
        """
        Nemotron-Personas-Japanデータを処理

        Args:
            nemotron_path: Nemotron データセットのパス

        Returns:
            処理済みデータセット
        """
        print("Nemotron-Personas-Japan データ処理中...")

        # データ読み込み
        nemotron_data = load_from_disk(nemotron_path)

        processed_data = []

        for item in nemotron_data:
            # ペルソナ情報からテキスト生成
            persona_text = self._extract_persona_text(item)

            # Big Five特性を抽出（Nemotronデータに含まれている場合）
            big_five = self._extract_big_five(item)

            if persona_text and big_five:
                processed_data.append({
                    'text': persona_text,
                    **big_five,
                    'source': 'nemotron'
                })

            # 進捗表示（10000件ごと）
            if len(processed_data) % 10000 == 0 and len(processed_data) > 0:
                print(f"  処理済み: {len(processed_data)}件")

        print(f"✓ Nemotron-Personas-Japan: {len(processed_data)}件処理")
        return Dataset.from_pandas(pd.DataFrame(processed_data))

    def _extract_persona_text(self, item: Dict[str, Any]) -> str:
        """ペルソナ情報からテキスト抽出"""
        # Nemotronデータの構造に応じて調整
        text_fields = ['persona', 'description', 'text', 'content', 'profile']

        for field in text_fields:
            if field in item and item[field]:
                return str(item[field])

        # フィールドが見つからない場合は全体をJSON文字列化
        return json.dumps(item, ensure_ascii=False)

    def _extract_big_five(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Big Five特性を抽出"""
        big_five = {}

        # 直接フィールドから取得
        for trait in self.big_five_keys:
            if trait in item:
                big_five[trait] = float(item[trait])

        # personality辞書から取得
        if 'personality' in item and isinstance(item['personality'], dict):
            for trait in self.big_five_keys:
                if trait in item['personality']:
                    big_five[trait] = float(item['personality'][trait])

        # traits辞書から取得
        if 'traits' in item and isinstance(item['traits'], dict):
            for trait in self.big_five_keys:
                if trait in item['traits']:
                    big_five[trait] = float(item['traits'][trait])

        # 全てのBig Five特性が揃っているか確認
        if len(big_five) == 5:
            return big_five

        # 揃っていない場合はNone（デフォルト値で補完するオプションもあり）
        return None

    def create_train_val_split(self, dataset: Dataset, val_ratio: float = 0.1) -> Dict[str, Dataset]:
        """訓練/検証データに分割"""
        split = dataset.train_test_split(test_size=val_ratio, seed=42)
        return {
            'train': split['train'],
            'validation': split['test']
        }


def main():
    """メイン処理"""
    print("=" * 60)
    print("Big Five性格特性推定 - データ前処理")
    print("=" * 60)

    preprocessor = BigFivePreprocessor()

    # 1. RealPersonaChat処理
    print("\n[1/2] RealPersonaChat データ処理")
    realpersona_dataset = preprocessor.process_realpersonachat(
        dialogue_path="data/realpersonachat/dialogue",
        interlocutor_path="data/realpersonachat/interlocutor"
    )

    # 2. Nemotron-Personas-Japan処理
    print("\n[2/2] Nemotron-Personas-Japan データ処理")
    nemotron_dataset = preprocessor.process_nemotron_personas(
        nemotron_path="data/nemotron/personas_japan"
    )

    # 3. データセット分割
    print("\n" + "=" * 60)
    print("データセット分割")
    print("=" * 60)

    # Stage 1用（Nemotron）
    nemotron_split = preprocessor.create_train_val_split(nemotron_dataset)
    print(f"Stage 1 (Nemotron) - Train: {len(nemotron_split['train'])}件, Val: {len(nemotron_split['validation'])}件")

    # Stage 2用（RealPersonaChat）
    realpersona_split = preprocessor.create_train_val_split(realpersona_dataset)
    print(f"Stage 2 (RealPersona) - Train: {len(realpersona_split['train'])}件, Val: {len(realpersona_split['validation'])}件")

    # 4. 保存
    print("\n保存中...")
    os.makedirs("data/processed", exist_ok=True)

    nemotron_split['train'].save_to_disk("data/processed/stage1_train")
    nemotron_split['validation'].save_to_disk("data/processed/stage1_val")
    realpersona_split['train'].save_to_disk("data/processed/stage2_train")
    realpersona_split['validation'].save_to_disk("data/processed/stage2_val")

    print("\n✓ 処理済みデータをdata/processed/に保存しました")

    # サンプル表示
    print("\n" + "=" * 60)
    print("サンプルデータ")
    print("=" * 60)
    print("\n[Stage 1 Sample]")
    print(json.dumps(nemotron_split['train'][0], indent=2, ensure_ascii=False))
    print("\n[Stage 2 Sample]")
    print(json.dumps(realpersona_split['train'][0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
