"""
ローカル学習テストスクリプト
小規模データセットで動作確認（SageMaker実行前の検証）
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json


# train.pyからクラスをインポート
sys.path.append(os.path.dirname(__file__))
from train import BigFiveRegressionModel, BigFiveDataset, compute_metrics


def test_data_loading():
    """データ読み込みテスト"""
    print("=" * 60)
    print("テスト1: データ読み込み")
    print("=" * 60)

    try:
        # Stage 1データ
        print("\n[Stage 1] データ読み込み中...")
        stage1_train = load_from_disk("data/small/processed/stage1_train")
        stage1_val = load_from_disk("data/small/processed/stage1_val")
        print(f"✓ Stage 1 - Train: {len(stage1_train)}件, Val: {len(stage1_val)}件")
        print(f"  カラム: {stage1_train.column_names}")

        # サンプル表示
        if len(stage1_train) > 0:
            print("\n  サンプル:")
            print(json.dumps(stage1_train[0], indent=4, ensure_ascii=False))

        # Stage 2データ
        print("\n[Stage 2] データ読み込み中...")
        stage2_train = load_from_disk("data/small/processed/stage2_train")
        stage2_val = load_from_disk("data/small/processed/stage2_val")
        print(f"✓ Stage 2 - Train: {len(stage2_train)}件, Val: {len(stage2_val)}件")
        print(f"  カラム: {stage2_train.column_names}")

        # Big Five特性の確認
        print("\n[Big Five特性確認]")
        if len(stage2_train) > 0:
            sample = stage2_train[0]
            traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
            print("  特性値:")
            for trait in traits:
                if trait in sample:
                    print(f"    {trait}: {sample[trait]}")
                else:
                    print(f"    ✗ {trait}: 存在しません")
                    return False

        print("\n✓ データ読み込みテスト成功")
        return True

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """モデル初期化テスト"""
    print("\n" + "=" * 60)
    print("テスト2: モデル初期化")
    print("=" * 60)

    try:
        # トークナイザー
        print("\nトークナイザー読み込み中...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")  # 軽量版でテスト
        print(f"✓ トークナイザー読み込み成功")

        # LoRA設定
        print("\nLoRA設定...")
        lora_config = LoraConfig(
            r=8,  # テスト用に小さめ
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        print(f"✓ LoRA設定完了: r={lora_config.r}, alpha={lora_config.lora_alpha}")

        # モデル構築（軽量版）
        print("\nモデル構築中（xlm-roberta-base）...")
        model = BigFiveRegressionModel("xlm-roberta-base", lora_config)
        print(f"✓ モデル構築成功")

        # パラメータ数確認
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  学習可能パラメータ: {trainable_params:,} / {total_params:,}")
        print(f"  削減率: {100 * (1 - trainable_params / total_params):.2f}%")

        # デバイス確認
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n  デバイス: {device}")
        if device.type == 'cpu':
            print("  ⚠ CPUで実行します（GPUがあれば高速化されます）")

        model.to(device)

        # ダミーデータでフォワードパステスト
        print("\nフォワードパステスト...")
        dummy_input_ids = torch.randint(0, 1000, (2, 128)).to(device)
        dummy_attention_mask = torch.ones(2, 128).to(device)
        dummy_labels = torch.rand(2, 5).to(device)

        with torch.no_grad():
            output = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                labels=dummy_labels
            )

        print(f"✓ フォワードパス成功")
        print(f"  出力形状: {output['logits'].shape}")
        print(f"  Loss: {output['loss'].item():.4f}")

        print("\n✓ モデル初期化テスト成功")
        return True, tokenizer, model

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_dataset_creation(tokenizer):
    """Datasetクラステスト"""
    print("\n" + "=" * 60)
    print("テスト3: Dataset作成")
    print("=" * 60)

    try:
        # データ読み込み
        print("\nデータ読み込み中...")
        stage2_train = load_from_disk("data/small/processed/stage2_train")

        # Dataset作成
        print("\nDataset作成中...")
        dataset = BigFiveDataset(stage2_train, tokenizer, max_length=128)  # テスト用に短め
        print(f"✓ Dataset作成成功: {len(dataset)}件")

        # サンプル取得
        print("\nサンプル取得テスト...")
        sample = dataset[0]
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  labels shape: {sample['labels'].shape}")
        print(f"  labels: {sample['labels'].tolist()}")

        print("\n✓ Dataset作成テスト成功")
        return True

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_training_test():
    """簡易学習テスト（1エポック）"""
    print("\n" + "=" * 60)
    print("テスト4: 簡易学習テスト（1エポック）")
    print("=" * 60)

    try:
        # データ読み込み
        print("\nデータ読み込み中...")
        train_data = load_from_disk("data/small/processed/stage2_train")
        val_data = load_from_disk("data/small/processed/stage2_val")

        # さらに小規模化（高速テスト用）
        if len(train_data) > 20:
            train_data = train_data.select(range(20))
        if len(val_data) > 5:
            val_data = val_data.select(range(5))

        print(f"✓ Train: {len(train_data)}件, Val: {len(val_data)}件")

        # トークナイザー・モデル
        print("\nモデル準備中...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = BigFiveRegressionModel("xlm-roberta-base", lora_config)

        # Dataset作成
        train_dataset = BigFiveDataset(train_data, tokenizer, max_length=128)
        val_dataset = BigFiveDataset(val_data, tokenizer, max_length=128)

        # トレーニング設定（超軽量）
        print("\n学習設定...")
        training_args = TrainingArguments(
            output_dir="output/local_test",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=2e-4,
            logging_steps=5,
            evaluation_strategy='epoch',
            save_strategy='no',  # 保存しない（テストのみ）
            report_to='none',
            disable_tqdm=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        # 学習実行
        print("\n学習開始（1エポックのみ）...")
        print("-" * 60)
        trainer.train()

        # 評価
        print("\n評価実行...")
        eval_results = trainer.evaluate()
        print("\n評価結果:")
        print(json.dumps(eval_results, indent=2))

        print("\n✓ 簡易学習テスト成功")
        return True

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全テスト実行"""
    print("=" * 60)
    print("Big Five性格特性推定 - ローカル学習テスト")
    print("=" * 60)
    print("\nSageMaker実行前の動作確認を行います")
    print("想定実行時間: 5-15分（環境により変動）\n")

    results = {}

    # テスト1: データ読み込み
    results['data_loading'] = test_data_loading()

    if not results['data_loading']:
        print("\n✗ データ読み込みに失敗しました")
        print("先に create_small_dataset.py を実行してください")
        return

    # テスト2: モデル初期化
    success, tokenizer, model = test_model_initialization()
    results['model_init'] = success

    if not success:
        print("\n✗ モデル初期化に失敗しました")
        return

    # テスト3: Dataset作成
    results['dataset_creation'] = test_dataset_creation(tokenizer)

    # テスト4: 簡易学習テスト
    print("\n簡易学習テストを実行しますか？（CPU環境では5-10分かかる場合があります）")
    response = input("実行する場合は 'y' を入力: ")
    if response.lower() == 'y':
        results['quick_training'] = quick_training_test()
    else:
        print("学習テストをスキップしました")
        results['quick_training'] = None

    # 結果サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    for test_name, result in results.items():
        if result is None:
            status = "⊘ スキップ"
        elif result:
            status = "✓ 成功"
        else:
            status = "✗ 失敗"
        print(f"{status} - {test_name}")

    if all(r in [True, None] for r in results.values()):
        print("\n" + "=" * 60)
        print("✓ 全テスト成功！SageMaker実行準備完了")
        print("=" * 60)
        print("\n次のステップ:")
        print("1. フルデータセットをダウンロード: python download_datasets.py")
        print("2. フルデータセットを前処理: python preprocess_data.py")
        print("3. SageMakerで本格学習: python run_sagemaker.py")
    else:
        print("\n" + "=" * 60)
        print("✗ いくつかのテストに失敗しました")
        print("=" * 60)
        print("エラーを修正してから再実行してください")


if __name__ == '__main__':
    main()
