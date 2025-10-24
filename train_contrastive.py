#!/usr/bin/env python3
"""
Stage 1: Contrastive Learning for Persona Representation
Nemotron-Personas-Japanデータでペルソナテキストの表現を学習
Big Fiveラベル不要！
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np


class ContrastivePersonaDataset(Dataset):
    """
    Contrastive Learning用データセット
    同じペルソナから2つの異なる視点（augmentation）を生成
    """

    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ペルソナの各側面カラム
        self.persona_columns = [
            'persona', 'professional_persona', 'sports_persona',
            'arts_persona', 'travel_persona', 'culinary_persona'
        ]

    def __len__(self):
        return len(self.dataset)

    def augment_persona(self, example):
        """
        データ拡張: 同じペルソナの異なる側面を選択
        これにより、同じ人物の表現が近くなるように学習
        """
        # 利用可能なペルソナテキストを収集
        available_texts = []
        for col in self.persona_columns:
            if col in example and example[col]:
                text = str(example[col]).strip()
                if text and len(text) > 10:  # 最小長チェック
                    available_texts.append(text)

        # 最低2つのテキストが必要
        if len(available_texts) < 2:
            # フォールバック: 同じテキストを使用
            text1 = text2 = available_texts[0] if available_texts else "No persona available"
        else:
            # ランダムに2つ選択
            indices = np.random.choice(len(available_texts), size=2, replace=False)
            text1 = available_texts[indices[0]]
            text2 = available_texts[indices[1]]

        return text1, text2

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # 同じペルソナから2つの視点を取得
        text1, text2 = self.augment_persona(example)

        # トークナイズ
        encoding1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids_1': encoding1['input_ids'].squeeze(0),
            'attention_mask_1': encoding1['attention_mask'].squeeze(0),
            'input_ids_2': encoding2['input_ids'].squeeze(0),
            'attention_mask_2': encoding2['attention_mask'].squeeze(0),
        }


class ContrastivePersonaModel(nn.Module):
    """
    Contrastive Learning用モデル
    SimCLRベースの対照学習
    """

    def __init__(self, base_model, projection_dim=128):
        super().__init__()
        self.encoder = base_model
        self.hidden_dim = base_model.config.hidden_size

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Project to contrastive space
        projection = self.projection(cls_embedding)

        return F.normalize(projection, dim=1)  # L2 normalize


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
    SimCLRで使用される対照損失
    """
    batch_size = z1.shape[0]

    # 全ての表現を結合
    z = torch.cat([z1, z2], dim=0)  # [2*batch_size, projection_dim]

    # 類似度行列を計算
    similarity_matrix = torch.matmul(z, z.T) / temperature  # [2*batch_size, 2*batch_size]

    # 対角線をマスク（自分自身との類似度を除外）
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])  # [2*batch_size]

    # Cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def train_epoch(model, dataloader, optimizer, scheduler, device, temperature=0.5):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # 2つの視点のペアを取得
        input_ids_1 = batch['input_ids_1'].to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        input_ids_2 = batch['input_ids_2'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)

        # Forward pass for both views
        z1 = model(input_ids_1, attention_mask_1)
        z2 = model(input_ids_2, attention_mask_2)

        # Contrastive loss
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device, temperature=0.5):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)

            z1 = model(input_ids_1, attention_mask_1)
            z2 = model(input_ids_2, attention_mask_2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model-name', type=str, default='xlm-roberta-large')
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)

    # LoRA parameters
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.1)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--warmup-steps', type=int, default=500)

    # Data parameters
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/local/processed/train'))
    parser.add_argument('--val-data', type=str, default=os.environ.get('SM_CHANNEL_VAL', 'data/local/processed/val'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './output/stage1'))

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load datasets
    print(f"Loading train data from: {args.train_data}")
    train_dataset = load_from_disk(args.train_data)
    print(f"Loading val data from: {args.val_data}")
    val_dataset = load_from_disk(args.val_data)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create contrastive datasets
    train_contrastive = ContrastivePersonaDataset(train_dataset, tokenizer, args.max_length)
    val_contrastive = ContrastivePersonaDataset(val_dataset, tokenizer, args.max_length)

    # Dataloaders
    train_loader = DataLoader(train_contrastive, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_contrastive, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load base model
    print(f"Loading base model: {args.model_name}")
    base_model = AutoModel.from_pretrained(args.model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Create contrastive model
    model = ContrastivePersonaModel(base_model, projection_dim=args.projection_dim)
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.temperature)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device, args.temperature)
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best val loss: {val_loss:.4f}")

            # Save model
            os.makedirs(args.model_dir, exist_ok=True)
            model.encoder.save_pretrained(args.model_dir)
            tokenizer.save_pretrained(args.model_dir)

            # Save training info
            with open(os.path.join(args.model_dir, 'training_info.json'), 'w') as f:
                json.dump({
                    'stage': 1,
                    'type': 'contrastive_learning',
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'args': vars(args)
                }, f, indent=2)

            print(f"Model saved to: {args.model_dir}")

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.model_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
