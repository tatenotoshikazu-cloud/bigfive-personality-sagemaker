"""
Big Five Personality Traits Dataset Loader
Quick access to Hugging Face datasets with Big Five labels

Usage:
    python load_big5_datasets.py --dataset essays
    python load_big5_datasets.py --dataset pandora --sample 1000
"""

import argparse
from datasets import load_dataset
import pandas as pd


def load_essays_big5(sample_size=None):
    """
    Load Essays-Big5 dataset (2,470 essays)

    Returns:
        dataset: HuggingFace Dataset with splits: train, validation, test

    Fields:
        - text: Essay text (string)
        - O, C, E, A, N: Binary labels (0/1)
        - ptype: Personality type (string)
    """
    print("Loading jingjietan/essays-big5...")
    dataset = load_dataset("jingjietan/essays-big5")

    if sample_size:
        dataset['train'] = dataset['train'].select(range(min(sample_size, len(dataset['train']))))

    print(f"✓ Loaded Essays-Big5")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")

    # Show example
    print("\nExample sample:")
    example = dataset['train'][0]
    print(f"  Text (first 100 chars): {example['text'][:100]}...")
    print(f"  Openness: {example['O']}")
    print(f"  Conscientiousness: {example['C']}")
    print(f"  Extraversion: {example['E']}")
    print(f"  Agreeableness: {example['A']}")
    print(f"  Neuroticism: {example['N']}")

    return dataset


def load_pandora_big5(sample_size=None):
    """
    Load PANDORA-Big5 dataset (3M+ Reddit comments)

    Returns:
        dataset: HuggingFace Dataset with splits: train, validation, test

    Fields:
        - text: Reddit comment (string)
        - O, C, E, A, N: Float scores (0-100)
        - ptype: Personality type (int 0-31)
    """
    print("Loading jingjietan/pandora-big5...")

    if sample_size:
        # Load only a subset
        dataset = load_dataset("jingjietan/pandora-big5", split={
            'train': f'train[:{sample_size}]',
            'validation': f'validation[:{sample_size//10}]',
            'test': f'test[:{sample_size//10}]'
        })
    else:
        dataset = load_dataset("jingjietan/pandora-big5")

    print(f"✓ Loaded PANDORA-Big5")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")

    # Show example
    print("\nExample sample:")
    example = dataset['train'][0]
    print(f"  Text (first 100 chars): {example['text'][:100]}...")
    print(f"  Openness: {example['O']:.1f}")
    print(f"  Conscientiousness: {example['C']:.1f}")
    print(f"  Extraversion: {example['E']:.1f}")
    print(f"  Agreeableness: {example['A']:.1f}")
    print(f"  Neuroticism: {example['N']:.1f}")

    return dataset


def load_automated_personality_prediction(sample_size=None):
    """
    Load Automated-Personality-Prediction dataset (20,877 Reddit comments)

    Returns:
        dataset: HuggingFace Dataset with splits: train, validation, test

    Fields:
        - text: Reddit comment (string)
        - agreeableness, openness, conscientiousness, extraversion, neuroticism: Float (0-99)
    """
    print("Loading Fatima0923/Automated-Personality-Prediction...")
    dataset = load_dataset("Fatima0923/Automated-Personality-Prediction")

    if sample_size:
        dataset['train'] = dataset['train'].select(range(min(sample_size, len(dataset['train']))))

    print(f"✓ Loaded Automated-Personality-Prediction")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")

    # Show example
    print("\nExample sample:")
    example = dataset['train'][0]
    print(f"  Text (first 100 chars): {example['text'][:100]}...")
    print(f"  Openness: {example['openness']:.1f}")
    print(f"  Conscientiousness: {example['conscientiousness']:.1f}")
    print(f"  Extraversion: {example['extraversion']:.1f}")
    print(f"  Agreeableness: {example['agreeableness']:.1f}")
    print(f"  Neuroticism: {example['neuroticism']:.1f}")

    return dataset


def load_synthetic_persona_chat(sample_size=None):
    """
    Load Synthetic-Persona-Chat dataset (20,000 conversations)

    Note: This dataset does NOT have explicit Big Five labels.
    It contains persona descriptions and conversations.

    Returns:
        dataset: HuggingFace Dataset with splits: train, validation, test

    Fields:
        - user_1_personas: Persona description (string)
        - user_2_personas: Persona description (string)
        - Best_Generated_Conversation: Conversation text (string)
    """
    print("Loading google/Synthetic-Persona-Chat...")
    dataset = load_dataset("google/Synthetic-Persona-Chat")

    if sample_size:
        dataset['train'] = dataset['train'].select(range(min(sample_size, len(dataset['train']))))

    print(f"✓ Loaded Synthetic-Persona-Chat")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")

    print("\n⚠️  Note: This dataset does NOT have Big Five labels.")
    print("    Use for persona-based conversation generation.")

    # Show example
    print("\nExample sample:")
    example = dataset['train'][0]
    print(f"  User 1 Persona: {example['user_1_personas'][:100]}...")
    print(f"  User 2 Persona: {example['user_2_personas'][:100]}...")
    print(f"  Conversation (first 100 chars): {example['Best_Generated_Conversation'][:100]}...")

    return dataset


def get_dataset_stats(dataset_name):
    """Print detailed statistics about a dataset"""

    loaders = {
        'essays': load_essays_big5,
        'pandora': load_pandora_big5,
        'automated': load_automated_personality_prediction,
        'persona': load_synthetic_persona_chat
    }

    if dataset_name not in loaders:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {', '.join(loaders.keys())}")
        return

    dataset = loaders[dataset_name](sample_size=1000)

    # Convert to pandas for analysis
    df = pd.DataFrame(dataset['train'])

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    # Text length statistics
    df['text_length'] = df['text'].apply(len)
    print("\nText Length Statistics:")
    print(df['text_length'].describe())

    # Big Five statistics (if available)
    if dataset_name == 'essays':
        traits = ['O', 'C', 'E', 'A', 'N']
        print("\nBig Five Distribution (Binary):")
        for trait in traits:
            count = df[trait].sum()
            percentage = (count / len(df)) * 100
            print(f"  {trait}: {count}/{len(df)} ({percentage:.1f}%)")

    elif dataset_name in ['pandora', 'automated']:
        if dataset_name == 'pandora':
            traits = ['O', 'C', 'E', 'A', 'N']
        else:
            traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

        print("\nBig Five Score Statistics (0-100):")
        print(df[traits].describe())

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Load Big Five Personality Datasets from Hugging Face')
    parser.add_argument('--dataset', type=str, default='essays',
                        choices=['essays', 'pandora', 'automated', 'persona'],
                        help='Dataset to load: essays, pandora, automated, or persona')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples to load (default: all)')
    parser.add_argument('--stats', action='store_true',
                        help='Show detailed statistics')

    args = parser.parse_args()

    print("="*60)
    print("BIG FIVE PERSONALITY TRAITS DATASET LOADER")
    print("="*60 + "\n")

    if args.stats:
        get_dataset_stats(args.dataset)
    else:
        loaders = {
            'essays': load_essays_big5,
            'pandora': load_pandora_big5,
            'automated': load_automated_personality_prediction,
            'persona': load_synthetic_persona_chat
        }

        dataset = loaders[args.dataset](sample_size=args.sample)

        print("\n" + "="*60)
        print("Dataset loaded successfully!")
        print("="*60)
        print("\nTo use this dataset in your code:")
        print(f"  from datasets import load_dataset")

        dataset_names = {
            'essays': 'jingjietan/essays-big5',
            'pandora': 'jingjietan/pandora-big5',
            'automated': 'Fatima0923/Automated-Personality-Prediction',
            'persona': 'google/Synthetic-Persona-Chat'
        }

        print(f"  dataset = load_dataset('{dataset_names[args.dataset]}')")

        print("\n" + "="*60)


if __name__ == "__main__":
    main()
