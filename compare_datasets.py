"""
Big Five Dataset Comparison Script
Compare all available datasets side-by-side
"""

import pandas as pd
from tabulate import tabulate


def compare_datasets():
    """Compare all Big Five datasets available on Hugging Face"""

    datasets_info = [
        {
            'Dataset Name': 'jingjietan/essays-big5',
            'Short Name': 'essays-big5',
            'Total Size': '2,470',
            'Train/Val/Test': '1,580 / 395 / 494',
            'Text Type': 'Essays',
            'Label Type': 'Binary (0/1)',
            'Text Source': 'College students',
            'Text Length': '217-12,900 chars',
            'License': 'Apache 2.0',
            'File Size': '~5 MB',
            'Download Speed': 'Fast',
            'Recommended For': 'Small projects, experiments',
            'Pros': 'Easy to use, clean data, binary labels',
            'Cons': 'Small size, limited diversity'
        },
        {
            'Dataset Name': 'jingjietan/pandora-big5',
            'Short Name': 'pandora-big5',
            'Total Size': '3,006,566',
            'Train/Val/Test': '1.92M / 481k / 601k',
            'Text Type': 'Reddit comments',
            'Label Type': 'Float (0-100)',
            'Text Source': 'Reddit users (10k+)',
            'Text Length': '1-48,000 chars',
            'License': 'Apache 2.0',
            'File Size': '511 MB',
            'Download Speed': 'Slow',
            'Recommended For': 'Large-scale DL, fine-tuning',
            'Pros': 'Very large, continuous scores, diverse',
            'Cons': 'Large size, slow download'
        },
        {
            'Dataset Name': 'Fatima0923/Automated-Personality-Prediction',
            'Short Name': 'automated-personality',
            'Total Size': '20,877',
            'Train/Val/Test': '16,000 / 2,420 / 2,420',
            'Text Type': 'Reddit comments',
            'Label Type': 'Float (0-99)',
            'Text Source': 'Reddit (1,608 users)',
            'Text Length': '16-8,880 chars',
            'License': 'Unknown',
            'File Size': '6.02 MB',
            'Download Speed': 'Fast',
            'Recommended For': 'Medium projects, Reddit analysis',
            'Pros': 'Good size, Reddit data, curated',
            'Cons': 'License unclear'
        },
        {
            'Dataset Name': 'google/Synthetic-Persona-Chat',
            'Short Name': 'persona-chat',
            'Total Size': '20,000',
            'Train/Val/Test': '8,940 / 1,000 / 968',
            'Text Type': 'Conversations',
            'Label Type': 'No Big Five labels',
            'Text Source': 'Synthetic personas',
            'Text Length': '388-3,460 chars',
            'License': 'CC-BY-4.0',
            'File Size': '~10 MB',
            'Download Speed': 'Fast',
            'Recommended For': 'Dialogue generation',
            'Pros': 'Persona-based, Google quality',
            'Cons': 'No Big Five labels'
        }
    ]

    df = pd.DataFrame(datasets_info)

    print("="*100)
    print("BIG FIVE PERSONALITY DATASETS COMPARISON")
    print("="*100 + "\n")

    # Basic info table
    basic_cols = ['Short Name', 'Total Size', 'Text Type', 'Label Type', 'File Size', 'License']
    print("\n📊 BASIC INFORMATION")
    print("-"*100)
    print(tabulate(df[basic_cols], headers='keys', tablefmt='grid', showindex=False))

    # Technical details
    tech_cols = ['Short Name', 'Train/Val/Test', 'Text Length', 'Download Speed']
    print("\n\n⚙️  TECHNICAL DETAILS")
    print("-"*100)
    print(tabulate(df[tech_cols], headers='keys', tablefmt='grid', showindex=False))

    # Recommendations
    rec_cols = ['Short Name', 'Recommended For', 'Pros', 'Cons']
    print("\n\n💡 RECOMMENDATIONS")
    print("-"*100)
    print(tabulate(df[rec_cols], headers='keys', tablefmt='grid', showindex=False))

    print("\n\n" + "="*100)
    print("QUICK SELECTION GUIDE")
    print("="*100)

    guide = """
    🎯 Use Case Based Selection:

    1. Learning & Small Projects:
       → essays-big5 (2,470 samples)
       ✓ Fast download, easy to work with
       ✓ Perfect for learning and prototyping

    2. Production Deep Learning:
       → pandora-big5 (3M+ samples)
       ✓ Large-scale training data
       ✓ Fine-tuning LLMs
       ⚠️  Requires significant compute resources

    3. Medium-Scale Projects:
       → automated-personality (20,877 samples)
       ✓ Good balance of size and quality
       ✓ Reddit conversation style

    4. Dialogue Systems:
       → persona-chat (20,000 conversations)
       ⚠️  No Big Five labels (need to add manually)
       ✓ Good for persona-based chatbots

    💾 Storage Requirements:
       - essays-big5: ~5 MB
       - automated-personality: ~6 MB
       - persona-chat: ~10 MB
       - pandora-big5: 511 MB ⚠️

    ⏱️  Download Time (estimated):
       - essays-big5: < 10 seconds
       - automated-personality: < 10 seconds
       - persona-chat: < 30 seconds
       - pandora-big5: 2-5 minutes (depending on connection)
    """

    print(guide)

    print("\n" + "="*100)
    print("LOAD COMMANDS")
    print("="*100 + "\n")

    for _, row in df.iterrows():
        print(f"# {row['Short Name']}")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{row['Dataset Name']}')")
        print()

    print("="*100)


if __name__ == "__main__":
    try:
        compare_datasets()
    except ImportError:
        print("Please install tabulate: pip install tabulate")
        print("\nFalling back to simple comparison...\n")

        # Simple version without tabulate
        datasets = [
            ("essays-big5", "jingjietan/essays-big5", "2,470", "Essays", "Binary", "Small projects"),
            ("pandora-big5", "jingjietan/pandora-big5", "3M+", "Reddit", "Float", "Large-scale DL"),
            ("automated-personality", "Fatima0923/Automated-Personality-Prediction", "20,877", "Reddit", "Float", "Medium projects"),
            ("persona-chat", "google/Synthetic-Persona-Chat", "20,000", "Conversations", "None", "Dialogue gen")
        ]

        print("BIG FIVE DATASETS COMPARISON\n")
        print(f"{'Short Name':<25} {'Full Name':<50} {'Size':<10} {'Type':<12} {'Labels':<8} {'Best For'}")
        print("-" * 140)

        for short, full, size, text_type, labels, best_for in datasets:
            print(f"{short:<25} {full:<50} {size:<10} {text_type:<12} {labels:<8} {best_for}")

        print("\n" + "="*140)
        print("RECOMMENDATION: Start with 'essays-big5' for learning, use 'pandora-big5' for production")
        print("="*140)
