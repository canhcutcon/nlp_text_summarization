"""
Load and explore Vietnamese Summarization Dataset from Hugging Face
Dataset: 8Opt/vietnamese-summarization-dataset-0001
"""

from datasets import load_dataset
import pandas as pd

def load_and_explore_dataset():
    """Load the Vietnamese summarization dataset and display basic information"""

    print("=" * 80)
    print("Loading Vietnamese Summarization Dataset from Hugging Face")
    print("Dataset: 8Opt/vietnamese-summarization-dataset-0001")
    print("=" * 80)

    # Load dataset from Hugging Face
    print("\n📥 Loading dataset...")
    dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

    print("\n✅ Dataset loaded successfully!")
    print("\n📊 Dataset Structure:")
    print(dataset)

    # Explore each split
    for split_name in dataset.keys():
        print(f"\n{'=' * 80}")
        print(f"📂 Split: {split_name.upper()}")
        print(f"{'=' * 80}")

        split_data = dataset[split_name]

        # Basic statistics
        print(f"Number of samples: {len(split_data):,}")
        print(f"Features: {split_data.features}")

        # Show first 3 examples
        print(f"\n📝 Sample data from {split_name}:")
        for i in range(min(3, len(split_data))):
            print(f"\n--- Example {i+1} ---")
            example = split_data[i]
            for key, value in example.items():
                if isinstance(value, str):
                    # Truncate long text for display
                    display_value = value[:200] + "..." if len(value) > 200 else value
                    print(f"{key}: {display_value}")
                else:
                    print(f"{key}: {value}")

    # Convert to pandas for additional analysis
    print(f"\n{'=' * 80}")
    print("📊 Converting to Pandas DataFrame for analysis")
    print(f"{'=' * 80}")

    for split_name in dataset.keys():
        df = dataset[split_name].to_pandas()
        print(f"\n{split_name.upper()} split:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Calculate text statistics
        if 'article' in df.columns or 'text' in df.columns:
            text_col = 'article' if 'article' in df.columns else 'text'
            summary_col = 'summary' if 'summary' in df.columns else 'highlights'

            if text_col in df.columns:
                df['text_length'] = df[text_col].str.len()
                print(f"\n{text_col.capitalize()} length statistics:")
                print(df['text_length'].describe())

            if summary_col in df.columns:
                df['summary_length'] = df[summary_col].str.len()
                print(f"\n{summary_col.capitalize()} length statistics:")
                print(df['summary_length'].describe())

    return dataset


def save_as_csv(dataset, output_dir='./data'):
    """Save dataset splits as CSV files"""
    import os

    print(f"\n{'=' * 80}")
    print(f"💾 Saving dataset as CSV files to {output_dir}")
    print(f"{'=' * 80}")

    os.makedirs(output_dir, exist_ok=True)

    for split_name in dataset.keys():
        output_path = os.path.join(output_dir, f'{split_name}.csv')
        df = dataset[split_name].to_pandas()
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Saved {split_name} split to {output_path} ({len(df):,} samples)")


if __name__ == "__main__":
    # Load and explore the dataset
    dataset = load_and_explore_dataset()

    # Optionally save as CSV
    print("\n" + "=" * 80)
    save_csv = input("Do you want to save the dataset as CSV files? (y/n): ")
    if save_csv.lower() == 'y':
        save_as_csv(dataset)

    print("\n✨ Done!")
