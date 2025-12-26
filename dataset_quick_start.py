"""
Quick Start: Load Vietnamese Summarization Dataset
Use this instead of curl or git clone commands
"""

from datasets import load_dataset
import pandas as pd

def load_vietnamese_dataset():
    """
    Load the Vietnamese summarization dataset from Hugging Face
    Dataset: 8Opt/vietnamese-summarization-dataset-0001

    Returns:
        datasets.DatasetDict: Dictionary with train, validation, and test splits
    """
    print("Loading Vietnamese Summarization Dataset...")
    print("=" * 80)

    # Load the entire dataset
    dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

    print(f"✅ Dataset loaded successfully!")
    print(f"\nDataset structure:")
    print(dataset)

    print(f"\nDataset splits:")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data):,} samples")

    print(f"\nFeatures: {list(dataset['train'].features.keys())}")

    return dataset


def show_examples(dataset, n=3):
    """Show sample examples from the dataset"""
    print("\n" + "=" * 80)
    print(f"Sample Examples (showing {n} from training set)")
    print("=" * 80)

    for i in range(n):
        example = dataset['train'][i]
        print(f"\n--- Example {i+1} ---")
        print(f"Document (first 150 chars): {example['document'][:150]}...")
        print(f"Summary: {example['summary'][:150]}...")
        print(f"Keywords: {example['keywords']}")


def get_dataloader_ready_format(dataset, tokenizer, max_length=512, max_target_length=128):
    """
    Prepare dataset for training with a tokenizer

    Args:
        dataset: The loaded dataset
        tokenizer: HuggingFace tokenizer
        max_length: Max length for input documents
        max_target_length: Max length for summaries

    Returns:
        Tokenized dataset ready for training
    """
    def preprocess_function(examples):
        # For T5-based models, add prefix
        inputs = [f"summarize: {doc}" for doc in examples['document']]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )

        # Tokenize targets
        labels = tokenizer(
            examples['summary'],
            max_length=max_target_length,
            truncation=True,
            padding='max_length'
        )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    return tokenized_dataset


def convert_to_csv(dataset, output_dir='./data_csv'):
    """
    Convert dataset to CSV files (optional)

    Args:
        dataset: The loaded dataset
        output_dir: Output directory for CSV files
    """
    import os

    print(f"\n{'=' * 80}")
    print(f"Converting dataset to CSV format...")
    print(f"{'=' * 80}")

    os.makedirs(output_dir, exist_ok=True)

    for split_name in dataset.keys():
        output_path = os.path.join(output_dir, f'{split_name}.csv')
        df = dataset[split_name].to_pandas()

        # Convert list column to string for CSV
        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Saved {split_name}.csv ({len(df):,} samples)")

    print(f"\nCSV files saved to: {output_dir}/")


if __name__ == "__main__":
    # Load dataset
    dataset = load_vietnamese_dataset()

    # Show examples
    show_examples(dataset)

    # Optional: Convert to CSV
    print("\n" + "=" * 80)
    convert = input("Convert to CSV format? (y/n): ").lower()
    if convert == 'y':
        convert_to_csv(dataset)

    print("\n" + "=" * 80)
    print("✅ Ready to use!")
    print("\nTo use in your training script:")
    print("""
from datasets import load_dataset

# Load dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Get a sample
sample = train_data[0]
print(sample["document"])
print(sample["summary"])
print(sample["keywords"])
    """)
