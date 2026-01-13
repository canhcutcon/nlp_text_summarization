"""
Data Augmentation for Vietnamese Text Summarization
Generates additional training data through back-translation and paraphrasing
"""
import pandas as pd
import random
import re
from typing import List, Tuple

class VietnameseDataAugmenter:
    """
    Data augmentation strategies for Vietnamese summarization:
    1. Sentence shuffling (for multi-sentence documents)
    2. Synonym replacement (basic word-level)
    3. Summary variation (rephrasing)
    """

    def __init__(self, seed=42):
        random.seed(seed)

        # Common Vietnamese synonyms for augmentation
        self.synonyms = {
            "tốt": ["tuyệt vời", "xuất sắc", "hay"],
            "xấu": ["tồi", "kém", "dở"],
            "lớn": ["to", "khổng lồ", "rộng lớn"],
            "nhỏ": ["bé", "tí hon", "nhỏ bé"],
            "nhanh": ["mau", "nhanh chóng", "tốc độ"],
            "chậm": ["lậu", "chậm chạp", "từ từ"],
            "đẹp": ["xinh", "đẹp đẽ", "lung linh"],
            "quan trọng": ["thiết yếu", "cần thiết", "chủ yếu"],
            "cần": ["cần thiết", "thiết yếu", "phải"],
            "nhiều": ["đông", "phong phú", "dồi dào"],
            "ít": ["hiếm", "khan hiếm", "thiếu"],
        }

    def sentence_shuffle(self, text: str, shuffle_prob=0.3) -> str:
        """
        Shuffle sentences in document (useful for extractive-style documents)
        Only shuffle with certain probability to maintain coherence
        """
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 2 or random.random() > shuffle_prob:
            return text

        # Keep first sentence fixed, shuffle the rest
        first = sentences[0]
        rest = sentences[1:]
        random.shuffle(rest)

        return first + " " + " ".join(rest)

    def synonym_replace(self, text: str, replace_prob=0.1) -> str:
        """
        Replace words with synonyms
        """
        words = text.split()
        new_words = []

        for word in words:
            # Check if word (lowercase) has synonyms
            word_lower = word.lower()
            if word_lower in self.synonyms and random.random() < replace_prob:
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)

        return " ".join(new_words)

    def augment_pair(self, document: str, summary: str) -> Tuple[str, str]:
        """
        Augment a document-summary pair
        Returns augmented (document, summary)
        """
        # Apply augmentation with certain probability
        aug_document = document
        aug_summary = summary

        # Document augmentation
        if random.random() < 0.3:
            aug_document = self.sentence_shuffle(aug_document)
        if random.random() < 0.2:
            aug_document = self.synonym_replace(aug_document, replace_prob=0.05)

        # Summary augmentation (lighter touch)
        if random.random() < 0.15:
            aug_summary = self.synonym_replace(aug_summary, replace_prob=0.05)

        return aug_document, aug_summary


def augment_dataset(input_csv: str, output_csv: str, num_augmented: int = 3000):
    """
    Augment dataset by creating variations of existing samples

    Args:
        input_csv: Path to original CSV file
        output_csv: Path to save augmented CSV
        num_augmented: Number of augmented samples to create
    """
    print(f"📊 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"   Original samples: {len(df):,}")

    augmenter = VietnameseDataAugmenter()

    augmented_data = []

    print(f"🔄 Generating {num_augmented:,} augmented samples...")
    for i in range(num_augmented):
        # Randomly select a sample
        idx = random.randint(0, len(df) - 1)
        original_doc = df.iloc[idx]['document']
        original_sum = df.iloc[idx]['summary']

        # Augment
        aug_doc, aug_sum = augmenter.augment_pair(original_doc, original_sum)

        augmented_data.append({
            'document': aug_doc,
            'summary': aug_sum
        })

        if (i + 1) % 500 == 0:
            print(f"   Generated {i+1:,} / {num_augmented:,} samples")

    # Create augmented dataframe
    aug_df = pd.DataFrame(augmented_data)

    # Combine with original
    combined_df = pd.concat([df, aug_df], ignore_index=True)

    print(f"\n💾 Saving to {output_csv}...")
    combined_df.to_csv(output_csv, index=False)

    print(f"✅ Done!")
    print(f"   Original: {len(df):,}")
    print(f"   Augmented: {len(aug_df):,}")
    print(f"   Total: {len(combined_df):,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment Vietnamese summarization dataset")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/train_augmented.csv", help="Output CSV file")
    parser.add_argument("--num", type=int, default=3000, help="Number of augmented samples")

    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.num)

    # Example usage
    print("\n" + "="*70)
    print("Example augmentation:")
    print("="*70)

    augmenter = VietnameseDataAugmenter()

    doc = "Hà Nội là thủ đô của Việt Nam. Đây là một thành phố lớn với nhiều di tích lịch sử quan trọng."
    summ = "Hà Nội là thủ đô Việt Nam, có nhiều di tích lịch sử."

    print(f"\nOriginal document:")
    print(f"  {doc}")
    print(f"\nOriginal summary:")
    print(f"  {summ}")

    aug_doc, aug_sum = augmenter.augment_pair(doc, summ)

    print(f"\nAugmented document:")
    print(f"  {aug_doc}")
    print(f"\nAugmented summary:")
    print(f"  {aug_sum}")
