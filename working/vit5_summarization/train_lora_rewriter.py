"""
Train LoRA adapter for Vietnamese summary rewriting

This script:
1. Loads your trained ViT5/mT5 model
2. Generates summaries for all training documents
3. Trains LoRA adapter to rewrite mT5 → human quality
4. Evaluates before/after rewriting

Usage:
    python train_lora_rewriter.py --epochs 3 --batch_size 4
"""

import torch
import pandas as pd
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import evaluate
from typing import Optional

class LoRARewriterTrainer:
    """Train LoRA adapter for rewriting mT5 summaries"""

    def __init__(
        self,
        stage1_model: str = "VietAI/vit5-base",
        stage2_model: str = "Qwen/Qwen2.5-7B-Instruct",
        stage1_checkpoint: Optional[str] = None
    ):
        """
        Args:
            stage1_model: mT5/ViT5 base model
            stage2_model: Vietnamese LLM for rewriting
            stage1_checkpoint: Path to trained Stage 1 model (if available)
        """
        self.stage1_model_name = stage1_model
        self.stage2_model_name = stage2_model
        self.stage1_checkpoint = stage1_checkpoint or stage1_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def generate_mt5_summaries(
        self,
        documents: list[str],
        batch_size: int = 8
    ) -> list[str]:
        """
        Generate summaries using trained mT5/ViT5 model

        Args:
            documents: List of documents to summarize
            batch_size: Batch size for generation

        Returns:
            List of generated summaries
        """
        print(f"\n🔄 Generating {len(documents)} summaries with {self.stage1_model_name}...")

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(self.stage1_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.stage1_checkpoint)
        model = model.to(self.device)
        model.eval()

        summaries = []

        with torch.no_grad():
            for i in tqdm(range(0, len(documents), batch_size), desc="Generating"):
                batch_docs = documents[i:i+batch_size]

                # Add prefix and tokenize
                inputs = tokenizer(
                    ["tóm tắt: " + doc for doc in batch_docs],
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                # Generate
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                # Decode
                batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                summaries.extend(batch_summaries)

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()

        print(f"✅ Generated {len(summaries)} summaries")
        return summaries

    def create_training_dataset(
        self,
        documents: list[str],
        mt5_summaries: list[str],
        target_summaries: list[str]
    ) -> Dataset:
        """
        Create training dataset with prompt format

        Format:
        [PROMPT]
        Văn bản gốc: ...
        Tóm tắt cần viết lại: ...
        Tóm tắt đã cải thiện: [TARGET]
        """
        examples = []

        for doc, mt5_sum, target_sum in zip(documents, mt5_summaries, target_summaries):
            prompt = self._create_prompt(doc, mt5_sum, target_sum)
            examples.append({"text": prompt})

        return Dataset.from_list(examples)

    def _create_prompt(
        self,
        original_doc: str,
        mt5_summary: str,
        target_summary: Optional[str] = None
    ) -> str:
        """Create prompt for training/inference"""

        # Truncate document to first 500 chars for context
        doc_preview = original_doc[:500] + "..." if len(original_doc) > 500 else original_doc

        prompt = f"""Bạn là chuyên gia viết lại văn bản tiếng Việt. Nhiệm vụ: cải thiện bản tóm tắt sau.

Yêu cầu:
- Giữ nguyên thông tin và ý nghĩa
- Cải thiện sự tự nhiên và mạch lạc
- Sử dụng từ ngữ phù hợp tiếng Việt
- Ngắn gọn, súc tích

VĂN BẢN GỐC:
{doc_preview}

TÓM TẮT CẦN VIẾT LẠI:
{mt5_summary}

TÓM TẮT ĐÃ CẢI THIỆN:
"""
        if target_summary:
            prompt += target_summary

        return prompt

    def train(
        self,
        train_data_path: str = "data/train.csv",
        val_data_path: str = "data/validation.csv",
        output_dir: str = "./lora_rewriter",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None
    ):
        """
        Main training function

        Args:
            train_data_path: Path to train.csv
            val_data_path: Path to validation.csv
            output_dir: Where to save LoRA weights
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_train_samples: Limit training samples (for testing)
            max_val_samples: Limit validation samples
        """
        print("=" * 80)
        print("🚀 TRAINING LORA ADAPTER FOR VIETNAMESE SUMMARY REWRITING")
        print("=" * 80)

        # Step 1: Load data
        print(f"\n📥 Loading data...")
        train_df = pd.read_csv(train_data_path)
        val_df = pd.read_csv(val_data_path)

        if max_train_samples:
            train_df = train_df.head(max_train_samples)
        if max_val_samples:
            val_df = val_df.head(max_val_samples)

        print(f"   Train: {len(train_df):,} samples")
        print(f"   Val: {len(val_df):,} samples")

        # Step 2: Generate mT5 summaries
        train_mt5_sums = self.generate_mt5_summaries(train_df['document'].tolist())
        val_mt5_sums = self.generate_mt5_summaries(val_df['document'].tolist())

        # Step 3: Create training datasets
        print(f"\n📝 Creating training datasets...")
        train_dataset = self.create_training_dataset(
            train_df['document'].tolist(),
            train_mt5_sums,
            train_df['summary'].tolist()
        )

        val_dataset = self.create_training_dataset(
            val_df['document'].tolist(),
            val_mt5_sums,
            val_df['summary'].tolist()
        )

        print(f"   Train examples: {len(train_dataset)}")
        print(f"   Val examples: {len(val_dataset)}")

        # Step 4: Load LLM with 4-bit quantization
        print(f"\n📥 Loading LLM: {self.stage2_model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.stage2_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(self.stage2_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Step 5: Apply LoRA
        print(f"\n🔧 Applying LoRA...")

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Step 6: Tokenize
        print(f"\n🔄 Tokenizing...")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding="max_length"
            )

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)

        # Step 7: Train
        print(f"\n🚀 Starting training...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        trainer.train()

        # Step 8: Save
        print(f"\n💾 Saving LoRA adapter...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✅ Training complete!")
        print(f"   Saved to: {output_dir}")

        return model, tokenizer

    def evaluate(
        self,
        test_data_path: str = "data/test.csv",
        lora_checkpoint: Optional[str] = None,
        num_samples: int = 100
    ):
        """
        Evaluate rewriting quality

        Compares:
        - Original mT5 summaries
        - Rewritten summaries (with LoRA)
        - Human summaries (ground truth)
        """
        print("\n" + "=" * 80)
        print("📊 EVALUATION")
        print("=" * 80)

        # Load test data
        test_df = pd.read_csv(test_data_path).head(num_samples)
        print(f"\nEvaluating on {len(test_df)} samples...")

        # Generate mT5 summaries
        mt5_summaries = self.generate_mt5_summaries(test_df['document'].tolist())

        # Rewrite with LLM (if LoRA checkpoint provided)
        if lora_checkpoint:
            print(f"\n🔄 Rewriting with LoRA model...")
            # TODO: Implement rewriting with trained LoRA
            rewritten_summaries = mt5_summaries  # Placeholder
        else:
            print(f"\n⚠️  No LoRA checkpoint - skipping rewriting")
            rewritten_summaries = None

        # Compute ROUGE scores
        rouge = evaluate.load("rouge")

        # mT5 vs Human
        mt5_results = rouge.compute(
            predictions=mt5_summaries,
            references=test_df['summary'].tolist()
        )

        print(f"\n📊 Results (mT5 only):")
        print(f"   ROUGE-1: {mt5_results['rouge1']:.4f}")
        print(f"   ROUGE-2: {mt5_results['rouge2']:.4f}")
        print(f"   ROUGE-L: {mt5_results['rougeL']:.4f}")

        if rewritten_summaries:
            rewritten_results = rouge.compute(
                predictions=rewritten_summaries,
                references=test_df['summary'].tolist()
            )

            print(f"\n📊 Results (mT5 + LoRA rewrite):")
            print(f"   ROUGE-1: {rewritten_results['rouge1']:.4f} ({rewritten_results['rouge1'] - mt5_results['rouge1']:+.4f})")
            print(f"   ROUGE-2: {rewritten_results['rouge2']:.4f} ({rewritten_results['rouge2'] - mt5_results['rouge2']:+.4f})")
            print(f"   ROUGE-L: {rewritten_results['rougeL']:.4f} ({rewritten_results['rougeL'] - mt5_results['rougeL']:+.4f})")

        # Show examples
        print(f"\n📝 Sample Comparisons:")
        for i in range(min(3, len(test_df))):
            print(f"\n{'='*80}")
            print(f"Example {i+1}")
            print(f"{'='*80}")
            print(f"\n📄 Original: {test_df.iloc[i]['document'][:200]}...")
            print(f"\n📝 mT5: {mt5_summaries[i]}")
            if rewritten_summaries:
                print(f"\n✨ Rewritten: {rewritten_summaries[i]}")
            print(f"\n👤 Human: {test_df.iloc[i]['summary']}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA for Vietnamese summary rewriting")
    parser.add_argument("--stage1_model", default="VietAI/vit5-base", help="Stage 1 model")
    parser.add_argument("--stage1_checkpoint", default=None, help="Trained Stage 1 checkpoint")
    parser.add_argument("--stage2_model", default="Qwen/Qwen2.5-7B-Instruct", help="Stage 2 LLM")
    parser.add_argument("--train_data", default="data/train.csv", help="Training data")
    parser.add_argument("--val_data", default="data/validation.csv", help="Validation data")
    parser.add_argument("--test_data", default="data/test.csv", help="Test data")
    parser.add_argument("--output_dir", default="./lora_rewriter", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_train", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val", type=int, default=None, help="Max val samples")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate")
    parser.add_argument("--lora_checkpoint", default=None, help="LoRA checkpoint for eval")

    args = parser.parse_args()

    # Initialize trainer
    trainer = LoRARewriterTrainer(
        stage1_model=args.stage1_model,
        stage2_model=args.stage2_model,
        stage1_checkpoint=args.stage1_checkpoint
    )

    if args.eval_only:
        # Evaluation only
        trainer.evaluate(
            test_data_path=args.test_data,
            lora_checkpoint=args.lora_checkpoint
        )
    else:
        # Training
        trainer.train(
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_train_samples=args.max_train,
            max_val_samples=args.max_val
        )

        # Evaluate after training
        trainer.evaluate(
            test_data_path=args.test_data,
            lora_checkpoint=args.output_dir
        )


if __name__ == "__main__":
    main()
