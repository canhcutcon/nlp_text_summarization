"""
Script to create a clean, refactored notebook for mT5 Vietnamese summarization
"""
import json

# Create clean notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def create_cell(cell_type, source, outputs=None):
    """Helper to create a cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell

# Add cells in order
cells = [
    # Header
    create_cell("markdown", """# Vietnamese Text Summarization - mT5-Small Fine-tuning

✅ **Model**: google/mt5-small (300M params)
✅ **Task**: Abstractive Summarization for Vietnamese
✅ **Strategy**: Properly structured seq2seq with optimized hyperparameters
✅ **Dataset**: Vietnamese documents with human-written summaries

---

## Key Improvements in This Version

1. **Standardized Summarization Task Format**
   - Proper prefix: "tóm tắt: " for all inputs
   - Consistent max lengths (input: 512, output: 128)

2. **Stable Training Configuration**
   - Learning rate: 2e-4 (optimal for mT5)
   - Batch size: 2 with gradient accumulation: 8 (effective batch: 16)
   - FP16 enabled on CUDA GPUs
   - Warmup steps: 500

3. **Comprehensive Evaluation**
   - ROUGE-1, ROUGE-2, ROUGE-L metrics
   - Sample output inspection (metrics aren't everything!)

4. **Optimized Inference**
   - Beam search: 4-6 beams
   - Length penalty: 1.0-1.5
   - Repetition penalty: 1.2"""),

    # 1. Install Packages
    create_cell("markdown", "## 1. Install Packages"),
    create_cell("code", """# Install required packages
!pip install -q transformers datasets accelerate sentencepiece evaluate rouge-score py-rouge torch --root-user-action=ignore

print("✅ All packages installed!")"""),

    # 2. GPU Setup
    create_cell("markdown", "## 2. GPU/Device Setup"),
    create_cell("code", """import torch
import gc

print("=" * 60)
print("🎯 GPU/Device Setup")
print("=" * 60)

# Clear GPU cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    USE_FP16 = True
    USE_GRAD_CHECKPOINT = True
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (training will be very slow)")
    print("   💡 Consider using Google Colab with free GPU")
    USE_FP16 = False
    USE_GRAD_CHECKPOINT = False

print(f"\\n📊 Configuration:")
print(f"   Device: {device}")
print(f"   FP16: {USE_FP16}")
print(f"   Gradient Checkpointing: {USE_GRAD_CHECKPOINT}")"""),

    # 3. Load Data
    create_cell("markdown", "## 3. Load and Verify Data"),
    create_cell("code", """import re
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

# Load dataset
print("📊 Loading Vietnamese Summarization Dataset...")
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print(f"✓ Train: {len(train_df):,} samples")
print(f"✓ Validation: {len(val_df):,} samples")
print(f"✓ Test: {len(test_df):,} samples")

# Analyze data statistics
def analyze_lengths(df: pd.DataFrame, name: str):
    doc_words = df['document'].apply(lambda x: len(x.split()))
    sum_words = df['summary'].apply(lambda x: len(x.split()))
    compression_ratio = (sum_words.mean() / doc_words.mean() * 100)

    print(f"\\n{name}:")
    print(f"  Avg document: {doc_words.mean():.0f} words, Avg summary: {sum_words.mean():.0f} words")
    print(f"  Compression ratio: {compression_ratio:.1f}%")

analyze_lengths(train_df, "Train")
analyze_lengths(val_df, "Validation")
analyze_lengths(test_df, "Test")

# Convert to HuggingFace Dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df[['document', 'summary']], preserve_index=False),
    'validation': Dataset.from_pandas(val_df[['document', 'summary']], preserve_index=False),
    'test': Dataset.from_pandas(test_df[['document', 'summary']], preserve_index=False)
})

print(f"\\n📝 Sample:")
sample = dataset['train'][0]
print(f"Document: {sample['document'][:200]}...")
print(f"Summary: {sample['summary'][:150]}...")"""),

    # 4. Load Model
    create_cell("markdown", "## 4. Load mT5-Small Model"),
    create_cell("code", """from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate

# Load mT5-Small model and tokenizer
MODEL_NAME = "google/mt5-small"

print(f"📥 Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"✅ Model loaded successfully!")
print(f"   Parameters: {model.num_parameters():,}")
print(f"   Vocab size: {tokenizer.vocab_size:,}")

# Move to device
model = model.to(device)
print(f"   Device: {device}")"""),

    # 5. Tokenize Data
    create_cell("markdown", """## 5. Chuẩn Hoá Dữ Liệu (Standardize Data)

⚠️ **RẤT QUAN TRỌNG**: Task prefix "tóm tắt: " để model hiểu đây là task summarization"""),
    create_cell("code", """def preprocess_function(examples):
    \"\"\"
    Chuẩn hoá dữ liệu cho bài toán tóm tắt:
    - Thêm prefix "tóm tắt: " vào đầu document
    - Tokenize input với max_length=512
    - Tokenize output (summary) với max_length=128
    \"\"\"
    # Thêm task prefix
    inputs = ["tóm tắt: " + doc for doc in examples["document"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False  # Dynamic padding sẽ được xử lý bởi DataCollator
    )

    # Tokenize targets/labels
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=128,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("🔄 Tokenizing dataset...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

# Verify tokenization
sample = tokenized_datasets["train"][0]
print(f"\\n✅ Tokenization complete!")
print(f"   Input length: {len(sample['input_ids'])} tokens")
print(f"   Label length: {len(sample['labels'])} tokens")
print(f"\\n   Sample input: {tokenizer.decode(sample['input_ids'][:100])}")
print(f"   Sample label: {tokenizer.decode(sample['labels'][:50])}")"""),

    # 6. Define Metrics
    create_cell("markdown", """## 6. Define Metrics

⚠️ **CHÚ Ý**: ROUGE cao ≠ tóm tắt hay. Luôn đọc sample output bằng mắt người!"""),
    create_cell("code", """rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    \"\"\"
    Compute ROUGE scores và hiển thị sample predictions
    để đánh giá chất lượng thực tế
    \"\"\"
    predictions, labels = eval_pred

    # Nếu predictions là logits, lấy argmax
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (padding tokens)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 👁️ LUÔN HIỂN THỊ SAMPLE để kiểm tra chất lượng thực tế
    if len(decoded_preds) > 0:
        print(f"\\n{'='*70}")
        print("📝 SAMPLE PREDICTION (để đánh giá chất lượng thực tế):")
        print(f"{'='*70}")
        print(f"Prediction: {decoded_preds[0][:200]}")
        print(f"Reference:  {decoded_labels[0][:200]}")
        print(f"{'='*70}\\n")

    # Clean text
    decoded_preds = ["\\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\\n".join(label.strip().split()) for label in decoded_labels]

    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False
    )

    # Return scores
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "rougeLsum": result["rougeLsum"],
    }

print("✅ Metrics defined")"""),

    # 7. Setup Training
    create_cell("markdown", """## 7. Setup Training (Baseline Stable Configuration)

**Learning rate cho mT5:**
- 1e-4 → ổn định
- 2e-4 → nhanh hơn (recommended)
- >3e-4 → dễ nổ loss 💣"""),
    create_cell("code", """# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# Training arguments (baseline ổn định)
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_vi_sum",

    # Batch size strategy
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Giả lập batch size 16

    # Learning rate
    learning_rate=2e-4,  # Optimal for mT5
    warmup_steps=500,
    num_train_epochs=3,
    weight_decay=0.01,

    # Evaluation strategy
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,

    # Generation settings for evaluation
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,

    # Optimization
    fp16=USE_FP16,  # Enable FP16 on CUDA
    gradient_checkpointing=USE_GRAD_CHECKPOINT,

    # Logging
    logging_steps=100,
    logging_first_step=True,
    save_total_limit=2,

    # Best model selection
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",  # ROUGE-L là quan trọng nhất
    greater_is_better=True,

    report_to="none",
)

# Create Seq2Seq Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized!")
print(f"\\n📊 Training Configuration:")
print(f"   Device: {device}")
print(f"   FP16: {USE_FP16}")
print(f"   Per-device batch size: {training_args.per_device_train_batch_size}")
print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Warmup steps: {training_args.warmup_steps}")
print(f"   Total epochs: {training_args.num_train_epochs}")
print(f"   Eval every: {training_args.eval_steps} steps")"""),

    # 8. Train
    create_cell("markdown", "## 8. Train Model 🚀"),
    create_cell("code", """print("🚀 Starting training...")
print("="*70)
print("Expected training time:")
print("  • RTX 4070 SUPER (12GB): ~1-1.5 hours")
print("  • CPU: ~10-15 hours (not recommended)")
print("  • Google Colab T4: ~2-3 hours")
print("="*70)

# Start training
trainer.train()

print("\\n" + "="*70)
print("✅ Training complete!")
print("="*70)"""),

    # 9. Evaluate
    create_cell("markdown", "## 9. Evaluate on Test Set"),
    create_cell("code", """print("📊 Evaluating on test set...")
results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

print("\\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
for key, value in results.items():
    if 'rouge' in key:
        print(f"{key.upper()}: {value:.4f}")"""),

    # 10. Optimized Inference
    create_cell("markdown", """## 10. Optimized Inference Function

**Beam Search Tips:**
- num_beams: 4-6
- length_penalty: 1.0-1.5
- repetition_penalty: 1.2"""),
    create_cell("code", """def generate_summary(text, max_length=150, min_length=40, num_beams=4,
                     length_penalty=1.2, repetition_penalty=1.2):
    \"\"\"
    Generate summary with optimized parameters

    Args:
        text: Input document
        max_length: Maximum summary length (default: 150)
        min_length: Minimum summary length (default: 40)
        num_beams: Beam search beams (4-6 recommended)
        length_penalty: Length penalty (1.0-1.5 recommended)
        repetition_penalty: Repetition penalty (1.2 recommended)
    \"\"\"
    inputs = tokenizer(
        "tóm tắt: " + text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with examples
print("\\n=== INFERENCE EXAMPLES ===")
for i in range(3):
    test_text = dataset['test'][i]['document']
    ground_truth = dataset['test'][i]['summary']

    print(f"\\n{'='*70}")
    print(f"Example {i+1}")
    print(f"{'='*70}")
    print(f"Original ({len(test_text)} chars):")
    print(test_text[:200], "...\\n")

    print("Generated Summary:")
    generated = generate_summary(test_text)
    print(generated)

    print("\\nGround Truth:")
    print(ground_truth)
    print("="*70)"""),

    # 11. Save Model
    create_cell("markdown", "## 11. Save Model"),
    create_cell("code", """output_dir = "./mt5-small-vietnamese-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")
print(f"\\nTo load later:")
print(f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')
print(f'model = AutoModelForSeq2SeqLM.from_pretrained("{output_dir}")')"""),

    # 12. Quick Test
    create_cell("markdown", "## 12. Quick Test with Custom Text"),
    create_cell("code", """# Test with your own text
custom_text = \"\"\"
Chiều 26/1, UBND TP Hà Nội tổ chức họp báo công bố kết quả thực hiện
nhiệm vụ phát triển kinh tế - xã hội năm 2024. Theo đó, tổng sản phẩm
trên địa bàn (GRDP) của Hà Nội năm 2024 ước tăng 7,5% so với năm 2023,
cao hơn mức tăng trưởng chung của cả nước (7,09%).
\"\"\"

print("Original text:")
print(custom_text)
print("\\nGenerated summary:")
summary = generate_summary(custom_text.strip())
print(summary)

# Try different parameters
print("\\n" + "="*70)
print("Testing different beam search parameters:")
print("="*70)

params = [
    {"num_beams": 4, "length_penalty": 1.0},
    {"num_beams": 6, "length_penalty": 1.2},
    {"num_beams": 4, "length_penalty": 1.5},
]

for i, param in enumerate(params, 1):
    print(f"\\n[Config {i}] num_beams={param['num_beams']}, length_penalty={param['length_penalty']}")
    summary = generate_summary(custom_text.strip(), **param)
    print(summary)"""),
]

notebook["cells"] = cells

# Save notebook
with open("vietnamese_summarization_mt5_CLEAN.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("✅ Clean notebook created: vietnamese_summarization_mt5_CLEAN.ipynb")
