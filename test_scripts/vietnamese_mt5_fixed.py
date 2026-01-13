#!/usr/bin/env python
# coding: utf-8
"""
Vietnamese Text Summarization - mT5-Small (FIXED VERSION)
- ✅ Proper CUDA device detection for RTX 3090
- ✅ Pre-training diagnostics to catch "loss = 0" issues
- ✅ Safety checks integrated
"""

# ============================================================================
# 1. Install Packages (if needed)
# ============================================================================
import sys

# Uncomment if running in Colab or fresh environment:
# !pip install -q transformers datasets accelerate sentencepiece evaluate rouge-score py-rouge scikit-learn protobuf

print("✅ Imports starting...")

import re
import pandas as pd
import numpy as np
import torch
import gc
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate

print("=" * 60)
print("Vietnamese Text Summarization - mT5-Small (FIXED)")
print("=" * 60)

# ============================================================================
# 2. Load and Verify Data
# ============================================================================
print("\n📊 Loading Data...")

def sent_tokenize(text: str) -> list[str]:
    """Vietnamese sentence tokenizer"""
    pattern = r'(?<=[.!?])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

# Load CSV
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

# Keep only document and summary columns
train_df = train_df[['document', 'summary']].dropna()
val_df = val_df[['document', 'summary']].dropna()
test_df = test_df[['document', 'summary']].dropna()

print(f"✓ Train: {len(train_df):,} samples")
print(f"✓ Validation: {len(val_df):,} samples")
print(f"✓ Test: {len(test_df):,} samples")

# Convert to Dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'validation': Dataset.from_pandas(val_df, preserve_index=False),
    'test': Dataset.from_pandas(test_df, preserve_index=False)
})

# Show sample
print(f"\n📝 Sample:")
sample = dataset['train'][0]
print(f"Document: {sample['document'][:200]}...")
print(f"Summary: {sample['summary'][:150]}...")

# ============================================================================
# 3. FIXED: Proper GPU Detection for RTX 3090
# ============================================================================
print("\n" + "=" * 60)
print("🎯 GPU Setup (FIXED for RTX 3090)")
print("=" * 60)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Available VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f} GB")
    USE_FP16 = True
    USE_GRAD_CHECKPOINT = True
else:
    device = torch.device("cpu")
    print("⚠️  No CUDA GPU detected - using CPU (will be slow!)")
    USE_FP16 = False
    USE_GRAD_CHECKPOINT = False

# ============================================================================
# 4. Load Model with Safety Checks
# ============================================================================
print("\n" + "=" * 60)
print("📦 Loading Model")
print("=" * 60)

MODEL_NAME = "google/mt5-small"
print(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if USE_FP16:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )
    print("✓ Loaded with FP16")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("✓ Loaded with FP32")

if USE_GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")

model.to(device)

# Verify model parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,}")

if trainable_params == 0:
    print("❌ CRITICAL: No trainable parameters! Model is frozen!")
    sys.exit(1)

# ============================================================================
# 5. Tokenize Data
# ============================================================================
print("\n" + "=" * 60)
print("🔄 Tokenizing Dataset")
print("=" * 60)

def preprocess_function(examples):
    """Tokenize inputs and targets"""
    # Add prefix
    inputs = ["tóm tắt: " + doc for doc in examples["document"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False
    )
    
    # IMPORTANT: Use text_target for labels
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=128,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

print("✓ Tokenization complete")

# ============================================================================
# 6. PRE-TRAINING DIAGNOSTICS (CRITICAL!)
# ============================================================================
print("\n" + "=" * 60)
print("🔍 PRE-TRAINING DIAGNOSTICS (CRITICAL)")
print("=" * 60)

# Create data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model, 
    label_pad_token_id=-100
)

# Test 1: Check Labels
print("\n[Test 1] Checking labels...")
sample = tokenized_datasets["train"][0]
valid_labels = [l for l in sample['labels'] if l != -100]

if len(valid_labels) == 0:
    print("❌ CRITICAL: All labels are -100! Training will fail!")
    sys.exit(1)
else:
    print(f"✅ Valid labels: {len(valid_labels)} tokens")
    print(f"   Sample: {tokenizer.decode(valid_labels[:20])}")

# Test 2: Forward Pass
print("\n[Test 2] Testing forward pass...")
from torch.utils.data import DataLoader

test_loader = DataLoader(
    tokenized_datasets["train"].select(range(2)),
    batch_size=2,
    collate_fn=data_collator
)

batch = next(iter(test_loader))
batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

model.eval()
with torch.no_grad():
    outputs = model(**batch_device)
    test_loss = outputs.loss.item()

print(f"Test loss: {test_loss:.4f}")

if test_loss == 0.0:
    print("❌ CRITICAL: Test loss is 0! Model cannot learn!")
    print("Possible causes:")
    print("  - All labels are -100")
    print("  - Model weights frozen")
    print("  - Data collator misconfigured")
    sys.exit(1)
elif torch.isnan(torch.tensor(test_loss)):
    print("❌ CRITICAL: Test loss is NaN! Numerical instability!")
    sys.exit(1)
elif test_loss < 0.1:
    print("⚠️  WARNING: Test loss is very low (<0.1), this is suspicious")
elif test_loss > 10:
    print("⚠️  WARNING: Test loss is very high (>10), might be unstable")
else:
    print("✅ Test loss looks normal!")

# Test 3: Generation
print("\n[Test 3] Testing generation...")
test_text = "Chiều 26/1, UBND TP Hà Nội tổ chức họp báo"
inputs = tokenizer("tóm tắt: " + test_text, return_tensors="pt").to(device)

with torch.no_grad():
    gen_outputs = model.generate(**inputs, max_length=50)
    generated = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)

print(f"Input: {test_text}")
print(f"Generated: '{generated}'")

if '<' in generated and '>' in generated:
    print("❌ WARNING: Generated text contains sentinel tokens!")
    print("   This might indicate tokenizer/model mismatch")
elif len(generated.strip()) == 0:
    print("❌ WARNING: Generated text is empty!")
else:
    print("✅ Generation looks reasonable!")

# Test 4: Gradients
print("\n[Test 4] Testing gradients...")
model.train()
outputs = model(**batch_device)
loss = outputs.loss
loss.backward()

has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
               for p in model.parameters() if p.requires_grad)

if not has_grad:
    print("❌ CRITICAL: No gradients computed! Model cannot learn!")
    sys.exit(1)
else:
    print("✅ Gradients computed successfully!")

print("\n" + "=" * 60)
print("✅ ALL DIAGNOSTIC TESTS PASSED!")
print("=" * 60)

# Clear gradients before training
model.zero_grad()

# ============================================================================
# 7. Define Metrics
# ============================================================================
print("\n📊 Setting Up Metrics...")

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Compute ROUGE scores"""
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "rougeLsum": result["rougeLsum"],
    }

print("✓ Metrics defined")

# ============================================================================
# 8. Setup Training (FIXED)
# ============================================================================
print("\n" + "=" * 60)
print("⚙️  Setting Up Training (FIXED)")
print("=" * 60)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-small-vn-fixed",
    
    # Batch sizes
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    
    # Learning
    learning_rate=5e-5,  # Slightly higher than before for better learning
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    
    # Generation
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    
    # Optimization
    fp16=USE_FP16,
    gradient_checkpointing=USE_GRAD_CHECKPOINT,
    max_grad_norm=1.0,  # Gradient clipping to prevent explosion
    
    # Logging
    logging_dir="./logs",
    logging_steps=10,  # Log frequently at start
    save_steps=500,
    save_total_limit=2,
    
    # Best model
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✓ Trainer ready")
print(f"  FP16: {USE_FP16}")
print(f"  Gradient checkpointing: {USE_GRAD_CHECKPOINT}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size (effective): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# ============================================================================
# 9. Train Model 🚀
# ============================================================================
print("\n" + "=" * 60)
print("🚀 Starting Training")
print("=" * 60)
print("\n⚠️  IMPORTANT: Watch the first 10 steps!")
print("Expected: Loss should start at 5-8 and decrease")
print("Stop immediately if you see loss = 0.000 or NaN!\n")

try:
    trainer.train()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 10. Save Model
# ============================================================================
print("\n💾 Saving Model...")

output_dir = "./mt5-small-vietnamese-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")
print("\n" + "=" * 60)
print("🏁 COMPLETE!")
print("=" * 60)
