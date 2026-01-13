#!/usr/bin/env python
# coding: utf-8
"""
🚨 EMERGENCY FIX - mT5 Training Loss = 0 Bug
This script diagnoses and fixes the root cause
"""

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import gc

print("=" * 60)
print("🚨 EMERGENCY FIX FOR LOSS = 0")
print("=" * 60)

# ============================================================================
# 1. Setup Device
# ============================================================================
print("\n[1/8] Setting up device...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

# ============================================================================
# 2. Load Model CORRECTLY
# ============================================================================
print("\n[2/8] Loading model...")

MODEL_NAME = "google/mt5-small"

# CRITICAL: Load tokenizer and model from SAME source
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# CRITICAL: Ensure model parameters are trainable
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)
model.train()

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# 3. Load Data
# ============================================================================
print("\n[3/8] Loading data...")

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")

train_df = train_df[['document', 'summary']].dropna()
val_df = val_df[['document', 'summary']].dropna()

print(f"✅ Train: {len(train_df):,} samples")
print(f"✅ Val: {len(val_df):,} samples")

# Convert to datasets
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'validation': Dataset.from_pandas(val_df, preserve_index=False),
})

# ============================================================================
# 4. FIXED Preprocessing Function
# ============================================================================
print("\n[4/8] Tokenizing (FIXED method)...")

def preprocess_function_FIXED(examples):
    """
    FIXED preprocessing - ensures labels are NOT all -100
    """
    inputs = ["tóm tắt: " + doc for doc in examples["document"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length",  # IMPORTANT: Use max_length padding
    )
    
    # CRITICAL: Tokenize targets WITHOUT text_target parameter
    # Some versions of transformers have issues with text_target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=128,
            truncation=True,
            padding="max_length",  # IMPORTANT: Use max_length padding
        )
    
    # Replace padding token id with -100 for loss computation
    label_ids = []
    for label in labels["input_ids"]:
        label_ids.append([
            (l if l != tokenizer.pad_token_id else -100) for l in label
        ])
    
    model_inputs["labels"] = label_ids
    return model_inputs

# Tokenize
tokenized_datasets = dataset.map(
    preprocess_function_FIXED,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

# ============================================================================
# 5. VERIFY Labels
# ============================================================================
print("\n[5/8] Verifying tokenization...")

sample = tokenized_datasets["train"][0]
labels = sample["labels"]
valid_count = sum(1 for l in labels if l != -100)
total_count = len(labels)

print(f"   Total labels: {total_count}")
print(f"   Valid labels: {valid_count} ({valid_count/total_count*100:.1f}%)")

if valid_count == 0:
    print("❌ CRITICAL: All labels are -100!")
    print("   Trying alternative preprocessing...")
    
    # Alternative preprocessing
    def preprocess_alt(examples):
        inputs = ["tóm tắt: " + doc for doc in examples["document"]]
        targets = examples["summary"]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        target_encodings = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        
        labels = []
        for ids in target_encodings["input_ids"]:
            labels.append([(id if id != tokenizer.pad_token_id else -100) for id in ids])
        
        model_inputs["labels"] = labels
        return model_inputs
    
    tokenized_datasets = dataset.map(
        preprocess_alt,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing (ALT)"
    )
    
    sample = tokenized_datasets["train"][0]
    labels = sample["labels"]
    valid_count = sum(1 for l in labels if l != -100)
    print(f"   After ALT: Valid labels = {valid_count}")
    
if valid_count == 0:
    print("\n❌❌❌ STILL NO VALID LABELS!")
    print("Check your data format!")
    exit(1)
else:
    valid_labels = [l for l in labels if l != -100]
    decoded = tokenizer.decode(valid_labels[:30])
    print(f"✅ Labels look good!")
    print(f"   Decoded sample: '{decoded[:100]}'")

# ============================================================================
# 6. Test Forward Pass
# ============================================================================
print("\n[6/8] Testing forward pass...")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

from torch.utils.data import DataLoader
test_loader = DataLoader(
    tokenized_datasets["train"].select(range(4)),
    batch_size=2,
    collate_fn=data_collator
)

batch = next(iter(test_loader))
batch = {k: v.to(device) for k, v in batch.items()}

# Check batch labels
batch_labels = batch["labels"]
valid_in_batch = (batch_labels != -100).sum().item()
print(f"   Valid labels in batch: {valid_in_batch}")

if valid_in_batch == 0:
    print("❌ CRITICAL: Batch has no valid labels!")
    exit(1)

# Forward pass
model.train()
outputs = model(**batch)
loss = outputs.loss

print(f"   Loss: {loss.item():.4f}")
print(f"   Loss requires_grad: {loss.requires_grad}")

if loss.item() == 0:
    print("❌ CRITICAL: Loss is still 0!")
    print("   Checking model internals...")
    
    # Debug
    logits = outputs.logits
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    
    exit(1)
elif torch.isnan(loss):
    print("❌ CRITICAL: Loss is NaN!")
    exit(1)
else:
    print("✅ Forward pass OK!")

# Test backward
loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
if has_grad:
    print("✅ Gradients computed!")
else:
    print("❌ No gradients!")
    exit(1)

model.zero_grad()

# ============================================================================
# 7. Setup Training
# ============================================================================
print("\n[7/8] Setting up training...")

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: v for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-emergency-fix",
    
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    
    learning_rate=1e-4,  # Higher LR
    num_train_epochs=3,
    warmup_steps=200,
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    eval_strategy="steps",
    eval_steps=500,
    
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=2,  # Reduced for speed
    
    fp16=False,  # DISABLED - can cause issues
    gradient_checkpointing=False,  # DISABLED - can cause issues
    
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    
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

print("✅ Trainer ready!")
print("   FP16: DISABLED (stability)")
print("   Gradient checkpointing: DISABLED (stability)")
print(f"   Learning rate: {training_args.learning_rate}")

# ============================================================================
# 8. TRAIN
# ============================================================================
print("\n[8/8] Starting training...")
print("=" * 60)
print("⚠️  WATCH CLOSELY:")
print("   - Step 1-10: Loss should be 2-8")
print("   - If loss = 0: STOP and report!")
print("=" * 60)

trainer.train()

print("\n✅ Training complete!")
trainer.save_model("./mt5-vietnamese-fixed")
print("✅ Model saved to ./mt5-vietnamese-fixed")
