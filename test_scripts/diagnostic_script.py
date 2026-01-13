"""
🔍 CHẨN ĐOÁN TOÀN DIỆN - MT5 TRAINING BUG
Chạy script này TRƯỚC KHI train để tìm lỗi
"""

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

print("="*60)
print("🔍 BẮT ĐẦU CHẨN ĐOÁN")
print("="*60)

# ============================================================================
# 1. KIỂM TRA THIẾT LẬP CƠ BẢN
# ============================================================================
print("\n📋 BƯỚC 1: Kiểm tra thiết lập cơ bản")
print("-"*60)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ KHÔNG CÓ GPU!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. LOAD VÀ KIỂM TRA DATA
# ============================================================================
print("\n📋 BƯỚC 2: Load và kiểm tra data")
print("-"*60)

try:
    train_df = pd.read_csv("data/train.csv")
    print(f"✅ Loaded train.csv: {len(train_df)} rows")
    
    # Check columns
    print(f"   Columns: {train_df.columns.tolist()}")
    
    # Check sample
    sample = train_df.iloc[0]
    print(f"\n   Sample document (first 100 chars):")
    print(f"   '{sample['document'][:100]}'")
    print(f"\n   Sample summary (first 100 chars):")
    print(f"   '{sample['summary'][:100]}'")
    
    # Check for NaN
    if train_df['document'].isna().any():
        print("   ❌ WARNING: NaN values in document column!")
    if train_df['summary'].isna().any():
        print("   ❌ WARNING: NaN values in summary column!")
    
except Exception as e:
    print(f"❌ ERROR loading data: {e}")
    exit(1)

# ============================================================================
# 3. LOAD VÀ KIỂM TRA MODEL
# ============================================================================
print("\n📋 BƯỚC 3: Load và kiểm tra model")
print("-"*60)

MODEL_NAME = "google/mt5-small"
print(f"Loading model: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    print(f"✅ Model loaded successfully")
    print(f"   Model class: {model.__class__.__name__}")
    print(f"   Parameters: {model.num_parameters():,}")
    print(f"   Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")
    print(f"   Pad token ID: {tokenizer.pad_token_id}")
    print(f"   EOS token ID: {tokenizer.eos_token_id}")
    
    # Move to GPU
    model = model.to(device)
    print(f"   Model on device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

# ============================================================================
# 4. KIỂM TRA TOKENIZATION
# ============================================================================
print("\n📋 BƯỚC 4: Kiểm tra tokenization")
print("-"*60)

test_doc = train_df.iloc[0]['document'][:200]
test_sum = train_df.iloc[0]['summary'][:100]

print(f"Test document: '{test_doc}'")
print(f"Test summary: '{test_sum}'")

# Tokenize input
input_encoding = tokenizer(
    "tóm tắt: " + test_doc,
    max_length=512,
    truncation=True,
    return_tensors="pt"
)

print(f"\n✅ Input tokenization:")
print(f"   Input IDs shape: {input_encoding['input_ids'].shape}")
print(f"   Input IDs (first 20): {input_encoding['input_ids'][0][:20].tolist()}")
print(f"   Decoded (first 100 chars): '{tokenizer.decode(input_encoding['input_ids'][0][:100])}'")

# Tokenize target
target_encoding = tokenizer(
    text_target=test_sum,
    max_length=128,
    truncation=True,
    return_tensors="pt"
)

print(f"\n✅ Target tokenization:")
print(f"   Target IDs shape: {target_encoding['input_ids'].shape}")
print(f"   Target IDs (first 20): {target_encoding['input_ids'][0][:20].tolist()}")
print(f"   Decoded: '{tokenizer.decode(target_encoding['input_ids'][0])}'")

# ============================================================================
# 5. KIỂM TRA FORWARD PASS
# ============================================================================
print("\n📋 BƯỚC 5: Kiểm tra forward pass")
print("-"*60)

# Test forward pass WITHOUT labels (generation mode)
print("Test 1: Forward pass WITHOUT labels (inference)")
with torch.no_grad():
    outputs_no_labels = model(
        input_ids=input_encoding['input_ids'].to(device),
        attention_mask=input_encoding['attention_mask'].to(device)
    )
    print(f"   Logits shape: {outputs_no_labels.logits.shape}")
    print(f"   Logits min/max: {outputs_no_labels.logits.min().item():.4f} / {outputs_no_labels.logits.max().item():.4f}")

# Test forward pass WITH labels (training mode)
print("\nTest 2: Forward pass WITH labels (training)")
with torch.no_grad():
    outputs_with_labels = model(
        input_ids=input_encoding['input_ids'].to(device),
        attention_mask=input_encoding['attention_mask'].to(device),
        labels=target_encoding['input_ids'].to(device)
    )
    
    loss = outputs_with_labels.loss.item()
    print(f"   Loss: {loss:.4f}")
    print(f"   Loss is finite: {torch.isfinite(outputs_with_labels.loss).item()}")
    
    if loss == 0.0:
        print("   ❌❌❌ CRITICAL ERROR: Loss is 0! This is WRONG!")
    elif torch.isnan(outputs_with_labels.loss):
        print("   ❌❌❌ CRITICAL ERROR: Loss is NaN! This is WRONG!")
    elif loss > 10:
        print("   ⚠️  WARNING: Loss is very high (>10)")
    elif loss < 0.1:
        print("   ⚠️  WARNING: Loss is suspiciously low (<0.1)")
    else:
        print("   ✅ Loss looks normal!")

# ============================================================================
# 6. KIỂM TRA GENERATION
# ============================================================================
print("\n📋 BƯỚC 6: Kiểm tra generation")
print("-"*60)

test_input = "Chiều 26/1, UBND TP Hà Nội tổ chức họp báo công bố kết quả phát triển kinh tế."
print(f"Test input: '{test_input}'")

inputs = tokenizer("tóm tắt: " + test_input, return_tensors="pt").to(device)

with torch.no_grad():
    # Greedy generation (fast)
    outputs_greedy = model.generate(
        **inputs,
        max_length=50,
        num_beams=1,
        do_sample=False
    )
    
    generated_greedy = tokenizer.decode(outputs_greedy[0], skip_special_tokens=True)
    generated_with_special = tokenizer.decode(outputs_greedy[0], skip_special_tokens=False)
    
    print(f"\n✅ Greedy generation:")
    print(f"   Output IDs: {outputs_greedy[0].tolist()[:30]}")
    print(f"   With special tokens: '{generated_with_special}'")
    print(f"   Without special tokens: '{generated_greedy}'")
    print(f"   Length: {len(generated_greedy)} chars")
    
    # Check for garbage output
    if len(generated_greedy.strip()) == 0:
        print("   ❌ CRITICAL: Generated text is EMPTY!")
    elif '<' in generated_greedy and '>' in generated_greedy:
        print("   ❌ CRITICAL: Generated text contains sentinel tokens!")
    elif any(c < ' ' for c in generated_greedy if c != '\n'):
        print("   ❌ CRITICAL: Generated text contains control characters!")
    elif all(ord(c) < 128 for c in generated_greedy):
        print("   ⚠️  WARNING: Generated text is all ASCII (expected Vietnamese)")
    else:
        print("   ✅ Generated text looks reasonable!")

# ============================================================================
# 7. KIỂM TRA DATA COLLATOR
# ============================================================================
print("\n📋 BƯỚC 7: Kiểm tra Data Collator")
print("-"*60)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# Create mini dataset
mini_df = train_df.head(2)
mini_dataset = Dataset.from_pandas(mini_df[['document', 'summary']], preserve_index=False)

def preprocess_function(examples):
    inputs = ["tóm tắt: " + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=False)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_mini = mini_dataset.map(preprocess_function, batched=True)

# Test collator
batch = [tokenized_mini[0], tokenized_mini[1]]
collated_batch = data_collator(batch)

print(f"✅ Data collator output:")
print(f"   Input IDs shape: {collated_batch['input_ids'].shape}")
print(f"   Labels shape: {collated_batch['labels'].shape}")
print(f"   Attention mask shape: {collated_batch['attention_mask'].shape}")

# Check labels
labels_sample = collated_batch['labels'][0]
num_neg100 = (labels_sample == -100).sum().item()
num_valid = (labels_sample != -100).sum().item()

print(f"\n   Labels analysis:")
print(f"   Total tokens: {len(labels_sample)}")
print(f"   Padding (-100): {num_neg100}")
print(f"   Valid tokens: {num_valid}")
print(f"   Valid percentage: {num_valid/len(labels_sample)*100:.1f}%")

if num_valid == 0:
    print("   ❌❌❌ CRITICAL: ALL LABELS ARE -100!")
elif num_valid < 5:
    print("   ❌ WARNING: Very few valid labels!")
else:
    print("   ✅ Labels look normal!")

# Decode labels
valid_labels = labels_sample[labels_sample != -100]
if len(valid_labels) > 0:
    decoded_labels = tokenizer.decode(valid_labels)
    print(f"   Decoded labels: '{decoded_labels[:100]}'")

# ============================================================================
# 8. KIỂM TRA TRAINING STEP
# ============================================================================
print("\n📋 BƯỚC 8: Kiểm tra một training step")
print("-"*60)

# Set model to training mode
model.train()
print(f"Model in training mode: {model.training}")

# Move batch to device
collated_batch = {k: v.to(device) for k, v in collated_batch.items()}

# Forward pass
outputs = model(**collated_batch)
loss = outputs.loss

print(f"\n✅ Training step:")
print(f"   Loss: {loss.item():.4f}")
print(f"   Loss requires_grad: {loss.requires_grad}")
print(f"   Loss is finite: {torch.isfinite(loss).item()}")

if loss.item() == 0.0:
    print("   ❌❌❌ CRITICAL: Training loss is 0!")
    
    # Debug why
    print("\n   Debugging zero loss:")
    print(f"   - Labels contain -100: {(collated_batch['labels'] == -100).any().item()}")
    print(f"   - All labels are -100: {(collated_batch['labels'] == -100).all().item()}")
    print(f"   - Input IDs shape: {collated_batch['input_ids'].shape}")
    print(f"   - Labels shape: {collated_batch['labels'].shape}")
    
elif torch.isnan(loss):
    print("   ❌❌❌ CRITICAL: Training loss is NaN!")
else:
    print("   ✅ Training loss looks normal!")

# Test backward
loss.backward()
print(f"\n✅ Backward pass successful")

# Check gradients
has_grad = False
total_params = 0
params_with_grad = 0

for name, param in model.named_parameters():
    total_params += 1
    if param.grad is not None and param.grad.abs().sum() > 0:
        has_grad = True
        params_with_grad += 1

print(f"   Total parameters: {total_params}")
print(f"   Parameters with gradients: {params_with_grad}")

if not has_grad:
    print("   ❌ CRITICAL: NO GRADIENTS COMPUTED!")
elif params_with_grad < total_params * 0.5:
    print("   ⚠️  WARNING: Less than 50% of parameters have gradients")
else:
    print("   ✅ Gradients look normal!")

# ============================================================================
# 9. FINAL DIAGNOSIS
# ============================================================================
print("\n" + "="*60)
print("📊 FINAL DIAGNOSIS")
print("="*60)

issues_found = []

# Check each component
if loss.item() == 0.0:
    issues_found.append("❌ Training loss is 0 - Model is not learning!")
    issues_found.append("   Possible causes:")
    issues_found.append("   - All labels are -100 (padding)")
    issues_found.append("   - Model weights are frozen")
    issues_found.append("   - Incorrect loss computation")

if torch.isnan(loss):
    issues_found.append("❌ Training loss is NaN - Numerical instability!")
    issues_found.append("   Possible causes:")
    issues_found.append("   - FP16 precision issues")
    issues_found.append("   - Exploding gradients")
    issues_found.append("   - Invalid input data")

if not has_grad:
    issues_found.append("❌ No gradients computed - Model cannot learn!")
    issues_found.append("   Possible causes:")
    issues_found.append("   - Model in eval mode")
    issues_found.append("   - Parameters frozen")
    issues_found.append("   - Loss computation error")

if '<' in generated_greedy and '>' in generated_greedy:
    issues_found.append("❌ Model generates sentinel tokens - Wrong tokenizer/model combo!")
    issues_found.append("   Possible causes:")
    issues_found.append("   - Tokenizer doesn't match model")
    issues_found.append("   - Model not properly initialized")

if num_valid == 0:
    issues_found.append("❌ All labels are -100 - Data collator error!")
    issues_found.append("   Possible causes:")
    issues_found.append("   - Wrong label_pad_token_id")
    issues_found.append("   - Preprocessing error")

if issues_found:
    print("\n🔴 ISSUES FOUND:")
    for issue in issues_found:
        print(issue)
    print("\n💡 Next steps:")
    print("1. Review the issues above")
    print("2. Check the corresponding sections for details")
    print("3. Fix issues before training")
else:
    print("\n✅ NO CRITICAL ISSUES FOUND!")
    print("Model should train correctly.")
    print("\n💡 Proceed with training and monitor:")
    print("- First step loss should be 2-8")
    print("- Loss should decrease over time")
    print("- ROUGE scores should be > 0 after first eval")

print("\n" + "="*60)
print("🏁 DIAGNOSIS COMPLETE")
print("="*60)
