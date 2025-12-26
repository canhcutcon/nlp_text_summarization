# Kaggle Setup Guide - Vietnamese Text Summarization

## Overview

This guide shows how to run your Vietnamese text summarization project on Kaggle with GPU acceleration.

## Why Kaggle?

- ✅ **Free GPU access** (T4, P100)
- ✅ **30 hours/week** of GPU time
- ✅ **Pre-installed packages** (most dependencies already available)
- ✅ **No local setup needed**
- ✅ **Easy dataset integration**

## Setup Steps

### 1. Create New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Give it a name: "Vietnamese Text Summarization"

### 2. Enable GPU

1. In the notebook, click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2** (recommended) or **GPU P100**
3. Click **Save**

### 3. Install Required Packages

Most packages are pre-installed on Kaggle, but you need to verify versions:

```python
# Cell 1: Check and install/upgrade packages
import sys
print(f"Python version: {sys.version}")

# Check transformers version
import transformers
print(f"Transformers version: {transformers.__version__}")

# Upgrade if needed (Kaggle might have older versions)
if transformers.__version__ < "4.35.0":
    !pip install --upgrade transformers>=4.35.0 -q

# Install any missing packages
!pip install -q sentencepiece rouge-score evaluate accelerate datasets

# Verify installations
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4. Load Dataset from Hugging Face

**Option A: Direct Load (Recommended)**

```python
# Cell 2: Load dataset from Hugging Face
from datasets import load_dataset

print("Loading Vietnamese Summarization Dataset...")
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

print(f"\nDataset loaded successfully!")
print(dataset)

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

print(f"\nTrain samples: {len(train_data):,}")
print(f"Validation samples: {len(val_data):,}")
print(f"Test samples: {len(test_data):,}")

# Show a sample
sample = train_data[0]
print(f"\nSample document: {sample['document'][:200]}...")
print(f"Sample summary: {sample['summary'][:200]}...")
print(f"Keywords: {sample['keywords']}")
```

**Option B: Upload Dataset as Kaggle Dataset**

If you want to use your local parquet files:

1. Create a new dataset on Kaggle:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click **"New Dataset"**
   - Upload your `train-00000-of-00001.parquet`, `validation-00000-of-00001.parquet`, `test-00000-of-00001.parquet` from the `data/` folder
   - Name it: "vietnamese-summarization-dataset"

2. In your notebook, add the dataset:
   - Click **"Add Data"** (right sidebar)
   - Search for your uploaded dataset
   - Click **"Add"**

3. Load the parquet files:

```python
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

# Read parquet files
train_df = pd.read_parquet('/kaggle/input/vietnamese-summarization-dataset/train-00000-of-00001.parquet')
val_df = pd.read_parquet('/kaggle/input/vietnamese-summarization-dataset/validation-00000-of-00001.parquet')
test_df = pd.read_parquet('/kaggle/input/vietnamese-summarization-dataset/test-00000-of-00001.parquet')

# Convert to Hugging Face Dataset format
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df),
    'test': Dataset.from_pandas(test_df)
})

print(dataset)
```

### 5. Training Configuration for Kaggle

```python
# Cell 3: Set up training configuration
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Choose model (ViT5 recommended for Vietnamese)
MODEL_NAME = 'VietAI/vit5-base'
# Alternative: 'google/mt5-base' or 'vinai/phobert-base'

# Load tokenizer and model
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"✅ Model loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model on device: {device}")
```

### 6. Preprocess Dataset

```python
# Cell 4: Tokenize dataset
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

def preprocess_function(examples):
    # Add "summarize: " prefix for T5-based models
    inputs = [f"summarize: {doc}" for doc in examples['document']]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

    # Tokenize targets
    labels = tokenizer(
        examples['summary'],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding='max_length'
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing"
)

print("✅ Tokenization complete!")
```

### 7. Training Arguments

```python
# Cell 5: Configure training
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=True,  # Enable mixed precision for faster training
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='rouge1',
    greater_is_better=True,
    push_to_hub=False,
    report_to=['tensorboard'],
    gradient_accumulation_steps=2,  # Effective batch size = 16
    warmup_steps=500,
)

print("Training configuration:")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  FP16: {training_args.fp16}")
```

### 8. Set Up Metrics

```python
# Cell 6: Define evaluation metrics
import numpy as np
from datasets import load_metric

# Load ROUGE metric
rouge = load_metric('rouge')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Extract F1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return {k: round(v, 4) for k, v in result.items()}
```

### 9. Initialize Trainer and Start Training

```python
# Cell 7: Create trainer and start training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
print("=" * 80)

# Train the model
trainer.train()

print("=" * 80)
print("✅ Training complete!")

# Save the final model
trainer.save_model('./final_model')
tokenizer.save_pretrained('./final_model')
print("✅ Model saved to ./final_model")
```

### 10. Evaluate on Test Set

```python
# Cell 8: Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_datasets['test'])

print("\nTest Results:")
print("=" * 80)
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")
```

### 11. Generate Sample Predictions

```python
# Cell 9: Test with sample predictions
def generate_summary(text, max_length=128):
    inputs = tokenizer(
        f"summarize: {text}",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        length_penalty=0.6,
        early_stopping=True
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Test with examples from test set
print("Sample Predictions:")
print("=" * 80)

for i in range(3):
    sample = dataset['test'][i]

    print(f"\n--- Example {i+1} ---")
    print(f"Document: {sample['document'][:200]}...")
    print(f"\nReference Summary: {sample['summary']}")

    predicted_summary = generate_summary(sample['document'])
    print(f"\nPredicted Summary: {predicted_summary}")
    print("-" * 80)
```

### 12. Save Results

```python
# Cell 10: Save results for download
import json

# Save training history
with open('training_results.json', 'w', encoding='utf-8') as f:
    json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)

# Save test results
with open('test_results.json', 'w', encoding='utf-8') as f:
    json.dump(test_results, f, indent=2)

print("✅ Results saved!")
print("\nYou can download:")
print("  - ./final_model/ (trained model)")
print("  - training_results.json")
print("  - test_results.json")
```

## Complete Notebook Structure

```
Cell 1: Install/upgrade packages
Cell 2: Load dataset
Cell 3: Load model and tokenizer
Cell 4: Preprocess dataset
Cell 5: Configure training arguments
Cell 6: Set up metrics
Cell 7: Train model
Cell 8: Evaluate on test set
Cell 9: Generate sample predictions
Cell 10: Save results
```

## Kaggle-Specific Tips

### Memory Management

If you run out of GPU memory:

```python
# Reduce batch size
per_device_train_batch_size=4  # Instead of 8

# Increase gradient accumulation
gradient_accumulation_steps=4  # Instead of 2

# Reduce sequence length
MAX_INPUT_LENGTH = 384  # Instead of 512
MAX_TARGET_LENGTH = 96   # Instead of 128

# Enable gradient checkpointing (saves memory)
model.gradient_checkpointing_enable()
```

### Speed Up Training

```python
# Use smaller dataset for testing
train_dataset = tokenized_datasets['train'].select(range(1000))  # Use first 1000 samples

# Reduce epochs
num_train_epochs=1

# Increase batch size if memory allows
per_device_train_batch_size=16
```

### Monitor Training

Kaggle automatically logs to TensorBoard. View it by:
1. Click on **"Logs"** in the right sidebar
2. You'll see training loss, evaluation metrics, etc.

## Dataset Loading Summary

**Recommended approach for Kaggle:**

```python
# Single line - no setup needed!
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")
```

**Alternative if you upload parquet files:**

```python
import pandas as pd
from datasets import Dataset, DatasetDict

dataset = DatasetDict({
    'train': Dataset.from_pandas(
        pd.read_parquet('/kaggle/input/YOUR-DATASET-NAME/train-00000-of-00001.parquet')
    ),
    'validation': Dataset.from_pandas(
        pd.read_parquet('/kaggle/input/YOUR-DATASET-NAME/validation-00000-of-00001.parquet')
    ),
    'test': Dataset.from_pandas(
        pd.read_parquet('/kaggle/input/YOUR-DATASET-NAME/test-00000-of-00001.parquet')
    )
})
```

## Key Differences from Local Setup

| Aspect | Local | Kaggle |
|--------|-------|--------|
| Environment | Virtual env (.venv) | Pre-configured |
| Kernel setup | Manual (ipykernel) | Automatic |
| GPU | MPS (Apple Silicon) | CUDA (NVIDIA) |
| Package install | `pip install -r requirements.txt` | Most pre-installed |
| Dataset path | `./data/` | `/kaggle/input/` or HF Hub |
| Output path | Local directory | `/kaggle/working/` |

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
- Reduce `per_device_train_batch_size` to 4 or 2
- Increase `gradient_accumulation_steps`
- Reduce `MAX_INPUT_LENGTH`

### Issue: "Session timeout" during training

**Solution:**
- Enable "Commit and Run All" to run entire notebook
- Kaggle will complete training even if browser closes
- Check back later to download results

### Issue: Old transformers version

**Solution:**
```python
!pip install --upgrade transformers>=4.35.0 -q
# Then restart kernel: Runtime → Restart Kernel
```

## Expected Training Time

| Setup | GPU | Samples | Epochs | Time |
|-------|-----|---------|--------|------|
| Small test | T4 | 1,000 | 1 | ~15 min |
| Full training | T4 | 15,620 | 3 | ~6-8 hours |
| Full training | P100 | 15,620 | 3 | ~4-5 hours |

## Download Trained Model

After training:
1. Click **Output** tab (right sidebar)
2. Find `final_model` folder
3. Click download icon
4. Extract and use locally

## Next Steps After Kaggle Training

1. ✅ Download trained model from Kaggle
2. Load it locally:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('./final_model')
tokenizer = AutoTokenizer.from_pretrained('./final_model')

# Use for inference
text = "Your Vietnamese text here..."
inputs = tokenizer(f"summarize: {text}", return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

---

**Ready to use on Kaggle!** 🚀

Just copy the code cells above into a new Kaggle notebook and run!
