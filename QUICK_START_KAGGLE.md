# 🚀 Quick Start: Kaggle in 5 Minutes

## Step 1: Create Kaggle Notebook (30 seconds)

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Name it: "Vietnamese Text Summarization"

## Step 2: Enable GPU (10 seconds)

1. Click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2**
3. Click **Save**

## Step 3: Copy & Paste Code (1 minute)

Open [kaggle_starter.py](kaggle_starter.py) and copy the cells.

Or use this minimal version:

### Cell 1: Setup
```python
# Install packages
!pip install -q --upgrade transformers datasets sentencepiece rouge-score evaluate accelerate

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 2: Load Dataset (THE KEY CHANGE!)
```python
from datasets import load_dataset

# THIS IS THE NEW WAY - No curl, no git clone!
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

print(dataset)
print(f"Train: {len(dataset['train']):,} samples")
print(f"Validation: {len(dataset['validation']):,} samples")
print(f"Test: {len(dataset['test']):,} samples")
```

### Cell 3: Load Model
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = 'VietAI/vit5-base'  # Best for Vietnamese

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"✅ Model loaded on {device}")
```

### Cell 4: Preprocess
```python
def preprocess_function(examples):
    inputs = [f"summarize: {doc}" for doc in examples['document']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(examples['summary'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True,
                                 remove_columns=dataset['train'].column_names)
print("✅ Tokenization complete!")
```

### Cell 5: Configure Training
```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import evaluate

# Setup metrics
rouge_metric = evaluate.load('rouge')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    fp16=True,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model='rouge1',
)

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

print("✅ Trainer configured")
```

### Cell 6: Train!
```python
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# Save model
trainer.save_model('./final_model')
tokenizer.save_pretrained('./final_model')
print("✅ Model saved!")
```

### Cell 7: Evaluate
```python
# Evaluate on test set
test_results = trainer.evaluate(tokenized_datasets['test'])

print("\n📊 Test Results:")
for key, value in test_results.items():
    if 'rouge' in key:
        print(f"  {key.upper()}: {value:.2f}")
```

### Cell 8: Test Predictions
```python
def generate_summary(text):
    inputs = tokenizer(f"summarize: {text}", max_length=512, truncation=True, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with examples
for i in range(3):
    sample = dataset['test'][i]
    print(f"\n{'='*80}")
    print(f"Example {i+1}")
    print(f"Document: {sample['document'][:200]}...")
    print(f"\nReference: {sample['summary']}")
    print(f"\nGenerated: {generate_summary(sample['document'])}")
```

## Step 4: Click "Run All" (3 minutes setup + training time)

Click **Cell** → **Run All** in the menu.

Training will take 6-8 hours, but you can close your browser!

## Step 5: Download Model (after training)

1. Click **Output** tab (right sidebar)
2. Find `final_model` folder
3. Click download icon
4. Use locally!

---

## The Big Change: Dataset Loading

### ❌ OLD (Don't use):
```bash
curl -X GET "https://datasets-server.huggingface.co/rows?dataset=..."
git clone https://huggingface.co/datasets/...
```

### ✅ NEW (Use this):
```python
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")
```

**That's it!** Works everywhere:
- ✅ Kaggle
- ✅ Google Colab
- ✅ Your local machine
- ✅ Any Python environment

---

## Troubleshooting

### "CUDA out of memory"
Change in Cell 5:
```python
per_device_train_batch_size=4,  # Instead of 8
gradient_accumulation_steps=4,  # Instead of 2
```

### "Old transformers version"
Add to Cell 1:
```python
!pip install --upgrade transformers>=4.35.0 -q
```

### Want faster testing?
Use smaller dataset in Cell 4:
```python
# After tokenization, before training
tokenized_datasets['train'] = tokenized_datasets['train'].select(range(1000))  # Just 1000 samples
training_args.num_train_epochs = 1  # Just 1 epoch
```

---

## Expected Timeline

| Task | Time |
|------|------|
| Setup notebook | 1 min |
| Install packages | 2 min |
| Load dataset | 30 sec |
| Load model | 1 min |
| Tokenize dataset | 3 min |
| **Training (3 epochs, full dataset)** | **6-8 hours** |
| Evaluation | 5 min |
| Generate predictions | 1 min |

**Total: ~7-8 hours** (mostly training, can close browser)

---

## What You Get

After training completes:

✅ **Trained model** (`./final_model/`) ready to download
✅ **Test results** showing ROUGE scores
✅ **Sample predictions** to verify quality
✅ **Training history** for analysis

---

## Next Steps After Training

### Download and Use Locally

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your trained model
model = AutoModelForSeq2SeqLM.from_pretrained('./final_model')
tokenizer = AutoTokenizer.from_pretrained('./final_model')

# Use for inference
text = "Your Vietnamese text here..."
inputs = tokenizer(f"summarize: {text}", return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=128, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

---

## Key Takeaways

1. **Dataset loading is simple**: One line with `load_dataset()`
2. **No kernel setup needed**: Kaggle handles it
3. **Free GPU**: T4 with 16GB memory
4. **Close browser**: Training continues
5. **Download results**: Get trained model back

---

## Files Reference

- **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)** - Detailed guide with explanations
- **[kaggle_starter.py](kaggle_starter.py)** - Complete code with all cells
- **[KAGGLE_VS_LOCAL.md](KAGGLE_VS_LOCAL.md)** - Comparison guide
- **This file** - Quick 5-minute start

---

## Ready? Let's Go!

1. Open [kaggle.com/code](https://www.kaggle.com/code)
2. Create new notebook
3. Enable GPU
4. Copy cells from above
5. Click "Run All"
6. Wait ~8 hours
7. Download trained model
8. Done! 🎉

**Questions?** Check [KAGGLE_SETUP.md](KAGGLE_SETUP.md) for detailed troubleshooting.
