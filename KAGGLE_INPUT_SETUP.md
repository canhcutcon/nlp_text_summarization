# 📦 Kaggle Input Setup Guide

## Overview

The notebook has been configured to load data from your Kaggle input dataset at:
```
/kaggle/input/vietnamese-sumary/
├── train.csv
├── validation.csv
└── test.csv
```

## ✅ What's Already Done

The notebook automatically:
1. ✅ Loads CSV files from `/kaggle/input/vietnamese-sumary/`
2. ✅ Falls back to local `./data/` if not on Kaggle
3. ✅ Uses correct column names (`document` and `summary`)
4. ✅ Handles pre-split train/val/test data
5. ✅ Validates data structure before training

## 🚀 How to Use on Kaggle

### Step 1: Upload Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Click **File → Upload Notebook**
4. Select `vietnamese_summarization.ipynb`

### Step 2: Add Your Dataset

Your dataset is already available at `/kaggle/input/vietnamese-sumary/`

**If you need to verify or add it:**
1. In the right sidebar, click **"Add Data"**
2. Search for "vietnamese-sumary" (your dataset)
3. Click **"Add"**
4. The path will be: `/kaggle/input/vietnamese-sumary/`

### Step 3: Enable GPU

1. Click **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2**
3. Click **Save**

### Step 4: Run the Notebook

**Option A: Run All (Full Training)**
```
Cell → Run All
```
- This will take 6-8 hours
- You can close the browser - Kaggle keeps running

**Option B: Quick Test (5 minutes)**

After Cell 7 (data loading), add a new cell:
```python
# Use small subset for quick testing
train = train.head(100)
val = val.head(20)
test = test.head(20)
print(f"Using subset: {len(train)} train, {len(val)} val, {len(test)} test")
```

Then change NUM_EPOCHS in Cell 16:
```python
NUM_EPOCHS = 1  # Instead of 3
```

Run all cells - completes in ~5 minutes.

## 📊 Dataset Structure

Your CSV files should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `document` | Vietnamese news article | "Hôm nay, Bộ Y tế công bố..." |
| `summary` | Human-written summary | "Bộ Y tế ghi nhận 15.527 ca..." |
| `keywords` | Keywords (optional) | "['COVID-19', 'Bộ Y tế']" |

**Required columns:** `document`, `summary`
**Optional columns:** `keywords`, any others (will be ignored)

## 🔍 Verification

When you run Cell 7 (data loading), you should see:

```
Loading Vietnamese Summarization Dataset from Kaggle Input...
================================================================================
✅ Found Kaggle input directory: /kaggle/input/vietnamese-sumary

📂 Loading data files:
  Train: /kaggle/input/vietnamese-sumary/train.csv
  Validation: /kaggle/input/vietnamese-sumary/validation.csv
  Test: /kaggle/input/vietnamese-sumary/test.csv

✅ Dataset loaded successfully!

📊 Dataset Summary:
  Train: 15,620 samples
  Validation: 1,952 samples
  Test: 1,953 samples
  Total: 19,525 samples

📋 Columns in dataset: ['document', 'summary', 'keywords']
```

If you see this, everything is working! ✅

## 🏠 Using Locally

The notebook also works on your local machine:

### Option 1: Use Kaggle CSV Files

1. Download your CSV files from Kaggle
2. Put them in `./data/` folder:
   ```
   nlp_text_summarization/
   ├── data/
   │   ├── train.csv
   │   ├── validation.csv
   │   └── test.csv
   └── vietnamese_summarization.ipynb
   ```
3. Open notebook with "Python 3.13 (NLP)" kernel
4. Run all cells

### Option 2: Use Hugging Face Dataset

If you prefer to use the Hugging Face dataset instead:

Replace Cell 7 with:
```python
from datasets import load_dataset

dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

train_df = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()
test_df = dataset['test'].to_pandas()

print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
```

## 📝 Cell-by-Cell Breakdown

| Cell | Purpose | What It Does |
|------|---------|--------------|
| **1-6** | Setup & Theory | Install packages, explain concepts |
| **7** | Load Data | Load from `/kaggle/input/vietnamese-sumary/` |
| **8** | Explore Data | Show statistics and sample |
| **9** | Visualize | Plot length distributions |
| **10-11** | Clean Data | Remove invalid samples |
| **12** | Split Data | Assign train/val/test (already split) |
| **13-14** | Dataset Class | Create PyTorch dataset |
| **15-16** | Config | Set model and hyperparameters |
| **17** | Load Model | Load ViT5-base |
| **18** | Create Datasets | Tokenize data |
| **19-22** | Training Setup | Configure trainer |
| **23** | Train | Run training (6-8 hours) |
| **24** | Save Model | Save to `./vit5_final` |
| **25-26** | Evaluate | Test on test set |
| **27-29** | Analyze | Plot results, show samples |
| **30-31** | Save Results | Export to CSV/JSON |

## ⚙️ Configuration Options

### For Quick Testing

```python
# In Cell 7 (after loading data)
train = train.head(100)
val = val.head(20)
test = test.head(20)

# In Cell 16
NUM_EPOCHS = 1
BATCH_SIZE = 4
```

### For Full Training (Recommended)

```python
# In Cell 16 - use defaults
MODEL_NAME = 'VietAI/vit5-base'
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 128
```

### For Better Performance (if you have GPU memory)

```python
# In Cell 16
BATCH_SIZE = 8  # Instead of 4
MAX_LENGTH = 768  # Instead of 512 (for longer documents)

# In Cell 21
gradient_accumulation_steps=1  # Instead of 2
```

### If Out of Memory

```python
# In Cell 16
BATCH_SIZE = 2  # Reduce batch size

# In Cell 21
gradient_accumulation_steps=4  # Increase accumulation
```

## 📈 Expected Output

### After Cell 7 (Data Loading)
```
✅ Dataset loaded successfully!
📊 Dataset Summary:
  Train: 15,620 samples
  Validation: 1,952 samples
  Test: 1,953 samples
```

### After Cell 8 (Data Exploration)
```
📝 Document length statistics:
count    15620.000000
mean      2293.567623
std       2862.606264
...
```

### After Cell 23 (Training)
```
Training: 100%|██████████| 5000/5000 [6:45:32<00:00, 4.87s/it]
✅ Training completed!
Final training loss: 1.2345
```

### After Cell 26 (Evaluation)
```
📊 TEST RESULTS
ROUGE-1: 45.23 ± 12.45
ROUGE-2: 23.45 ± 8.67
ROUGE-L: 39.87 ± 10.23
```

## 🎯 Expected Results

With full dataset on Kaggle T4:

| Metric | Target | Excellent |
|--------|--------|-----------|
| ROUGE-1 | 40-45% | 45-50% |
| ROUGE-2 | 20-23% | 23-28% |
| ROUGE-L | 35-40% | 40-45% |

**Training time:** 6-8 hours on T4 GPU

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: train.csv"

**Cause:** Dataset not added to Kaggle notebook

**Solution:**
1. Right sidebar → Add Data
2. Search for your dataset name
3. Click Add
4. Verify path: `/kaggle/input/vietnamese-sumary/`

### Issue: "KeyError: 'document'"

**Cause:** CSV columns don't match expected names

**Solution:**
Check your CSV has columns named `document` and `summary`:
```python
# Add this after Cell 7
print(train_df.columns.tolist())
```

If different, rename them:
```python
# If your columns are 'article' and 'abstract':
train_df = train_df.rename(columns={'article': 'document', 'abstract': 'summary'})
val_df = val_df.rename(columns={'article': 'document', 'abstract': 'summary'})
test_df = test_df.rename(columns={'article': 'document', 'abstract': 'summary'})
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size
```python
# In Cell 16
BATCH_SIZE = 2  # Instead of 4

# In Cell 21
gradient_accumulation_steps=4  # Instead of 2
```

### Issue: "Kernel appears to be dead"

**Cause:** Usually memory issue

**Solution:**
1. Restart kernel: Kernel → Restart
2. Reduce batch size (see above)
3. Enable T4 x2 GPU if not already

## 📁 Output Files

After training completes:

```
/kaggle/working/
├── vit5_final/                    # Trained model
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
├── vit5_summarization/            # Checkpoints
│   └── checkpoint-*/
├── test_results.csv               # Detailed results
└── summary_statistics.json        # Summary metrics
```

**Download these files:**
1. Click **Output** tab (right sidebar)
2. Find files/folders
3. Click download icon
4. Use locally for inference

## 🎓 Next Steps After Training

### 1. Download Model
- Click Output tab
- Download `vit5_final` folder
- Size: ~1.5GB

### 2. Use Locally

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your trained model
model = AutoModelForSeq2SeqLM.from_pretrained('./vit5_final')
tokenizer = AutoTokenizer.from_pretrained('./vit5_final')

# Generate summary
text = "Your Vietnamese text here..."
inputs = tokenizer(f"summarize: {text}", return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=128, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

### 3. Share or Deploy
- Upload to Hugging Face Hub
- Create API endpoint
- Integrate into application

## 📞 Support

**Common Questions:**

**Q: Can I use my own dataset?**
A: Yes! Just make sure it has `document` and `summary` columns in CSV format.

**Q: How do I change the model?**
A: In Cell 16, change `MODEL_NAME`:
```python
MODEL_NAME = 'google/mt5-base'  # or 'VietAI/vit5-large'
```

**Q: Can I train on CPU?**
A: Yes, but it will take 10-20x longer. Not recommended for full dataset.

**Q: What if I have different column names?**
A: Rename them after loading:
```python
train_df = train_df.rename(columns={'old_name': 'document', 'old_summary': 'summary'})
```

---

**Status:** ✅ Ready to use with Kaggle input
**Dataset Path:** `/kaggle/input/vietnamese-sumary/`
**Required Files:** `train.csv`, `validation.csv`, `test.csv`
**Required Columns:** `document`, `summary`
