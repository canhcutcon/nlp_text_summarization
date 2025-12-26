# 🚀 Quick Reference - Vietnamese Text Summarization

## 📍 Your Setup

**Dataset Location (Kaggle):**
```
/kaggle/input/vietnamese-sumary/
├── train.csv        (15,620 samples)
├── validation.csv   (1,952 samples)
└── test.csv         (1,953 samples)
```

**Required Columns:**
- `document` - Vietnamese text to summarize
- `summary` - Target summary

---

## ⚡ Quick Start (3 Steps)

### On Kaggle:

**1. Upload Notebook**
- Go to kaggle.com/code → New Notebook
- Upload `vietnamese_summarization.ipynb`

**2. Enable GPU**
- Settings → GPU T4 x2 → Save

**3. Run**
- Cell → Run All
- Wait 6-8 hours (can close browser)

### On Local:

**1. Select Kernel**
- Jupyter → Kernel → "Python 3.13 (NLP)"

**2. Prepare Data**
- Put CSV files in `./data/` folder
- OR modify Cell 7 to use Hugging Face dataset

**3. Run**
- Cell → Run All

---

## 🎯 What You Get

After training:

| Output | Location | Size |
|--------|----------|------|
| Trained Model | `./vit5_final/` | ~1.5GB |
| Test Results | `test_results.csv` | ~5MB |
| Statistics | `summary_statistics.json` | <1MB |
| Checkpoints | `./vit5_summarization/` | ~5GB |

**Expected Performance:**
- ROUGE-1: 42-48%
- ROUGE-2: 20-25%
- ROUGE-L: 36-42%

---

## 🔧 Common Modifications

### Quick Test (5 min instead of 8 hours)

After Cell 7:
```python
train = train.head(100)
val = val.head(20)
test = test.head(20)
```

Cell 16:
```python
NUM_EPOCHS = 1
```

### Better Performance

Cell 16:
```python
BATCH_SIZE = 8  # If you have memory
MAX_LENGTH = 768  # For longer documents
```

### Fix Out of Memory

Cell 16:
```python
BATCH_SIZE = 2  # Reduce
```

Cell 21:
```python
gradient_accumulation_steps=4  # Increase
```

---

## 📊 Cell Guide

| Cells | Purpose | Time |
|-------|---------|------|
| 1-6 | Setup & theory | <1 min |
| **7** | **Load data from Kaggle** | **10 sec** |
| 8-9 | Explore & visualize | 30 sec |
| 10-12 | Clean & split | 10 sec |
| 13-22 | Setup training | 2 min |
| **23** | **TRAIN MODEL** | **6-8 hrs** |
| 24-26 | Save & evaluate | 10 min |
| 27-31 | Analyze & export | 2 min |

**Total:** ~7-8 hours (mostly training)

---

## 💡 Key Code Snippets

### Load Data (Cell 7)
```python
DATA_PATH = '/kaggle/input/vietnamese-sumary'
train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
val_df = pd.read_csv(f'{DATA_PATH}/validation.csv')
test_df = pd.read_csv(f'{DATA_PATH}/test.csv')
```

### Model Config (Cell 16)
```python
MODEL_NAME = 'VietAI/vit5-base'
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 128
```

### Use Trained Model (After training)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('./vit5_final')
tokenizer = AutoTokenizer.from_pretrained('./vit5_final')

text = "Your Vietnamese text..."
inputs = tokenizer(f"summarize: {text}", return_tensors='pt', truncation=True)
outputs = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "FileNotFoundError" | Add dataset in Kaggle (right sidebar → Add Data) |
| "KeyError: 'document'" | Check CSV has `document` column (not `article`) |
| "CUDA out of memory" | Reduce `BATCH_SIZE = 2` |
| "Kernel dead" | Enable GPU, restart kernel |
| Slow training | Make sure GPU is enabled (should be ~4s/it, not 40s/it) |

---

## 📁 File Index

| File | Purpose |
|------|---------|
| **vietnamese_summarization.ipynb** | Main notebook (USE THIS) |
| **KAGGLE_INPUT_SETUP.md** | Detailed setup guide |
| QUICK_REFERENCE.md | This file |
| KAGGLE_SETUP.md | Alternative: HuggingFace dataset |
| kaggle_starter.py | Alternative: Python script |

---

## ✅ Verification Checklist

Before running full training:

- [ ] Dataset added to Kaggle notebook
- [ ] GPU T4 x2 enabled
- [ ] Cell 7 shows: "✅ Found Kaggle input directory"
- [ ] Cell 7 shows: "Train: 15,620 samples"
- [ ] Cell 8 shows: "document" and "summary" columns
- [ ] Cell 12 shows: All splits have correct columns
- [ ] Ready to train!

---

## 🎓 After Training

1. **Download model** from Output tab
2. **Check results** in `test_results.csv`
3. **Use model** locally with code above
4. **Share** (optional) on Hugging Face Hub

---

## 🆘 Need Help?

**Common Issues:**
1. Check [KAGGLE_INPUT_SETUP.md](KAGGLE_INPUT_SETUP.md) - Detailed troubleshooting
2. Verify GPU is enabled: Should see "CUDA available: True"
3. Check data path: Should be `/kaggle/input/vietnamese-sumary/`

**Still stuck?**
- Make sure CSV files have `document` and `summary` columns
- Try quick test mode first (100 samples, 1 epoch)
- Check Cell 7 output for errors

---

**Last Updated:** 2024-12-24
**Dataset:** Kaggle Input `/kaggle/input/vietnamese-sumary/`
**Model:** VietAI/vit5-base
**Status:** ✅ Ready to use
