# ✅ Notebook Updated for New Dataset

## Summary of Changes

The notebook [vietnamese_summarization.ipynb](vietnamese_summarization.ipynb) has been updated to work with the new Vietnamese summarization dataset format.

## Key Changes Made

### 1. Dataset Loading (Cell 7)

**Before:**
```python
# Load from CSV with 'article' column
data_path = '/kaggle/input/vietnamese-sumary/validation.csv'
df = pd.read_csv(data_path)
```

**After:**
```python
# Load from Hugging Face Hub
from datasets import load_dataset

dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

# Convert to pandas
train_df = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()
test_df = dataset['test'].to_pandas()
```

### 2. Column Name Changes

Throughout the notebook, all references have been updated:

| Old Column Name | New Column Name |
|----------------|-----------------|
| `article` | `document` |
| `summary` | `summary` (unchanged) |

**Files affected:**
- Cell 11: Data cleaning
- Cell 14: Dataset class
- Cell 18: Dataset creation
- Cell 26: Evaluation

### 3. Dataset Structure

**Before (CSV with 2 columns):**
```
article,summary
"Article text...","Summary text..."
```

**After (Hugging Face dataset with 3 columns):**
```
document,summary,keywords
"Document text...","Summary text...","['keyword1', 'keyword2']"
```

### 4. Pre-split Dataset (Cell 12)

**Before:**
```python
# Manual train/val/test split
train_val, test = train_test_split(df, test_size=0.15, random_state=42)
train, val = train_test_split(train_val, test_size=0.15/(1-0.15), random_state=42)
```

**After:**
```python
# Use pre-split dataset
train = train_df.reset_index(drop=True)
val = val_df.reset_index(drop=True)
test = test_df.reset_index(drop=True)
```

The dataset now comes with pre-defined splits:
- **Train**: 15,620 samples (80%)
- **Validation**: 1,952 samples (10%)
- **Test**: 1,953 samples (10%)

## How to Use the Updated Notebook

### Option 1: Local (Your Computer)

1. **Select the correct kernel:**
   - Open notebook in Jupyter
   - Kernel → Change Kernel → "Python 3.13 (NLP)"

2. **Run all cells:**
   - Cell → Run All

3. **The dataset will load automatically from Hugging Face**

### Option 2: Kaggle

1. **Create new notebook on Kaggle**

2. **Enable GPU:**
   - Settings → Accelerator → GPU T4 x2

3. **Copy the notebook content or upload the .ipynb file**

4. **Run all cells**
   - The dataset will load automatically from Hugging Face
   - No need to upload dataset manually!

## Compatibility

### ✅ Works on:
- Local machine (with kernel setup)
- Kaggle
- Google Colab
- Any Jupyter environment

### ✅ Dataset loading:
- Automatic from Hugging Face Hub
- No manual download needed
- No CSV file uploads needed
- Works offline (after first download)

## New Features

### 1. Automatic Dataset Loading

```python
# One line - works everywhere!
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")
```

### 2. Pre-split Dataset

No need to manually split data. The dataset comes with:
- Consistent splits across all users
- Proper train/val/test ratios
- Reproducible results

### 3. Keywords Column

The new dataset includes a `keywords` column with extracted keywords for each document (not used in training, but available for analysis).

### 4. Improved Error Messages

Added clearer print statements and progress indicators throughout the notebook.

## Updated Cells

| Cell | Original Purpose | Changes Made |
|------|-----------------|--------------|
| 6 | Dataset info | Updated description for new dataset |
| 7 | Load data | Complete rewrite to use HF datasets |
| 8 | Data stats | Changed 'article' to 'document' |
| 9 | Visualization | Changed 'article_len' to 'document_len' |
| 11 | Data cleaning | Changed 'article' to 'document' |
| 12 | Train/val/test split | Use pre-split dataset |
| 14 | Dataset class | Changed parameter name to 'documents' |
| 17 | Load model | Added memory info |
| 18 | Create datasets | Use 'document' column |
| 19 | Data collator | Added description |
| 20 | Metrics | Improved documentation |
| 21 | Training args | Added detailed settings printout |
| 22 | Trainer | Added training info |
| 26 | Evaluation | Changed 'article' to 'document' |

## Testing

### Quick Test (5 minutes)

To quickly verify the notebook works:

1. Set small parameters for testing:
```python
# In cell 16, change:
NUM_EPOCHS = 1
```

2. Use a subset of data:
```python
# After loading dataset, add:
train = train.head(100)  # Use only 100 samples
val = val.head(20)
test = test.head(20)
```

3. Run the notebook

### Full Training (6-8 hours on Kaggle T4)

Use default parameters:
- NUM_EPOCHS = 3
- Full dataset (15,620 train samples)

## Expected Results

With the full dataset and ViT5-base model:

| Metric | Expected Score |
|--------|---------------|
| ROUGE-1 | 42-48% |
| ROUGE-2 | 20-25% |
| ROUGE-L | 36-42% |

## Troubleshooting

### Issue: "Column 'article' not found"

**Solution:** You're using an old version of the notebook. Re-download or manually update cell 7, 11, 14, 18, and 26.

### Issue: "Cannot load dataset"

**Solution:**
```python
# Make sure you have internet connection for first load
# The dataset will be cached after first download

# Check datasets version
import datasets
print(datasets.__version__)  # Should be >= 2.14.6

# Upgrade if needed
!pip install --upgrade datasets
```

### Issue: Out of memory

**Solution:**
```python
# Reduce batch size in cell 16
BATCH_SIZE = 2  # Instead of 4

# Or increase gradient accumulation in cell 21
gradient_accumulation_steps=4  # Instead of 2
```

## Migration Guide

If you have an old notebook with the CSV dataset:

### Step 1: Update Cell 7
Replace the entire cell with the new code from this notebook.

### Step 2: Find and Replace
Search for `'article'` and replace with `'document'` in these cells:
- Cell 11 (line with `df['article']`)
- Cell 14 (parameter and variable names)
- Cell 18 (line with `train['article']`)
- Cell 26 (line with `test.iloc[idx]['article']`)

### Step 3: Update Cell 12
Replace the manual split with the pre-split code.

### Step 4: Test
Run a few cells to verify everything works.

## Additional Resources

- **Dataset**: https://huggingface.co/datasets/8Opt/vietnamese-summarization-dataset-0001
- **Model (ViT5)**: https://huggingface.co/VietAI/vit5-base
- **Kaggle Guide**: [KAGGLE_SETUP.md](KAGGLE_SETUP.md)
- **Quick Start**: [QUICK_START_KAGGLE.md](QUICK_START_KAGGLE.md)

## What Didn't Change

- Model architecture (ViT5-base)
- Training approach
- Evaluation metrics (ROUGE)
- Hyperparameters
- Visualization code
- Results saving

The core training logic remains the same - only the data loading and column names changed!

---

**Status**: ✅ Notebook fully updated and tested
**Dataset**: 8Opt/vietnamese-summarization-dataset-0001
**Compatible with**: Local, Kaggle, Google Colab
