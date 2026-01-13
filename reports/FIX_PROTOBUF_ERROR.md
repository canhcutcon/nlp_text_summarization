# Fix Protobuf Import Error for mT5 Models ✅

**Status**: FIXED in main notebook

## Problem
When loading mT5 models, you encounter:
```
ImportError: T5Converter requires the protobuf library but it was not found in your environment.
```

## Solution

Add this cell **at the very beginning** of your notebook (before importing transformers):

```python
# Install required protobuf library
!pip install -q protobuf sentencepiece

# Restart may be required - if error persists, restart kernel and run again
```

## Complete Installation Cell

Add this as your **first code cell** in the notebook:

```python
# ============================================================
# INSTALL DEPENDENCIES
# ============================================================

print("Installing required packages...")

# Install protobuf (required for T5 tokenizer)
!pip install -q protobuf>=3.20.0

# Install sentencepiece (required for T5 tokenizer)
!pip install -q sentencepiece

# Install other required packages
!pip install -q transformers datasets torch
!pip install -q rouge-score py-rouge evaluate scikit-learn
!pip install -q underthesea networkx matplotlib seaborn pandas tqdm

print("✅ All packages installed successfully!")
print("\n⚠️  If you see 'protobuf' errors, please RESTART THE KERNEL and run cells again.")
```

## Quick Fix for Existing Notebooks

If you're already in the middle of running a notebook:

1. **Add this cell and run it:**
   ```python
   !pip install -q protobuf sentencepiece
   ```

2. **Restart the kernel** (Kernel → Restart in Jupyter, or Runtime → Restart runtime in Colab)

3. **Run all cells from the beginning**

## Environment-Specific Instructions

### Kaggle
```python
# Add at the beginning of your notebook
!pip install -q --upgrade protobuf sentencepiece
```

### Google Colab
```python
# Add at the beginning of your notebook
!pip install -q protobuf sentencepiece
# You may need to restart runtime: Runtime → Restart runtime
```

### Local Jupyter
```bash
# In terminal
pip install protobuf sentencepiece

# Or in notebook cell
!pip install protobuf sentencepiece
```

### Conda Environment
```bash
# In terminal
conda install -c conda-forge protobuf sentencepiece
```

## Why This Error Occurs

The mT5 model uses SentencePiece tokenization, which requires:
1. **protobuf** library - for loading the .spiece model file
2. **sentencepiece** library - for tokenization operations

Both must be installed before importing AutoTokenizer for mT5 models.

## Verification

After installation, verify it works:

```python
import transformers
print(f"Transformers version: {transformers.__version__}")

try:
    import google.protobuf
    print(f"✅ Protobuf version: {google.protobuf.__version__}")
except ImportError:
    print("❌ Protobuf not found")

try:
    import sentencepiece
    print(f"✅ SentencePiece installed")
except ImportError:
    print("❌ SentencePiece not found")

# Test loading mT5 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
print("✅ mT5 tokenizer loaded successfully!")
```

## Alternative: Use Slow Tokenizer

If protobuf installation fails, you can force using the slow tokenizer:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Force use of slow tokenizer (no protobuf required)
tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False  # Use slow tokenizer instead
)

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
```

**Note:** The slow tokenizer is slightly slower but doesn't require protobuf.

---

## Updated Installation for vietnamese_summarization_mt5_rtx_4070.ipynb

Our main notebook should have this cell at the beginning:

```python
# ============================================================
# 1. INSTALL PACKAGES
# ============================================================

print("Installing required packages...")
print("This may take 2-3 minutes on first run...")

# Core dependencies
!pip install -q protobuf>=3.20.0 sentencepiece

# Transformers and ML libraries
!pip install -q transformers datasets torch

# Evaluation metrics
!pip install -q rouge-score py-rouge evaluate scikit-learn

# Vietnamese NLP and utilities
!pip install -q underthesea networkx

# Visualization
!pip install -q matplotlib seaborn pandas tqdm

print("\n" + "="*60)
print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
print("="*60)
```

This ensures protobuf is installed before any model loading attempts.
