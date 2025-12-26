# Kaggle vs Local Setup - Quick Reference

## Dataset Loading Comparison

### ❌ OLD WAY (Don't use these)

```bash
# Don't use curl
curl -X GET "https://datasets-server.huggingface.co/rows?dataset=8Opt%2Fvietnamese-summarization-dataset-0001&config=default&split=train&offset=0&length=100"

# Don't use git clone
git clone https://huggingface.co/datasets/8Opt/vietnamese-summarization-dataset-0001
```

### ✅ NEW WAY (Use this instead)

**Both Local and Kaggle:**
```python
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")
```

That's it! One line works everywhere.

---

## Environment Setup Comparison

### Local Setup (Your Computer)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Jupyter kernel
pip install ipykernel jupyter
python -m ipykernel install --user --name=nlp_venv --display-name="Python 3.13 (NLP)"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open notebook and select "Python 3.13 (NLP)" kernel
jupyter notebook vietnamese_summarization.ipynb
```

**Hardware:**
- Your own computer
- Apple Silicon (MPS) or CPU
- Limited memory

**Best for:**
- Development
- Testing
- Small experiments

---

### Kaggle Setup (Cloud)

```python
# 1. Create new notebook on kaggle.com

# 2. Enable GPU (Settings → Accelerator → GPU T4 x2)

# 3. Install/upgrade packages (single cell)
!pip install -q --upgrade transformers>=4.35.0 datasets sentencepiece rouge-score

# 4. Load dataset (single cell)
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

# 5. Start training immediately!
```

**Hardware:**
- Free NVIDIA T4/P100 GPU
- 16GB+ GPU memory
- 30 hours/week free

**Best for:**
- Training large models
- Full dataset training
- Production runs

---

## Code Differences

### Loading Dataset

**Local (both work):**
```python
# Option 1: From Hugging Face Hub (recommended)
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

# Option 2: From local parquet files
import pandas as pd
from datasets import Dataset
df = pd.read_parquet('data/train-00000-of-00001.parquet')
dataset = Dataset.from_pandas(df)
```

**Kaggle (same!):**
```python
# Same as Option 1 above - just works!
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")
```

---

### GPU Configuration

**Local (Apple Silicon):**
```python
import torch

# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Training args
training_args = Seq2SeqTrainingArguments(
    # ...
    use_mps_device=True,  # Enable MPS
    fp16=False,  # MPS doesn't support fp16 yet
)
```

**Kaggle (NVIDIA GPU):**
```python
import torch

# Use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training args
training_args = Seq2SeqTrainingArguments(
    # ...
    fp16=True,  # Enable mixed precision for speed
)
```

---

## Training Configuration

### Local (Conservative settings for limited resources)

```python
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Quick test
    per_device_train_batch_size=2,  # Small batch
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Compensate for small batch
    fp16=False,  # Not for MPS
    learning_rate=5e-5,
    eval_steps=1000,
    save_steps=1000,
)

# Use subset for testing
train_dataset = dataset['train'].select(range(1000))  # Just 1000 samples
```

### Kaggle (Aggressive settings for full training)

```python
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Full training
    per_device_train_batch_size=8,  # Larger batch
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,  # Mixed precision
    learning_rate=5e-5,
    eval_steps=500,
    save_steps=500,
)

# Use full dataset
train_dataset = dataset['train']  # All 15,620 samples
```

---

## File Paths

### Local

```python
# Dataset
data_path = './data/train-00000-of-00001.parquet'

# Model output
output_dir = './results'
model_dir = './final_model'

# Results
results_file = './test_results.json'
```

### Kaggle

```python
# Dataset (if uploaded)
data_path = '/kaggle/input/vietnamese-summarization-dataset/train-00000-of-00001.parquet'

# Model output (in working directory)
output_dir = './results'  # or '/kaggle/working/results'
model_dir = './final_model'  # or '/kaggle/working/final_model'

# Results
results_file = './test_results.json'
```

---

## Quick Start Commands

### Local

```bash
# Activate environment
source .venv/bin/activate

# Verify setup
python verify_setup.py

# Load dataset
python load_dataset.py

# Start Jupyter
jupyter notebook
# Then select "Python 3.13 (NLP)" kernel
```

### Kaggle

```python
# Cell 1: Load dataset
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")

# Cell 2: Load model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('VietAI/vit5-base')
tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-base')

# Cell 3: Start training
# (See kaggle_starter.py for complete code)
```

---

## Expected Performance

### Local (Apple Silicon M1/M2)

| Task | Time |
|------|------|
| Load dataset | ~5 seconds |
| Load model | ~30 seconds |
| Train 1 epoch (1K samples) | ~15 minutes |
| Train 3 epochs (full) | 6-8 hours |
| Inference (1 sample) | ~2 seconds |

### Kaggle (T4 GPU)

| Task | Time |
|------|------|
| Load dataset | ~5 seconds |
| Load model | ~30 seconds |
| Train 1 epoch (1K samples) | ~3 minutes |
| Train 3 epochs (full) | 6-8 hours |
| Inference (1 sample) | ~0.5 seconds |

---

## Package Installation

### Local

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify
pip list | grep transformers
pip list | grep torch
pip list | grep datasets
```

### Kaggle

```python
# Most packages pre-installed, just upgrade
!pip install -q --upgrade transformers>=4.35.0

# Check versions
import transformers
print(transformers.__version__)
```

---

## When to Use Each

### Use Local When:
- ✅ Developing and debugging code
- ✅ Testing with small datasets
- ✅ Quick experiments
- ✅ You have good hardware (M1/M2 Max)
- ✅ Need frequent code changes

### Use Kaggle When:
- ✅ Training with full dataset
- ✅ Need GPU acceleration
- ✅ Long training runs (hours)
- ✅ Want to train overnight
- ✅ Limited local resources
- ✅ Need reproducible environment

### Best Practice:
1. **Develop locally** - write and test your code
2. **Train on Kaggle** - run full training with GPU
3. **Download model** - bring trained model back to local
4. **Use locally** - inference with trained model

---

## Troubleshooting

### Local Issues

**Issue: Kernel not found**
```bash
# Solution: Reinstall kernel
python -m ipykernel install --user --name=nlp_venv --display-name="Python 3.13 (NLP)" --force
```

**Issue: Out of memory**
```python
# Solution: Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

### Kaggle Issues

**Issue: CUDA out of memory**
```python
# Solution: Reduce batch size
per_device_train_batch_size=4  # Instead of 8
gradient_accumulation_steps=4  # Instead of 2
```

**Issue: Session timeout**
```
# Solution: Use "Commit and Run All"
# Kaggle will complete even if browser closes
```

---

## Summary Table

| Feature | Local | Kaggle |
|---------|-------|--------|
| **Setup** | Manual (venv + kernel) | Automatic |
| **GPU** | MPS (Apple) or CPU | CUDA (NVIDIA T4/P100) |
| **Memory** | Limited by your RAM | 16GB+ GPU memory |
| **Time Limit** | Unlimited | 9-12 hours per session |
| **Dataset Load** | Same code | Same code |
| **FP16** | No (MPS limitation) | Yes (2x faster) |
| **Cost** | Free (your hardware) | Free (30h GPU/week) |
| **Best For** | Development | Training |

---

## The Dataset Loading Solution (TL;DR)

**❌ DON'T:**
- Use `curl` to hit API endpoints
- Use `git clone` to download repos
- Manually download files

**✅ DO:**
```python
from datasets import load_dataset
dataset = load_dataset("8Opt/vietnamese-summarization-dataset-0001")
```

**This works:**
- ✅ On your local machine
- ✅ On Kaggle
- ✅ On Google Colab
- ✅ Anywhere Python runs

**One line. Everywhere. Always.**

---

## Ready to Start?

### Local Development
1. Read: [SETUP_SUCCESS.md](SETUP_SUCCESS.md)
2. Run: `python verify_setup.py`
3. Start: Open Jupyter with "Python 3.13 (NLP)" kernel

### Kaggle Training
1. Read: [KAGGLE_SETUP.md](KAGGLE_SETUP.md)
2. Copy: Code from [kaggle_starter.py](kaggle_starter.py)
3. Start: Paste into Kaggle notebook and run!

**Both use the same dataset loading code!** 🎉
