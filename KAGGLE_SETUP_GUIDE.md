# 🚀 HƯỚNG DẪN SETUP VÀ TRAINING TRÊN KAGGLE

## 📋 MỤC LỤC

1. [Setup Dataset trên Kaggle](#1-setup-dataset)
2. [Create Notebook](#2-create-notebook)
3. [Configuration & Training](#3-training)
4. [Troubleshooting](#4-troubleshooting)
5. [Best Practices](#5-best-practices)

---

## 1️⃣ SETUP DATASET

### Cách 1: Upload VLSP Dataset lên Kaggle

1. **Prepare dataset locally:**
   ```
   vlsp2021-summarization/
   ├── train.csv
   ├── test.csv
   └── README.md
   ```

2. **Create Kaggle Dataset:**
   - Vào https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload folder hoặc ZIP file
   - Set title: "VLSP 2021 Text Summarization"
   - Set visibility: Private (nếu data riêng)
   - Click "Create"

3. **Dataset format:**
   ```csv
   article,summary
   "Văn bản tin tức dài...","Tóm tắt ngắn gọn..."
   ```

### Cách 2: Download từ VLSP Official

```python
# Trong Kaggle notebook
!wget https://vlsp.org.vn/download/summarization-2021.zip
!unzip summarization-2021.zip
```

---

## 2️⃣ CREATE KAGGLE NOTEBOOK

### Bước 1: Create New Notebook

1. Vào https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings:
   - **Type:** Notebook
   - **Language:** Python
   - **Accelerator:** GPU T4 x2 (hoặc P100 nếu có)
   - **Internet:** ON
   - **Environment:** Pin to reproducible environment

### Bước 2: Add Dataset

1. Trong notebook, click "Add Data" (bên phải)
2. Search dataset bạn vừa upload: "vlsp2021-summarization"
3. Click "Add"
4. Dataset sẽ xuất hiện tại: `/kaggle/input/vlsp2021-summarization/`

### Bước 3: Install Dependencies

```python
# Cell 1: Install packages
!pip install transformers==4.35.0 -q
!pip install datasets==2.14.6 -q
!pip install rouge-score==0.1.2 -q
!pip install sentencepiece==0.1.99 -q
!pip install accelerate==0.24.1 -q

print("✅ All packages installed!")
```

---

## 3️⃣ TRAINING

### Configuration cho Kaggle GPU

```python
import torch

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
# Kaggle GPU specs:
# - T4: 16GB VRAM
# - P100: 16GB VRAM
# - GPU limit: 30 hours/week
```

### Hyperparameters cho Kaggle

```python
# RECOMMENDED SETTINGS CHO KAGGLE T4 GPU

# Model: ViT5-base
MODEL_NAME = 'VietAI/vit5-base'

# Batch size - điều chỉnh theo GPU memory
BATCH_SIZE = 4              # T4: 4-8, P100: 8-16
GRADIENT_ACCUMULATION = 2    # Effective batch = 4*2 = 8

# Learning rate
LEARNING_RATE = 5e-5        # Standard cho fine-tuning T5

# Epochs
NUM_EPOCHS = 3              # 3-5 epochs đủ cho summarization

# Sequence lengths
MAX_INPUT_LENGTH = 512      # Article length
MAX_TARGET_LENGTH = 128     # Summary length

# Mixed precision training (quan trọng!)
FP16 = True                 # Giảm memory usage ~50%

# Logging
LOGGING_STEPS = 100
EVAL_STEPS = 500
SAVE_STEPS = 500
```

### Training Arguments

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir='./vit5_summarization',
    
    # Training config
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    
    # Optimization
    weight_decay=0.01,
    warmup_steps=500,
    fp16=FP16,  # CRITICAL cho Kaggle
    
    # Evaluation
    evaluation_strategy='steps',
    eval_steps=EVAL_STEPS,
    predict_with_generate=True,
    
    # Logging & Saving
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,  # Chỉ giữ 3 checkpoints mới nhất
    load_best_model_at_end=True,
    metric_for_best_model='rouge1',
    greater_is_better=True,
    
    # Disable external logging
    report_to='none',  # Không dùng wandb
    
    # Memory optimization
    gradient_checkpointing=True,  # Giảm memory
    optim='adamw_torch',
)
```

### Training Time Estimates

**Với ViT5-base trên Kaggle T4:**

| Dataset Size | Batch Size | Epochs | Training Time |
|--------------|------------|--------|---------------|
| 10K samples  | 4          | 3      | ~2-3 hours    |
| 50K samples  | 4          | 3      | ~8-10 hours   |
| 100K samples | 4          | 3      | ~15-20 hours  |

**Tips:**
- Kaggle có giới hạn **30 giờ GPU/tuần**
- Save checkpoints thường xuyên
- Monitor GPU usage: `!nvidia-smi`

---

## 4️⃣ TROUBLESHOOTING

### ❌ Problem: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Giảm batch size:**
   ```python
   BATCH_SIZE = 2  # Thay vì 4-8
   ```

2. **Tăng gradient accumulation:**
   ```python
   GRADIENT_ACCUMULATION = 4  # Maintain effective batch size
   ```

3. **Giảm sequence length:**
   ```python
   MAX_INPUT_LENGTH = 384  # Thay vì 512
   MAX_TARGET_LENGTH = 96   # Thay vì 128
   ```

4. **Enable gradient checkpointing:**
   ```python
   training_args = Seq2SeqTrainingArguments(
       gradient_checkpointing=True,  # Giảm ~50% memory
       ...
   )
   ```

5. **Clear cache thường xuyên:**
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

### ❌ Problem: Training quá chậm

**Solutions:**

1. **Enable mixed precision:**
   ```python
   fp16=True  # CRITICAL!
   ```

2. **Giảm eval frequency:**
   ```python
   eval_steps=1000  # Thay vì 500
   ```

3. **Disable some logging:**
   ```python
   logging_steps=500  # Thay vì 100
   ```

4. **Use smaller validation set:**
   ```python
   val_df = val_df.sample(frac=0.5)  # Chỉ dùng 50% val data
   ```

### ❌ Problem: Model không learn (loss không giảm)

**Diagnosis & Solutions:**

1. **Check learning rate:**
   ```python
   # Too high: loss explodes
   # Too low: loss không giảm
   LEARNING_RATE = 5e-5  # Standard starting point
   ```

2. **Check data quality:**
   ```python
   # Print samples để verify
   print(train_df.iloc[0]['article'])
   print(train_df.iloc[0]['summary'])
   ```

3. **Check labels:**
   ```python
   # Labels không được là -100 hết
   sample = train_dataset[0]
   print((sample['labels'] != -100).sum())  # Should be > 0
   ```

4. **Warm-up steps:**
   ```python
   warmup_steps=500  # Rất quan trọng cho stability
   ```

### ❌ Problem: Kaggle session timeout

**Prevention:**

1. **Save checkpoints thường xuyên:**
   ```python
   save_steps=500  # Save mỗi 500 steps
   ```

2. **Enable auto-resume:**
   ```python
   # Check if checkpoint exists
   import os
   checkpoint_dir = './vit5_summarization/checkpoint-*'
   if os.path.exists(checkpoint_dir):
       trainer.train(resume_from_checkpoint=checkpoint_dir)
   else:
       trainer.train()
   ```

3. **Monitor time:**
   ```python
   import time
   start_time = time.time()
   trainer.train()
   elapsed = (time.time() - start_time) / 3600
   print(f"Training took {elapsed:.2f} hours")
   ```

---

## 5️⃣ BEST PRACTICES

### 📊 Monitoring Training

```python
# Cell để monitor GPU usage
!nvidia-smi --loop=10  # Update mỗi 10 giây
```

```python
# Cell để check training progress
import matplotlib.pyplot as plt

# Plot training history
history = trainer.state.log_history
losses = [x['loss'] for x in history if 'loss' in x]
plt.plot(losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

### 💾 Saving Best Practices

```python
# Save model sau training
trainer.save_model('./vit5_final')
tokenizer.save_pretrained('./vit5_final')

# Save training history
import json
with open('./training_history.json', 'w') as f:
    json.dump(trainer.state.log_history, f)

# Save results
results_df.to_csv('./test_results.csv', index=False)
```

### 🔄 Version Control

```python
# Document experiment
experiment_config = {
    'model': MODEL_NAME,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'epochs': NUM_EPOCHS,
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'final_rouge1': final_rouge1,
    'final_rouge2': final_rouge2,
    'final_rougeL': final_rougeL,
    'training_time_hours': training_time
}

with open('./experiment_config.json', 'w') as f:
    json.dump(experiment_config, f, indent=2)
```

### 🎯 Optimization Tips

1. **Start Small:**
   - Train on subset trước (1K samples)
   - Verify pipeline hoạt động
   - Scale up gradually

2. **Use Checkpoints:**
   - Enable `save_total_limit=3`
   - Always `load_best_model_at_end=True`

3. **Monitor Metrics:**
   - Check validation loss mỗi epoch
   - Early stopping nếu không improve

4. **Resource Management:**
   ```python
   # Clear memory after each experiment
   del model
   del trainer
   gc.collect()
   torch.cuda.empty_cache()
   ```

---

## 📈 EXPECTED RESULTS

### Baseline Performance (VLSP 2021)

| Model       | ROUGE-1 | ROUGE-2 | ROUGE-L | Training Time |
|-------------|---------|---------|---------|---------------|
| PhoBERT     | 0.35    | 0.15    | 0.30    | ~4 hours      |
| mT5-base    | 0.42    | 0.20    | 0.36    | ~8 hours      |
| ViT5-base   | **0.45**| **0.23**| **0.39**| ~8 hours      |
| ViT5-large  | 0.48    | 0.26    | 0.42    | ~16 hours     |

### Your Target

- **Minimum:** ROUGE-1 > 0.40
- **Good:** ROUGE-1 > 0.43
- **Excellent:** ROUGE-1 > 0.45

---

## 🔗 USEFUL LINKS

- **VLSP Website:** https://vlsp.org.vn/
- **ViT5 Model:** https://huggingface.co/VietAI/vit5-base
- **mT5 Model:** https://huggingface.co/google/mt5-base
- **ROUGE Documentation:** https://github.com/google-research/google-research/tree/master/rouge
- **Transformers Docs:** https://huggingface.co/docs/transformers

---

## ✅ FINAL CHECKLIST

Trước khi submit hoặc commit:

- [ ] Dataset đã upload và accessible
- [ ] All dependencies installed
- [ ] GPU đã được enable
- [ ] Training arguments đã configure
- [ ] Checkpoints được save properly
- [ ] Results được evaluate và visualize
- [ ] Model được save to output
- [ ] Training time < 30 hours/week limit

---

## 💡 PRO TIPS

1. **Use Kaggle Datasets API:**
   ```bash
   # Download dataset programmatically
   kaggle datasets download -d username/vlsp2021
   ```

2. **Version your notebooks:**
   - Click "Save Version" regularly
   - Add meaningful commit messages

3. **Share your work:**
   - Make notebook public sau khi verify results
   - Add comprehensive markdown explanations

4. **Compare with others:**
   - Check Kaggle leaderboard
   - Learn from top submissions

---

**Good luck với training! 🚀**

**Need help?** 
- Kaggle Forums: https://www.kaggle.com/discussions
- Discord: Vietnamese NLP Community
- Email: support@kaggle.com
