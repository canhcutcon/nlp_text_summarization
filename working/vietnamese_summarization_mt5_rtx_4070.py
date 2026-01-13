#!/usr/bin/env python
# coding: utf-8

# # Vietnamese Text Summarization - mT5-Small (FINAL)
# 
# ✅ **Model**: google/mt5-small (300M params)  
# ✅ **GPU**: RTX 4070 SUPER 12GB  
# ✅ **Optimized**: Batch size 8, FP16, Gradient Checkpointing  
# ✅ **Fixed**: All previous errors (CSV, tokenization, OOM)  
# ✅ **Training time**: ~1-1.5 hours  
# 
# ---
# 
# ## Why mT5-Small?
# - ✅ Fast training (~1.5h vs 3h for base)
# - ✅ Good results (only 1-2% lower ROUGE than base)
# - ✅ No OOM issues
# - ✅ Perfect for Vietnamese summarization

# In[ ]:


# # Trong terminal hoặc notebook cell
# !pkill -9 python
# !pkill -9 jupyter

# # Hoặc kill specific PIDs
# !kill -9 2048433
# !kill -9 2059574


# ## 1. Install Packages

# In[ ]:


get_ipython().system('pip install -q transformers datasets accelerate sentencepiece evaluate rouge-score py-rouge scikit-learn protobuf --root-user-action=ignore')
# ============================================================================
# Install Required Packages
# ============================================================================
get_ipython().system('pip install -q torch torchvision torchaudio')
get_ipython().system('pip install -q underthesea  # For Vietnamese text processing')
get_ipython().system('pip install -q scikit-learn networkx  # For TextRank')
get_ipython().system('pip install --upgrade --force-reinstall underthesea')
print("✅ All packages installed!")


# ## 2. Load and Verify Data

# In[ ]:


import re
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np

def sent_tokenize(text: str) -> list[str]:  # ← Use lowercase 'list'
    """Vietnamese sentence tokenizer"""
    pattern = r'(?<=[.!?])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

# Load CSV - FIXED: Bỏ header=None vì CSV có header
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print(f"\n📊 Dataset Statistics:")
print(f"  Train: {len(train_df):,} samples")
print(f"  Validation: {len(val_df):,} samples")
print(f"  Test: {len(test_df):,} samples")

# Show sample
print(f"\n📄 Sample data:")
sample = train_df.iloc[0]
print(f"\nDocument (first 200 chars):\n{sample['document'][:200]}...")
print(f"\nSummary:\n{sample['summary']}")


# In[ ]:


# ============================================================================
# Data Statistics
# ============================================================================
def analyze_text_lengths(df: pd.DataFrame, name: str):
    """Analyze document and summary lengths"""
    doc_words = df['document'].apply(lambda x: len(x.split()))
    sum_words = df['summary'].apply(lambda x: len(x.split()))

    doc_sents = df['document'].apply(lambda x: len(sent_tokenize(x)))
    sum_sents = df['summary'].apply(lambda x: len(sent_tokenize(x)))

    print(f"\n{name} Statistics:")
    print(f"  Document words: mean={doc_words.mean():.1f}, median={doc_words.median():.1f}")
    print(f"  Summary words: mean={sum_words.mean():.1f}, median={sum_words.median():.1f}")
    print(f"  Document sentences: mean={doc_sents.mean():.1f}, median={doc_sents.median():.1f}")
    print(f"  Summary sentences: mean={sum_sents.mean():.1f}, median={sum_sents.median():.1f}")
    print(f"  Compression ratio: {(sum_words.mean() / doc_words.mean() * 100):.1f}%")

analyze_text_lengths(train_df, "Train")
analyze_text_lengths(val_df, "Validation")
analyze_text_lengths(test_df, "Test")


# ## 3. Clear GPU Memory

# In[ ]:


import torch
import gc

# Clear any existing models
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✓ GPU memory cleared")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# In[ ]:


# ============================================================================
# TextRank Implementation
# ============================================================================
class TextRankSummarizer:
    """TextRank algorithm for extractive summarization"""

    def __init__(self, top_n: int = 3, damping: float = 0.85):
        self.top_n = top_n
        self.damping = damping
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.model = AutoModel.from_pretrained('vinai/phobert-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get PhoBERT embedding for a sentence"""
        inputs = self.tokenizer(
            sentence, 
            return_tensors='pt', 
            truncation=True, 
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0]

    def build_similarity_matrix(self, sentences: list[str]) -> np.ndarray:
        """Build similarity matrix between sentences"""
        print(f"  Computing embeddings for {len(sentences)} sentences...")
        embeddings = []

        for sent in tqdm(sentences, desc="Encoding"):
            emb = self.get_sentence_embedding(sent)
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings)

        return similarity_matrix

    def textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Run TextRank algorithm (PageRank on sentence graph)"""
        # Create graph from similarity matrix
        nx_graph = nx.from_numpy_array(similarity_matrix)

        # Compute PageRank scores
        scores = nx.pagerank(nx_graph, alpha=self.damping)

        return np.array(list(scores.values()))

    def summarize(self, document: str, num_sentences: int = None) -> str:
        """Generate extractive summary using TextRank"""
        if num_sentences is None:
            num_sentences = self.top_n

        # Split into sentences
        sentences = sent_tokenize(document)

        if len(sentences) <= num_sentences:
            return document

        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentences)

        # Run TextRank
        scores = self.textrank(similarity_matrix)

        # Select top sentences
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]

        # Sort by original order to maintain coherence
        ranked_indices = sorted(ranked_indices)

        # Extract summary
        summary_sentences = [sentences[i] for i in ranked_indices]
        summary = ' '.join(summary_sentences)

        return summary

print("✅ TextRank Summarizer created!")


# In[ ]:


# ============================================================
# 2. Load Data from CSV (FIXED VERSION)
# ============================================================
import pandas as pd
from datasets import Dataset, DatasetDict

data_path = "data"
print("📊 Loading Vietnamese Text Summarization Dataset...")

# Option 1: If CSV has headers (recommended)
train_df = pd.read_csv(f"{data_path}/train.csv")
val_df = pd.read_csv(f"{data_path}/validation.csv")
test_df = pd.read_csv(f"{data_path}/test.csv")

# Option 2: If CSV has NO headers, use names parameter
# train_df = pd.read_csv(f"{data_path}/train.csv", names=['document', 'summary', 'keywords'], header=None)
# val_df = pd.read_csv(f"{data_path}/validation.csv", names=['document', 'summary', 'keywords'], header=None)
# test_df = pd.read_csv(f"{data_path}/test.csv", names=['document', 'summary', 'keywords'], header=None)

# Keep only document and summary columns
train_df = train_df[['document', 'summary']].dropna()
val_df = val_df[['document', 'summary']].dropna()
test_df = test_df[['document', 'summary']].dropna()

print(f"✓ Train: {len(train_df):,} samples")
print(f"✓ Validation: {len(val_df):,} samples")
print(f"✓ Test: {len(test_df):,} samples")

# Convert to Dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'validation': Dataset.from_pandas(val_df, preserve_index=False),
    'test': Dataset.from_pandas(test_df, preserve_index=False)
})

print(f"\n{dataset}")

# Show actual sample
print(f"\n📝 Sample:")
sample = dataset['train'][0]
print(f"Document ({len(sample['document'])} chars): {sample['document'][:300]}...")
print(f"\nSummary ({len(sample['summary'])} chars): {sample['summary']}")

# Show statistics
print(f"\n📊 Text Length Statistics:")
train_doc_lens = [len(doc) for doc in dataset['train']['document']]
train_sum_lens = [len(summ) for summ in dataset['train']['summary']]
print(f"  Avg document length: {np.mean(train_doc_lens):.0f} chars")
print(f"  Avg summary length: {np.mean(train_sum_lens):.0f} chars")
print(f"  Compression ratio: {np.mean(train_doc_lens)/np.mean(train_sum_lens):.1f}x")


# ## 4. Load Model (mT5-Small)

# In[ ]:


get_ipython().system('pip install transformers -U')


# In[ ]:


# Load model with MEMORY OPTIMIZATION
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/mt5-small"  # MUST use small for T4

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load with FP16 from start
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# Enable gradient checkpointing (saves 30% memory)
model.gradient_checkpointing_enable()

device = torch.device("cuda")
model.to(device)

print(f"✓ Model: {model_name}")
print(f"✓ Gradient checkpointing: ON")
print(f"✓ GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")


# ## 5. Tokenize Data

# In[ ]:


def preprocess_function(examples):
    """Tokenize inputs and targets"""
    # Add prefix
    inputs = ["tóm tắt: " + doc for doc in examples["document"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False  # Dynamic padding with DataCollator
    )

    # Tokenize targets
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=128,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
print("🔄 Tokenizing dataset...")
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

# Verify tokenization
sample = tokenized_datasets["train"][0]
print("\n=== Tokenization Verification ===")
print(f"Input length: {len(sample['input_ids'])}")
print(f"Label length: {len(sample['labels'])}")
print(f"Decoded input: {tokenizer.decode(sample['input_ids'][:100])}")
print(f"Decoded label: {tokenizer.decode([l for l in sample['labels'][:50] if l != -100])}")
print("\n✓ Tokenization complete!")


# Cell 1: Check labels có đúng không
sample = tokenized_datasets["train"][0]

print("=== LABEL CHECK ===")
print(f"Labels (first 30): {sample['labels'][:30]}")
print(f"Number of -100: {sum(1 for l in sample['labels'] if l == -100)}")
print(f"Total length: {len(sample['labels'])}")
print(f"Non -100 tokens: {len([l for l in sample['labels'] if l != -100])}")

# Nếu toàn -100 → Labels bị SAI!
# Nếu có tokens bình thường → Labels đúng

# Decode non-padding tokens
valid_labels = [l for l in sample['labels'] if l != -100]
if valid_labels:
    print(f"\nDecoded labels: {tokenizer.decode(valid_labels)}")
else:
    print("❌ NO VALID LABELS! All are -100!")


# In[ ]:


print("=== DETAILED LABEL CHECK ===")
sample = tokenized_datasets["train"][0]

print(f"Input IDs (first 20): {sample['input_ids'][:20]}")
print(f"Labels (first 20): {sample['labels'][:20]}")

# Count -100
num_neg100 = sum(1 for l in sample['labels'] if l == -100)
num_valid = len(sample['labels']) - num_neg100

print(f"\nTotal labels: {len(sample['labels'])}")
print(f"Number of -100: {num_neg100}")
print(f"Valid labels: {num_valid}")
print(f"Percentage valid: {num_valid/len(sample['labels'])*100:.1f}%")

# Decode valid labels
valid_labels = [l for l in sample['labels'] if l != -100]
if valid_labels:
    decoded = tokenizer.decode(valid_labels)
    print(f"\nDecoded valid labels: {decoded}")
else:
    print("\n❌❌❌ NO VALID LABELS - ALL ARE -100! ❌❌❌")


# ## 6. Define Metrics

# In[ ]:


import evaluate
import numpy as np

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Compute ROUGE scores"""
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean text
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "rougeLsum": result["rougeLsum"],
    }

print("✓ Metrics defined")


# ## 7. Setup Training

# In[ ]:


print("=== DETAILED LABEL CHECK ===")
sample = tokenized_datasets["train"][0]

print(f"Input IDs (first 20): {sample['input_ids'][:20]}")
print(f"Labels (first 20): {sample['labels'][:20]}")

# Count -100
num_neg100 = sum(1 for l in sample['labels'] if l == -100)
num_valid = len(sample['labels']) - num_neg100

print(f"\nTotal labels: {len(sample['labels'])}")
print(f"Number of -100: {num_neg100}")
print(f"Valid labels: {num_valid}")
print(f"Percentage valid: {num_valid/len(sample['labels'])*100:.1f}%")

# Decode valid labels
valid_labels = [l for l in sample['labels'] if l != -100]
if valid_labels:
    decoded = tokenizer.decode(valid_labels)
    print(f"\nDecoded valid labels: {decoded}")
else:
    print("\n❌❌❌ NO VALID LABELS - ALL ARE -100! ❌❌❌")


# In[ ]:


from torch.utils.data import DataLoader

print("\n=== BATCH CHECK ===")
test_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=2,
    collate_fn=data_collator
)

batch = next(iter(test_loader))

print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Labels shape: {batch['labels'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")

print(f"\nLabels[0] first 30: {batch['labels'][0][:30]}")
print(f"Num -100 in batch[0]: {(batch['labels'][0] == -100).sum().item()}")
print(f"Num valid in batch[0]: {(batch['labels'][0] != -100).sum().item()}")

# Check if labels are all padding
if (batch['labels'] != -100).sum().item() == 0:
    print("\n❌❌❌ ENTIRE BATCH LABELS ARE -100! ❌❌❌")
else:
    print(f"\n✓ Batch has {(batch['labels'] != -100).sum().item()} valid label tokens")


# In[ ]:


print("\n=== MANUAL FORWARD PASS ===")
model.eval()

# Move batch to cuda if not already
batch_cuda = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.no_grad():
    outputs = model(**batch_cuda)

print(f"Loss: {outputs.loss.item()}")
print(f"Logits shape: {outputs.logits.shape}")
print(f"Loss is nan: {torch.isnan(outputs.loss).item()}")
print(f"Loss is zero: {outputs.loss.item() < 0.001}")

if outputs.loss.item() < 0.001:
    print("\n❌ LOSS TOO LOW - SOMETHING WRONG!")
elif torch.isnan(outputs.loss).item():
    print("\n❌ LOSS IS NAN - LABELS ISSUE!")
else:
    print(f"\n✓ Loss looks normal: {outputs.loss.item():.4f}")


# In[ ]:


print("\n=== PREPROCESSING CHECK ===")

# Get ONE example manually
raw_example = {
    'document': [train_df.iloc[0]['document']],
    'summary': [train_df.iloc[0]['summary']]
}

print(f"Raw document (first 100): {raw_example['document'][0][:100]}")
print(f"Raw summary (first 100): {raw_example['summary'][0][:100]}")

# Apply preprocessing
processed = preprocess_function(raw_example)

print(f"\nProcessed input_ids length: {len(processed['input_ids'][0])}")
print(f"Processed labels length: {len(processed['labels'][0])}")

print(f"\nInput_ids (first 20): {processed['input_ids'][0][:20]}")
print(f"Labels (first 20): {processed['labels'][0][:20]}")

# Decode
decoded_input = tokenizer.decode(processed['input_ids'][0][:100])
decoded_label = tokenizer.decode([l for l in processed['labels'][0] if l != -100][:50])

print(f"\nDecoded input: {decoded_input}")
print(f"Decoded label: {decoded_label}")


# In[ ]:


from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model, 
    label_pad_token_id=-100
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5-small-vn-v2",

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    learning_rate=1e-5,  # GIẢM từ 3e-5 → 1e-5
    num_train_epochs=3,
    warmup_steps=1000,   # TĂNG warmup
    weight_decay=0.01,

    eval_strategy="steps",
    eval_steps=500,      # Eval sớm hơn để phát hiện lỗi

    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,

    fp16=True,
    gradient_checkpointing=True,

    logging_steps=50,    # Log nhiều hơn
    save_steps=500,
    save_total_limit=1,

    load_best_model_at_end=False,
    metric_for_best_model="rouge1",
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

print("✓ Trainer ready")

# CRITICAL: Test generation TRƯỚC
print("=== Testing generation BEFORE training ===")
test_text = "Chiều 26/1, UBND TP Hà Nội tổ chức họp báo"
inputs = tokenizer("tóm tắt: " + test_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {test_text}")
print(f"Generated: '{generated}'")
print(f"Length: {len(generated)}")

# Nếu generated là garbage → vẫn có vấn đề
# Nếu generated có nghĩa (ngay cả khi không chính xác) → OK


# ## 8. Train Model 🚀

# In[ ]:


print("🚀 Starting training...")
print("This will take ~1-1.5 hours on RTX 4070 SUPER")
print("="*50)

trainer.train()

print("\n" + "="*50)
print("✅ Training complete!")


# ## 9. Evaluate on Test Set

# In[ ]:


print("📊 Evaluating on test set...")
results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
for key, value in results.items():
    if 'rouge' in key:
        print(f"{key.upper()}: {value:.4f}")


# ## 10. Test Inference

# In[ ]:


def generate_summary(text, max_length=128, num_beams=4):
    """Generate summary for input text"""
    inputs = tokenizer(
        "tóm tắt: " + text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with examples
print("\n=== INFERENCE EXAMPLES ===")
for i in range(3):
    test_text = dataset['test'][i]['document']
    ground_truth = dataset['test'][i]['summary']

    print(f"\n--- Example {i+1} ---")
    print(f"Original ({len(test_text)} chars):")
    print(test_text[:200], "...\n")

    print("Generated Summary:")
    generated = generate_summary(test_text)
    print(generated)

    print("\nGround Truth:")
    print(ground_truth)
    print("\n" + "="*50)


# ## 11. Save Model

# In[ ]:


output_dir = "./mt5-small-vietnamese-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")
print(f"\nTo load later:")
print(f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')
print(f'model = AutoModelForSeq2SeqLM.from_pretrained("{output_dir}")')


# ## 12. Quick Test with New Text

# In[ ]:


# Test with your own text
custom_text = """
Chiều 26/1, UBND TP Hà Nội tổ chức họp báo công bố kết quả thực hiện 
nhiệm vụ phát triển kinh tế - xã hội năm 2024. Theo đó, tổng sản phẩm 
trên địa bàn (GRDP) của Hà Nội năm 2024 ước tăng 7,5% so với năm 2023, 
cao hơn mức tăng trưởng chung của cả nước (7,09%).
"""

print("Original text:")
print(custom_text)
print("\nGenerated summary:")
print(generate_summary(custom_text.strip()))

