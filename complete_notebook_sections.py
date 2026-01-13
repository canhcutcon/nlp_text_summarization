#!/usr/bin/env python3
"""
Complete the Vietnamese Text Summarization notebook by adding Sections 4-7
"""

import json

# Read existing notebook
with open('vietnamese_summarization_mt5_rtx_4070.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Starting with {len(nb['cells'])} existing cells")

# Helper functions
def mk(lines):
    """Create markdown cell"""
    if isinstance(lines, str):
        lines = lines.strip().split('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in lines]
    }

def code(lines):
    """Create code cell"""
    if isinstance(lines, str):
        lines = lines.strip().split('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in lines]
    }

# ============================================================================
# SECTION 4: ABSTRACTIVE SUMMARIZATION
# ============================================================================

section4 = [
    mk("""---

# 4. Abstractive Summarization

## 4.1 Sequence-to-Sequence Models - Theory

### Nguyên lý / Principle:

**Abstractive Summarization** sử dụng mô hình **Sequence-to-Sequence (Seq2Seq)** để tạo ra bản tóm tắt mới thay vì chọn câu có sẵn.

### Kiến trúc Encoder-Decoder:

**Encoder:**
- Nhận văn bản đầu vào
- Chuyển thành vector biểu diễn (context vector)
- Capture toàn bộ ngữ nghĩa của văn bản

**Decoder:**
- Nhận context vector từ encoder
- Sinh ra từng từ một của bản tóm tắt
- Sử dụng thông tin từ encoder và từ đã sinh trước đó

### Attention Mechanism:

**Vấn đề / Problem:**
- Context vector cố định không thể chứa toàn bộ thông tin
- Với văn bản dài, thông tin bị mất

**Giải pháp / Solution:**
- **Attention** cho phép decoder "tập trung" vào phần khác nhau của input
- Tại mỗi bước sinh từ, decoder xem lại toàn bộ encoder outputs
- Tính trọng số quan trọng cho từng vị trí trong input

### Generation Strategies:

#### 1. Greedy Decoding
- Chọn từ có xác suất cao nhất tại mỗi bước
- **Ưu điểm**: Nhanh
- **Nhược điểm**: Không tối ưu toàn cục

#### 2. Beam Search
- Giữ K candidates tốt nhất (beam width = K)
- **Ưu điểm**: Chất lượng cao hơn greedy
- **Nhược điểm**: Chậm hơn (K lần)

#### 3. Top-k Sampling
- Chọn ngẫu nhiên từ K từ có xác suất cao nhất
- **Ưu điểm**: Đa dạng hơn
- **Nhược điểm**: Có thể không ổn định

#### 4. Top-p (Nucleus) Sampling
- Chọn ngẫu nhiên từ các từ có tổng xác suất >= p
- **Ưu điểm**: Cân bằng giữa đa dạng và chất lượng
- **Nhược điểm**: Cần điều chỉnh p cẩn thận"""),

    mk("## 4.2 Load mT5-small Model (Inference Only)"),

    code("""# Clear GPU memory
import gc
torch.cuda.empty_cache()
gc.collect()

print("="*60)
print("LOADING MT5-SMALL MODEL")
print("="*60)

# Load mT5-small model
print("\\nLoading mT5-small model from HuggingFace...")
mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/mt5-small",
    torch_dtype=torch.float16
)

mt5_model.to(device)
mt5_model.eval()

print(f"✓ mT5-small loaded on {device}")
print(f"✓ Model size: ~300M parameters")
if torch.cuda.is_available():
    print(f"✓ GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")"""),

    mk("## 4.3 Load ViT5 Model from vit5_final/"),

    code("""print("="*60)
print("LOADING VIT5 MODEL")
print("="*60)

# Load ViT5 model from local directory
print("\\nLoading ViT5 model from vit5_final/...")
vit5_tokenizer = AutoTokenizer.from_pretrained("./vit5_final")
vit5_model = AutoModelForSeq2SeqLM.from_pretrained(
    "./vit5_final",
    torch_dtype=torch.float16
)

vit5_model.to(device)
vit5_model.eval()

print(f"✓ ViT5 loaded on {device}")
print(f"✓ Model: Vietnamese-specific T5")
if torch.cuda.is_available():
    print(f"✓ GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

print(f"\\n✅ Both models loaded successfully!")"""),

    mk("## 4.4 Inference Functions"),

    code("""def generate_summary_mt5(text, max_length=128, min_length=30, num_beams=4, strategy="beam_search"):
    input_text = f"tóm tắt: {text}"
    inputs = mt5_tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        if strategy == "beam_search":
            outputs = mt5_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        elif strategy == "sampling":
            outputs = mt5_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7
            )
        elif strategy == "top_k":
            outputs = mt5_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_k=50,
                temperature=0.8
            )
        elif strategy == "top_p":
            outputs = mt5_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
        else:
            outputs = mt5_model.generate(**inputs, max_length=max_length, min_length=min_length)

    return mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_summary_vit5(text, max_length=256, min_length=50, num_beams=4):
    input_text = f"tóm tắt: {text}"
    inputs = vit5_tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = vit5_model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return vit5_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("✅ Inference functions defined!")"""),

    mk("## 4.5 Test Abstractive Summarization"),

    code("""# Test both models on examples
print("="*60)
print("ABSTRACTIVE SUMMARIZATION EXAMPLES")
print("="*60)

num_examples = 3

for i in range(num_examples):
    test_doc = dataset['test'][i]['document']
    test_ref = dataset['test'][i]['summary']

    print(f"\\n{'='*60}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*60}")

    print(f"\\n📄 Original Document ({len(test_doc.split())} words):")
    print(test_doc[:300] + "...")

    print(f"\\n🤖 mT5-small Summary:")
    mt5_summary = generate_summary_mt5(test_doc)
    print(mt5_summary)

    print(f"\\n🤖 ViT5 Summary:")
    vit5_summary = generate_summary_vit5(test_doc)
    print(vit5_summary)

    print(f"\\n📝 Reference Summary:")
    print(test_ref)

    print(f"\\n📊 Statistics:")
    print(f"  Original: {len(test_doc.split())} words")
    print(f"  mT5: {len(mt5_summary.split())} words")
    print(f"  ViT5: {len(vit5_summary.split())} words")
    print(f"  Reference: {len(test_ref.split())} words")

print("\\n✅ Abstractive summarization demo complete!")"""),

    mk("## 4.6 Generation Strategy Comparison"),

    code("""# Compare different generation strategies
print("="*60)
print("GENERATION STRATEGY COMPARISON")
print("="*60)

test_text = dataset['test'][0]['document']

print(f"\\nTest Document ({len(test_text.split())} words):")
print(test_text[:200] + "...\\n")

strategies = ["beam_search", "sampling", "top_k", "top_p"]

print("\\nComparing generation strategies with mT5-small:\\n")

for strategy in strategies:
    summary = generate_summary_mt5(test_text, strategy=strategy)
    print(f"{'─'*60}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'─'*60}")
    print(summary)
    print(f"Length: {len(summary.split())} words\\n")

print("✅ Strategy comparison complete!")"""),
]

# ============================================================================
# SECTION 5: EVALUATION & COMPARISON
# ============================================================================

section5 = [
    mk("""---

# 5. Evaluation & Comparison

## 5.1 ROUGE Metrics Implementation"""),

    code("""def compute_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores[metric]['precision'].append(result[metric].precision)
            scores[metric]['recall'].append(result[metric].recall)
            scores[metric]['fmeasure'].append(result[metric].fmeasure)

    return scores

print("✅ ROUGE computation function defined!")"""),

    mk("## 5.2 Generate Predictions on Test Set"),

    code("""print("="*60)
print("GENERATING PREDICTIONS ON TEST SET")
print("="*60)

# Use subset for faster execution (adjust as needed)
sample_size = 500
print(f"\\nUsing {sample_size} samples from test set")
print("This will take approximately 10-15 minutes...")

test_docs_sample = dataset['test']['document'][:sample_size]
test_refs_sample = dataset['test']['summary'][:sample_size]

# Initialize lists
mt5_predictions = []
vit5_predictions = []
extractive_predictions = []

# Generate predictions with progress bar
print("\\nGenerating predictions...")

for i, doc in enumerate(tqdm(test_docs_sample, desc="Processing")):
    # mT5 predictions
    mt5_pred = generate_summary_mt5(doc)
    mt5_predictions.append(mt5_pred)

    # ViT5 predictions
    vit5_pred = generate_summary_vit5(doc)
    vit5_predictions.append(vit5_pred)

    # Extractive predictions
    extractive_pred = textrank.summarize(doc, num_sentences=3)
    extractive_predictions.append(extractive_pred)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{sample_size} samples...")

print(f"\\n✅ All {sample_size} predictions generated!")"""),

    mk("## 5.3 Compute ROUGE Scores"),

    code("""print("="*60)
print("COMPUTING ROUGE SCORES")
print("="*60)

# Compute ROUGE scores for all models
mt5_scores = compute_rouge_scores(mt5_predictions, test_refs_sample)
vit5_scores = compute_rouge_scores(vit5_predictions, test_refs_sample)
extractive_scores = compute_rouge_scores(extractive_predictions, test_refs_sample)

# Create models dictionary
models = {
    'mT5-small': mt5_scores,
    'ViT5': vit5_scores,
    'TextRank (Extractive)': extractive_scores
}

# Print results
print("\\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

for model_name, scores in models.items():
    print(f"\\n{model_name}:")
    print(f"  ROUGE-1 F1: {np.mean(scores['rouge1']['fmeasure']):.4f}")
    print(f"  ROUGE-2 F1: {np.mean(scores['rouge2']['fmeasure']):.4f}")
    print(f"  ROUGE-L F1: {np.mean(scores['rougeL']['fmeasure']):.4f}")

print("\\n✅ ROUGE evaluation complete!")"""),

    mk("## 5.4 Detailed Comparison Table"),

    code("""# Create detailed comparison table
comparison_data = []

for model_name, scores in models.items():
    comparison_data.append({
        'Model': model_name,
        'ROUGE-1': f"{np.mean(scores['rouge1']['fmeasure']):.4f} ± {np.std(scores['rouge1']['fmeasure']):.4f}",
        'ROUGE-2': f"{np.mean(scores['rouge2']['fmeasure']):.4f} ± {np.std(scores['rouge2']['fmeasure']):.4f}",
        'ROUGE-L': f"{np.mean(scores['rougeL']['fmeasure']):.4f} ± {np.std(scores['rougeL']['fmeasure']):.4f}",
        'Avg': np.mean([
            np.mean(scores['rouge1']['fmeasure']),
            np.mean(scores['rouge2']['fmeasure']),
            np.mean(scores['rougeL']['fmeasure'])
        ])
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Avg', ascending=False)

print("\\n" + "="*80)
print("DETAILED MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("\\n✅ Comparison table created!")"""),

    mk("## 5.5 Side-by-Side Examples"),

    code("""# Show side-by-side examples
print("="*60)
print("SIDE-BY-SIDE COMPARISON EXAMPLES")
print("="*60)

num_examples = 5

for i in range(num_examples):
    print(f"\\n{'='*80}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*80}")

    print(f"\\n📄 Original Document ({len(test_docs_sample[i].split())} words):")
    print(test_docs_sample[i][:200] + "...")

    print(f"\\n🤖 mT5-small:")
    print(mt5_predictions[i])

    print(f"\\n🤖 ViT5:")
    print(vit5_predictions[i])

    print(f"\\n🤖 TextRank (Extractive):")
    print(extractive_predictions[i][:200] if len(extractive_predictions[i]) > 200 else extractive_predictions[i])

    print(f"\\n📝 Reference:")
    print(test_refs_sample[i])

print("\\n✅ Side-by-side comparison complete!")"""),
]

# Add sections to notebook
nb['cells'].extend(section4)
nb['cells'].extend(section5)

print(f"Added Section 4: {len(section4)} cells")
print(f"Added Section 5: {len(section5)} cells")
print(f"Total cells now: {len(nb['cells'])}")

# Save progress
with open('vietnamese_summarization_mt5_rtx_4070.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\\n✅ Sections 4-5 added successfully!")
print("Next: Adding Sections 6-7...")
