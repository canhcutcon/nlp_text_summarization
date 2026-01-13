#!/usr/bin/env python3
"""
Build complete Vietnamese Text Summarization notebook with all 7 sections
"""

import json

# Read existing notebook (has sections 1-3)
with open('vietnamese_summarization_mt5_rtx_4070.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Starting with {len(nb['cells'])} existing cells")

# Helper function to create cells
def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

# Section 4: Abstractive Summarization
section4_cells = [
    markdown_cell("""## 4.2 Load mT5-small Model (Inference Only)"""),

    code_cell("""# Clear GPU memory
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
print(f"✓ GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "")"""),

    markdown_cell("""## 4.3 Load ViT5 Model from vit5_final/"""),

    code_cell("""print("="*60)
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
print(f"✓ GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
print(f"\\n✅ Both models loaded successfully!")"""),

    markdown_cell("""## 4.4 Inference Functions"""),

    code_cell("""def generate_summary_mt5(text, max_length=128, min_length=30,
                         num_beams=4, strategy="beam_search"):
    \\"""
    Generate summary with mT5-small

    Args:
        text (str): Input document
        max_length (int): Maximum length of summary
        min_length (int): Minimum length of summary
        num_beams (int): Number of beams for beam search
        strategy (str): Generation strategy ("beam_search", "sampling", "top_k", "top_p")

    Returns:
        str: Generated summary
    \\"""
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
            # Greedy decoding
            outputs = mt5_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length
            )

    return mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_summary_vit5(text, max_length=256, min_length=50, num_beams=4):
    \\"""
    Generate summary with ViT5

    Args:
        text (str): Input document
        max_length (int): Maximum length of summary
        min_length (int): Minimum length of summary
        num_beams (int): Number of beams for beam search

    Returns:
        str: Generated summary
    \\"""
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

    markdown_cell("""## 4.5 Test Abstractive Summarization"""),

    code_cell("""# Test both models on examples
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

    # Statistics
    print(f"\\n📊 Statistics:")
    print(f"  Original: {len(test_doc.split())} words")
    print(f"  mT5: {len(mt5_summary.split())} words")
    print(f"  ViT5: {len(vit5_summary.split())} words")
    print(f"  Reference: {len(test_ref.split())} words")

print("\\n✅ Abstractive summarization demo complete!")"""),

    markdown_cell("""## 4.6 Generation Strategy Comparison"""),

    code_cell("""# Compare different generation strategies
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

# Add Section 4 cells
nb['cells'].extend(section4_cells)

print(f"Added Section 4: {len(section4_cells)} cells")
print(f"Total cells now: {len(nb['cells'])}")

# Continue with remaining sections in next parts...
# Save progress
with open('vietnamese_summarization_mt5_rtx_4070.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Section 4 added successfully!")
print("\\nNext: Will add Sections 5-7 in continuation...")
