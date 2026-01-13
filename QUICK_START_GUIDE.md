# Quick Start Guide - Vietnamese Text Summarization Notebook

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Core dependencies (protobuf required for T5 tokenizers)
pip install protobuf sentencepiece

# Install all required packages
pip install transformers datasets torch
pip install rouge-score py-rouge evaluate scikit-learn
pip install underthesea networkx matplotlib seaborn pandas tqdm
```

### Step 2: Verify Data Files
Ensure these files exist:
- `data/train.csv`
- `data/validation.csv`
- `data/test.csv`

Note: Models (mT5-small and ViT5) will be downloaded automatically from HuggingFace

### Step 3: Run the Notebook
```bash
jupyter notebook vietnamese_summarization_mt5_rtx_4070.ipynb
```
Or use JupyterLab, VS Code, or Google Colab.

---

## 📊 Notebook Structure (69 Cells)

### Section 1: Theory (4 markdown + 0 code)
- What is text summarization
- Extractive vs Abstractive
- T5, mT5, ViT5 architectures
- ROUGE metrics explanation

### Section 2: Setup & Data (4 markdown + 5 code)
- Install packages
- Load CSV data
- Data exploration
- Statistics and visualizations

### Section 3: Extractive Summarization (4 markdown + 3 code)
- TextRank theory
- PhoBERT + PageRank implementation
- Test examples

### Section 4: Abstractive Summarization (5 markdown + 6 code)
- Seq2Seq theory
- Load mT5-small and ViT5
- Generation strategies
- Test examples

### Section 5: Evaluation (5 markdown + 5 code)
- ROUGE implementation
- Generate 500 predictions
- Compare all 3 models
- Side-by-side examples

### Section 6: Visualizations (8 markdown + 8 code)
- Chart 1: ROUGE-1 distribution
- Chart 2: ROUGE-2 distribution
- Chart 3: ROUGE-L distribution
- Chart 4: Box plots
- Chart 5: Document length vs ROUGE-L
- Chart 6: Prediction vs reference length
- Chart 7: Performance by category
- Chart 8: Heatmap + table + ranking

### Section 7: Applications (6 markdown + 6 code)
- News summarization
- Long documents
- Multiple lengths
- Batch processing
- Quality comparison
- Conclusion

---

## ⏱️ Execution Time

| Section | Time Estimate |
|---------|---------------|
| 1. Theory | 0 min (read only) |
| 2. Setup & Data | 2-3 min |
| 3. Extractive | 5-7 min |
| 4. Abstractive | 3-5 min |
| 5. Evaluation | 10-15 min |
| 6. Visualizations | 1-2 min |
| 7. Applications | 3-5 min |
| **TOTAL** | **~25-35 min** |

---

## 💾 GPU Memory Usage

| Component | Memory |
|-----------|--------|
| PhoBERT (TextRank) | ~1.5 GB |
| mT5-small (FP16) | ~2.5 GB |
| ViT5 (FP16) | ~2.5 GB |
| Data + Activations | ~2 GB |
| **Total Peak** | **~8-9 GB** |

✅ Safe for RTX 4070 SUPER (12GB)
✅ Works on RTX 3080 (10GB)
⚠️ May struggle on RTX 3060 (8GB) - reduce batch size

---

## 🎯 Key Features

### Models
- **mT5-small**: HuggingFace `google/mt5-small` (300M params)
- **ViT5**: HuggingFace `YangYang0203/vi5_summarize` (Vietnamese T5)
- **TextRank**: PhoBERT embeddings + PageRank

### Evaluation
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L
- **Sample Size**: 500 test documents (adjustable)
- **Comparison**: All 3 models side-by-side

### Visualizations
- **8 Charts**: Histograms, box plots, scatter plots, heatmap
- **Professional**: Seaborn styling, proper labels
- **Informative**: Correlations, statistics, rankings

---

## 🔧 Common Adjustments

### Reduce Execution Time
```python
# In Section 5.2, change:
sample_size = 500  # → change to 100 or 200
```

### Generate Longer Summaries
```python
# In Section 4, change:
generate_summary_mt5(text, max_length=128)  # → 200
generate_summary_vit5(text, max_length=256)  # → 400
```

### Extract More Sentences (TextRank)
```python
# In Section 3.3, change:
textrank = TextRankSummarizer(top_n=3)  # → 5 or 7
```

---

## 📈 Expected Results

### ROUGE Scores (on VLSP 2021 dataset)
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| mT5-small | 0.30-0.35 | 0.15-0.20 | 0.28-0.33 |
| ViT5 | 0.35-0.40 | 0.18-0.23 | 0.32-0.37 |
| TextRank | 0.25-0.30 | 0.10-0.15 | 0.23-0.28 |

### Summary Characteristics
- **mT5-small**: Fast, good quality, multilingual
- **ViT5**: Best Vietnamese quality, most natural
- **TextRank**: Fastest, factual, less fluent

---

## 🐛 Troubleshooting

### Error: "No such file or directory: data/train.csv"
**Solution**: Ensure CSV files are in `data/` directory

### Error: "CUDA out of memory"
**Solution**:
1. Reduce `sample_size` to 100
2. Restart notebook kernel
3. Close other GPU applications

### Error: "Connection error downloading ViT5"
**Solution**: Check internet connection - ViT5 downloads from HuggingFace

### Slow Execution
**Solution**:
1. Use GPU (10-20x faster than CPU)
2. Reduce `sample_size` in Section 5
3. Use fewer TextRank sentences

---

## 📚 What You'll Learn

1. **Theory**: Text summarization approaches and models
2. **Implementation**: TextRank, Transformers, ROUGE
3. **Evaluation**: How to assess summarization quality
4. **Visualization**: Data analysis and presentation
5. **Applications**: Real-world use cases

---

## 🎓 Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of NLP concepts (helpful but not required)
- Familiarity with Jupyter notebooks

### Not Required
- Deep learning expertise
- Vietnamese language proficiency
- Prior experience with transformers

---

## 📞 Next Steps After Completion

1. **Experiment**: Try with your own Vietnamese documents
2. **Customize**: Adjust parameters for your use case
3. **Fine-tune**: Train models on domain-specific data
4. **Deploy**: Create API or web interface
5. **Extend**: Add more models or metrics

---

## 📖 Additional Resources

### HuggingFace Models
- mT5: https://huggingface.co/google/mt5-small
- PhoBERT: https://huggingface.co/vinai/phobert-base

### Documentation
- Transformers: https://huggingface.co/docs/transformers
- ROUGE: https://github.com/google-research/google-research/tree/master/rouge

### Papers
- T5: "Exploring the Limits of Transfer Learning" (Raffel et al., 2019)
- TextRank: "TextRank: Bringing Order into Text" (Mihalcea & Tarau, 2004)

---

## ✅ Checklist Before Running

- [ ] Python 3.8+ installed
- [ ] All packages installed (`pip install ...`)
- [ ] Data files in `data/` directory
- [ ] Internet connection (for downloading models from HuggingFace)
- [ ] GPU available (recommended)
- [ ] At least 30 minutes available for full execution

---

**Ready? Let's begin! 🚀**

Open the notebook and run cells sequentially. The notebook is self-contained with explanations in both Vietnamese and English.

**Happy Summarizing! 🎉**
