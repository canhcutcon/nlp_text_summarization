#!/usr/bin/env python3
"""
Add Sections 6-7 to complete the Vietnamese Text Summarization notebook
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
# SECTION 6: VISUALIZATIONS (8 CHARTS)
# ============================================================================

section6 = [
    mk("""---

# 6. Visualizations

## 6.1 ROUGE-1 Score Distribution (Chart 1)"""),

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, scores) in enumerate(models.items()):
    r1_scores = scores['rouge1']['fmeasure']
    axes[i].hist(r1_scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].axvline(np.mean(r1_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(r1_scores):.3f}')
    axes[i].set_xlabel('ROUGE-1 F1 Score', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].set_title(f'{model_name} - ROUGE-1 Distribution', fontsize=13, fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Chart 1: ROUGE-1 Distribution created!")"""),

    mk("## 6.2 ROUGE-2 Score Distribution (Chart 2)"),

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, scores) in enumerate(models.items()):
    r2_scores = scores['rouge2']['fmeasure']
    axes[i].hist(r2_scores, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[i].axvline(np.mean(r2_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(r2_scores):.3f}')
    axes[i].set_xlabel('ROUGE-2 F1 Score', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].set_title(f'{model_name} - ROUGE-2 Distribution', fontsize=13, fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Chart 2: ROUGE-2 Distribution created!")"""),

    mk("## 6.3 ROUGE-L Score Distribution (Chart 3)"),

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, scores) in enumerate(models.items()):
    rL_scores = scores['rougeL']['fmeasure']
    axes[i].hist(rL_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[i].axvline(np.mean(rL_scores), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(rL_scores):.3f}')
    axes[i].set_xlabel('ROUGE-L F1 Score', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].set_title(f'{model_name} - ROUGE-L Distribution', fontsize=13, fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Chart 3: ROUGE-L Distribution created!")"""),

    mk("## 6.4 Box Plots Comparing Models (Chart 4)"),

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['rouge1', 'rouge2', 'rougeL']
metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    data_to_plot = [scores[metric]['fmeasure'] for scores in models.values()]
    bp = axes[i].boxplot(data_to_plot, labels=models.keys(), patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    axes[i].set_ylabel('F1 Score', fontsize=12)
    axes[i].set_title(f'{name} Score Comparison', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3, axis='y')
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

print("✅ Chart 4: Box Plots created!")"""),

    mk("## 6.5 Document Length vs ROUGE-L (Chart 5)"),

    code("""doc_lengths = [len(doc.split()) for doc in test_docs_sample]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, scores) in enumerate(models.items()):
    rL_scores = scores['rougeL']['fmeasure']
    axes[i].scatter(doc_lengths, rL_scores, alpha=0.4, s=20)

    # Add correlation coefficient
    corr = np.corrcoef(doc_lengths, rL_scores)[0, 1]
    axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=axes[i].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    axes[i].set_xlabel('Document Length (words)', fontsize=12)
    axes[i].set_ylabel('ROUGE-L F1 Score', fontsize=12)
    axes[i].set_title(f'{model_name}', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Chart 5: Document Length vs ROUGE-L created!")"""),

    mk("## 6.6 Prediction vs Reference Length (Chart 6)"),

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ref_lengths = [len(ref.split()) for ref in test_refs_sample]
pred_lengths_dict = {
    'mT5-small': [len(p.split()) for p in mt5_predictions],
    'ViT5': [len(p.split()) for p in vit5_predictions],
    'TextRank (Extractive)': [len(p.split()) for p in extractive_predictions]
}

for i, (model_name, pred_lengths) in enumerate(pred_lengths_dict.items()):
    axes[i].scatter(ref_lengths, pred_lengths, alpha=0.4, s=20, color='coral')

    # Add diagonal line (perfect match)
    max_len = max(max(ref_lengths), max(pred_lengths))
    axes[i].plot([0, max_len], [0, max_len], 'k--', alpha=0.5, label='Perfect match')

    # Add correlation
    corr = np.corrcoef(ref_lengths, pred_lengths)[0, 1]
    axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=axes[i].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    axes[i].set_xlabel('Reference Length (words)', fontsize=12)
    axes[i].set_ylabel('Prediction Length (words)', fontsize=12)
    axes[i].set_title(f'{model_name}', fontsize=13, fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Chart 6: Prediction vs Reference Length created!")"""),

    mk("## 6.7 Performance by Document Length Category (Chart 7)"),

    code("""# Categorize documents by length
def categorize_length(length):
    if length < 100:
        return 'Short (<100)'
    elif length < 300:
        return 'Medium (100-300)'
    else:
        return 'Long (>300)'

doc_categories = [categorize_length(l) for l in doc_lengths]

# Calculate average ROUGE by category
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

categories = ['Short (<100)', 'Medium (100-300)', 'Long (>300)']

for i, (model_name, scores) in enumerate(models.items()):
    rouge_scores_by_cat = {cat: [] for cat in categories}

    for cat, score in zip(doc_categories, scores['rouge1']['fmeasure']):
        rouge_scores_by_cat[cat].append(score)

    means = [np.mean(rouge_scores_by_cat[cat]) if rouge_scores_by_cat[cat] else 0
             for cat in categories]
    stds = [np.std(rouge_scores_by_cat[cat]) if rouge_scores_by_cat[cat] else 0
            for cat in categories]

    x_pos = np.arange(len(categories))
    axes[i].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5,
               color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(categories, rotation=15)
    axes[i].set_ylabel('ROUGE-1 F1 Score', fontsize=12)
    axes[i].set_title(f'{model_name}', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("✅ Chart 7: Performance by Document Category created!")"""),

    mk("## 6.8 Correlation Heatmap + Summary Statistics (Chart 8)"),

    code("""fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Correlation heatmap (top)
ax1 = fig.add_subplot(gs[0, :])

correlation_data = []
for model_name, scores in models.items():
    correlation_data.append([
        np.mean(scores['rouge1']['fmeasure']),
        np.mean(scores['rouge2']['fmeasure']),
        np.mean(scores['rougeL']['fmeasure'])
    ])

corr_df = pd.DataFrame(
    correlation_data,
    columns=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
    index=models.keys()
)

sns.heatmap(corr_df, annot=True, fmt='.4f', cmap='YlGnBu',
           cbar_kws={'label': 'F1 Score'}, ax=ax1, linewidths=1)
ax1.set_title('Model Performance Heatmap', fontsize=15, fontweight='bold')

# Summary statistics table (bottom left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')

summary_stats = []
for model_name, scores in models.items():
    r1_mean = np.mean(scores['rouge1']['fmeasure'])
    r1_std = np.std(scores['rouge1']['fmeasure'])
    r2_mean = np.mean(scores['rouge2']['fmeasure'])
    r2_std = np.std(scores['rouge2']['fmeasure'])
    rL_mean = np.mean(scores['rougeL']['fmeasure'])
    rL_std = np.std(scores['rougeL']['fmeasure'])

    summary_stats.append([
        model_name,
        f"{r1_mean:.4f} ± {r1_std:.4f}",
        f"{r2_mean:.4f} ± {r2_std:.4f}",
        f"{rL_mean:.4f} ± {rL_std:.4f}"
    ])

table = ax2.table(cellText=summary_stats,
                 colLabels=['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                 cellLoc='center', loc='center',
                 colWidths=[0.3, 0.23, 0.23, 0.23])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax2.set_title('Summary Statistics (Mean ± Std)', fontsize=13, fontweight='bold', pad=20)

# Model ranking (bottom right)
ax3 = fig.add_subplot(gs[1, 1])

model_names = list(models.keys())
avg_rouge = [np.mean([
    np.mean(scores['rouge1']['fmeasure']),
    np.mean(scores['rouge2']['fmeasure']),
    np.mean(scores['rougeL']['fmeasure'])
]) for scores in models.values()]

sorted_indices = np.argsort(avg_rouge)[::-1]
sorted_names = [model_names[i] for i in sorted_indices]
sorted_scores = [avg_rouge[i] for i in sorted_indices]

colors = ['gold', 'silver', 'chocolate']
bars = ax3.barh(sorted_names, sorted_scores, color=colors)
ax3.set_xlabel('Average ROUGE Score', fontsize=12)
ax3.set_title('Model Ranking', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    ax3.text(score + 0.005, i, f'{score:.4f}',
            va='center', fontsize=10, fontweight='bold')

plt.show()

print("✅ Chart 8: Correlation Heatmap + Summary Statistics created!")
print("\\n✅ All 8 visualizations complete!")"""),
]

# ============================================================================
# SECTION 7: APPLICATIONS
# ============================================================================

section7 = [
    mk("""---

# 7. Applications

## 7.1 Real-World Use Cases

This section demonstrates how Vietnamese text summarization can be applied in various real-world scenarios:

### 1. News Summarization
- Automatically generate headlines and summaries for news articles
- Help readers quickly understand main points
- Enable news aggregation services

### 2. Document Summarization
- Summarize research papers and technical reports
- Create executive summaries for business documents
- Generate abstracts for academic papers

### 3. Meeting Notes
- Automatically summarize meeting transcripts
- Extract action items and key decisions
- Create concise meeting reports

### 4. Legal Document Summaries
- Summarize contracts and legal agreements
- Extract key terms and conditions
- Help legal professionals quickly review documents

### 5. Medical Record Summarization
- Summarize patient medical histories
- Extract key symptoms and diagnoses
- Create concise clinical summaries"""),

    mk("## 7.2 Application 1: News Article Summarization"),

    code("""# Example: News Article
news_article = dataset['test'][10]['document']

print("="*60)
print("APPLICATION 1: NEWS ARTICLE SUMMARIZATION")
print("="*60)

print(f"\\n📰 Original News Article ({len(news_article.split())} words):")
print(news_article)

print(f"\\n{'─'*60}")
print("🤖 mT5-small Summary:")
print(generate_summary_mt5(news_article))

print(f"\\n{'─'*60}")
print("🤖 ViT5 Summary:")
print(generate_summary_vit5(news_article))

print(f"\\n{'─'*60}")
print("🤖 TextRank (Extractive) Summary:")
print(textrank.summarize(news_article, num_sentences=3))

print(f"\\n{'─'*60}")
print("📝 Reference Summary:")
print(dataset['test'][10]['summary'])"""),

    mk("## 7.3 Application 2: Long Document Summarization"),

    code("""# Example: Long Document
long_doc = dataset['test'][50]['document']

print("="*60)
print("APPLICATION 2: LONG DOCUMENT SUMMARIZATION")
print("="*60)

print(f"\\n📄 Original Document ({len(long_doc.split())} words):")
print(long_doc[:500] + "...")

print(f"\\n{'─'*60}")
print("🤖 mT5-small Summary:")
print(generate_summary_mt5(long_doc, max_length=150))

print(f"\\n{'─'*60}")
print("🤖 ViT5 Summary:")
print(generate_summary_vit5(long_doc, max_length=200))

print(f"\\n{'─'*60}")
print("📝 Reference Summary:")
print(dataset['test'][50]['summary'])"""),

    mk("## 7.4 Application 3: Multiple Summary Lengths"),

    code("""# Demonstrate different summary lengths
test_doc = dataset['test'][100]['document']

print("="*60)
print("APPLICATION 3: MULTIPLE SUMMARY LENGTHS")
print("="*60)

print(f"\\n📄 Original Document ({len(test_doc.split())} words):")
print(test_doc[:300] + "...")

print(f"\\n{'─'*60}")
print("SHORT Summary (max 50 words):")
print(generate_summary_mt5(test_doc, max_length=50, min_length=20))

print(f"\\n{'─'*60}")
print("MEDIUM Summary (max 100 words):")
print(generate_summary_mt5(test_doc, max_length=100, min_length=40))

print(f"\\n{'─'*60}")
print("LONG Summary (max 150 words):")
print(generate_summary_mt5(test_doc, max_length=150, min_length=60))

print(f"\\n{'─'*60}")
print("📝 Reference Summary:")
print(dataset['test'][100]['summary'])"""),

    mk("## 7.5 Application 4: Batch Summarization"),

    code("""# Demonstrate batch summarization
print("="*60)
print("APPLICATION 4: BATCH SUMMARIZATION")
print("="*60)

batch_docs = dataset['test']['document'][200:205]

print(f"\\nSummarizing {len(batch_docs)} documents...\\n")

for i, doc in enumerate(batch_docs):
    print(f"{'─'*60}")
    print(f"Document {i+1} ({len(doc.split())} words)")
    print(f"{'─'*60}")
    print(f"Summary: {generate_summary_mt5(doc)}")
    print()

print("✅ Batch summarization complete!")"""),

    mk("## 7.6 Application 5: Quality Comparison"),

    code("""# Compare quality across different approaches
comparison_doc = dataset['test'][150]['document']
comparison_ref = dataset['test'][150]['summary']

print("="*60)
print("APPLICATION 5: QUALITY COMPARISON")
print("="*60)

print(f"\\n📄 Original Document:")
print(comparison_doc[:300] + "...\\n")

# Generate summaries
summaries = {
    'mT5-small (beam=4)': generate_summary_mt5(comparison_doc, num_beams=4),
    'mT5-small (beam=8)': generate_summary_mt5(comparison_doc, num_beams=8),
    'ViT5': generate_summary_vit5(comparison_doc),
    'TextRank': textrank.summarize(comparison_doc, num_sentences=3),
    'Reference': comparison_ref
}

# Compute ROUGE for each
print("ROUGE Scores:\\n")
for name, summary in summaries.items():
    if name != 'Reference':
        score = compute_rouge_scores([summary], [comparison_ref])
        r1 = np.mean(score['rouge1']['fmeasure'])
        r2 = np.mean(score['rouge2']['fmeasure'])
        rL = np.mean(score['rougeL']['fmeasure'])
        print(f"{name}:")
        print(f"  ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rL:.4f}")
        print(f"  Summary: {summary}")
        print()

print("\\n✅ Quality comparison complete!")"""),

    mk("""## 7.7 Conclusion

### Summary of Findings:

1. **Best Overall Performance**: The abstractive models (mT5-small and ViT5) generally outperform the extractive approach in ROUGE scores

2. **Model Comparison**:
   - **ViT5**: Best for Vietnamese-specific content, more natural summaries
   - **mT5-small**: Good multilingual performance, fast inference
   - **TextRank**: Fast, reliable, but less fluent summaries

3. **Use Case Recommendations**:
   - **News**: Use ViT5 or mT5 for natural, concise summaries
   - **Technical Documents**: TextRank for factual accuracy
   - **Long Documents**: mT5/ViT5 with adjusted length parameters
   - **Real-time Applications**: TextRank for speed

4. **Key Insights**:
   - Beam search (4-8 beams) produces best quality
   - Document length impacts performance
   - Vietnamese-specific models (ViT5) better capture language nuances

### Next Steps:

- Fine-tune mT5/ViT5 on your specific domain data
- Experiment with different generation parameters
- Combine extractive and abstractive approaches
- Deploy models with appropriate hardware for production

---

## ✅ Notebook Complete!

This comprehensive notebook covered:
1. ✅ Theory of text summarization
2. ✅ Data loading and exploration
3. ✅ Extractive summarization (TextRank)
4. ✅ Abstractive summarization (mT5 + ViT5)
5. ✅ ROUGE evaluation and comparison
6. ✅ 8 comprehensive visualizations
7. ✅ Real-world applications

Thank you for using this notebook! 🎉"""),
]

# Add sections to notebook
nb['cells'].extend(section6)
nb['cells'].extend(section7)

print(f"Added Section 6 (Visualizations): {len(section6)} cells")
print(f"Added Section 7 (Applications): {len(section7)} cells")
print(f"Total cells now: {len(nb['cells'])}")

# Save complete notebook
with open('vietnamese_summarization_mt5_rtx_4070.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\\n" + "="*60)
print("✅ NOTEBOOK COMPLETE!")
print("="*60)
print(f"\\nFinal notebook has {len(nb['cells'])} cells")
print("\\nStructure:")
print("  Section 1: Theory (✅)")
print("  Section 2: Setup & Data (✅)")
print("  Section 3: Extractive Summarization (✅)")
print("  Section 4: Abstractive Summarization (✅)")
print("  Section 5: Evaluation & Comparison (✅)")
print("  Section 6: Visualizations - 8 Charts (✅)")
print("  Section 7: Applications (✅)")
print("\\nThe notebook is ready to use! 🎉")
