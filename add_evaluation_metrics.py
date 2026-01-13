"""
Script to add comprehensive evaluation metrics to demo sections in the Vietnamese summarization notebook.
This script modifies the notebook to include ROUGE, BLEU scores, and detailed statistics in demo cells.
"""

import json
import sys

def load_notebook(path):
    """Load a Jupyter notebook"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook, path):
    """Save a Jupyter notebook"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

def create_evaluation_helper_cell():
    """Create a new cell with evaluation helper functions"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sacrebleu.metrics import BLEU\n",
            "from tabulate import tabulate\n",
            "\n",
            "def evaluate_summary(prediction, reference, original_doc):\n",
            "    \"\"\"\n",
            "    Compute comprehensive evaluation metrics for a summary\n",
            "    \n",
            "    Args:\n",
            "        prediction (str): Generated summary\n",
            "        reference (str): Reference/gold summary\n",
            "        original_doc (str): Original document\n",
            "    \n",
            "    Returns:\n",
            "        dict: Dictionary containing all evaluation metrics\n",
            "    \"\"\"\n",
            "    # Initialize ROUGE scorer\n",
            "    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)\n",
            "    rouge_scores = scorer.score(reference, prediction)\n",
            "    \n",
            "    # Initialize BLEU scorer\n",
            "    bleu = BLEU()\n",
            "    bleu_score = bleu.sentence_score(prediction, [reference])\n",
            "    \n",
            "    # Calculate statistics\n",
            "    doc_words = len(original_doc.split())\n",
            "    pred_words = len(prediction.split())\n",
            "    ref_words = len(reference.split())\n",
            "    compression = (pred_words / doc_words * 100) if doc_words > 0 else 0\n",
            "    \n",
            "    return {\n",
            "        'rouge1_f1': rouge_scores['rouge1'].fmeasure,\n",
            "        'rouge1_p': rouge_scores['rouge1'].precision,\n",
            "        'rouge1_r': rouge_scores['rouge1'].recall,\n",
            "        'rouge2_f1': rouge_scores['rouge2'].fmeasure,\n",
            "        'rouge2_p': rouge_scores['rouge2'].precision,\n",
            "        'rouge2_r': rouge_scores['rouge2'].recall,\n",
            "        'rougeL_f1': rouge_scores['rougeL'].fmeasure,\n",
            "        'rougeL_p': rouge_scores['rougeL'].precision,\n",
            "        'rougeL_r': rouge_scores['rougeL'].recall,\n",
            "        'bleu': bleu_score.score,\n",
            "        'doc_words': doc_words,\n",
            "        'pred_words': pred_words,\n",
            "        'ref_words': ref_words,\n",
            "        'compression': compression\n",
            "    }\n",
            "\n",
            "def display_evaluation_table(metrics, model_name=\"Model\"):\n",
            "    \"\"\"\n",
            "    Display evaluation metrics in a formatted table\n",
            "    \n",
            "    Args:\n",
            "        metrics (dict): Evaluation metrics from evaluate_summary()\n",
            "        model_name (str): Name of the model for display\n",
            "    \"\"\"\n",
            "    print(f\"\\n📊 Evaluation Metrics for {model_name}\")\n",
            "    print(\"=\" * 70)\n",
            "    \n",
            "    # ROUGE scores table\n",
            "    rouge_table = [\n",
            "        ['ROUGE-1', f\"{metrics['rouge1_p']:.4f}\", f\"{metrics['rouge1_r']:.4f}\", f\"{metrics['rouge1_f1']:.4f}\"],\n",
            "        ['ROUGE-2', f\"{metrics['rouge2_p']:.4f}\", f\"{metrics['rouge2_r']:.4f}\", f\"{metrics['rouge2_f1']:.4f}\"],\n",
            "        ['ROUGE-L', f\"{metrics['rougeL_p']:.4f}\", f\"{metrics['rougeL_r']:.4f}\", f\"{metrics['rougeL_f1']:.4f}\"]\n",
            "    ]\n",
            "    print(\"\\nROUGE Scores:\")\n",
            "    print(tabulate(rouge_table, headers=['Metric', 'Precision', 'Recall', 'F1-Score'], tablefmt='grid'))\n",
            "    \n",
            "    # BLEU and statistics\n",
            "    stats_table = [\n",
            "        ['BLEU Score', f\"{metrics['bleu']:.2f}\"],\n",
            "        ['Original Length', f\"{metrics['doc_words']} words\"],\n",
            "        ['Prediction Length', f\"{metrics['pred_words']} words\"],\n",
            "        ['Reference Length', f\"{metrics['ref_words']} words\"],\n",
            "        ['Compression Ratio', f\"{metrics['compression']:.1f}%\"]\n",
            "    ]\n",
            "    print(\"\\nAdditional Metrics:\")\n",
            "    print(tabulate(stats_table, headers=['Metric', 'Value'], tablefmt='grid'))\n",
            "\n",
            "def compare_models(metrics_list, model_names):\n",
            "    \"\"\"\n",
            "    Compare multiple models side by side\n",
            "    \n",
            "    Args:\n",
            "        metrics_list (list): List of metrics dictionaries\n",
            "        model_names (list): List of model names\n",
            "    \"\"\"\n",
            "    print(\"\\n📊 Model Comparison\")\n",
            "    print(\"=\" * 100)\n",
            "    \n",
            "    comparison_table = [\n",
            "        ['ROUGE-1 F1'] + [f\"{m['rouge1_f1']:.4f}\" for m in metrics_list],\n",
            "        ['ROUGE-2 F1'] + [f\"{m['rouge2_f1']:.4f}\" for m in metrics_list],\n",
            "        ['ROUGE-L F1'] + [f\"{m['rougeL_f1']:.4f}\" for m in metrics_list],\n",
            "        ['BLEU'] + [f\"{m['bleu']:.2f}\" for m in metrics_list],\n",
            "        ['Length (words)'] + [f\"{m['pred_words']}\" for m in metrics_list],\n",
            "        ['Compression'] + [f\"{m['compression']:.1f}%\" for m in metrics_list]\n",
            "    ]\n",
            "    \n",
            "    print(tabulate(comparison_table, headers=['Metric'] + model_names, tablefmt='grid'))\n",
            "    \n",
            "    # Highlight best scores\n",
            "    print(\"\\n🏆 Best Scores:\")\n",
            "    best_rouge1 = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['rouge1_f1'])\n",
            "    best_rouge2 = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['rouge2_f1'])\n",
            "    best_rougeL = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['rougeL_f1'])\n",
            "    best_bleu = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['bleu'])\n",
            "    \n",
            "    print(f\"  • ROUGE-1: {model_names[best_rouge1]} ({metrics_list[best_rouge1]['rouge1_f1']:.4f})\")\n",
            "    print(f\"  • ROUGE-2: {model_names[best_rouge2]} ({metrics_list[best_rouge2]['rouge2_f1']:.4f})\")\n",
            "    print(f\"  • ROUGE-L: {model_names[best_rougeL]} ({metrics_list[best_rougeL]['rougeL_f1']:.4f})\")\n",
            "    print(f\"  • BLEU: {model_names[best_bleu]} ({metrics_list[best_bleu]['bleu']:.2f})\")\n",
            "\n",
            "print(\"✅ Evaluation helper functions loaded!\")\n"
        ]
    }

def create_evaluation_helper_markdown():
    """Create markdown cell for evaluation helper section"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5.0 Evaluation Helper Functions\n",
            "\n",
            "These helper functions compute comprehensive evaluation metrics:\n",
            "\n",
            "- **ROUGE scores**: Precision, Recall, F1 for ROUGE-1, ROUGE-2, ROUGE-L\n",
            "- **BLEU score**: Machine translation quality metric\n",
            "- **Statistics**: Length comparison and compression ratio\n"
        ]
    }

def update_package_installation(notebook):
    """Update package installation cell to include sacrebleu and tabulate"""
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('rouge-score' in line for line in cell.get('source', [])):
            # Find the evaluation packages line
            for j, line in enumerate(cell['source']):
                if 'rouge-score' in line and 'scikit-learn' in line:
                    cell['source'][j] = line.replace('scikit-learn', 'scikit-learn sacrebleu tabulate')
                    print(f"✓ Updated package installation cell (cell {i})")
                    return True
    return False

def add_evaluation_helpers(notebook):
    """Add evaluation helper functions before section 5.1"""
    # Find section 5.1 (ROUGE Metrics Implementation)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell.get('source', []))
            if '5.1 ROUGE Metrics Implementation' in source_text:
                # Insert markdown and code cells before this
                notebook['cells'].insert(i, create_evaluation_helper_cell())
                notebook['cells'].insert(i, create_evaluation_helper_markdown())
                print(f"✓ Added evaluation helper functions before section 5.1 (position {i})")
                return True
    return False

def update_extractive_demo(notebook):
    """Update extractive summarization demo with evaluation metrics"""
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('EXTRACTIVE SUMMARIZATION EXAMPLES' in line for line in cell.get('source', [])):
            # Replace the demo cell with enhanced version
            cell['source'] = [
                "# Test on a few examples with evaluation metrics\n",
                "print(\"=\"*60)\n",
                "print(\"EXTRACTIVE SUMMARIZATION EXAMPLES\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "num_examples = 3\n",
                "\n",
                "for i in range(num_examples):\n",
                "    test_doc = dataset['test'][i]['document']\n",
                "    test_ref = dataset['test'][i]['summary']\n",
                "    \n",
                "    print(f\"\\n{'='*60}\")\n",
                "    print(f\"EXAMPLE {i+1}\")\n",
                "    print(f\"{'='*60}\")\n",
                "    \n",
                "    print(f\"\\n📄 Original Document ({len(test_doc.split())} words):\")\n",
                "    print(test_doc[:300] + \"...\")\n",
                "    \n",
                "    print(f\"\\n🤖 Extractive Summary (TextRank):\")\n",
                "    extractive_summary = textrank.summarize(test_doc, num_sentences=3)\n",
                "    print(extractive_summary)\n",
                "    \n",
                "    print(f\"\\n📝 Reference Summary:\")\n",
                "    print(test_ref)\n",
                "    \n",
                "    # Evaluate the extractive summary\n",
                "    metrics = evaluate_summary(extractive_summary, test_ref, test_doc)\n",
                "    display_evaluation_table(metrics, \"TextRank Extractive\")\n",
                "\n",
                "print(\"\\n✅ Extractive summarization demo complete!\")\n"
            ]
            print(f"✓ Updated extractive demo cell (cell {i})")
            return True
    return False

def update_abstractive_demo(notebook):
    """Update abstractive summarization demo with evaluation metrics"""
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('ABSTRACTIVE SUMMARIZATION EXAMPLES' in line for line in cell.get('source', [])):
            # Replace with enhanced version
            cell['source'] = [
                "# Test both models on examples with evaluation\n",
                "print(\"=\"*60)\n",
                "print(\"ABSTRACTIVE SUMMARIZATION EXAMPLES\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "num_examples = 3\n",
                "\n",
                "for i in range(num_examples):\n",
                "    test_doc = dataset['test'][i]['document']\n",
                "    test_ref = dataset['test'][i]['summary']\n",
                "\n",
                "    print(f\"\\n{'='*60}\")\n",
                "    print(f\"EXAMPLE {i+1}\")\n",
                "    print(f\"{'='*60}\")\n",
                "\n",
                "    print(f\"\\n📄 Original Document ({len(test_doc.split())} words):\")\n",
                "    print(test_doc[:300] + \"...\")\n",
                "\n",
                "    print(f\"\\n🤖 mT5-small Summary:\")\n",
                "    mt5_summary = generate_summary_mt5(test_doc)\n",
                "    print(mt5_summary)\n",
                "\n",
                "    print(f\"\\n🤖 ViT5 Summary:\")\n",
                "    vit5_summary = generate_summary_vit5(test_doc)\n",
                "    print(vit5_summary)\n",
                "\n",
                "    print(f\"\\n📝 Reference Summary:\")\n",
                "    print(test_ref)\n",
                "    \n",
                "    # Evaluate both models\n",
                "    mt5_metrics = evaluate_summary(mt5_summary, test_ref, test_doc)\n",
                "    vit5_metrics = evaluate_summary(vit5_summary, test_ref, test_doc)\n",
                "    \n",
                "    # Display comparison\n",
                "    compare_models([mt5_metrics, vit5_metrics], ['mT5-small', 'ViT5'])\n",
                "\n",
                "print(\"\\n✅ Abstractive summarization demo complete!\")\n"
            ]
            print(f"✓ Updated abstractive demo cell (cell {i})")
            return True
    return False

def update_generation_strategy_demo(notebook):
    """Update generation strategy comparison with evaluation metrics"""
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and any('GENERATION STRATEGY COMPARISON' in line for line in cell.get('source', [])):
            cell['source'] = [
                "# Compare different generation strategies with evaluation\n",
                "print(\"=\"*60)\n",
                "print(\"GENERATION STRATEGY COMPARISON\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "test_text = dataset['test'][0]['document']\n",
                "test_ref = dataset['test'][0]['summary']\n",
                "\n",
                "print(f\"\\nTest Document ({len(test_text.split())} words):\")\n",
                "print(test_text[:200] + \"...\\n\")\n",
                "\n",
                "print(f\"\\nReference Summary:\")\n",
                "print(test_ref)\n",
                "\n",
                "strategies = [\"beam_search\", \"sampling\", \"top_k\", \"top_p\"]\n",
                "all_metrics = []\n",
                "all_summaries = []\n",
                "\n",
                "print(\"\\n\\nComparing generation strategies with mT5-small:\\n\")\n",
                "\n",
                "for strategy in strategies:\n",
                "    summary = generate_summary_mt5(test_text, strategy=strategy)\n",
                "    all_summaries.append(summary)\n",
                "    \n",
                "    print(f\"{'─'*60}\")\n",
                "    print(f\"Strategy: {strategy.upper()}\")\n",
                "    print(f\"{'─'*60}\")\n",
                "    print(summary)\n",
                "    print(f\"Length: {len(summary.split())} words\")\n",
                "    \n",
                "    # Evaluate this strategy\n",
                "    metrics = evaluate_summary(summary, test_ref, test_text)\n",
                "    all_metrics.append(metrics)\n",
                "\n",
                "# Compare all strategies\n",
                "compare_models(all_metrics, [s.upper() for s in strategies])\n",
                "\n",
                "print(\"\\n✅ Strategy comparison complete!\")\n"
            ]
            print(f"✓ Updated generation strategy demo cell (cell {i})")
            return True
    return False

def main():
    notebook_path = 'vietnamese_summarization_mt5_rtx_4070.ipynb'
    
    print("Loading notebook...")
    notebook = load_notebook(notebook_path)
    
    print("\nApplying updates:")
    print("-" * 60)
    
    # Apply all updates
    update_package_installation(notebook)
    add_evaluation_helpers(notebook)
    update_extractive_demo(notebook)
    update_abstractive_demo(notebook)
    update_generation_strategy_demo(notebook)
    
    # Save updated notebook
    print("\nSaving updated notebook...")
    save_notebook(notebook, notebook_path)
    
    print("\n" + "=" * 60)
    print("✅ Notebook successfully updated with evaluation metrics!")
    print("=" * 60)
    print("\nUpdates applied:")
    print("  • Added sacrebleu and tabulate packages")
    print("  • Added evaluation helper functions (Section 5.0)")
    print("  • Enhanced extractive demo with ROUGE & BLEU scores")
    print("  • Enhanced abstractive demo with model comparison")
    print("  • Enhanced generation strategy comparison with metrics")
    print("\nYou can now run the notebook to see detailed evaluation metrics!")

if __name__ == "__main__":
    main()
