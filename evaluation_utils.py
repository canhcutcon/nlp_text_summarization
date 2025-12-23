"""
Advanced Evaluation & Analysis Utilities
=========================================

Các công cụ để:
1. Detailed ROUGE analysis
2. Error analysis
3. Model comparison
4. Statistical significance testing
5. Qualitative evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import re


class AdvancedEvaluator:
    """
    Comprehensive evaluation toolkit cho text summarization
    """
    
    def __init__(self, predictions, references):
        """
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
        """
        self.predictions = predictions
        self.references = references
        assert len(predictions) == len(references), "Predictions và references phải có cùng length"
        
    # ========================================================================
    # 1. ROUGE ANALYSIS
    # ========================================================================
    
    def compute_rouge_scores(self):
        """Compute comprehensive ROUGE scores"""
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        scores = {
            'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
            'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
            'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
        }
        
        for pred, ref in zip(self.predictions, self.references):
            result = scorer.score(ref, pred)
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                scores[metric]['precision'].append(result[metric].precision)
                scores[metric]['recall'].append(result[metric].recall)
                scores[metric]['fmeasure'].append(result[metric].fmeasure)
        
        return scores
    
    def plot_rouge_detailed(self, scores):
        """Plot detailed ROUGE analysis"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        metrics = ['rouge1', 'rouge2', 'rougeL']
        measures = ['precision', 'recall', 'fmeasure']
        
        for i, metric in enumerate(metrics):
            for j, measure in enumerate(measures):
                data = scores[metric][measure]
                
                # Histogram
                axes[i, j].hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                axes[i, j].axvline(np.mean(data), color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {np.mean(data):.3f}')
                axes[i, j].axvline(np.median(data), color='green', linestyle='--',
                                 linewidth=2, label=f'Median: {np.median(data):.3f}')
                
                axes[i, j].set_xlabel('Score', fontsize=11)
                axes[i, j].set_ylabel('Frequency', fontsize=11)
                axes[i, j].set_title(f'{metric.upper()} - {measure.title()}', 
                                   fontsize=12, fontweight='bold')
                axes[i, j].legend(fontsize=10)
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # 2. ERROR ANALYSIS
    # ========================================================================
    
    def analyze_errors(self, rouge_scores, threshold=0.3):
        """
        Phân tích các trường hợp model perform kém
        
        Args:
            rouge_scores: Dict of ROUGE scores
            threshold: ROUGE-1 F1 threshold để classify as "error"
        """
        errors = []
        
        for idx, (pred, ref, score) in enumerate(zip(
            self.predictions, 
            self.references, 
            rouge_scores['rouge1']['fmeasure']
        )):
            if score < threshold:
                errors.append({
                    'index': idx,
                    'prediction': pred,
                    'reference': ref,
                    'rouge1': score,
                    'pred_len': len(pred.split()),
                    'ref_len': len(ref.split()),
                    'length_ratio': len(pred.split()) / max(len(ref.split()), 1)
                })
        
        error_df = pd.DataFrame(errors)
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS (ROUGE-1 < {threshold})")
        print(f"{'='*60}")
        print(f"\nTotal errors: {len(errors)} / {len(self.predictions)} ({len(errors)/len(self.predictions):.1%})")
        
        if len(errors) > 0:
            print(f"\nError statistics:")
            print(f"  Mean ROUGE-1: {error_df['rouge1'].mean():.3f}")
            print(f"  Mean length ratio: {error_df['length_ratio'].mean():.2f}")
            
            # Categorize errors
            too_short = sum(error_df['length_ratio'] < 0.5)
            too_long = sum(error_df['length_ratio'] > 2.0)
            wrong_content = len(errors) - too_short - too_long
            
            print(f"\nError categories:")
            print(f"  Too short: {too_short} ({too_short/len(errors):.1%})")
            print(f"  Too long: {too_long} ({too_long/len(errors):.1%})")
            print(f"  Wrong content: {wrong_content} ({wrong_content/len(errors):.1%})")
        
        return error_df
    
    def show_worst_cases(self, rouge_scores, n=5):
        """Show n worst performing examples"""
        # Get indices of worst cases
        f1_scores = rouge_scores['rouge1']['fmeasure']
        worst_indices = np.argsort(f1_scores)[:n]
        
        print(f"\n{'='*80}")
        print(f"TOP {n} WORST CASES")
        print(f"{'='*80}\n")
        
        for rank, idx in enumerate(worst_indices):
            print(f"\n{'─'*80}")
            print(f"Rank {rank+1} (ROUGE-1: {f1_scores[idx]:.3f})")
            print(f"{'─'*80}")
            print(f"\n📝 Reference:")
            print(f"{self.references[idx]}")
            print(f"\n🤖 Generated:")
            print(f"{self.predictions[idx]}")
    
    def show_best_cases(self, rouge_scores, n=5):
        """Show n best performing examples"""
        f1_scores = rouge_scores['rouge1']['fmeasure']
        best_indices = np.argsort(f1_scores)[-n:][::-1]
        
        print(f"\n{'='*80}")
        print(f"TOP {n} BEST CASES")
        print(f"{'='*80}\n")
        
        for rank, idx in enumerate(best_indices):
            print(f"\n{'─'*80}")
            print(f"Rank {rank+1} (ROUGE-1: {f1_scores[idx]:.3f})")
            print(f"{'─'*80}")
            print(f"\n📝 Reference:")
            print(f"{self.references[idx]}")
            print(f"\n🤖 Generated:")
            print(f"{self.predictions[idx]}")
    
    # ========================================================================
    # 3. LENGTH ANALYSIS
    # ========================================================================
    
    def analyze_length_correlation(self, rouge_scores):
        """Analyze correlation giữa length và ROUGE scores"""
        pred_lengths = [len(p.split()) for p in self.predictions]
        ref_lengths = [len(r.split()) for r in self.references]
        f1_scores = rouge_scores['rouge1']['fmeasure']
        
        # Compute correlations
        corr_pred_rouge = np.corrcoef(pred_lengths, f1_scores)[0, 1]
        corr_ref_rouge = np.corrcoef(ref_lengths, f1_scores)[0, 1]
        
        print(f"\n{'='*60}")
        print(f"LENGTH CORRELATION ANALYSIS")
        print(f"{'='*60}")
        print(f"\nCorrelation với ROUGE-1:")
        print(f"  Prediction length: {corr_pred_rouge:.3f}")
        print(f"  Reference length: {corr_ref_rouge:.3f}")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prediction length vs ROUGE
        axes[0].scatter(pred_lengths, f1_scores, alpha=0.3, s=20)
        axes[0].set_xlabel('Prediction Length (words)', fontsize=12)
        axes[0].set_ylabel('ROUGE-1 F1', fontsize=12)
        axes[0].set_title(f'Prediction Length vs ROUGE-1 (r={corr_pred_rouge:.3f})', 
                        fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Reference length vs ROUGE
        axes[1].scatter(ref_lengths, f1_scores, alpha=0.3, s=20, color='coral')
        axes[1].set_xlabel('Reference Length (words)', fontsize=12)
        axes[1].set_ylabel('ROUGE-1 F1', fontsize=12)
        axes[1].set_title(f'Reference Length vs ROUGE-1 (r={corr_ref_rouge:.3f})', 
                        fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # 4. NGRAM OVERLAP ANALYSIS
    # ========================================================================
    
    def analyze_ngram_overlap(self, n=2):
        """Analyze n-gram overlap giữa prediction và reference"""
        overlaps = []
        
        for pred, ref in zip(self.predictions, self.references):
            pred_ngrams = self._get_ngrams(pred, n)
            ref_ngrams = self._get_ngrams(ref, n)
            
            if len(ref_ngrams) > 0:
                overlap = len(pred_ngrams & ref_ngrams) / len(ref_ngrams)
            else:
                overlap = 0.0
            
            overlaps.append(overlap)
        
        print(f"\n{'='*60}")
        print(f"{n}-GRAM OVERLAP ANALYSIS")
        print(f"{'='*60}")
        print(f"\nMean overlap: {np.mean(overlaps):.3f}")
        print(f"Median overlap: {np.median(overlaps):.3f}")
        print(f"Std: {np.std(overlaps):.3f}")
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.hist(overlaps, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
        plt.axvline(np.mean(overlaps), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(overlaps):.3f}')
        plt.xlabel(f'{n}-gram Overlap', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{n}-gram Overlap Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return overlaps
    
    def _get_ngrams(self, text, n):
        """Extract n-grams from text"""
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    # ========================================================================
    # 5. VOCABULARY ANALYSIS
    # ========================================================================
    
    def analyze_vocabulary(self):
        """Analyze vocabulary usage"""
        # Tokenize
        pred_words = [word.lower() for pred in self.predictions for word in pred.split()]
        ref_words = [word.lower() for ref in self.references for word in ref.split()]
        
        # Vocabulary sizes
        pred_vocab = set(pred_words)
        ref_vocab = set(ref_words)
        
        # Common words
        common_vocab = pred_vocab & ref_vocab
        
        print(f"\n{'='*60}")
        print(f"VOCABULARY ANALYSIS")
        print(f"{'='*60}")
        print(f"\nVocabulary sizes:")
        print(f"  Predictions: {len(pred_vocab):,} unique words")
        print(f"  References: {len(ref_vocab):,} unique words")
        print(f"  Common: {len(common_vocab):,} words ({len(common_vocab)/len(ref_vocab):.1%})")
        
        # Word frequency
        pred_freq = Counter(pred_words)
        ref_freq = Counter(ref_words)
        
        print(f"\nTop 10 words in predictions:")
        for word, count in pred_freq.most_common(10):
            print(f"  {word}: {count:,}")
        
        print(f"\nTop 10 words in references:")
        for word, count in ref_freq.most_common(10):
            print(f"  {word}: {count:,}")
        
        return pred_vocab, ref_vocab, common_vocab


class ModelComparator:
    """
    So sánh performance giữa nhiều models
    """
    
    def __init__(self, results_dict):
        """
        Args:
            results_dict: Dict of {model_name: rouge_scores}
        """
        self.results = results_dict
        
    def compare_models(self):
        """Generate comprehensive comparison"""
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON")
        print(f"{'='*70}\n")
        
        # Create comparison table
        data = []
        for model_name, scores in self.results.items():
            data.append({
                'Model': model_name,
                'ROUGE-1': np.mean(scores['rouge1']['fmeasure']),
                'ROUGE-2': np.mean(scores['rouge2']['fmeasure']),
                'ROUGE-L': np.mean(scores['rougeL']['fmeasure']),
                'R1_std': np.std(scores['rouge1']['fmeasure']),
                'R2_std': np.std(scores['rouge2']['fmeasure']),
                'RL_std': np.std(scores['rougeL']['fmeasure'])
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('ROUGE-1', ascending=False)
        
        print(df.to_string(index=False))
        
        return df
    
    def plot_comparison(self):
        """Visualize model comparison"""
        models = list(self.results.keys())
        metrics = ['rouge1', 'rouge2', 'rougeL']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            scores = [np.mean(self.results[model][m]['fmeasure']) for m in metrics]
            stds = [np.std(self.results[model][m]['fmeasure']) for m in metrics]
            
            ax.bar(x + i * width, scores, width, 
                  label=model, yerr=stds, capsize=5)
        
        ax.set_xlabel('Metrics', fontsize=13)
        ax.set_ylabel('F1 Score', fontsize=13)
        ax.set_title('Model Comparison - ROUGE Scores', fontsize=15, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def statistical_test(self, model1, model2):
        """
        Kiểm tra statistical significance giữa 2 models
        
        Uses paired t-test
        """
        scores1 = self.results[model1]['rouge1']['fmeasure']
        scores2 = self.results[model2]['rouge1']['fmeasure']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        print(f"\n{'='*60}")
        print(f"STATISTICAL SIGNIFICANCE TEST")
        print(f"{'='*60}")
        print(f"\nComparing: {model1} vs {model2}")
        print(f"Metric: ROUGE-1 F1")
        print(f"\n{model1}: {np.mean(scores1):.4f} ± {np.std(scores1):.4f}")
        print(f"{model2}: {np.mean(scores2):.4f} ± {np.std(scores2):.4f}")
        print(f"\nT-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.001:
            print(f"✓ Difference is HIGHLY significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"✓ Difference is significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"✓ Difference is marginally significant (p < 0.05)")
        else:
            print(f"✗ No significant difference (p >= 0.05)")
        
        return t_stat, p_value


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Advanced Evaluation Utilities")
    print("=" * 60)
    print("\nUsage example:")
    print("""
    # 1. Single model evaluation
    evaluator = AdvancedEvaluator(predictions, references)
    
    # Compute ROUGE
    scores = evaluator.compute_rouge_scores()
    
    # Plot detailed ROUGE
    evaluator.plot_rouge_detailed(scores)
    
    # Error analysis
    errors = evaluator.analyze_errors(scores)
    evaluator.show_worst_cases(scores, n=5)
    evaluator.show_best_cases(scores, n=5)
    
    # Length analysis
    evaluator.analyze_length_correlation(scores)
    
    # N-gram overlap
    evaluator.analyze_ngram_overlap(n=2)
    
    # Vocabulary analysis
    evaluator.analyze_vocabulary()
    
    # 2. Multiple model comparison
    results = {
        'ViT5': vit5_scores,
        'mT5': mt5_scores,
        'PhoBERT': phobert_scores
    }
    
    comparator = ModelComparator(results)
    comparison_df = comparator.compare_models()
    comparator.plot_comparison()
    comparator.statistical_test('ViT5', 'mT5')
    """)
