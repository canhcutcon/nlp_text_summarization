# PowerPoint Update Summary

**Date**: 2026-01-10
**File**: Vietnamese_Text_Summarization_Applications.pptx
**Source**: vietnamese_summarization_mt5_rtx_4070.ipynb (notebook results)

---

## Updates Applied

### 1. **Corrected ROUGE Scores** (Slides 1, 4, 5, 8)

#### Original (Incorrect):
- ViT5 ROUGE-1: 0.448

#### Updated (Actual Results from Notebook):
- **ViT5**: ROUGE-1 = **0.7781** ± 0.0457
- **TextRank**: ROUGE-1 = **0.5924** ± 0.1170
- **mT5-small**: ROUGE-1 = **0.1269** ± 0.0544

---

## Detailed Results from Notebook (500 test samples)

### Model Performance Comparison

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| **ViT5** | 0.7781 ± 0.0457 | 0.4963 ± 0.0897 | 0.4915 ± 0.0912 |
| **TextRank (Extractive)** | 0.5924 ± 0.1170 | 0.3267 ± 0.0874 | 0.3587 ± 0.0689 |
| **mT5-small** | 0.1269 ± 0.0544 | 0.0571 ± 0.0344 | 0.1074 ± 0.0410 |

### Model Ranking (by Average ROUGE)

1. **ViT5**: 0.5886 (Best overall)
2. **TextRank**: 0.4259 (Good extractive baseline)
3. **mT5-small**: 0.0971 (Needs domain fine-tuning)

---

## Slides Updated

### Existing Slides Modified:

#### Slide 1: Title Slide
- **Before**: "ROUGE-1: 0.448" / "Near SOTA (BARTpho: 0.450)"
- **After**: "ROUGE-1: 0.7781" / "Excellent Performance on VLSP Dataset"

#### Slide 4: Model Comparison
- **Before**: "ROUGE-1: 0.448 (only 0.002 behind SOTA)"
- **After**: Updated to show ViT5 0.778 with accurate comparison

#### Slide 5: Performance Metrics
- **Before**: Main metric 0.448, compared with BARTpho 0.450
- **After**:
  ```
  ViT5: 0.778 ✓
  TextRank: 0.592
  mT5-small: 0.127
  ```

#### Slide 8: Conclusion
- **Before**: "Near SOTA (0.448 vs 0.450)"
- **After**: "Strong ROUGE-1: 0.778"

---

## New Slides Added (Slides 9-14)

### Slide 9: ROUGE Score Distributions
- **Image**: ROUGE-1 Distribution.png
- **Shows**: Distribution of ROUGE-1 scores across all 3 models
- **Key Insight**: ViT5 has highest and most consistent scores

### Slide 10: Model Performance Comparison
- **Image**: Box Plots.png
- **Shows**: Box plots for ROUGE-1, ROUGE-2, ROUGE-L
- **Key Insight**: Clear performance hierarchy: ViT5 > TextRank > mT5-small

### Slide 11: Model Performance Summary ⭐
- **Image**: Correlation Heatmap + Summary.png
- **Shows**:
  - Heatmap of all ROUGE metrics
  - Summary statistics table
  - Model ranking bar chart
- **Key Insight**: Comprehensive visual summary of all results

### Slide 12: Document Length vs Performance
- **Image**: Document Length vs ROUGE-L.png
- **Shows**: Scatter plots showing how document length affects ROUGE-L
- **Key Insights**:
  - ViT5: Negative correlation (-0.295) - performs well on all lengths
  - TextRank: Negative correlation (-0.380) - struggles with long docs
  - mT5-small: Weak correlation (-0.020) - inconsistent

### Slide 13: Performance by Document Category
- **Image**: Performance by Document.png
- **Shows**: Bar charts for Short (<100), Medium (100-300), Long (>300) documents
- **Key Insights**:
  - ViT5: Consistent across all lengths (0.75-0.80)
  - TextRank: Better on short/medium, worse on long (0.45-0.65)
  - mT5-small: Consistently low (~0.12) across all lengths

### Slide 14: Dataset Statistics & Analysis
- **Image**: data_analytic.png
- **Shows**:
  - Document length distributions
  - Summary length distributions
  - Compression ratios
  - Dataset split statistics
- **Key Stats**:
  - Train: 34,198 samples
  - Validation: 4,476 samples
  - Test: 4,218 samples (500 used for evaluation)
  - Average compression: ~13%

---

## Key Findings from Notebook Results

### ✅ **ViT5 Performance**
- **Excellent ROUGE scores** (0.778 ROUGE-1)
- Significantly outperforms baseline models
- Consistent performance across document lengths
- Best for Vietnamese abstractive summarization

### ⚠️ **mT5-small Performance**
- **Poor ROUGE scores** (0.127 ROUGE-1)
- Likely due to:
  - No domain-specific fine-tuning
  - Multilingual model dilution
  - Vietnamese language complexity
- **Recommendation**: Fine-tune on VLSP dataset for better results

### ✓ **TextRank Performance**
- **Good extractive baseline** (0.592 ROUGE-1)
- Fast and reliable
- No training required
- Good for factual accuracy, but less fluent than ViT5

---

## Technical Notes

### Evaluation Setup
- **Test Set**: 500 samples from VLSP 2021 test set (4,218 total)
- **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)
- **Models**:
  - ViT5: YangYang0203/vi5_summarize (HuggingFace)
  - mT5-small: google/mt5-small (HuggingFace)
  - TextRank: PhoBERT embeddings + PageRank

### Generation Parameters
- **ViT5**:
  - Prefix: "tóm tắt:"
  - Max length: 256 tokens
  - Min length: 50 tokens
  - Num beams: 4
  - FP16 precision

- **mT5-small**:
  - Prefix: "tóm tắt:"
  - Max length: 128 tokens
  - Min length: 30 tokens
  - Num beams: 4
  - FP16 precision

- **TextRank**:
  - Top-N sentences: 3
  - Damping factor: 0.85
  - PhoBERT embeddings

---

## Visual Assets Added

All images are sourced from the notebook execution results:

1. ✅ [ROUGE-1 Distribution.png](ROUGE-1 Distribution.png) - Histogram of ROUGE-1 scores
2. ✅ [ROUGE-2 Distribution.png](ROUGE-2 Distribution.png) - Histogram of ROUGE-2 scores
3. ✅ [ROUGE-L Distribution.png](ROUGE-L Distribution.png) - Histogram of ROUGE-L scores
4. ✅ [Box Plots.png](Box Plots.png) - Box plots comparing all metrics
5. ✅ [Correlation Heatmap + Summary.png](Correlation Heatmap + Summary.png) - Comprehensive summary
6. ✅ [Document Length vs ROUGE-L.png](Document Length vs ROUGE-L.png) - Performance vs length
7. ✅ [Performance by Document.png](Performance by Document.png) - Performance by category
8. ✅ [Prediction vs Reference Length.png](Prediction vs Reference Length.png) - Length comparison
9. ✅ [data_analytic.png](data_analytic.png) - Dataset statistics

---

## Recommendations

### For Presentation:
1. **Focus on Slide 11** (Correlation Heatmap + Summary) - Most comprehensive visual
2. **Highlight ViT5's superiority** over baseline models
3. **Explain mT5-small's poor performance** - needs fine-tuning, not a fair comparison
4. **Emphasize practical applications** - ViT5 is production-ready

### For Future Work:
1. **Fine-tune mT5-small** on VLSP dataset to improve Vietnamese performance
2. **Test on full test set** (4,218 samples) for more robust evaluation
3. **Add BLEU and METEOR metrics** for comprehensive evaluation
4. **Domain adaptation** - test on news, legal, medical texts separately
5. **Hybrid approach** - combine TextRank (factual) with ViT5 (fluency)

---

## Files Modified

1. ✅ **Vietnamese_Text_Summarization_Applications.pptx** - Updated with correct results
2. ✅ **Vietnamese_Text_Summarization_Applications_Updated.pptx** - Backup with all changes

---

## Summary

The PowerPoint presentation has been successfully updated with:
- ✅ Correct ROUGE scores from notebook results
- ✅ 6 new visualization slides with charts and analysis
- ✅ Comprehensive model performance comparison
- ✅ Dataset statistics and insights
- ✅ 14 total slides (8 original + 6 new)

**Key Message**: ViT5 achieves excellent performance (0.778 ROUGE-1) on Vietnamese text summarization, significantly outperforming extractive baselines and demonstrating strong production readiness for real-world applications.

---

**Last Updated**: 2026-01-10
**Status**: ✅ Complete and ready for presentation
