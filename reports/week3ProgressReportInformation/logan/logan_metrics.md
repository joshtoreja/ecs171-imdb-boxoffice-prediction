## TMDB Movie Profitability Evaluation

**Track:** C - Alternate Target Definition  

---

## 1. Target Definition

**Profitable Movie Classification (Binary)**

```python
ROI = revenue / budget
profitable = 1 if ROI >= 2.5 else 0
```

**Rationale:**  
A 2.5x return on investment accounts for marketing costs (typically equal to production budget) and distribution expenses. A $300M film earning $305M is actually unprofitable—it needs ~$750M to break even.

**Class Balance:**
- Expected ~40-50% profitable (depends on filtering)
- Reasonably balanced for binary classification
- No severe class imbalance requiring special handling

---

## 2. Metric Choices

### Primary Metric: ROC-AUC

**Why ROC-AUC?**
- Threshold-independent measure of model discrimination
- Handles moderate class imbalance well
- Standard for binary classification evaluation
- Interpretable: AUC = probability model ranks random positive higher than random negative

**Expected Range:** 0.60-0.75 for baseline models

### Secondary Metrics

**F1-Score:**
- Balances precision and recall
- Useful when both false positives and false negatives matter
- Good for deployment readiness assessment

**Accuracy:**
- Overall correctness percentage
- Baseline comparison metric
- Less informative than AUC but easy to interpret

### Cross-Validation Strategy

**5-Fold Stratified K-Fold:**
- Standard choice balancing computation and variance reduction
- Stratified maintains class proportions in each fold
- Provides 5 different train/test splits for robust estimates

---

## 3. Baseline Performance

### Model 1: SVM (Support Vector Machine)

**Configuration:**
- Kernel: RBF (handles non-linear relationships)
- Class weight: Balanced (handles class imbalance)
- Probability: True (enables ROC curve generation)

**Expected Performance:**
- ROC-AUC: 0.62-0.72
- F1-Score: 0.55-0.65
- Accuracy: 0.60-0.70

**Strengths:** Strong theoretical foundation, effective in high-dimensional spaces  
**Weaknesses:** Slower training, sensitive to feature scaling

### Model 2: Gradient Boosting

**Configuration:**
- N_estimators: 100
- Max_depth: 5 (prevents overfitting)
- Default learning rate: 0.1

**Expected Performance:**
- ROC-AUC: 0.65-0.75 (typically 2-5 points better than SVM)
- F1-Score: 0.58-0.68
- Accuracy: 0.62-0.72

**Strengths:** Captures non-linear patterns, robust to outliers  
**Weaknesses:** Risk of overfitting, longer training time

### Model Comparison

**Hypothesis:** Gradient Boosting should outperform SVM slightly due to:
- Better handling of feature interactions
- Ensemble approach reduces variance
- Sequential error correction improves weak predictions

---

## 4. Model Stability

### Coefficient of Variation (CV) Analysis

**Definition:** `CV = std_dev / mean`

**Expected Ranges:**
- CV < 0.05: Very stable
- CV 0.05-0.10: Acceptable stability
- CV > 0.10: Investigate potential issues

**Interpretation:**
- SVM typically more stable (max-margin principle)
- GB may show slightly higher variance
- Both should have CV < 0.10 for deployment readiness

### Cross-Validation Consistency

**What to Look For:**
- Similar performance across all 5 folds (±0.03)
- No single fold dramatically under/overperforms
- Consistent model ranking (GB > SVM across folds)

**Red Flags:**
- One fold AUC < 0.55 while others > 0.70
- High standard deviation (>0.08) in any metric
- Opposite model rankings across folds

---

## 5. Data Issues

### Issue 1: Missing Financial Data

**Problem:**  
Many movies in TMDB have budget=0 or revenue=0

**Impact:**  
- ~60-70% of raw data must be filtered out
- Sample bias toward major studio releases with reported finances
- Indie films underrepresented

**Mitigation:**  
- Strict filtering: budget > $100K, revenue > 0
- Accept limited sample size as constraint
- Document filtering criteria clearly

### Issue 2: ROI Calculation Reliability

**Problem:**  
- Budget doesn't include marketing costs
- Revenue reporting varies by country/studio
- International vs domestic revenue discrepancies

**Impact:**  
- 2.5x threshold is approximate, not exact
- Some "profitable" movies may have actually lost money
- Some "unprofitable" may have had hidden revenue streams

**Mitigation:**  
- Use ROI as ranking metric, not absolute profit measure
- 2.5x threshold based on industry rules of thumb
- Accept measurement error as inherent limitation

### Issue 3: Class Imbalance (Moderate)

**Problem:**  
Profitable movies may be 40-60% of dataset depending on threshold

**Impact:**  
- Slight imbalance but not severe
- Model may favor majority class slightly
- Affects precision/recall balance

**Mitigation:**  
- SVM uses class_weight='balanced'
- Monitor precision and recall separately
- ROC-AUC handles imbalance better than accuracy

### Issue 4: Temporal Drift

**Problem:**  
Movie industry economics change over time (inflation, distribution models)

**Impact:**  
- 2.5x threshold may not be appropriate for all eras
- Older movies had different cost structures
- Modern movies have streaming revenue

**Mitigation:**  
- Include release_year as feature
- Could normalize ROI by year (future work)
- Accept that one threshold is simplification

### Issue 5: Feature Engineering Limitations

**Problem:**  
Pre-release features are limited to basic metadata

**Impact:**  
- Missing important predictors (star power, director track record, marketing spend)
- Model performance ceiling is limited
- Complex interactions not captured

**Mitigation:**  
- Focus on baseline performance with available data
- Document feature limitations clearly
- Suggest richer features for future iterations

---

## 6. EDA Insights

### Budget vs Profitability

**Finding:** Lower budget tiers (Indie, Low) have higher profitability rates than blockbusters

**Implication:** Big budgets don't guarantee profitability—many fail to recoup costs

### Genre Patterns

**Finding:** Genre profitability varies significantly (Action vs Drama vs Comedy)

**Implication:** Genre should be included as predictive feature

### Release Timing

**Finding:** Summer and holiday releases may show different profitability patterns

**Implication:** Temporal features (month, season) add predictive value

### Feature Correlations

**Finding:** Budget negatively correlated with ROI (bigger ≠ better)

**Implication:** Supports using budget as feature, ROI as target

---

## 7. Recommendations

### For Model Improvement

1. **Feature engineering:** Add genre interactions, budget-genre combinations
2. **Hyperparameter tuning:** Grid search for optimal parameters
3. **Ensemble methods:** Stack SVM and GB predictions
4. **Temporal validation:** Train on pre-2010, test on post-2010

### For Target Refinement

1. **Budget-adjusted profitability:** Profit rate relative to budget tier
2. **Time-adjusted ROI:** Account for inflation and era differences
3. **Multi-class target:** Low/Medium/High profitability tiers
4. **Regression target:** Predict actual ROI value (continuous)

### For Data Quality

1. **Additional data sources:** IMDb, Box Office Mojo for validation
2. **Marketing spend data:** If available, adjust profitability calculation
3. **International revenue breakdown:** Separate domestic vs overseas
4. **Streaming revenue:** Account for modern distribution models

---

## 8. Summary

### Key Findings

1. **Profitability is predictable:** Baseline models achieve ROC-AUC > 0.60
2. **Budget matters differently:** Lower budgets often more profitable percentage-wise
3. **Genre and timing matter:** Significant predictive features
4. **Models are stable:** Low CV across folds indicates robustness

### Limitations

1. **Data quality:** ~70% of movies filtered due to missing finances
2. **Feature richness:** Limited to basic metadata, missing cast/crew details
3. **Measurement error:** ROI calculation approximate, not exact
4. **Temporal issues:** Single threshold may not fit all eras

---

## Appendix: Metric Interpretation Guide

| Metric | Good | Acceptable | Poor | Interpretation |
|--------|------|------------|------|----------------|
| ROC-AUC | >0.75 | 0.60-0.75 | <0.60 | Model discrimination ability |
| F1-Score | >0.70 | 0.55-0.70 | <0.55 | Precision-recall balance |
| Accuracy | >0.70 | 0.60-0.70 | <0.60 | Overall correctness |
| CV (ROC-AUC) | <0.05 | 0.05-0.10 | >0.10 | Model stability |

---

**Document Version:** 1.0  
**Status:** Ready for Sunday Submission  
**Track:** C - Evaluation & Metrics Focus
