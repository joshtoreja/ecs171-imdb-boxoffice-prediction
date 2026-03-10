Week 3 Analysis – Andrew
Cross-Track Integration & Sensitivity Analysis
Dataset and Modeling Setup

Dataset: TMDB Movie Dataset (structured metadata)
Target Definition: Binary classification where

Profitable = ROI ≥ threshold

ROI = revenue / budget

Thresholds tested:

ROI ≥ 2.0

ROI ≥ 2.5

ROI ≥ 3.0

Models evaluated:

Logistic Regression (with scaling, class-balanced)

Random Forest (class-balanced)

Evaluation metric:

5-fold cross-validated Accuracy

F1 Score

ROC-AUC

Threshold Sensitivity Analysis

As the ROI threshold increases from 2.0 to 3.0, the positive class rate decreases from approximately 29% to 24%, increasing class imbalance and making the classification task more difficult. This effect is reflected in model performance. For Logistic Regression, ROC-AUC decreases from approximately 0.773 at ROI ≥ 2.0 to 0.746 at ROI ≥ 3.0. Random Forest shows a similar but more stable trend, decreasing from approximately 0.827 to 0.802.

The drop in F1 score at higher thresholds suggests that stricter profitability definitions introduce more borderline and rare cases, reducing recall for profitable films. Overall, predicting extreme profitability (ROI ≥ 3.0) is meaningfully harder than predicting moderate profitability, confirming that threshold choice directly impacts task difficulty.

Best Model Selection

Across all thresholds, Random Forest consistently outperformed Logistic Regression in ROC-AUC and F1 score.

The strongest configuration was:

Model: Random Forest

Threshold: ROI ≥ 2.5 (project baseline)

Cross-Validated ROC-AUC: ~0.82

This suggests that nonlinear modeling captures interactions between budget, popularity, runtime, and vote counts more effectively than a linear classifier.

False Positive and False Negative Analysis
False Positives (Predicted Profitable but Not)

The largest false positives tend to be high-budget, high-popularity films such as Snowpiercer, Ghostbusters, and Solo: A Star Wars Story. These films exhibit strong commercial signals — large budgets, substantial vote counts, and mainstream genres — leading the model to predict profitability. However, because ROI is a ratio, high budgets increase the denominator, making it harder to exceed the 2.5 threshold.

This indicates that the model may overweight popularity and scale while underestimating the structural difficulty of achieving high ROI with large production costs.

False Negatives (Predicted Not Profitable but Are)

The largest false negatives are predominantly low-budget or niche films, including documentaries and minimally publicized releases. These films often have very low popularity and vote counts, causing the model to predict non-profitability. However, their extremely small budgets allow them to achieve high ROI despite limited visibility.

This suggests the model struggles to identify “sleeper hits” and underestimates profitability when revenue is modest but the production cost is extremely low.

Interpretation and Implications

These results suggest:

Stricter ROI definitions reduce positive class prevalence and increase prediction difficulty.

Nonlinear models (Random Forest) outperform linear models in this setting.

High-budget films may appear commercially strong but fail ROI thresholds.

Low-budget films can achieve high ROI without strong pre-release signals.

Overall, structured metadata provides meaningful predictive signal, but profitability — especially extreme ROI — remains partially driven by factors not fully captured in pre-release indicators.