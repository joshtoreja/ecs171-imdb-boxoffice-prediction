# Alex Week 3: Robustness & Feature Analysis

## Task 1: Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | runtime | 0.188 |
| 2 | popularity | 0.163 |
| 3 | budget | 0.127 |
| 4 | log_budget | 0.125 |
| 5 | vote_count | 0.121 |

**runtime (0.188):** Proxy for film type. Mainstream 90 to 130 min films tend toward wide release and higher ROI. Very short or very long films occupy niche markets.

**popularity (0.163):** Captures audience awareness and media buzz. High popularity correlates with strong marketing and franchise recognition, driving higher returns relative to cost.

**budget (0.127):** Non-linear relationship with ROI. Large budgets can dilute ROI while low budget films that find audiences achieve high ROI. The model exploits both patterns.

**log_budget (0.125):** Compresses the wide budget range and captures diminishing returns to spending. Nearly equal importance to raw budget confirms the tree uses both representations.

**vote_count (0.121):** Reflects post-release audience engagement volume. Widely seen films generate more revenue while budget stays fixed, boosting ROI.

## Task 2: Feature Ablation

Removed: `log_budget`, `popularity_x_votes`

| Model | Features | Holdout Acc | Holdout F1 | Holdout AUC |
|-------|----------|-------------|------------|-------------|
| Full Model | 9 | 0.8167 | 0.6419 | 0.8352 |
| Ablation | 7 | 0.8167 | 0.6433 | 0.8341 |

Removing `log_budget` and `popularity_x_votes` had negligible impact. AUC dropped by 0.001 and F1 slightly increased. These engineered features are redundant since the Random Forest already captures the non-linear patterns from the raw inputs. The model is robust to their removal.

## Task 3: Pre-Release Model

Removed: `vote_average`, `vote_count`, `popularity`, `popularity_x_votes`

| Model | Features | Holdout Acc | Holdout F1 | Holdout AUC |
|-------|----------|-------------|------------|-------------|
| Full Model | 9 | 0.8167 | 0.6419 | 0.8352 |
| Baseline | 4 | 0.7974 | 0.6223 | 0.8176 |
| Pre-Release | 5 | 0.7457 | 0.5236 | 0.7646 |

Removing post-release features dropped AUC by 0.071 and F1 by 0.118. Audience reception signals are critical for ROI prediction. Pre-release prediction using only budget, runtime, and scheduling features is significantly harder. Additional features like cast, franchise status, or social media data would be needed to close the gap.
