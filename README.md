# ECS 171 — Predicting Movie Profitability Using Pre-Release Metadata

This project explores whether a movie will be profitable using structured metadata available before release. Our goal was to build a realistic machine learning pipeline that supports pre-release decision-making without relying on information that would only be known after a film comes out.

## Team
- Andrew Toreja
- Alexander Stroev
- Logan Tomasetti

## Project Overview
Movie studios often make greenlighting and budgeting decisions before a film is released. This project uses machine learning to predict profitability from pre-release features such as budget, runtime, release timing, genres, production companies, and production countries.

Rather than predicting vague “success,” we frame the task as a **binary classification problem**:
- **1 = profitable**
- **0 = not profitable**

Profitability was defined using a return-on-investment threshold based on revenue and budget.

## Dataset
We used movie metadata from the TMDB dataset, including features such as:
- budget
- runtime
- release year
- genres
- production companies
- production countries

Post-release variables were excluded from the main model to avoid data leakage and preserve real-world usefulness.

## Methods
Our workflow included:
- data cleaning and preprocessing
- ROI-based target construction
- feature engineering
- model training and comparison
- cross-validation and evaluation

We tested multiple models, including:
- Logistic Regression
- Random Forest
- XGBoost

## Evaluation Metrics
We evaluated performance using:
- Accuracy
- F1-score
- ROC-AUC

These metrics helped us assess overall correctness, class balance performance, and ranking ability.

## Repository Structure
```text
ECS171-IMDB-BoxOffice-Prediction/
├── data/
│   ├── raw/          # raw datasets
│   └── processed/    # cleaned datasets
├── notebooks/        # EDA, preprocessing, and modeling notebooks
├── src/              # reusable preprocessing and modeling code
├── webapp/           # Flask demo application
└── paper/            # LaTeX report and figures