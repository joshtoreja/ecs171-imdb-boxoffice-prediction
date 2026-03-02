from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

RAW_PATH = Path("../data/raw/TMDB_movie_dataset_v11.csv")
OUT_DIR = Path("../reports/alex_week3_outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BASE_THRESHOLD = 2.5

print("Loading data...")
df = pd.read_csv(RAW_PATH)
df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
df = df[df["budget"].notna() & df["revenue"].notna()]
df = df[(df["budget"] > 0) & (df["revenue"] >= 0)]
df["roi"] = df["revenue"] / df["budget"]
df["profitable"] = (df["roi"] >= BASE_THRESHOLD).astype(int)

if len(df) > 200_000:
    df = df.sample(n=200_000, random_state=RANDOM_STATE)

print("Creating balanced sample...")
pos = df[df["profitable"] == 1].sample(
    n=min(len(df[df["profitable"] == 1]), 15_000), random_state=RANDOM_STATE)
neg = df[df["profitable"] == 0].sample(
    n=min(len(df[df["profitable"] == 0]), 15_000), random_state=RANDOM_STATE)
df_sample = pd.concat([pos, neg]).copy()

print(f"Sample shape: {df_sample.shape}")
print(df_sample["profitable"].value_counts())

print("Engineering features...")
df_sample["log_budget"] = np.log1p(df_sample["budget"])
df_sample["popularity_x_votes"] = df_sample["popularity"] * \
    df_sample["vote_count"]
df_sample["release_month"] = pd.to_datetime(
    df_sample["release_date"], errors="coerce").dt.month
df_sample["has_homepage"] = df_sample["homepage"].notna().astype(
    int) if "homepage" in df_sample.columns else 0

# Feature sets
FULL_FEATURES = ["budget", "runtime", "popularity", "vote_count",
                 "log_budget", "popularity_x_votes", "vote_average",
                 "release_month", "has_homepage"]
ABLATION_FEATURES = [f for f in FULL_FEATURES if f not in (
    "log_budget", "popularity_x_votes")]
PRE_RELEASE_FEATURES = [f for f in FULL_FEATURES if f not in (
    "vote_average", "vote_count", "popularity", "popularity_x_votes")]
BASELINE_FEATURES = ["budget", "runtime", "popularity", "vote_count"]


def evaluate_rf(X, y, label="model"):
    X = X.fillna(X.median())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                random_state=RANDOM_STATE, n_jobs=-1)
    cv = cross_validate(rf, X, y, cv=skf, scoring={
                        "acc": "accuracy", "f1": "f1", "auc": "roc_auc"})

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    rf_full = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    rf_full.fit(X_tr, y_tr)
    probs = rf_full.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "label": label, "n_features": X.shape[1],
        "cv_acc": cv["test_acc"].mean(), "cv_f1": cv["test_f1"].mean(), "cv_auc": cv["test_auc"].mean(),
        "holdout_acc": accuracy_score(y_te, preds), "holdout_f1": f1_score(y_te, preds),
        "holdout_auc": roc_auc_score(y_te, probs),
    }, rf_full


y = df_sample["profitable"]

# Task 1: Feature Importance
print("\n--- FEATURE IMPORTANCES ---")
print("\nTraining full model for feature importances...")
full_metrics, rf_full_model = evaluate_rf(
    df_sample[FULL_FEATURES], y, "Full Model")
importances = pd.Series(rf_full_model.feature_importances_,
                        index=FULL_FEATURES).sort_values(ascending=False)
print(importances.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Feature Importances")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIG_DIR / "feature_importances.png", dpi=200)
print(f"Saved: {FIG_DIR / 'feature_importances.png'}")

# Task 2: Ablation
print("\n--- ABLATION TEST (removed: log_budget, popularity_x_votes) ---")
print("\nTraining ablation model...")
abl_metrics, _ = evaluate_rf(df_sample[ABLATION_FEATURES], y, "Ablation")
ablation_table = pd.DataFrame([full_metrics, abl_metrics])
print(ablation_table.to_string(index=False, float_format="%.4f"))

# Task 3: Pre-Release
print("\n--- PRE-RELEASE TEST (removed: vote_average, vote_count, popularity, popularity_x_votes) ---")
print("\nTraining pre-release model...")
pre_metrics, _ = evaluate_rf(df_sample[PRE_RELEASE_FEATURES], y, "Pre-Release")
print("Training baseline model...")
base_metrics, _ = evaluate_rf(df_sample[BASELINE_FEATURES], y, "Baseline")
comparison_table = pd.DataFrame([full_metrics, base_metrics, pre_metrics])
print(comparison_table.to_string(index=False, float_format="%.4f"))

print("\nSaving CSVs and figure...")
importances.to_csv(OUT_DIR / "feature_importances.csv", header=["importance"])
ablation_table.to_csv(OUT_DIR / "ablation_comparison.csv", index=False)
comparison_table.to_csv(OUT_DIR / "prerelease_comparison.csv", index=False)

print(f"\nOutputs saved to: {OUT_DIR.resolve()}")
print("Done!")
