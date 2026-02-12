"""
Alexander — Track B: TMDB 5000 Movie Dataset Mini-Pipeline

Goal: Run an independent mini-pipeline on a movie dataset and assess feasibility.

Dataset: TMDB 5000 Movie Dataset (Kaggle)
    - tmdb_5000_movies.csv
    - tmdb_5000_credits.csv
Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Prediction Target: Binary classification — will a movie be "profitable"?
    (revenue > budget)

Instructions:
    1. Download the two CSVs from Kaggle and place them in  data/raw/
    2. Run:  python alexander_track_b_pipeline.py
    3. Outputs saved to  reports/figures/alex/  and  data/processed/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# setup directory structure
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures/alex")

for d in [RAW, PROCESSED, FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# load raw data
movies = pd.read_csv(RAW / "tmdb_5000_movies.csv")
credits = pd.read_csv(RAW / "tmdb_5000_credits.csv")

print(f"movies.csv  — shape: {movies.shape}")
print(f"credits.csv — shape: {credits.shape}")
print(f"\nmovies columns:\n{movies.dtypes.to_string()}")
print(f"\ncredits columns:\n{credits.dtypes.to_string()}")

# data quality assessment
print(f"\nsample size (movies): {len(movies)}")
print(f"sample size (credits): {len(credits)}")

missing_movies = movies.isnull().sum()
missing_pct = (missing_movies / len(movies) * 100).round(2)
missing_df = pd.DataFrame(
    {"missing_count": missing_movies, "missing_pct": missing_pct})
print(f"\nmissingness (movies):")
print(missing_df[missing_df["missing_count"] > 0].to_string())

missing_credits = credits.isnull().sum()
print(f"\nmissingness (credits):")
print(missing_credits[missing_credits > 0].to_string(
) if missing_credits.sum() > 0 else "no missing values in credits.")

print(f"\nfeature richness:")
print(f"  movies table: {movies.shape[1]} columns")
print(f"  credits table: {credits.shape[1]} columns")
print(
    f"  numeric features: {movies.select_dtypes(include=np.number).columns.tolist()}")
print(f"  json-embedded features: genres, keywords, production_companies, production_countries, spoken_languages")

# join feasibility — merge the two tables
print(f"\njoin keys: movies['id'] <-> credits['movie_id']")
print(f"  movies['id'] unique:       {movies['id'].nunique()}")
print(f"  credits['movie_id'] unique: {credits['movie_id'].nunique()}")

overlap = set(movies["id"]) & set(credits["movie_id"])
print(f"  overlapping IDs:            {len(overlap)}")
print(f"  join coverage:              {len(overlap)/len(movies)*100:.1f}%")

df = movies.merge(credits, left_on="id", right_on="movie_id",
                  how="inner", suffixes=("", "_credit"))
print(f"  merged shape: {df.shape}")
print("  join: success — clean inner join with high coverage.\n")

# feature engineering and target definition


def extract_names(json_str, max_items=3):
    """Extract top-N names from a JSON-encoded list of dicts."""
    try:
        items = json.loads(json_str)
        return [item["name"] for item in items[:max_items]]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def get_director(crew_json):
    """Extract director name from crew JSON."""
    try:
        crew = json.loads(crew_json)
        for member in crew:
            if member.get("job") == "Director":
                return member["name"]
    except (json.JSONDecodeError, TypeError):
        pass
    return "Unknown"


def get_crew_size(crew_json):
    """Get total crew size."""
    try:
        return len(json.loads(crew_json))
    except (json.JSONDecodeError, TypeError):
        return 0


def get_top_cast_gender_ratio(cast_json):
    """Get ratio of female cast in the top 10 billed actors."""
    try:
        cast = json.loads(cast_json)
        top = cast[:10]
        if not top:
            return 0.5
        female = sum(1 for c in top if c.get("gender") == 1)
        return female / len(top)
    except (json.JSONDecodeError, TypeError):
        return 0.5


def count_production_companies(json_str):
    """Count number of production companies."""
    try:
        return len(json.loads(json_str))
    except (json.JSONDecodeError, TypeError):
        return 0


def count_production_countries(json_str):
    """Count number of production countries."""
    try:
        return len(json.loads(json_str))
    except (json.JSONDecodeError, TypeError):
        return 0


def count_spoken_languages(json_str):
    """Count number of spoken languages."""
    try:
        return len(json.loads(json_str))
    except (json.JSONDecodeError, TypeError):
        return 0


def get_keyword_count(json_str):
    """Count number of keywords tagged to the movie."""
    try:
        return len(json.loads(json_str))
    except (json.JSONDecodeError, TypeError):
        return 0


# core features
df["genre_list"] = df["genres"].apply(extract_names)
df["primary_genre"] = df["genre_list"].apply(
    lambda x: x[0] if x else "Unknown")
df["director"] = df["crew"].apply(get_director)
df["cast_size"] = df["cast"].apply(
    lambda x: len(json.loads(x)) if pd.notna(x) else 0)
df["num_genres"] = df["genre_list"].apply(len)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year

# new features to boost accuracy
df["release_month"] = df["release_date"].dt.month
df["crew_size"] = df["crew"].apply(get_crew_size)
df["num_production_companies"] = df["production_companies"].apply(
    count_production_companies)
df["num_production_countries"] = df["production_countries"].apply(
    count_production_countries)
df["num_spoken_languages"] = df["spoken_languages"].apply(
    count_spoken_languages)
df["num_keywords"] = df["keywords"].apply(get_keyword_count)
df["cast_gender_ratio"] = df["cast"].apply(get_top_cast_gender_ratio)
df["has_homepage"] = df["homepage"].notna().astype(int)
df["overview_length"] = df["overview"].fillna("").apply(len)
df["title_length"] = df["title"].fillna("").apply(len)
df["is_english"] = (df["original_language"] == "en").astype(int)

# filter to movies with valid budget and revenue
print(f"rows before filtering (budget > 0 & revenue > 0): {len(df)}")
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
print(f"rows after filtering: {len(df)}")

# prediction target: profitable = revenue > budget
df["profitable"] = (df["revenue"] > df["budget"]).astype(int)
print(f"\ntarget distribution (profitable):")
print(df["profitable"].value_counts().to_string())
print(f"  profitable:     {df['profitable'].mean()*100:.1f}%")
print(f"  not profitable: {(1-df['profitable'].mean())*100:.1f}%")

# interaction features
df["budget_per_genre"] = df["budget"] / df["num_genres"].clip(lower=1)
df["popularity_x_votes"] = df["popularity"] * df["vote_count"]
df["budget_per_cast"] = df["budget"] / df["cast_size"].clip(lower=1)
df["runtime_x_budget"] = df["runtime"] * df["budget"]

# cleaning and encoding for modeling
feature_cols = [
    "budget", "popularity", "runtime", "vote_average", "vote_count",
    "num_genres", "cast_size", "release_year", "primary_genre",
    "release_month", "crew_size", "num_production_companies",
    "num_production_countries", "num_spoken_languages", "num_keywords",
    "cast_gender_ratio", "has_homepage", "overview_length", "title_length",
    "is_english", "budget_per_genre", "popularity_x_votes",
    "budget_per_cast", "runtime_x_budget",
]

model_df = df[feature_cols + ["profitable"]].dropna().copy()

le = LabelEncoder()
model_df["primary_genre_enc"] = le.fit_transform(model_df["primary_genre"])

model_df["log_budget"] = np.log1p(model_df["budget"])
model_df["log_popularity"] = np.log1p(model_df["popularity"])
model_df["log_vote_count"] = np.log1p(model_df["vote_count"])
model_df["log_budget_per_genre"] = np.log1p(model_df["budget_per_genre"])
model_df["log_popularity_x_votes"] = np.log1p(model_df["popularity_x_votes"])
model_df["log_budget_per_cast"] = np.log1p(model_df["budget_per_cast"])
model_df["log_runtime_x_budget"] = np.log1p(model_df["runtime_x_budget"])

final_features = [
    "log_budget", "log_popularity", "runtime", "vote_average",
    "log_vote_count", "num_genres", "cast_size", "release_year",
    "primary_genre_enc", "release_month", "crew_size",
    "num_production_companies", "num_production_countries",
    "num_spoken_languages", "num_keywords", "cast_gender_ratio",
    "has_homepage", "overview_length", "title_length", "is_english",
    "log_budget_per_genre", "log_popularity_x_votes",
    "log_budget_per_cast", "log_runtime_x_budget",
]

print(
    f"\nfinal feature set ({len(final_features)} features): {final_features}")
print(f"modeling dataset shape: {model_df.shape}")

# train-test split
X = model_df[final_features]
y = model_df["profitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\ntrain: {X_train.shape[0]}  |  test: {X_test.shape[0]}")

# model 1: random forest (tuned)
print("\n--- random forest (tuned) ---")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"accuracy: {acc_rf:.3f}")

cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(
    f"5-fold cv accuracy: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std():.3f})")
print(classification_report(y_test, y_pred_rf,
      target_names=["Not Profitable", "Profitable"]))

# model 2: gradient boosting
print("--- gradient boosting ---")
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"accuracy: {acc_gb:.3f}")

cv_scores_gb = cross_val_score(gb, X, y, cv=5, scoring="accuracy")
print(
    f"5-fold cv accuracy: {cv_scores_gb.mean():.3f} (+/- {cv_scores_gb.std():.3f})")
print(classification_report(y_test, y_pred_gb,
      target_names=["Not Profitable", "Profitable"]))

# pick the best model for plots and reporting
if acc_gb >= acc_rf:
    best_model = gb
    best_name = "Gradient Boosting"
    best_acc = acc_gb
    y_pred_best = y_pred_gb
    best_importances = pd.Series(gb.feature_importances_, index=final_features)
else:
    best_model = rf
    best_name = "Random Forest"
    best_acc = acc_rf
    y_pred_best = y_pred_rf
    best_importances = pd.Series(rf.feature_importances_, index=final_features)

print(f"\nbest model: {best_name} — accuracy: {best_acc:.3f}")

# generate eda plots

# budget vs revenue colored by profitability
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    np.log1p(df["budget"]), np.log1p(df["revenue"]),
    c=df["profitable"], cmap="RdYlGn", alpha=0.5, s=20, edgecolors="none"
)
ax.plot([0, 22], [0, 22], "k--", alpha=0.4, label="Break-even line")
ax.set_xlabel("Log(Budget)", fontsize=12)
ax.set_ylabel("Log(Revenue)", fontsize=12)
ax.set_title("Budget vs Revenue (Green = Profitable)", fontsize=14)
ax.legend()
plt.colorbar(scatter, ax=ax, label="Profitable")
fig.tight_layout()
fig.savefig(FIGURES / "01_budget_vs_revenue.png")
plt.close(fig)
print("saved: 01_budget_vs_revenue.png")

# top 15 genres by count and profitability rate
genre_stats = df.groupby("primary_genre").agg(
    count=("profitable", "size"),
    profit_rate=("profitable", "mean")
).sort_values("count", ascending=False).head(15)

fig, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(genre_stats.index,
               genre_stats["count"], color="steelblue", alpha=0.7, label="Count")
ax1.set_ylabel("Number of Movies", fontsize=12, color="steelblue")
ax1.tick_params(axis="x", rotation=45)

ax2 = ax1.twinx()
ax2.plot(genre_stats.index, genre_stats["profit_rate"]
         * 100, "ro-", markersize=8, label="Profit Rate %")
ax2.set_ylabel("Profitability Rate (%)", fontsize=12, color="red")
ax2.set_ylim(0, 100)

ax1.set_title("Top 15 Genres: Volume & Profitability Rate", fontsize=14)
fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
fig.tight_layout()
fig.savefig(FIGURES / "02_genre_profitability.png")
plt.close(fig)
print("saved: 02_genre_profitability.png")

# feature importance from best model (top 15)
top_importances = best_importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(8, 6))
top_importances.plot(kind="barh", ax=ax, color="teal")
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title(f"{best_name} — Top 15 Feature Importances", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURES / "03_feature_importance.png")
plt.close(fig)
print("saved: 03_feature_importance.png")

# confusion matrix for best model
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Profitable", "Profitable"],
            yticklabels=["Not Profitable", "Profitable"], ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(
    f"Confusion Matrix — {best_name} (Accuracy: {best_acc:.1%})", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURES / "04_confusion_matrix.png")
plt.close(fig)
print("saved: 04_confusion_matrix.png")

# save processed data
save_cols = ["id", "title", "budget", "revenue", "popularity", "runtime",
             "vote_average", "vote_count", "primary_genre", "director",
             "cast_size", "num_genres", "release_year", "profitable"]
df[save_cols].to_csv(PROCESSED / "tmdb_5000_cleaned.csv", index=False)
print(f"saved processed data: {PROCESSED / 'tmdb_5000_cleaned.csv'}")

# dataset assessment notes
notes = f"""
(Mostly Somewhat AI Generated full disclosure)
Dataset Assessment Notes — TMDB 5000 Movie Dataset

Model performance:
  Random Forest accuracy:      {acc_rf:.3f}  (5-fold CV: {cv_scores_rf.mean():.3f})
  Gradient Boosting accuracy:  {acc_gb:.3f}  (5-fold CV: {cv_scores_gb.mean():.3f})
  Best model selected:         {best_name}

Pros:
  Two-table structure (movies + credits) enables a real join exercise.
  Rich features: budget, revenue, genres, cast, crew, popularity, ratings.
  Manageable size (~4,800 rows) — fast iteration, no memory issues.
  Clean join: near-100% ID overlap between the two tables.
  Well-documented and widely used in the data science community.

Cons:
  Many movies have budget=0 or revenue=0 (not truly zero — just missing).
  After filtering, usable rows drop to ~3,000 from ~4,800.
  JSON-embedded columns (genres, cast, crew, keywords) require parsing,
  which adds preprocessing complexity.
  No streaming/VOD revenue — only theatrical box office.
  Data skews toward English-language Hollywood films.
  Temporal coverage is uneven (more recent films overrepresented).

Join difficulty:
  Easy. Clean integer ID match between movies.id and credits.movie_id.
  Near-perfect coverage (~99%+). No fuzzy matching needed.

Feature limitations:
  No marketing spend, social media buzz, or pre-release hype metrics.
  Cast/crew encoded as JSON — extracting "star power" requires extra work.
  Genre is multi-label (stored as list); we simplified to primary genre only.
  No audience demographics or regional box office breakdowns.

Deployment friendliness:
  Model is lightweight (Random Forest / Gradient Boosting) — easy to serialize with joblib.
  Feature pipeline is straightforward (log transforms + label encoding).
  Main deployment risk: budget and revenue are only known post-release,
  so a production model would need proxy features (e.g., studio, cast,
  genre, release month) available before release for true prediction.
  Retraining would require periodic data refreshes from TMDB API.
"""

print(notes)
print("\npipeline complete — check reports/figures/alex/ for all plots.")
