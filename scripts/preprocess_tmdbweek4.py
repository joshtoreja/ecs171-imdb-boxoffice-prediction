import pandas as pd
import os


def main():
    """
    Preprocess TMDB dataset for profitability prediction.

    Steps:
    1. Load raw dataset
    2. Remove invalid rows (budget/revenue <= 0)
    3. Drop rows with missing critical features
    4. Create ROI
    5. Create binary profitability target (ROI >= 2.5)
    6. Save cleaned dataset to data/processed/
    """

    print("Loading raw dataset...")

    # Load raw dataset
    df = pd.read_csv("data/raw/TMDB_movie_dataset_v11.csv")

    print("Original shape:", df.shape)


    # Basic cleaning


    # Remove movies with zero or missing budget/revenue
    df = df[df["budget"] > 0]
    df = df[df["revenue"] > 0]

    # Drop rows with missing important features
    df = df.dropna(subset=[
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average",
        "vote_count"
    ])

    print("After cleaning:", df.shape)

    # Create ROI + target


    df["ROI"] = df["revenue"] / df["budget"]

    # Binary classification target
    df["profitable"] = (df["ROI"] >= 2.5).astype(int)

    print("Class distribution:")
    print(df["profitable"].value_counts())

    
    # Save processed dataset


    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/tmdb_clean.csv"
    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()