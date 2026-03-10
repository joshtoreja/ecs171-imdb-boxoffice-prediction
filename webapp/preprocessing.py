import numpy as np
import pandas as pd
from datetime import datetime

NUMERICAL = [
    "log_budget", "runtime", "release_year",
    "is_summer", "is_holiday", "long_movie",
]
CATEGORICAL = [
    "genres", "production_companies", "production_countries",
]


def preprocess_input(form_data: dict) -> pd.DataFrame:
    """
    Take the raw form dictionary and return a single-row DataFrame
    with the 9 columns the pipeline expects.
    """
    budget = float(form_data.get("budget", 0))
    runtime = float(form_data.get("runtime", 0))
    release_season = form_data.get("release_season", "Other")

    genres = form_data.get("genres", "Drama")
    production_companies = form_data.get("production_companies", "Other")
    production_countries = form_data.get("production_countries", "United States of America")

    log_budget = np.log1p(budget)
    release_year = datetime.now().year
    is_summer = 1 if release_season == "Summer" else 0
    is_holiday = 1 if release_season == "Holiday" else 0
    long_movie = 1 if runtime > 120 else 0

    row = {
        "log_budget": log_budget,
        "runtime": runtime,
        "release_year": release_year,
        "is_summer": is_summer,
        "is_holiday": is_holiday,
        "long_movie": long_movie,
        "genres": genres,
        "production_companies": production_companies,
        "production_countries": production_countries,
    }

    return pd.DataFrame([row], columns=NUMERICAL + CATEGORICAL)