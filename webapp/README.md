# Movie Profitability Predictor — Flask App

ECS 171 Final Project · Gradient Boosting on TMDB Data · ROI ≥ 2.5× threshold

## Project Structure

```
app/
├── app.py                    # Flask backend (routes + prediction endpoint)
├── preprocessing.py          # Converts form input → model feature vector
├── final_model.pkl           # Trained Gradient Boosting model
├── templates/
│   └── index.html            # Frontend UI
├── static/
│   └── style.css             # Stylesheet
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run the app
uv run python app.py
```

Then open **http://localhost:5000** in your browser.