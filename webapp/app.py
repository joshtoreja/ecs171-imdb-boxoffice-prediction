"""
app.py
Flask backend for the Movie Profitability Predictor.
Loads the trained Gradient Boosting model and serves predictions.
"""

import os
import joblib
from flask import Flask, render_template, request, jsonify
from preprocessing import preprocess_input

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # --- Basic validation ---
        errors = []
        budget = data.get("budget")
        runtime = data.get("runtime")

        if budget is None or budget == "":
            errors.append("Budget is required.")
        elif float(budget) <= 0:
            errors.append("Budget must be positive.")

        if runtime is None or runtime == "":
            errors.append("Runtime is required.")
        elif float(runtime) <= 0:
            errors.append("Runtime must be positive.")

        if not data.get("genres"):
            errors.append("Genre is required.")

        if not data.get("production_countries"):
            errors.append("Country is required.")

        if data.get("release_season") not in ("Summer", "Holiday", "Other"):
            errors.append("Invalid release season.")

        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        features = preprocess_input(data)
        proba = model.predict_proba(features)[0]  # [P(0), P(1)]
        prob_profitable = float(proba[1])
        label = "Likely Profitable" if prob_profitable >= 0.5 else "High Risk"

        return jsonify({
            "success": True,
            "probability": round(prob_profitable * 100, 1),
            "label": label,
        })

    except Exception as exc:
        return jsonify({"success": False, "errors": [str(exc)]}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)