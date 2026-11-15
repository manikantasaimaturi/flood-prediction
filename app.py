
from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Load selected feature names
with open(MODEL_DIR / "features.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)


# NumPy pickle compatibility
import numpy.random._pickle as np_random_pickle

_original_bit_generator_ctor = np_random_pickle.__bit_generator_ctor


def _compatible_bit_generator_ctor(bit_generator_name="MT19937"):
    if isinstance(bit_generator_name, type):
        if issubclass(bit_generator_name, np.random.BitGenerator):
            return bit_generator_name()

    return _original_bit_generator_ctor(bit_generator_name)


np_random_pickle.__bit_generator_ctor = _compatible_bit_generator_ctor


# Load saved model artifacts
ARTIFACTS = joblib.load(MODEL_DIR / "model_artifacts.pkl")

DEFAULTS = ARTIFACTS["input_defaults"]


# Validate feature configuration
if FEATURES != list(ARTIFACTS["input_features"]):
    raise RuntimeError(
        "features.json and model_artifacts.pkl feature order do not match."
    )

if len(FEATURES) != 15:
    raise RuntimeError(
        f"Expected 15 selected features, found {len(FEATURES)}."
    )

if getattr(ARTIFACTS["scaler"], "n_features_in_", None) != len(FEATURES):
    raise RuntimeError(
        "Scaler feature count does not match features.json."
    )


app = Flask(__name__)

_cnn_feature_model = None

# Project demonstration threshold
THRESHOLD = 0.50

AVAILABLE_MODELS = [
    "SVM + Boosting",
    "Random Forest + Boosting",
    "Decision Tree + Boosting",
    "CNN + Boosting"
]


# Load CNN feature extractor only when required
def get_cnn_feature_model():
    global _cnn_feature_model

    if _cnn_feature_model is None:
        try:
            from tensorflow.keras.models import load_model

        except Exception as exc:
            raise RuntimeError(
                "TensorFlow is not installed or failed to load. "
                "Check requirements.txt and Python runtime."
            ) from exc

        _cnn_feature_model = load_model(
            MODEL_DIR / "cnn_feature_extractor.keras",
            compile=False
        )

    return _cnn_feature_model


# Read and validate user input
def make_input(form):
    values = []

    for feature in FEATURES:
        raw = form.get(feature, "")

        if raw is None or str(raw).strip() == "":
            raise ValueError(
                f"Please enter a value for {feature}."
            )

        try:
            value = float(raw)

        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        values.append(value)

    return pd.DataFrame(
        [values],
        columns=FEATURES
    )


# Generate predictions from all four model branches
def predict_all(row):

    scaled = ARTIFACTS["scaler"].transform(row)

    result = {}

    # 1. SVM + Boosting
    svm_base = float(
        ARTIFACTS["svm_model"].predict(scaled)[0]
    )

    svm_correction = float(
        ARTIFACTS["svm_boost"].predict(row)[0]
    )

    result["SVM + Boosting"] = (
        svm_base + svm_correction
    )


    # 2. Random Forest + Boosting
    rf_base = float(
        ARTIFACTS["rf_model"].predict(row)[0]
    )

    rf_correction = float(
        ARTIFACTS["rf_boost"].predict(row)[0]
    )

    result["Random Forest + Boosting"] = (
        rf_base + rf_correction
    )


    # 3. Decision Tree + Boosting
    result["Decision Tree + Boosting"] = float(
        ARTIFACTS["dt_boost"].predict(row)[0]
    )


    # 4. CNN + Boosting
    cnn = get_cnn_feature_model()

    cnn_input = scaled.reshape(
        1,
        len(FEATURES),
        1
    )

    latent = cnn.predict(
        cnn_input,
        verbose=0
    )

    if latent.shape[-1] != ARTIFACTS["cnn_boost"].n_features_in_:
        raise RuntimeError(
            f"CNN produced {latent.shape[-1]} features but "
            f"CNN booster expects "
            f"{ARTIFACTS['cnn_boost'].n_features_in_}."
        )

    result["CNN + Boosting"] = float(
        ARTIFACTS["cnn_boost"].predict(latent)[0]
    )

    return result


# Main application route
@app.route("/", methods=["GET", "POST"])
def index():

    values = {
        f: float(DEFAULTS.get(f, 5.0))
        for f in FEATURES
    }

    predictions = None
    error = None

    selected_model = "SVM + Boosting"

    # Selected model result variables
    selected_prediction = None
    selected_label = None
    selected_percentage = None

    if request.method == "POST":

        selected_model = request.form.get(
            "model",
            "SVM + Boosting"
        )

        # Validate selected model
        if selected_model not in AVAILABLE_MODELS:
            selected_model = "SVM + Boosting"

        try:
            # Get input values
            row = make_input(request.form)

            values = {
                f: float(row.iloc[0][f])
                for f in FEATURES
            }

            # Evaluate all four models
            predictions = predict_all(row)

            # Extract selected model prediction
            selected_prediction = float(
                predictions[selected_model]
            )

            # Classify using illustrative project threshold
            if selected_prediction >= THRESHOLD:
                selected_label = "FLOOD"
            else:
                selected_label = "NO FLOOD"

            # Convert prediction into percentage
            selected_percentage = round(
                selected_prediction * 100,
                2
            )

        except Exception as exc:
            error = str(exc)

            predictions = None
            selected_prediction = None
            selected_label = None
            selected_percentage = None


    return render_template(
        "index.html",

        features=FEATURES,
        values=values,

        predictions=predictions,
        error=error,

        selected_model=selected_model,

        selected_prediction=selected_prediction,
        selected_label=selected_label,
        selected_percentage=selected_percentage
    )


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", "5000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
