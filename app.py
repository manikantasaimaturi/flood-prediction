from pathlib import Path
import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

with open(MODEL_DIR / "features.json", "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

import numpy as np
import numpy.random._pickle as np_random_pickle

_original_bit_generator_ctor = np_random_pickle.__bit_generator_ctor

def _compatible_bit_generator_ctor(bit_generator_name="MT19937"):
    if isinstance(bit_generator_name, type):
        if issubclass(bit_generator_name, np.random.BitGenerator):
            return bit_generator_name()
    return _original_bit_generator_ctor(bit_generator_name)

np_random_pickle.__bit_generator_ctor = _compatible_bit_generator_ctor

ARTIFACTS = joblib.load(MODEL_DIR / "model_artifacts.pkl")
DEFAULTS = ARTIFACTS["input_defaults"]

# The saved scaler and the four prediction branches were trained on this exact
# 15-feature order. The saved imputer was fitted on the original 20-column
# matrix, so it must not be applied to this 15-column selected-feature input.
if FEATURES != list(ARTIFACTS["input_features"]):
    raise RuntimeError("features.json and model_artifacts.pkl feature order do not match.")
if len(FEATURES) != 15:
    raise RuntimeError(f"Expected 15 selected features, found {len(FEATURES)}.")
if getattr(ARTIFACTS["scaler"], "n_features_in_", None) != len(FEATURES):
    raise RuntimeError("Scaler feature count does not match features.json.")

app = Flask(__name__)

_cnn_feature_model = None

def get_cnn_feature_model():
    global _cnn_feature_model
    if _cnn_feature_model is None:
        try:
            from tensorflow.keras.models import load_model
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow is not installed or failed to load. Install requirements.txt "
                "using the supported Python 3.11 runtime."
            ) from exc
        _cnn_feature_model = load_model(
            MODEL_DIR / "cnn_feature_extractor.keras",
            compile=False
        )
    return _cnn_feature_model

def make_input(form):
    values = []
    for feature in FEATURES:
        raw = form.get(feature, "")
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Please enter a value for {feature}.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be a number.")
        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")
        values.append(value)
    return pd.DataFrame([values], columns=FEATURES)

def predict_all(row):
    scaled = ARTIFACTS["scaler"].transform(row)
    result = {}

    # SVM base prediction + residual correction, exactly as in the notebook.
    svm_base = float(ARTIFACTS["svm_model"].predict(scaled)[0])
    svm_correction = float(ARTIFACTS["svm_boost"].predict(row)[0])
    result["SVM + Boosting"] = svm_base + svm_correction

    # Random Forest base prediction + out-of-fold residual correction.
    rf_base = float(ARTIFACTS["rf_model"].predict(row)[0])
    rf_correction = float(ARTIFACTS["rf_boost"].predict(row)[0])
    result["Random Forest + Boosting"] = rf_base + rf_correction

    # Decision Tree weak learners combined by AdaBoost.
    result["Decision Tree + Boosting"] = float(ARTIFACTS["dt_boost"].predict(row)[0])

    # CNN feature extractor produces 32 values; HistGradientBoosting expects 32.
    cnn = get_cnn_feature_model()
    cnn_input = scaled.reshape(1, len(FEATURES), 1)
    latent = cnn.predict(cnn_input, verbose=0)
    if latent.shape[-1] != ARTIFACTS["cnn_boost"].n_features_in_:
        raise RuntimeError(
            f"CNN produced {latent.shape[-1]} features but CNN booster expects "
            f"{ARTIFACTS['cnn_boost'].n_features_in_}."
        )
    result["CNN + Boosting"] = float(ARTIFACTS["cnn_boost"].predict(latent)[0])
    return result

@app.route("/", methods=["GET", "POST"])
def index():
    values = {f: float(DEFAULTS.get(f, 5.0)) for f in FEATURES}
    predictions = None
    error = None
    selected_model = "SVM + Boosting"

    if request.method == "POST":
        selected_model = request.form.get("model", "SVM + Boosting")
        try:
            row = make_input(request.form)
            values = {f: float(row.iloc[0][f]) for f in FEATURES}
            predictions = predict_all(row)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        features=FEATURES,
        values=values,
        predictions=predictions,
        error=error,
        selected_model=selected_model
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
