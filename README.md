# Flood Prediction Flask Deployment

This package wraps the supplied trained model artifacts in a Flask web application.

## Model files used
- `model/model_artifacts.pkl` — supplied sklearn models, scaler, selected feature order, and saved defaults.
- `model/cnn_feature_extractor.keras` — supplied 1D CNN feature extractor.
- `model/features.json` — exact 15 selected feature names and order.

## Run locally (Windows PowerShell)
Use Python 3.11. Create a virtual environment and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

Do not use the user's existing Python 3.14 environment for this model bundle. The pickle was created with scikit-learn 1.6.1, so this deployment pins that version. TensorFlow is pinned to a Python 3.11-compatible CPU build.

## Deploy on Render
1. Extract this ZIP and push the folder contents to a GitHub repository.
2. In Render, create a **New Web Service** and connect that repository.
3. Runtime: Python; Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy. Render reads the Python version from `runtime.txt`.

## Prediction pipeline (matches the notebook)
- SVM + Boosting: scaled 15-feature input -> LinearSVR base prediction + HistGradientBoosting residual correction on the raw selected features.
- Random Forest + Boosting: RandomForest base prediction + HistGradientBoosting residual correction on the raw selected features.
- Decision Tree + Boosting: saved AdaBoostRegressor prediction.
- CNN + Boosting: scaled 15-feature input -> shape `(1, 15, 1)` -> CNN 32-dimensional feature representation -> HistGradientBoostingRegressor.

The application calculates each of the four model outputs independently. It does **not** invent an average/ensemble score because the supplied notebook evaluates the four models separately and does not define a deployed final blend.

## Important limitations
The dataset described in the project is synthetic. The displayed 0.50 threshold is illustrative only, not a real-world flood alert threshold. Do not use this application for emergency decisions.

The original saved imputer was fitted on 20 numeric columns, while the final saved scaler and model inputs use the selected 15 features. This app follows the notebook's selected-feature inference path and does not incorrectly pass the 15-column form data through the 20-column imputer.
