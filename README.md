# PM2.5 One-Step Forecasting Dashboard

This project connects the dashboard to the supplied `air_quality.csv` dataset and trained models.

## Forecast definition

Given observations through time **T**, the system predicts PM2.5 at **T+1** only.

## Models

Ridge, Lasso, Linear Regression, Gradient Boosting, Random Forest, XGBoost, Extra Trees, RNN, KNN, SVR, LSTM, GRU, CNN-LSTM, CNN, MLP, Decision Tree.

## Run on Windows

Open a terminal in this folder and install dependencies:

```bash
pip install -r requirements.txt
```

Then start the backend:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000
```

Do **not** open `index.html` by double-clicking it; the page should be served by FastAPI so the JavaScript can call `/api/...`.

## Project structure

```text
project/
├── app.py
├── index.html
├── style.css
├── script.js
├── air_quality.csv
├── requirements.txt
└── model_artifacts/
    ├── *.joblib
    ├── *.pt
    ├── metrics.json
    ├── test_predictions.csv
    ├── shap_global.csv
    └── shap_local.csv
```

## Notes

The model artifacts were trained from the supplied dataset using a chronological train/validation/test split. The dashboard reads the saved test metrics and predictions rather than using placeholder numbers.
