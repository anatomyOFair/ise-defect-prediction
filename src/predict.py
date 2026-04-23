"""CLI prediction tool — given a CSV of code metrics, output defect predictions.

Usage:
    python src/predict.py --input my_module_metrics.csv --model rf
    python src/predict.py --input my_module_metrics.csv --model xgb
    python src/predict.py --input my_module_metrics.csv --model lr

The tool trains on all available datasets and predicts on the input file.
"""

import argparse
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models import make_lr, make_rf, make_xgb
from utils import load_dataset

warnings.filterwarnings("ignore")

DATASETS = [
    "data/ant-1.7.csv",
    "data/camel-1.6.csv",
    "data/ivy-2.0.csv",
    "data/jedit-4.3.csv",
]


def train_on_all(model_name: str):
    """Train a model on the union of all available datasets."""
    base = os.path.dirname(os.path.dirname(__file__))
    Xs, ys = [], []
    for path in DATASETS:
        full = os.path.join(base, path)
        if os.path.exists(full):
            X, y = load_dataset(full)
            Xs.append(X)
            ys.append(y)

    import pandas as pd
    import numpy as np
    X_all = pd.concat(Xs, ignore_index=True)
    y_all = pd.concat(ys, ignore_index=True)

    pos = y_all.sum()
    neg = (y_all == 0).sum()
    spw = neg / pos if pos > 0 else 1.0

    if model_name == "rf":
        model = make_rf()
    elif model_name == "xgb":
        model = make_xgb(spw)
    else:
        model = make_lr()

    model.fit(X_all, y_all)
    return model, X_all.columns.tolist()


def predict(input_path: str, model_name: str) -> None:
    model, train_cols = train_on_all(model_name)

    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Keep only columns the model was trained on
    missing = [c for c in train_cols if c not in df.columns]
    if missing:
        print(f"Warning: missing columns filled with 0: {missing}")
    for col in missing:
        df[col] = 0

    X = df[train_cols].fillna(0)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    df["predicted_defective"] = preds
    df["defect_probability"] = proba.round(3)

    out = df[["predicted_defective", "defect_probability"]]
    print(out.to_string(index=True))

    out_path = input_path.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict software defects from code metrics.")
    parser.add_argument("--input", required=True, help="CSV file with code metrics")
    parser.add_argument("--model", choices=["lr", "rf", "xgb"], default="rf",
                        help="Model to use: lr (Logistic Regression), rf (Random Forest), xgb (XGBoost)")
    args = parser.parse_args()
    predict(args.input, args.model)
