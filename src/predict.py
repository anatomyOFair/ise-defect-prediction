"""CLI prediction tool — given a CSV of code metrics, output defect predictions.

Usage:
    python src/predict.py --input my_module_metrics.csv --model xgb-hp
    python src/predict.py --input my_module_metrics.csv --model rf-hp --family aeeem

Supported models: lr, rf, xgb, lgb, stacking, rf-hp, xgb-hp, lgb-hp, rf-smote.
The tool trains the chosen model on all datasets in the chosen family, then
predicts on the input file.
"""

import argparse
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from models import (make_lr, make_rf, make_xgb, make_lgbm, make_stacking,
                    make_rf_tuned, make_xgb_tuned, make_lgbm_tuned, make_rf_smote)
from utils import load_dataset

MODEL_CHOICES = ["lr", "rf", "xgb", "lgb", "stacking",
                 "rf-hp", "xgb-hp", "lgb-hp", "rf-smote"]

warnings.filterwarnings("ignore")

FAMILY_DATASETS = {
    "promise-ck": [
        "data/ant-1.7.csv", "data/camel-1.6.csv", "data/ivy-2.0.csv",
        "data/jedit-4.3.csv", "data/lucene-2.4.csv", "data/poi-3.0.csv",
        "data/synapse-1.2.csv", "data/xalan-2.6.csv", "data/xerces-1.4.csv",
        "data/velocity-1.6.csv",
    ],
    "aeeem": [
        "data/aeeem/equinox.csv", "data/aeeem/jdt.csv", "data/aeeem/lucene.csv",
        "data/aeeem/mylyn.csv", "data/aeeem/pde.csv",
    ],
    "nasa": [
        "data/nasa/cm1.csv", "data/nasa/jm1.csv", "data/nasa/kc1.csv",
        "data/nasa/mw1.csv", "data/nasa/pc1.csv",
    ],
}


def train_on_all(model_name: str, family: str = "promise-ck"):
    """Train a model on the union of all datasets in the given family."""
    base = os.path.dirname(os.path.dirname(__file__))
    Xs, ys = [], []
    for path in FAMILY_DATASETS.get(family, FAMILY_DATASETS["promise-ck"]):
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

    makers = {
        "lr":       lambda: make_lr(),
        "rf":       lambda: make_rf(),
        "xgb":      lambda: make_xgb(spw),
        "lgb":      lambda: make_lgbm(spw),
        "stacking": lambda: make_stacking(spw),
        "rf-hp":    lambda: make_rf_tuned(family),
        "xgb-hp":   lambda: make_xgb_tuned(family, spw),
        "lgb-hp":   lambda: make_lgbm_tuned(family, spw),
        "rf-smote": lambda: make_rf_smote(),
    }
    if model_name not in makers:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_CHOICES)}")
    model = makers[model_name]()

    model.fit(X_all, y_all)
    return model, X_all.columns.tolist()


def predict(input_path: str, model_name: str, family: str = "promise-ck") -> None:
    import re
    model, train_cols = train_on_all(model_name, family)

    df = pd.read_csv(input_path)
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', c.strip().lower()) for c in df.columns]

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
    parser.add_argument("--model", choices=MODEL_CHOICES, default="xgb-hp",
                        help="Model to use. Default is xgb-hp (best within-project performer).")
    parser.add_argument("--family", choices=["promise-ck", "aeeem", "nasa"], default="promise-ck",
                        help="Dataset family to train on (default: promise-ck)")
    args = parser.parse_args()
    predict(args.input, args.model, args.family)
