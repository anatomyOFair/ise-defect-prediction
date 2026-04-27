"""Hyperparameter tuning for RF and XGB using RandomizedSearchCV.

Pools all datasets within each family, runs randomised search with
stratified 5-fold CV, optimising for F1 on the defective class.
Saves best parameters to results/best_params.json.

Usage:
    python src/tune.py                      # tune all three families
    python src/tune.py --family promise-ck  # one family only
    python src/tune.py --n-iter 40          # more thorough (slower)

After running, hand results/best_params.json back for analysis.
Expected runtime: ~10 min (default 20 iter) to ~25 min (40 iter).
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_dataset

warnings.filterwarnings("ignore")

DATASETS = {
    "promise-ck": {
        "ant-1.7":      "data/ant-1.7.csv",
        "camel-1.6":    "data/camel-1.6.csv",
        "ivy-2.0":      "data/ivy-2.0.csv",
        "jedit-4.3":    "data/jedit-4.3.csv",
        "lucene-2.4":   "data/lucene-2.4.csv",
        "poi-3.0":      "data/poi-3.0.csv",
        "synapse-1.2":  "data/synapse-1.2.csv",
        "xalan-2.6":    "data/xalan-2.6.csv",
        "xerces-1.4":   "data/xerces-1.4.csv",
        "velocity-1.6": "data/velocity-1.6.csv",
    },
    "aeeem": {
        "EQ":  "data/aeeem/equinox.csv",
        "JDT": "data/aeeem/jdt.csv",
        "LC":  "data/aeeem/lucene.csv",
        "ML":  "data/aeeem/mylyn.csv",
        "PDE": "data/aeeem/pde.csv",
    },
    "nasa": {
        "CM1": "data/nasa/cm1.csv",
        "JM1": "data/nasa/jm1.csv",
        "KC1": "data/nasa/kc1.csv",
        "MW1": "data/nasa/mw1.csv",
        "PC1": "data/nasa/pc1.csv",
    },
}

RF_PARAM_DIST = {
    "n_estimators":    [50, 100, 200, 300, 500],
    "max_depth":       [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":    ["sqrt", "log2", None],
}

XGB_PARAM_DIST = {
    "n_estimators":      [50, 100, 200, 300],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth":         [3, 5, 7, 10],
    "subsample":         [0.6, 0.8, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "min_child_weight":  [1, 3, 5, 10],
}

LGB_PARAM_DIST = {
    "n_estimators":       [50, 100, 200, 300],
    "learning_rate":      [0.01, 0.05, 0.1, 0.2],
    "max_depth":          [3, 5, 7, 10, -1],
    "num_leaves":         [15, 31, 63, 127],
    "subsample":          [0.6, 0.8, 1.0],
    "colsample_bytree":   [0.6, 0.8, 1.0],
    "min_child_samples":  [5, 10, 20],
}


def load_family(family: str, base: str) -> tuple[pd.DataFrame, pd.Series]:
    """Pool all available datasets in a family into one frame."""
    Xs, ys = [], []
    for name, path in DATASETS[family].items():
        full = os.path.join(base, path)
        if not os.path.exists(full):
            print(f"  SKIP {name}: not found")
            continue
        X, y = load_dataset(full)
        Xs.append(X)
        ys.append(y)

    if not Xs:
        raise FileNotFoundError(f"No datasets found for family '{family}'")

    common_cols = sorted(
        set(Xs[0].columns).intersection(*[set(X.columns) for X in Xs[1:]])
    )
    X_all = pd.concat([X[common_cols] for X in Xs], ignore_index=True)
    y_all = pd.concat(ys, ignore_index=True)
    return X_all, y_all


def tune_family(family: str, n_iter: int, cv: int, base: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Tuning: {family}  ({n_iter} iterations, {cv}-fold CV)")
    print(f"{'='*60}")

    X, y = load_family(family, base)
    pos = int(y.sum())
    neg = int((y == 0).sum())
    spw = neg / pos if pos > 0 else 1.0

    print(f"  Pooled: {len(X)} instances, {pos} defective ({100 * y.mean():.1f}%)")

    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    # --- Random Forest ---
    print("\n  Tuning RF...")
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf_search = RandomizedSearchCV(
        rf_base,
        RF_PARAM_DIST,
        n_iter=n_iter,
        scoring="f1",
        cv=cv_strat,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X, y)
    results["rf"] = rf_search.best_params_
    results["rf_cv_f1"] = round(rf_search.best_score_, 4)
    print(f"  RF  best CV F1 : {rf_search.best_score_:.4f}")
    print(f"  RF  best params: {rf_search.best_params_}")

    # --- XGBoost ---
    print("\n  Tuning XGB...")
    xgb_base = XGBClassifier(
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
        n_jobs=1,
    )
    xgb_search = RandomizedSearchCV(
        xgb_base,
        XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring="f1",
        cv=cv_strat,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search.fit(X, y)
    results["xgb"] = xgb_search.best_params_
    results["xgb_cv_f1"] = round(xgb_search.best_score_, 4)
    print(f"  XGB best CV F1 : {xgb_search.best_score_:.4f}")
    print(f"  XGB best params: {xgb_search.best_params_}")

    # --- LightGBM ---
    print("\n  Tuning LGB...")
    lgb_base = LGBMClassifier(
        scale_pos_weight=spw,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    lgb_search = RandomizedSearchCV(
        lgb_base,
        LGB_PARAM_DIST,
        n_iter=n_iter,
        scoring="f1",
        cv=cv_strat,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    lgb_search.fit(X, y)
    results["lgb"] = lgb_search.best_params_
    results["lgb_cv_f1"] = round(lgb_search.best_score_, 4)
    print(f"  LGB best CV F1 : {lgb_search.best_score_:.4f}")
    print(f"  LGB best params: {lgb_search.best_params_}")

    return results


def run(family_filter: str | None = None, n_iter: int = 20,
        cv: int = 5, results_dir: str = "results") -> None:
    base = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base, results_dir)
    os.makedirs(results_path, exist_ok=True)
    params_path = os.path.join(results_path, "best_params.json")

    # Load existing results so partial runs don't overwrite prior families
    best_params: dict = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            best_params = json.load(f)

    families = [family_filter] if family_filter else list(DATASETS.keys())

    for family in families:
        if family not in DATASETS:
            print(f"Unknown family '{family}'. Options: {list(DATASETS.keys())}")
            continue
        best_params[family] = tune_family(family, n_iter, cv, base)
        # Save after each family so a crash doesn't lose earlier work
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n  Saved to {params_path}")

    print(f"\n{'='*60}")
    print("DONE — full best_params.json:")
    print(f"{'='*60}")
    print(json.dumps(best_params, indent=2))
    print(f"\nFile: {params_path}")
    print("Hand this file (or the JSON above) back for analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune RF and XGB hyperparameters.")
    parser.add_argument("--family",      type=str, default=None,
                        help="Family to tune: promise-ck, aeeem, nasa (default: all)")
    parser.add_argument("--n-iter",      type=int, default=20,
                        help="RandomizedSearchCV iterations per model (default: 20)")
    parser.add_argument("--cv",          type=int, default=5,
                        help="CV folds (default: 5)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Output directory (default: results)")
    args = parser.parse_args()
    run(args.family, args.n_iter, args.cv, args.results_dir)
