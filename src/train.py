"""Training and evaluation pipeline.

Usage:
    python src/train.py                     # run all families
    python src/train.py --family promise-ck # run one family
    python src/train.py --dataset ant-1.7   # run one dataset
    python src/train.py --results-dir results/
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from models import make_lr, make_rf, make_xgb
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
        "EQ":    "data/aeeem/equinox.csv",
        "JDT":   "data/aeeem/jdt.csv",
        "LC":    "data/aeeem/lucene.csv",
        "ML":    "data/aeeem/mylyn.csv",
        "PDE":   "data/aeeem/pde.csv",
    },
    "nasa": {
        "CM1": "data/nasa/cm1.csv",
        "JM1": "data/nasa/jm1.csv",
        "KC1": "data/nasa/kc1.csv",
        "MW1": "data/nasa/mw1.csv",
        "PC1": "data/nasa/pc1.csv",
    },
}

N_REPEATS = 30
TEST_SIZE = 0.30


def evaluate_dataset(name: str, path: str, family: str,
                     n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """Run n_repeats train/test splits and return per-repeat metrics."""
    base = os.path.dirname(os.path.dirname(__file__))
    X, y = load_dataset(os.path.join(base, path))

    pos = y.sum()
    neg = (y == 0).sum()
    spw = neg / pos if pos > 0 else 1.0

    records = []
    for seed in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        for model_name, model in [
            ("LR",  make_lr(seed)),
            ("RF",  make_rf(seed)),
            ("XGB", make_xgb(spw, seed)),
        ]:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            records.append({
                "family":    family,
                "dataset":   name,
                "model":     model_name,
                "seed":      seed,
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall":    recall_score(y_te, y_pred, zero_division=0),
                "f1":        f1_score(y_te, y_pred, zero_division=0),
            })

    return pd.DataFrame(records)


def wilcoxon_test(df: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    """Return (statistic, p-value) for Wilcoxon signed-rank on F1 scores.
    Returns (0.0, 1.0) when all differences are zero (no evidence of difference).
    """
    a = df[df["model"] == model_a]["f1"].values
    b = df[df["model"] == model_b]["f1"].values
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        if np.isnan(p):
            return 0.0, 1.0
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of metrics per family/dataset/model."""
    agg = (
        df.groupby(["family", "dataset", "model"])[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .round(3)
    )
    return agg


def run(family_filter: str | None = None, dataset_filter: str | None = None,
        results_dir: str = "results") -> None:
    base = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base, results_dir)
    os.makedirs(results_path, exist_ok=True)

    # Determine which families/datasets to run
    if dataset_filter:
        families_to_run = {
            fam: {ds: path for ds, path in datasets.items() if ds == dataset_filter}
            for fam, datasets in DATASETS.items()
        }
        families_to_run = {f: d for f, d in families_to_run.items() if d}
    elif family_filter:
        if family_filter not in DATASETS:
            print(f"Unknown family '{family_filter}'. Available: {list(DATASETS.keys())}")
            return
        families_to_run = {family_filter: DATASETS[family_filter]}
    else:
        families_to_run = DATASETS

    all_records = []
    wilcoxon_rows = []

    for family, datasets in families_to_run.items():
        print(f"\n{'#'*60}")
        print(f"Family: {family}")
        print(f"{'#'*60}")

        for name, path in datasets.items():
            full_path = os.path.join(base, path)
            if not os.path.exists(full_path):
                print(f"  SKIP {name}: file not found ({path})")
                continue

            print(f"\n{'='*50}")
            print(f"Dataset: {name}  ({family})")
            df = evaluate_dataset(name, path, family)
            all_records.append(df)

            summary = summarise(df)
            print(summary.to_string())

            for comp in [("RF", "LR"), ("XGB", "LR")]:
                stat, p = wilcoxon_test(df, comp[0], comp[1])
                sig = "*" if p < 0.05 else ""
                print(f"Wilcoxon {comp[0]} vs {comp[1]}: stat={stat:.1f}, p={p:.4f} {sig}")
                wilcoxon_rows.append({
                    "family":     family,
                    "dataset":    name,
                    "comparison": f"{comp[0]} vs {comp[1]}",
                    "statistic":  round(stat, 1),
                    "p_value":    round(p, 4),
                    "significant": p < 0.05,
                })

    if not all_records:
        print("No datasets were run.")
        return

    all_df = pd.concat(all_records, ignore_index=True)

    out_path = os.path.join(results_path, "results.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    summary_path = os.path.join(results_path, "summary.csv")
    summarise(all_df).to_csv(summary_path)
    print(f"Summary saved to {summary_path}")

    wilcoxon_path = os.path.join(results_path, "wilcoxon.csv")
    pd.DataFrame(wilcoxon_rows).to_csv(wilcoxon_path, index=False)
    print(f"Wilcoxon results saved to {wilcoxon_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defect prediction experiments.")
    parser.add_argument("--family",      type=str, default=None,
                        help="Dataset family to run: promise-ck, aeeem, nasa")
    parser.add_argument("--dataset",     type=str, default=None,
                        help="Single dataset name to run")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()
    run(args.family, args.dataset, args.results_dir)
