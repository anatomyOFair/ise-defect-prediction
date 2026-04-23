"""Training and evaluation pipeline.

Usage:
    python src/train.py                  # run all datasets
    python src/train.py --dataset ant-1.7
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

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from models import make_lr, make_rf, make_xgb
from utils import defect_rate, load_dataset

warnings.filterwarnings("ignore")

DATASETS = {
    "ant-1.7": "data/ant-1.7.csv",
    "camel-1.6": "data/camel-1.6.csv",
    "ivy-2.0": "data/ivy-2.0.csv",
    "jedit-4.3": "data/jedit-4.3.csv",
}

N_REPEATS = 30
TEST_SIZE = 0.30


def evaluate_dataset(name: str, path: str, n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """Run n_repeats train/test splits and return per-repeat metrics."""
    base = os.path.dirname(os.path.dirname(__file__))
    X, y = load_dataset(os.path.join(base, path))

    pos = y.sum()
    neg = (y == 0).sum()
    spw = neg / pos if pos > 0 else 1.0  # scale_pos_weight for XGBoost

    records = []
    for seed in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )

        for model_name, model in [
            ("LR", make_lr(seed)),
            ("RF", make_rf(seed)),
            ("XGB", make_xgb(spw, seed)),
        ]:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            records.append({
                "dataset": name,
                "model": model_name,
                "seed": seed,
                "precision": precision_score(y_te, y_pred, zero_division=0),
                "recall": recall_score(y_te, y_pred, zero_division=0),
                "f1": f1_score(y_te, y_pred, zero_division=0),
            })

    return pd.DataFrame(records)


def wilcoxon_test(df: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    """Return (statistic, p-value) for Wilcoxon signed-rank on F1 scores."""
    a = df[df["model"] == model_a]["f1"].values
    b = df[df["model"] == model_b]["f1"].values
    stat, p = wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(p)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of metrics per dataset/model."""
    agg = (
        df.groupby(["dataset", "model"])[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .round(3)
    )
    return agg


def run(dataset_filter: str | None = None, results_dir: str = "results") -> None:
    base = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base, results_dir)
    os.makedirs(results_path, exist_ok=True)

    datasets = (
        {dataset_filter: DATASETS[dataset_filter]}
        if dataset_filter and dataset_filter in DATASETS
        else DATASETS
    )

    all_records = []
    for name, path in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        df = evaluate_dataset(name, path)
        all_records.append(df)

        summary = summarise(df)
        print(summary.to_string())

        # Wilcoxon tests
        print(f"\nWilcoxon RF vs LR:")
        stat, p = wilcoxon_test(df, "RF", "LR")
        print(f"  stat={stat:.1f}, p={p:.4f} {'*' if p < 0.05 else ''}")

        print(f"Wilcoxon XGB vs LR:")
        stat, p = wilcoxon_test(df, "XGB", "LR")
        print(f"  stat={stat:.1f}, p={p:.4f} {'*' if p < 0.05 else ''}")

    all_df = pd.concat(all_records, ignore_index=True)
    out_path = os.path.join(results_path, "results.csv")
    all_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Save summary
    summary_path = os.path.join(results_path, "summary.csv")
    summarise(all_df).to_csv(summary_path)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defect prediction experiments.")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name to run")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    run(args.dataset, args.results_dir)
