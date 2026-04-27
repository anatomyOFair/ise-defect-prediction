"""Training and evaluation pipeline.

Usage:
    python src/train.py                     # run all families
    python src/train.py --family promise-ck # run one family
    python src/train.py --dataset ant-1.7   # run one dataset
    python src/train.py --lopo              # run leave-one-project-out on PROMISE CK
    python src/train.py --results-dir results/
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from models import (make_lr, make_rf, make_rf_tuned, make_stacking, make_xgb, make_xgb_tuned,
                    make_lgbm, make_lgbm_tuned, make_rf_smote)
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

N_REPEATS = 30
TEST_SIZE = 0.30

COMPARISONS = [
    ("RF",       "LR"),
    ("XGB",      "LR"),
    ("LGB",      "LR"),
    ("Stacking", "LR"),
    ("RF-HP",    "LR"),
    ("XGB-HP",   "LR"),
    ("LGB-HP",   "LR"),
    ("RF-SMOTE", "LR"),
]


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size. Range [-1, +1]: positive means a tends to exceed b."""
    a, b = list(a), list(b)
    n = len(a) * len(b)
    if n == 0:
        return 0.0
    return (sum(1 for x in a for y in b if x > y) -
            sum(1 for x in a for y in b if x < y)) / n


def cliffs_magnitude(d: float) -> str:
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def evaluate_dataset(name: str, path: str, family: str,
                     n_repeats: int = N_REPEATS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run n_repeats train/test splits and return (per-repeat metrics, RF importances)."""
    base = os.path.dirname(os.path.dirname(__file__))
    X, y = load_dataset(os.path.join(base, path))

    pos = y.sum()
    neg = (y == 0).sum()
    spw = neg / pos if pos > 0 else 1.0

    records = []
    importance_records = []

    for seed in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        models = [
            ("LR",       make_lr(seed)),
            ("RF",       make_rf(seed)),
            ("XGB",      make_xgb(spw, seed)),
            ("LGB",      make_lgbm(spw, seed)),
            ("Stacking", make_stacking(spw, seed)),
            ("RF-HP",    make_rf_tuned(family, seed)),
            ("XGB-HP",   make_xgb_tuned(family, spw, seed)),
            ("LGB-HP",   make_lgbm_tuned(family, spw, seed)),
            ("RF-SMOTE", make_rf_smote(seed)),
        ]
        for model_name, model in models:
            try:
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
                if model_name in ("RF", "LGB-HP"):
                    estimator = (model.named_steps["clf"]
                                 if hasattr(model, "named_steps") else model)
                    for feat, imp in zip(X_tr.columns, estimator.feature_importances_):
                        importance_records.append({
                            "family":    family,
                            "dataset":   name,
                            "model":     model_name,
                            "feature":   feat,
                            "importance": float(imp),
                            "seed":      seed,
                        })
            except Exception as exc:
                print(f"  WARNING: {model_name} seed={seed} on {name}: {exc}")

    return pd.DataFrame(records), pd.DataFrame(importance_records)


def wilcoxon_test(df: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    """Wilcoxon signed-rank on F1 scores. Returns (0.0, 1.0) when no difference."""
    a = df[df["model"] == model_a]["f1"].values
    b = df[df["model"] == model_b]["f1"].values
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    try:
        stat, p = scipy_wilcoxon(a, b, alternative="two-sided")
        if np.isnan(p):
            return 0.0, 1.0
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of metrics per family/dataset/model."""
    return (
        df.groupby(["family", "dataset", "model"])[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .round(3)
    )


def run_lopo(results_dir: str = "results") -> pd.DataFrame:
    """Leave-one-project-out cross-project evaluation on PROMISE CK."""
    base = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base, results_dir)
    os.makedirs(results_path, exist_ok=True)

    datasets_list = list(DATASETS["promise-ck"].items())
    records = []

    print(f"\n{'#'*60}")
    print("Leave-one-project-out (PROMISE CK)")
    print(f"{'#'*60}")

    for i, (test_name, test_path) in enumerate(datasets_list):
        test_full = os.path.join(base, test_path)
        if not os.path.exists(test_full):
            print(f"  SKIP {test_name}: file not found")
            continue

        train_items = [(n, p) for j, (n, p) in enumerate(datasets_list) if j != i]

        Xs, ys = [], []
        for tn, tp in train_items:
            full = os.path.join(base, tp)
            if os.path.exists(full):
                X, y = load_dataset(full)
                Xs.append(X)
                ys.append(y)
        if not Xs:
            continue

        common_cols = sorted(set(Xs[0].columns).intersection(*[set(X.columns) for X in Xs[1:]]))
        X_train = pd.concat([X[common_cols] for X in Xs], ignore_index=True)
        y_train = pd.concat(ys, ignore_index=True)

        X_test, y_test = load_dataset(test_full)
        for col in common_cols:
            if col not in X_test.columns:
                X_test[col] = 0.0
        X_test = X_test[common_cols]

        spw = (y_train == 0).sum() / y_train.sum() if y_train.sum() > 0 else 1.0

        for model_name, model in [
            ("LR",  make_lr()),
            ("RF",  make_rf()),
            ("XGB", make_xgb(spw)),
        ]:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                records.append({
                    "dataset":   test_name,
                    "model":     model_name,
                    "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                    "recall":    round(recall_score(y_test, y_pred, zero_division=0), 3),
                    "f1":        round(f1_score(y_test, y_pred, zero_division=0), 3),
                })
            except Exception as exc:
                print(f"  LOPO WARNING: {model_name} on {test_name}: {exc}")

        print(f"  Done: {test_name}")

    lopo_df = pd.DataFrame(records)
    lopo_path = os.path.join(results_path, "lopo.csv")
    lopo_df.to_csv(lopo_path, index=False)
    print(f"\nLOPO results saved to {lopo_path}")
    print(lopo_df.to_string(index=False))
    return lopo_df


def compute_shap(families_to_run: dict, results_path: str, base: str) -> None:
    """Compute mean |SHAP| per feature using LGB-HP on pooled family data (seed=0)."""
    try:
        import shap as shap_lib
    except ImportError:
        print("shap not installed — skipping SHAP computation")
        return

    rows = []
    for family, datasets in families_to_run.items():
        Xs, ys = [], []
        for name, path in datasets.items():
            full = os.path.join(base, path)
            if os.path.exists(full):
                X, y = load_dataset(full)
                Xs.append(X)
                ys.append(y)
        if not Xs:
            continue
        common_cols = sorted(set(Xs[0].columns).intersection(*[set(X.columns) for X in Xs[1:]]))
        X_all = pd.concat([X[common_cols] for X in Xs], ignore_index=True)
        y_all = pd.concat(ys, ignore_index=True)
        spw = (y_all == 0).sum() / y_all.sum() if y_all.sum() > 0 else 1.0

        model = make_lgbm_tuned(family, spw, 0)
        model.fit(X_all, y_all)

        explainer = shap_lib.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_all)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)

        for feat, val in sorted(zip(common_cols, mean_abs), key=lambda x: -x[1]):
            rows.append({"family": family, "feature": feat,
                         "mean_abs_shap": round(float(val), 5)})

    if rows:
        shap_df = pd.DataFrame(rows)
        shap_path = os.path.join(results_path, "shap_importance.csv")
        shap_df.to_csv(shap_path, index=False)
        print(f"SHAP importance saved to {shap_path}")
        for fam in shap_df["family"].unique():
            top5 = shap_df[shap_df["family"] == fam].head(5)
            print(f"\nTop 5 SHAP features — {fam}:")
            print(top5.to_string(index=False))


def run(family_filter: str | None = None, dataset_filter: str | None = None,
        results_dir: str = "results", run_lopo_flag: bool = False) -> None:
    base = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base, results_dir)
    os.makedirs(results_path, exist_ok=True)

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
    all_importance = []
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
            df, imp_df = evaluate_dataset(name, path, family)
            all_records.append(df)
            if len(imp_df) > 0:
                all_importance.append(imp_df)

            summary = summarise(df)
            print(summary.to_string())

            for comp in COMPARISONS:
                a_vals = df[df["model"] == comp[0]]["f1"].values
                b_vals = df[df["model"] == comp[1]]["f1"].values
                if len(a_vals) == 0 or len(b_vals) == 0:
                    continue
                stat, p = wilcoxon_test(df, comp[0], comp[1])
                delta = cliffs_delta(a_vals, b_vals)
                mag = cliffs_magnitude(delta)
                sig = "*" if p < 0.05 else ""
                direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                print(f"  Wilcoxon {comp[0]} vs {comp[1]}: "
                      f"p={p:.4f}{sig}  d={delta:.3f} ({mag}) {direction}")
                wilcoxon_rows.append({
                    "family":       family,
                    "dataset":      name,
                    "comparison":   f"{comp[0]} vs {comp[1]}",
                    "statistic":    round(stat, 1),
                    "p_value":      round(p, 4),
                    "significant":  p < 0.05,
                    "cliffs_delta": round(delta, 4),
                    "magnitude":    mag,
                    "direction":    direction,
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
    print(f"Wilcoxon + Cliff's delta saved to {wilcoxon_path}")

    if all_importance:
        imp_all = pd.concat(all_importance, ignore_index=True)
        imp_summary = (
            imp_all.groupby(["family", "model", "feature"])["importance"]
            .mean()
            .reset_index()
            .sort_values(["family", "model", "importance"], ascending=[True, True, False])
            .round(4)
        )
        imp_path = os.path.join(results_path, "feature_importance.csv")
        imp_summary.to_csv(imp_path, index=False)
        print(f"Feature importance saved to {imp_path}")

        for fam in imp_summary["family"].unique():
            for mdl in imp_summary["model"].unique():
                top5 = imp_summary[
                    (imp_summary["family"] == fam) & (imp_summary["model"] == mdl)
                ].head(5)
                if not top5.empty:
                    print(f"\nTop 5 features — {fam} / {mdl}:")
                    print(top5.to_string(index=False))

    compute_shap(families_to_run, results_path, base)

    if run_lopo_flag or (not family_filter and not dataset_filter):
        if not family_filter or family_filter == "promise-ck":
            run_lopo(results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defect prediction experiments.")
    parser.add_argument("--family",      type=str, default=None,
                        help="Dataset family: promise-ck, aeeem, nasa")
    parser.add_argument("--dataset",     type=str, default=None,
                        help="Single dataset name to run")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--lopo",        action="store_true",
                        help="Run leave-one-project-out evaluation only")
    args = parser.parse_args()

    if args.lopo:
        run_lopo(args.results_dir)
    else:
        run(args.family, args.dataset, args.results_dir)
