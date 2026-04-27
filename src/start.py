"""Interactive terminal menu for the Software Defect Prediction tool.

Run this from the repo root:
    python src/start.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

MODELS = ["lr", "rf", "xgb", "lgb", "stacking",
          "rf-hp", "xgb-hp", "lgb-hp", "rf-smote"]

FAMILY_KEYS = {"1": "promise-ck", "2": "aeeem", "3": "nasa"}


DETAILS = {
    "1": {
        "title": "Run full evaluation",
        "what": ("Trains and evaluates all 9 models on all 20 datasets across\n"
                 "the 3 families (PROMISE CK, AEEEM, NASA MDP), with 30 repeated\n"
                 "stratified 70/30 splits. Reproduces every number in the report."),
        "reads": [
            "data/*.csv             (10 PROMISE CK projects)",
            "data/aeeem/*.csv       (5 AEEEM projects)",
            "data/nasa/*.csv        (5 NASA MDP projects)",
            "src/models.py          (tuned hyperparameters)",
        ],
        "writes": [
            "results/results.csv             (5,401 per-repeat rows)",
            "results/summary.csv             (mean +/- std per dataset/model)",
            "results/wilcoxon.csv            (p-values, Cliff's delta)",
            "results/feature_importance.csv  (RF tree importances)",
            "results/shap_importance.csv     (mean |SHAP| per feature)",
        ],
        "time": "~45 minutes",
        "warnings": ["Overwrites every file in results/."],
    },
    "2": {
        "title": "Run one family only",
        "what": ("Like option 1 but only on the family you choose next.\n"
                 "Useful for a quick partial reproduction (~10-25 min)."),
        "reads": ["The 5-10 CSVs of the chosen family"],
        "writes": [
            "results/results.csv, summary.csv, wilcoxon.csv, ...",
            "(only that family's rows)",
        ],
        "time": "~10-25 minutes depending on family",
        "warnings": [
            "OVERWRITES the main results files with only this family's rows.",
            "If you ran option 1 first, the other families' results are lost.",
            "Re-run option 1 to restore the full picture.",
        ],
    },
    "3": {
        "title": "Cross-project test (LOPO)",
        "what": ("Leave-one-project-out on PROMISE CK. For each of the 10\n"
                 "projects, trains on the other 9 and predicts on the held-out\n"
                 "one. This backs Table 5 of the report."),
        "reads": ["data/*.csv (10 PROMISE CK projects)"],
        "writes": ["results/lopo.csv (cross-project F1)"],
        "time": "~5 minutes",
        "warnings": ["Overwrites results/lopo.csv only. Main results files untouched."],
    },
    "4": {
        "title": "Predict on your CSV",
        "what": ("Trains the chosen model on a full dataset family, then\n"
                 "predicts on a CSV you provide. Outputs predictions and\n"
                 "probabilities alongside the input rows."),
        "reads": [
            "Your CSV (any path)",
            "All datasets in the chosen family (used as training data)",
        ],
        "writes": [
            "<your_input>_predictions.csv (next to your input file)",
        ],
        "time": "~30 sec (tree models), ~2 min (stacking)",
        "warnings": ["You will be prompted for path, model, and family."],
    },
    "5": {
        "title": "View results summary",
        "what": ("Reads the latest result files and prints a table showing\n"
                 "wins/20 vs LR and mean F1 per model. Lets you verify the\n"
                 "report's Tables 3 and 4 without rerunning anything."),
        "reads": [
            "results/summary.csv",
            "results/wilcoxon.csv",
        ],
        "writes": ["Nothing -- terminal output only."],
        "time": "Instant",
    },
    "6": {
        "title": "Regenerate SHAP figure",
        "what": ("Reads SHAP importances and produces the bar chart used as\n"
                 "Figure 1 in the report."),
        "reads": ["results/shap_importance.csv"],
        "writes": ["results/shap_importance_fig.pdf"],
        "time": "~1 second",
    },
    "7": {
        "title": "Re-run hyperparameter tuning",
        "what": ("Runs RandomizedSearchCV (20 iterations, 5-fold stratified CV)\n"
                 "over RF, XGB, and LGB for each family.\n"
                 "Verification only -- the tuned params used by the report\n"
                 "are already hardcoded in src/models.py."),
        "reads": ["All 20 dataset CSVs"],
        "writes": ["results/best_params.json"],
        "time": "~25 minutes",
        "warnings": [
            "Optional. The report's results do NOT depend on this step.",
            "train.py uses params from src/models.py, not from this JSON.",
        ],
    },
}


def show_and_confirm(key: str) -> bool:
    """Print full details for an option and ask y/N. Returns True if user confirms."""
    d = DETAILS.get(key)
    if d is None:
        return True
    print(f"\n{BOLD}{CYAN}{'-' * 62}{RESET}")
    print(f"{BOLD}  {d['title']}{RESET}\n")
    print(f"  {d['what']}")
    if d.get("reads"):
        print(f"\n  {BOLD}Reads:{RESET}")
        for r in d["reads"]:
            print(f"    {DIM}-{RESET} {r}")
    if d.get("writes"):
        print(f"\n  {BOLD}Writes:{RESET}")
        for w in d["writes"]:
            print(f"    {DIM}-{RESET} {w}")
    if d.get("time"):
        print(f"\n  {BOLD}Time:{RESET}  {d['time']}")
    for w in d.get("warnings", []):
        print(f"\n  {YELLOW}! {w}{RESET}")
    print(f"{BOLD}{CYAN}{'-' * 62}{RESET}")
    ans = input(f"\n{BOLD}Continue? [y/N]: {RESET}").strip().lower()
    if ans != "y":
        print(f"{DIM}Cancelled.{RESET}")
        return False
    return True


def banner() -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 62}{RESET}")
    print(f"{BOLD}  Software Defect Prediction Tool{RESET}")
    print(f"{DIM}  ISE Coursework 2026{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 62}{RESET}\n")


def menu() -> str:
    print(f"  {BOLD}1){RESET} Run full evaluation     {DIM}(9 models, 20 datasets, ~45 min){RESET}")
    print(f"  {BOLD}2){RESET} Run one family only     {DIM}(promise-ck / aeeem / nasa){RESET}")
    print(f"  {BOLD}3){RESET} Run cross-project test  {DIM}(leave-one-project-out){RESET}")
    print(f"  {BOLD}4){RESET} Predict on your CSV     {DIM}(point at a code-metrics file){RESET}")
    print(f"  {BOLD}5){RESET} View results summary    {DIM}(win counts per model){RESET}")
    print(f"  {BOLD}6){RESET} Regenerate SHAP figure")
    print(f"  {BOLD}7){RESET} Re-run hyperparameter tuning {DIM}(~25 min, optional){RESET}")
    print(f"  {BOLD}q){RESET} Quit\n")
    return input(f"{BOLD}Choose an option: {RESET}").strip().lower()


def run(cmd: list) -> None:
    """Run a subprocess from the repo root and stream output. Ignore Ctrl+C cleanly."""
    print(f"\n{DIM}> {' '.join(cmd)}{RESET}\n")
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Cancelled.{RESET}")


def opt_full() -> None:
    if not show_and_confirm("1"):
        return
    run([sys.executable, "src/train.py"])


def opt_family() -> None:
    if not show_and_confirm("2"):
        return
    print(f"\n  1) promise-ck   2) aeeem   3) nasa")
    choice = input("Pick family [1/2/3]: ").strip()
    fam = FAMILY_KEYS.get(choice)
    if not fam:
        print(f"{YELLOW}Invalid choice.{RESET}")
        return
    run([sys.executable, "src/train.py", "--family", fam])


def opt_lopo() -> None:
    if not show_and_confirm("3"):
        return
    run([sys.executable, "src/train.py", "--lopo"])


def opt_predict() -> None:
    if not show_and_confirm("4"):
        return
    print(f"\n{BOLD}Predict on your own CSV{RESET}")
    print(f"""
Your CSV must use the column naming of one of the three dataset families.
Three example files are provided — copy the one matching your metric type:

  {GREEN}examples/promise_ck_example.csv{RESET}  Java OO/CK metrics (wmc, cbo, rfc, loc, ...)
  {GREEN}examples/aeeem_example.csv{RESET}       change/process metrics (linesAddedUntil, age, ...)
  {GREEN}examples/nasa_example.csv{RESET}        Halstead/McCabe metrics (loc_total, halstead_*, ...)

Replace the rows with your own module data. The target column is NOT required.
Any missing columns will be filled with zero (a warning is printed).
""")
    path = input(f"{BOLD}Path to your CSV: {RESET}").strip()
    if not path:
        return
    if not os.path.exists(path):
        # try repo-relative
        rel = os.path.join(ROOT, path)
        if os.path.exists(rel):
            path = rel
        else:
            print(f"{YELLOW}File not found: {path}{RESET}")
            return

    print(f"\nAvailable models:")
    for i, m in enumerate(MODELS, 1):
        tag = f"  {DIM}<- recommended (within-project){RESET}" if m == "xgb-hp" else \
              f"  {DIM}<- recommended (cross-project / new project){RESET}" if m == "lr" else ""
        print(f"  {i}) {m}{tag}")
    mchoice = input(f"\n{BOLD}Pick model [1-9, default 7=xgb-hp]: {RESET}").strip() or "7"
    try:
        model = MODELS[int(mchoice) - 1]
    except (ValueError, IndexError):
        print(f"{YELLOW}Invalid model choice.{RESET}")
        return

    print(f"\n  1) promise-ck   2) aeeem   3) nasa")
    fchoice = input(f"{BOLD}Pick family (must match the example you used) [1/2/3]: {RESET}").strip()
    family = FAMILY_KEYS.get(fchoice)
    if not family:
        print(f"{YELLOW}Invalid family choice.{RESET}")
        return

    run([sys.executable, "src/predict.py", "--input", path,
         "--model", model, "--family", family])


def opt_summary() -> None:
    """Read summary.csv and wilcoxon.csv, print win counts vs LR per model."""
    if not show_and_confirm("5"):
        return
    summary_path = os.path.join(ROOT, "results", "summary.csv")
    wilcoxon_path = os.path.join(ROOT, "results", "wilcoxon.csv")
    if not (os.path.exists(summary_path) and os.path.exists(wilcoxon_path)):
        print(f"\n{YELLOW}No results yet. Run option 1 (full evaluation) first.{RESET}")
        return

    import pandas as pd
    cols = ["family", "dataset", "model",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std"]
    summary = pd.read_csv(summary_path, skiprows=3, names=cols)
    f1_by_model = summary.groupby("model")["f1_mean"].mean().round(3)

    wil = pd.read_csv(wilcoxon_path)
    wil = wil[wil['significant'] & (wil['direction'] == '↑')]
    wins = wil['comparison'].str.replace(' vs LR', '', regex=False).value_counts()

    notes = {
        "XGB-HP":   "Recommended (within-project)",
        "LR":       "Recommended (cross-project / new project)",
        "Stacking": "Avoid on small/imbalanced datasets",
    }

    rows = []
    for model in ["XGB-HP", "RF-HP", "LGB", "RF-SMOTE", "XGB", "LGB-HP", "RF", "Stacking", "LR"]:
        w = wins.get(model, "—" if model == "LR" else 0)
        f1 = f1_by_model.get(model, float("nan"))
        rows.append((model, w, f"{f1:.3f}" if f1 == f1 else "—", notes.get(model, "")))

    print(f"\n{BOLD}Results summary — significant wins (Wilcoxon p<=0.05, d>0) vs LR{RESET}\n")
    print(f"  {'Model':<10} {'Wins/20':<10} {'Mean F1':<10} {'Note'}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*40}")
    for model, w, f1, note in rows:
        ws = f"{w}/20" if w != "—" else "—"
        print(f"  {model:<10} {ws:<10} {f1:<10} {DIM}{note}{RESET}")
    print()


def opt_shap() -> None:
    if not show_and_confirm("6"):
        return
    run([sys.executable, "src/plot_shap.py"])


def opt_tune() -> None:
    if not show_and_confirm("7"):
        return
    run([sys.executable, "src/tune.py"])


HANDLERS = {
    "1": opt_full, "2": opt_family, "3": opt_lopo,
    "4": opt_predict, "5": opt_summary, "6": opt_shap, "7": opt_tune,
}


def main() -> None:
    while True:
        banner()
        choice = menu()
        if choice == "q":
            print(f"\n{DIM}Goodbye.{RESET}\n")
            return
        handler = HANDLERS.get(choice)
        if handler is None:
            print(f"\n{YELLOW}Unknown option: {choice}{RESET}")
        else:
            try:
                handler()
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Cancelled.{RESET}")
        input(f"\n{DIM}Press Enter to continue...{RESET}")


if __name__ == "__main__":
    main()
