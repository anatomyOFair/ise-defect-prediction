# Software Defect Prediction Tool

ISE Coursework 2026 — Binary classification tool that predicts whether a software module is defective from static code metrics. Evaluates nine models (Logistic Regression baseline plus RF, XGB, LGB, Stacking, hyperparameter-tuned variants, and RF-SMOTE) across 20 datasets from three benchmark families.

See [`report.pdf`](report.pdf) for the full methodology, results, and discussion.

## Quick Start

```bash
git clone https://github.com/anatomyOFair/ise-defect-prediction.git
cd ise-defect-prediction
pip install -r requirements.txt
python src/start.py
```

`start.py` opens a numbered terminal menu — pick an option to run experiments, predict on your own CSV, view the win-counts table, and so on. Each option shows a detail screen (inputs, outputs, runtime, side effects) and asks for confirmation before running.

## Requirements

- Python 3.11+
- macOS: `brew install libomp` may be needed for XGBoost

Pinned dependency versions are in `requirements.txt`. Full breakdown in [`requirements.pdf`](requirements.pdf).

## Documentation

| File | Purpose |
|---|---|
| `report.pdf` | Main report (6 pages) |
| `manual.pdf` | How to use the tool |
| `replication.pdf` | How to reproduce the reported results |
| `requirements.pdf` | Dependency list with package roles |

## Project Structure

```
src/
  start.py       Terminal menu (entry point)
  train.py       Training and evaluation pipeline
  predict.py     Prediction CLI (all 9 models)
  tune.py        Hyperparameter tuning (RandomizedSearchCV)
  plot_shap.py   SHAP figure generator
  models.py      Model factories
  utils.py       Data loading and preprocessing
data/            20 dataset CSVs (PROMISE CK, AEEEM, NASA MDP)
examples/        Sample CSVs for prediction (one per family)
results/         Experimental output (committed)
```

## Models

| Model | Notes |
|---|---|
| `lr` | Baseline (Logistic Regression with `class_weight='balanced'`) |
| `rf` | Random Forest, 100 trees, balanced class weights |
| `xgb` | XGBoost, default, `scale_pos_weight` |
| `lgb` | LightGBM, default, `scale_pos_weight` |
| `stacking` | LR + RF + XGB base, LR meta-learner, `cv=3` |
| `rf-hp` | Random Forest with tuned hyperparameters |
| `xgb-hp` | XGBoost with tuned hyperparameters |
| `lgb-hp` | LightGBM with tuned hyperparameters |
| `rf-smote` | Random Forest after SMOTE oversampling |

**Recommended models** (from report Conclusion):

- `xgb-hp` for within-project prediction (significant wins on 13/20 datasets, robust across all three families)
- `lr` for cross-project use on a new project (LOPO results show better generalisation)
- Avoid `stacking` on datasets with fewer than ~50 defective examples

## Datasets

| Family | Datasets (n) | Instances | Features | Defect rate |
|---|---|---|---|---|
| PROMISE CK | 10 Java OO projects | 229–965 | 20 | 2.2–74.3% |
| AEEEM | 5 Eclipse systems | 324–1862 | 15 | 9.3–39.8% |
| NASA MDP | 5 aerospace modules | 403–10878 | 21–40 | 6.9–19.3% |

Sources: Jureczko and Madeyski (2010) for PROMISE CK, D'Ambros et al. (2012) for AEEEM, Menzies et al. (2004) for NASA MDP. See `report.pdf` Section 4.1.

## Predicting on Your Own CSV

Three example files are provided in `examples/`, one per dataset family. Copy the example matching your metric type, replace the rows with your own module data, then point option 4 of `start.py` at the file (or run `predict.py` directly):

```bash
python src/predict.py --input my_metrics.csv --model xgb-hp --family promise-ck
```

The target column is not required (this is prediction, not training). Column names are normalised on load. Output is written to `<input>_predictions.csv` with appended columns `predicted_defective` (0/1) and `defect_probability` (0–1).

## Reproducing the Results

```bash
python src/train.py
```

This runs all 9 models on all 20 datasets with 30 repeated stratified 70/30 splits (~45 minutes). Tuned hyperparameters are already committed to `src/models.py`, so this single command reproduces every number in the report. Verification table in `replication.pdf`.
