# Software Defect Prediction Tool

Binary classification tool that predicts whether a software module is defective using static code metrics. Uses Random Forest and XGBoost ensemble models evaluated against a Logistic Regression baseline.

## Requirements

- Python 3.10+
- macOS: `brew install libomp` (required for XGBoost)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
ISE/
  src/
    train.py      # Training and evaluation pipeline
    predict.py    # CLI prediction tool
    models.py     # Model definitions
    utils.py      # Data loading and preprocessing
  data/           # PROMISE repository CSV datasets
  results/        # Experiment output (generated)
  report.pdf      # Final report
```

## Running Experiments

Run all datasets:

```bash
python src/train.py
```

Run a single dataset:

```bash
python src/train.py --dataset ant-1.7
```

Results are saved to `results/results.csv` and `results/summary.csv`.

## Predicting on New Data

Given a CSV file containing CK metrics (wmc, dit, noc, cbo, rfc, lcom, loc, etc.):

```bash
python src/predict.py --input path/to/metrics.csv --model rf
python src/predict.py --input path/to/metrics.csv --model xgb
python src/predict.py --input path/to/metrics.csv --model lr
```

The tool trains on all available datasets and outputs `predicted_defective` (0/1) and `defect_probability` columns. Predictions are saved to `<input>_predictions.csv`.

## Datasets

PROMISE repository datasets used (from Jureczko and Madeyski, 2010):

| Dataset   | Instances | Features | Defect Rate |
|-----------|-----------|----------|-------------|
| ant-1.7   | 745       | 20       | 22.4%       |
| camel-1.6 | 965       | 20       | 19.5%       |
| ivy-2.0   | 352       | 20       | 11.1%       |
| jedit-4.3 | 492       | 20       | 2.0%        |
