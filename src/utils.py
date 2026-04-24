"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np

# Metadata columns to drop (not features)
META_COLS = {"name", "version", "name.1"}

# Target column
TARGET = "bug"


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a PROMISE CSV, drop metadata, binarise target."""
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop metadata columns
    drop = [c for c in df.columns if c in META_COLS or c.startswith("name")]
    df = df.drop(columns=drop, errors="ignore")

    # Find target column (bug / bugs / defects / defective / label)
    target_col = None
    for candidate in ["bug", "bugs", "defects", "defect", "defective", "label"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        raise ValueError(f"No target column found in {path}. Columns: {df.columns.tolist()}")

    # Binarise: handle Y/N strings (NASA), 'buggy'/'clean' strings, or numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        val = df[target_col].str.lower().str.strip()
        y = val.isin(["y", "yes", "true", "1", "buggy"]).astype(int)
    else:
        y = (df[target_col] > 0).astype(int)
    X = df.drop(columns=[target_col])

    # Median imputation for missing values
    X = X.fillna(X.median(numeric_only=True))

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    return X, y


def defect_rate(y: pd.Series) -> float:
    return y.mean()
