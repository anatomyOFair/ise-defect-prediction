"""Model definitions: Logistic Regression baseline, Random Forest, XGBoost."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_lr(random_state: int = 42) -> Pipeline:
    """Logistic Regression baseline with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def make_rf(random_state: int = 42) -> RandomForestClassifier:
    """Random Forest with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def make_xgb(scale_pos_weight: float = 1.0, random_state: int = 42) -> XGBClassifier:
    """XGBoost with scale_pos_weight for class imbalance."""
    return XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
