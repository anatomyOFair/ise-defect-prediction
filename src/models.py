"""Model definitions: Logistic Regression baseline, Random Forest, XGBoost, LightGBM, Stacking, SMOTE."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


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


def make_lgbm(scale_pos_weight: float = 1.0, random_state: int = 42) -> LGBMClassifier:
    """LightGBM with scale_pos_weight for class imbalance."""
    return LGBMClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )


def make_rf_smote(random_state: int = 42) -> ImbPipeline:
    """Random Forest with SMOTE oversampling (no class_weight — SMOTE handles balance)."""
    return ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def make_stacking(scale_pos_weight: float = 1.0, random_state: int = 42) -> StackingClassifier:
    """Stacking ensemble: LR, RF, XGB as base learners; logistic regression meta-learner."""
    base_estimators = [
        ("lr",  make_lr(random_state)),
        ("rf",  make_rf(random_state)),
        ("xgb", make_xgb(scale_pos_weight, random_state)),
    ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        cv=3,
        n_jobs=-1,
    )


# Best hyperparameters from tune.py (RandomizedSearchCV, 20 iter, 5-fold CV, scoring=f1)
_RF_TUNED = {
    "promise-ck": dict(n_estimators=100, min_samples_split=5, min_samples_leaf=4,
                       max_features=None, max_depth=10),
    "aeeem":      dict(n_estimators=100, min_samples_split=5, min_samples_leaf=4,
                       max_features=None, max_depth=10),
    "nasa":       dict(n_estimators=200, min_samples_split=2, min_samples_leaf=4,
                       max_features="sqrt", max_depth=10),
}

_XGB_TUNED = {
    "promise-ck": dict(n_estimators=300, learning_rate=0.01, max_depth=7,
                       subsample=0.8, colsample_bytree=0.6, min_child_weight=5),
    "aeeem":      dict(n_estimators=300, learning_rate=0.01, max_depth=7,
                       subsample=0.8, colsample_bytree=0.6, min_child_weight=5),
    "nasa":       dict(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.6, colsample_bytree=1.0, min_child_weight=1),
}


def make_rf_tuned(family: str, random_state: int = 42) -> RandomForestClassifier:
    """Random Forest with family-specific tuned hyperparameters."""
    params = _RF_TUNED.get(family, _RF_TUNED["promise-ck"])
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        **params,
    )


def make_xgb_tuned(family: str, scale_pos_weight: float = 1.0,
                   random_state: int = 42) -> XGBClassifier:
    """XGBoost with family-specific tuned hyperparameters."""
    params = _XGB_TUNED.get(family, _XGB_TUNED["promise-ck"])
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
        **params,
    )


# Best hyperparameters from tune.py (RandomizedSearchCV, 20 iter, 5-fold CV, scoring=f1)
_LGB_TUNED = {
    "promise-ck": dict(n_estimators=300, learning_rate=0.05, max_depth=10,
                       num_leaves=15, subsample=0.8, colsample_bytree=1.0,
                       min_child_samples=5),
    "aeeem":      dict(n_estimators=300, learning_rate=0.05, max_depth=10,
                       num_leaves=15, subsample=0.8, colsample_bytree=1.0,
                       min_child_samples=5),
    "nasa":       dict(n_estimators=200, learning_rate=0.05, max_depth=10,
                       num_leaves=63, subsample=1.0, colsample_bytree=0.6,
                       min_child_samples=5),
}


def make_lgbm_tuned(family: str, scale_pos_weight: float = 1.0,
                    random_state: int = 42) -> LGBMClassifier:
    """LightGBM with family-specific tuned hyperparameters."""
    params = _LGB_TUNED.get(family, _LGB_TUNED["promise-ck"])
    return LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
        **params,
    )


class ThresholdTunedClassifier:
    """Wraps a classifier and tunes the decision threshold to maximise F1 on the training set."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.threshold_ = 0.5

    def fit(self, X, y):
        self.estimator.fit(X, y)
        proba = self.estimator.predict_proba(X)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y, proba)
        with np.errstate(invalid="ignore"):
            f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        if len(thresholds) > 0:
            self.threshold_ = float(thresholds[f1s[:-1].argmax()])
        return self

    def predict(self, X):
        proba = self.estimator.predict_proba(X)[:, 1]
        return (proba >= self.threshold_).astype(int)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
