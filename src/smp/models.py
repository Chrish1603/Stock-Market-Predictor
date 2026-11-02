"""Model registry: naive baselines, linear models and gradient boosting."""

from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

BASELINE = "baseline"

LGBM_DEFAULTS = dict(
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)
XGB_DEFAULTS = dict(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
)


def classifiers(random_state: int = 42, lgbm_params: dict | None = None) -> dict:
    """Candidates for next-day direction. Scaling sits inside the pipeline so it's fit per fold."""
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.1, max_iter=2000)),
            ]
        ),
        "lightgbm": LGBMClassifier(**{**LGBM_DEFAULTS, **(lgbm_params or {})}, random_state=random_state),
        "xgboost": XGBClassifier(**XGB_DEFAULTS, eval_metric="logloss", random_state=random_state),
    }


def regressors(random_state: int = 42, lgbm_params: dict | None = None) -> dict:
    """Candidates for next-day log return. The baseline always predicts zero return."""
    return {
        BASELINE: DummyRegressor(strategy="constant", constant=0.0),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=100.0))]),
        "lightgbm": LGBMRegressor(**{**LGBM_DEFAULTS, **(lgbm_params or {})}, random_state=random_state),
        "xgboost": XGBRegressor(**XGB_DEFAULTS, random_state=random_state),
    }


def feature_importance(model, columns: list[str]) -> dict[str, float]:
    """Normalised importance: tree gain/splits, or absolute coefficients for linear models."""
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = abs(est.coef_).ravel()
    else:
        return {}
    total = float(sum(values)) or 1.0
    pairs = sorted(zip(columns, values, strict=False), key=lambda p: -p[1])
    return {name: float(v) / total for name, v in pairs}
