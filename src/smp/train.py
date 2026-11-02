"""Walk-forward training, model selection, hold-out evaluation and artifact saving."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from smp.config import Config, ticker_to_filename
from smp.evaluate import backtest, backtest_stats, classification_metrics, regression_metrics
from smp.features import TARGET_DIRECTION, TARGET_RETURN, feature_columns, features_path
from smp.models import BASELINE, classifiers, feature_importance, regressors

log = logging.getLogger(__name__)

# Model selection criterion per task: (metric, higher_is_better)
SELECTION = {"classifier": ("roc_auc", True), "regressor": ("rmse", False)}


def walk_forward_splits(n: int, n_splits: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits: each test block comes strictly after its training data."""
    return list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.arange(n)))


def _fit_predict(model, task: str, X_tr, y_tr, X_te) -> np.ndarray:
    m = clone(model).fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1] if task == "classifier" else m.predict(X_te)


def _score(task: str, y, pred) -> dict[str, float]:
    return classification_metrics(y, pred) if task == "classifier" else regression_metrics(y, pred)


def cross_validate(model, task: str, X: pd.DataFrame, y: pd.Series, splits) -> dict[str, float]:
    """Mean metrics across walk-forward folds."""
    fold_scores = [
        _score(task, y.iloc[te], _fit_predict(model, task, X.iloc[tr], y.iloc[tr], X.iloc[te])) for tr, te in splits
    ]
    return {k: float(np.nanmean([s[k] for s in fold_scores])) for k in fold_scores[0]}


def tune_lightgbm(task: str, X, y, splits, n_trials: int, random_state: int) -> dict:
    """Small Optuna search over LightGBM parameters using walk-forward CV."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric, higher = SELECTION[task]
    factory = classifiers if task == "classifier" else regressors

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = factory(random_state, params)["lightgbm"]
        return cross_validate(model, task, X, y, splits)[metric]

    study = optuna.create_study(
        direction="maximize" if higher else "minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def model_dir(cfg: Config, ticker: str) -> Path:
    return cfg.paths.models / ticker_to_filename(ticker)


def train_ticker(cfg: Config, ticker: str, feats: pd.DataFrame | None = None) -> dict:
    """Train, select and evaluate both tasks for one ticker; save artifacts. Returns metrics."""
    if feats is None:
        feats = pd.read_parquet(features_path(cfg, ticker))
    cols = feature_columns(feats)
    data = feats.dropna(subset=cols + [TARGET_RETURN, TARGET_DIRECTION])
    tc = cfg.train
    if len(data) < tc.holdout_days + 200:
        raise ValueError(f"{ticker}: only {len(data)} usable rows; need more history")

    dev, hold = data.iloc[: -tc.holdout_days], data.iloc[-tc.holdout_days :]
    X_dev, X_hold = dev[cols], hold[cols]
    splits = walk_forward_splits(len(dev), tc.n_splits, tc.gap)
    out_dir = model_dir(cfg, ticker)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "ticker": ticker,
        "features": cols,
        "n_rows": len(data),
        "dev_period": [str(dev.index[0].date()), str(dev.index[-1].date())],
        "holdout_period": [str(hold.index[0].date()), str(hold.index[-1].date())],
    }
    holdout_preds = pd.DataFrame({"actual_return": hold[TARGET_RETURN], "actual_direction": hold[TARGET_DIRECTION]})

    for task, target in (("classifier", TARGET_DIRECTION), ("regressor", TARGET_RETURN)):
        y_dev, y_hold = dev[target], hold[target]
        lgbm_params = None
        if tc.tune:
            lgbm_params = tune_lightgbm(task, X_dev, y_dev, splits, tc.n_trials, tc.random_state)
            log.info("%s %s: tuned LightGBM params %s", ticker, task, lgbm_params)
        candidates = (classifiers if task == "classifier" else regressors)(tc.random_state, lgbm_params)

        cv, hold_scores, hold_raw = {}, {}, {}
        for name, model in candidates.items():
            cv[name] = cross_validate(model, task, X_dev, y_dev, splits)
            hold_raw[name] = _fit_predict(model, task, X_dev, y_dev, X_hold)
            hold_scores[name] = _score(task, y_hold, hold_raw[name])

        metric, higher = SELECTION[task]
        ranked = sorted(
            (n for n in candidates if n != BASELINE),
            key=lambda n: cv[n][metric],
            reverse=higher,
        )
        best = ranked[0]
        base_cv = cv[BASELINE][metric]
        beats = cv[best][metric] > base_cv if higher else cv[best][metric] < base_cv
        log.info(
            "%s %s: selected %s (CV %s=%.4f vs baseline %.4f; hold-out %.4f)",
            ticker,
            task,
            best,
            metric,
            cv[best][metric],
            base_cv,
            hold_scores[best][metric],
        )

        # Final model is refit on all labelled data for live predictions.
        final = clone(candidates[best]).fit(data[cols], data[target])
        joblib.dump(final, out_dir / f"{task}.joblib")

        col = "p_up" if task == "classifier" else "pred_return"
        holdout_preds[col] = hold_raw[best]
        if task == "classifier":
            signal = (holdout_preds[col] > cfg.backtest.prob_threshold).astype(float)
        else:
            signal = (holdout_preds[col] > 0).astype(float)
        bt = backtest(hold[TARGET_RETURN], signal, cfg.backtest.cost)
        bt.to_parquet(out_dir / f"backtest_{task}.parquet")

        metrics[task] = {
            "selected": best,
            "selection_metric": metric,
            "beats_baseline_cv": bool(beats),
            "lgbm_params": lgbm_params,
            "cv": cv,
            "holdout": hold_scores,
            "backtest": backtest_stats(bt),
            "feature_importance": feature_importance(final, cols),
        }

    holdout_preds.to_parquet(out_dir / "holdout_predictions.parquet")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def train_all(cfg: Config) -> dict[str, dict]:
    results = {}
    for ticker in cfg.data.tickers:
        if not features_path(cfg, ticker).exists():
            log.warning("%s: no features, skipping training", ticker)
            continue
        try:
            results[ticker] = train_ticker(cfg, ticker)
        except Exception as exc:
            log.error("%s: training failed: %s", ticker, exc)
    return results
