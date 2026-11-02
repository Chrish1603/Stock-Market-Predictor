"""Risk forecasts: future volatility and the chance of a big drop.

Unlike next-day direction, volatility is persistent (calm and stormy periods last), so it can be
forecast with real skill. Models are compared against the standard baselines:

- persistence: next month's volatility equals the last month's
- HAR: a linear blend of last week's, month's and quarter's volatility (Corsi, 2009)

Out-of-fold predictions from every walk-forward fold are kept, which gives a ~13-year
out-of-sample record for the allocation backtest, not just the final hold-out year.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from smp.config import Config, ticker_to_filename
from smp.evaluate import classification_metrics
from smp.features import TRADING_DAYS, feature_columns, features_path, target_drop, target_logvol
from smp.models import LGBM_DEFAULTS, feature_importance
from smp.train import split_dev_holdout, walk_forward_splits

log = logging.getLogger(__name__)

BASELINE_VOL = "persistence"
BASELINE_DROP = "baseline"
HAR_COLUMNS = ["vol_5", "vol_20", "vol_60"]
IMPLIED_VOLS = ["vol_5", "vol_20", "vol_60", "ewm_vol_20", "parkinson_vol_20"]
# Models that use a fixed subset of inputs (for labelling feature importance)
MODEL_INPUTS = {"har": HAR_COLUMNS, "vol_implied": [f"drop_p_from_{v}" for v in IMPLIED_VOLS] + ["drawdown_60"]}


def _log_annualised(v):
    return np.log(np.asarray(v, dtype=float) * np.sqrt(TRADING_DAYS))


class Persistence(BaseEstimator, RegressorMixin):
    """Predicts log annualised volatility = that of the trailing `column` window. Nothing to fit."""

    def __init__(self, column: str = "vol_20"):
        self.column = column

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return _log_annualised(X[self.column])


def _har_inputs(X, columns):
    return _log_annualised(X[columns])


def drop_probability(annual_vol, horizon: int, threshold: float):
    """Chance that a driftless random walk with this volatility falls `threshold` below its start
    within `horizon` trading days (reflection principle: 2 * P(end below the barrier))."""
    s = np.asarray(annual_vol, float) * np.sqrt(horizon / TRADING_DAYS)
    return np.clip(2 * norm.cdf(np.log(1 - threshold) / s), 1e-4, 1 - 1e-4)


def _implied_drop_logits(X, horizon, threshold):
    """Random-walk drop probabilities from several trailing vol estimates (as logits), plus drawdown."""
    p = drop_probability(X[IMPLIED_VOLS].to_numpy() * np.sqrt(TRADING_DAYS), horizon, threshold)
    return np.column_stack([np.log(p / (1 - p)), X[["drawdown_60"]].to_numpy()])


def vol_models(random_state: int = 42) -> dict:
    from lightgbm import LGBMRegressor

    return {
        BASELINE_VOL: Persistence("vol_20"),
        "har": Pipeline(
            [
                ("har", FunctionTransformer(_har_inputs, kw_args={"columns": HAR_COLUMNS})),
                ("model", LinearRegression()),
            ]
        ),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "lightgbm": LGBMRegressor(**LGBM_DEFAULTS, random_state=random_state),
    }


def drop_models(random_state: int = 42, horizon: int = 20, threshold: float = 0.05) -> dict:
    """Candidates for P(big drop). `vol_implied` turns trailing volatility into a random-walk drop
    probability and calibrates it with logistic regression; it is usually the best calibrated."""
    from lightgbm import LGBMClassifier

    return {
        BASELINE_DROP: DummyClassifier(strategy="prior"),  # always predicts the historical drop rate
        "vol_implied": Pipeline(
            [
                (
                    "implied",
                    FunctionTransformer(_implied_drop_logits, kw_args={"horizon": horizon, "threshold": threshold}),
                ),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        ),
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=2000))]),
        "lightgbm": LGBMClassifier(  # cautious settings: big drops are rare, defaults overfit
            n_estimators=200,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            verbose=-1,
            random_state=random_state,
        ),
    }


def vol_metrics(y_log, pred_log) -> dict[str, float]:
    """Errors on log vol, plus mean error in annualised volatility points (0.05 = 5 points)."""
    y_log, pred_log = np.asarray(y_log, float), np.asarray(pred_log, float)
    resid = y_log - pred_log
    ss_tot = ((y_log - y_log.mean()) ** 2).sum()
    return {
        "r2": float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt((resid**2).mean())),
        "mae_vol_points": float(np.abs(np.exp(y_log) - np.exp(pred_log)).mean()),
        "corr": float(np.corrcoef(y_log, pred_log)[0, 1]) if len(y_log) > 2 else float("nan"),
        "bias": float(-resid.mean()),  # > 0: forecasts too high on average
    }


def drop_metrics(y, proba) -> dict[str, float]:
    m = classification_metrics(y, proba)
    p = np.clip(np.asarray(proba, float), 1e-6, 1 - 1e-6)
    m["brier"] = float(((p - np.asarray(y, float)) ** 2).mean())
    return {k: m[k] for k in ("roc_auc", "brier", "log_loss", "up_rate")} | {"drop_rate": m["up_rate"]}


def _predict(model, task, X):
    return model.predict_proba(X)[:, 1] if task == "drop" else model.predict(X)


def _score(task, y, pred):
    return drop_metrics(y, pred) if task == "drop" else vol_metrics(y, pred)


def risk_dir(cfg: Config, ticker: str, horizon: int):
    return cfg.paths.models / ticker_to_filename(ticker) / f"risk_h{horizon}"


def train_risk_ticker(cfg: Config, ticker: str, feats: pd.DataFrame, horizon: int) -> dict:
    """Train volatility and drop models for one ticker and horizon; save models, metrics and OOS predictions."""
    cols = feature_columns(feats)
    vol_col, drop_col = target_logvol(horizon), target_drop(horizon)
    data = feats.dropna(subset=[*cols, vol_col, drop_col])
    dev, hold = split_dev_holdout(data, cfg.train.holdout_days, horizon)
    splits = walk_forward_splits(len(dev), cfg.train.n_splits, max(cfg.train.gap, horizon))
    out_dir = risk_dir(cfg, ticker, horizon)
    out_dir.mkdir(parents=True, exist_ok=True)
    rs = cfg.train.random_state

    metrics: dict = {
        "ticker": ticker,
        "horizon": horizon,
        "features": cols,
        "drop_threshold": cfg.risk.drop_threshold,
        "dev_period": [str(dev.index[0].date()), str(dev.index[-1].date())],
        "holdout_period": [str(hold.index[0].date()), str(hold.index[-1].date())],
    }
    oos = pd.DataFrame(index=dev.index[np.concatenate([te for _, te in splits])])
    oos["fold"] = np.concatenate([np.full(len(te), i) for i, (_, te) in enumerate(splits)])
    hold_out = pd.DataFrame(index=hold.index)
    for frame, src in ((oos, dev), (hold_out, hold)):
        frame["actual_logvol"] = src.loc[frame.index, vol_col]
        frame["actual_drop"] = src.loc[frame.index, drop_col]
        frame["trailing_logvol"] = _log_annualised(src.loc[frame.index, "vol_20"])

    thr = cfg.risk.drop_threshold
    for task, target, candidates, baseline, (metric, higher), allow_baseline in (
        ("vol", vol_col, vol_models(rs), BASELINE_VOL, ("rmse", False), False),
        # Probabilities are shown to users, so pick by Brier score (calibration + ranking). If nothing
        # beats the historical drop rate, use the historical rate.
        ("drop", drop_col, drop_models(rs, horizon, thr), BASELINE_DROP, ("brier", False), True),
    ):
        cv, ho, oof_pred, ho_pred = {}, {}, {}, {}
        for name, model in candidates.items():
            folds, preds = [], []
            for tr, te in splits:
                m = clone(model).fit(dev.iloc[tr][cols], dev.iloc[tr][target])
                p = _predict(m, task, dev.iloc[te][cols])
                folds.append(_score(task, dev.iloc[te][target], p))
                preds.append(p)
            cv[name] = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
            oof_pred[name] = np.concatenate(preds)
            m = clone(model).fit(dev[cols], dev[target])
            ho_pred[name] = _predict(m, task, hold[cols])
            ho[name] = _score(task, hold[target], ho_pred[name])

        pool = [n for n in candidates if allow_baseline or n != baseline]
        best = sorted(pool, key=lambda n: cv[n][metric], reverse=higher)[0]
        final = clone(candidates[best]).fit(data[cols], data[target])
        joblib.dump(final, out_dir / f"{task}.joblib")
        col = "pred_logvol" if task == "vol" else "p_drop"
        oos[col], hold_out[col] = oof_pred[best], ho_pred[best]
        if task == "vol":
            oos["har_logvol"], hold_out["har_logvol"] = oof_pred["har"], ho_pred["har"]
        log.info(
            "%s risk %dd %s: selected %s (CV %s=%.4f vs %s %.4f; hold-out %.4f)",
            ticker,
            horizon,
            task,
            best,
            metric,
            cv[best][metric],
            baseline,
            cv[baseline][metric],
            ho[best][metric],
        )
        metrics[task] = {
            "selected": best,
            "selection_metric": metric,
            "cv": cv,
            "holdout": ho,
            "beats_baseline_cv": bool(
                cv[best][metric] < cv[baseline][metric] if not higher else cv[best][metric] > cv[baseline][metric]
            ),
            "feature_importance": feature_importance(final, MODEL_INPUTS.get(best, cols)),
        }

    oos.to_parquet(out_dir / "oos_predictions.parquet")
    hold_out.to_parquet(out_dir / "holdout_predictions.parquet")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def train_risk_all(cfg: Config) -> dict:
    if not cfg.risk.enabled:
        return {}
    results: dict = {}
    for ticker in cfg.data.tickers:
        p = features_path(cfg, ticker)
        if not p.exists():
            continue
        feats = pd.read_parquet(p)
        for h in cfg.risk.horizons:
            try:
                results.setdefault(ticker, {})[h] = train_risk_ticker(cfg, ticker, feats, h)
            except Exception as exc:
                log.error("%s risk %dd: training failed: %s", ticker, h, exc)
    return results


def risk_predictions_path(cfg: Config):
    return cfg.paths.models / "risk_predictions.parquet"


def predict_risk_ticker(cfg: Config, ticker: str, feats: pd.DataFrame, horizon: int) -> dict | None:
    d = risk_dir(cfg, ticker, horizon)
    if not (d / "metrics.json").exists():
        return None
    meta = json.loads((d / "metrics.json").read_text())
    latest = feats[meta["features"]].dropna().tail(1)
    if latest.empty:
        return None
    vol = float(np.exp(joblib.load(d / "vol.joblib").predict(latest)[0]))
    p_drop = float(joblib.load(d / "drop.joblib").predict_proba(latest)[0, 1])
    return {
        "ticker": ticker,
        "horizon": horizon,
        "as_of": latest.index[0],
        "forecast_vol": vol,
        "trailing_vol": float(latest["vol_20"].iloc[0] * np.sqrt(TRADING_DAYS)),
        "p_drop": p_drop,
        "drop_threshold": meta["drop_threshold"],
        "vol_model": meta["vol"]["selected"],
        "drop_model": meta["drop"]["selected"],
    }


def predict_risk_all(cfg: Config) -> pd.DataFrame:
    """Latest risk forecasts and suggested position size for every ticker."""
    from smp.allocation import suggested_weight

    if not cfg.risk.enabled:
        return pd.DataFrame()
    rows = []
    for ticker in cfg.data.tickers:
        p = features_path(cfg, ticker)
        if not p.exists():
            continue
        feats = pd.read_parquet(p)
        for h in cfg.risk.horizons:
            if r := predict_risk_ticker(cfg, ticker, feats, h):
                rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    alloc = df[df["horizon"] == cfg.risk.allocation_horizon].set_index("ticker")
    df["suggested_weight"] = df["ticker"].map(
        {t: suggested_weight(cfg, r.trailing_vol, r.forecast_vol) for t, r in alloc.iterrows()}
    )
    df.to_parquet(risk_predictions_path(cfg), index=False)
    mkt = df[(df["ticker"] == cfg.risk.market_ticker) & (df["horizon"] == 20)]
    if not mkt.empty:
        from smp.regime import current_regime

        current_regime(cfg, float(mkt["forecast_vol"].iloc[0]))
    for r in df[df["horizon"] == cfg.risk.allocation_horizon].itertuples():
        log.info(
            "%s risk (%dd): forecast vol %.0f%% (trailing %.0f%%), P(-%.0f%%) %.0f%%, suggested weight %.0f%%",
            r.ticker,
            r.horizon,
            r.forecast_vol * 100,
            r.trailing_vol * 100,
            r.drop_threshold * 100,
            r.p_drop * 100,
            r.suggested_weight * 100,
        )
    return df
