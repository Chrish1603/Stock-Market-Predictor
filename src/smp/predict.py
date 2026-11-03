"""Generate the latest predictions (for every horizon) from saved models."""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from smp.config import Config
from smp.features import features_path
from smp.history import update_history
from smp.safeio import atomic_write_parquet
from smp.train import model_dir

log = logging.getLogger(__name__)


def predictions_path(cfg: Config):
    return cfg.paths.models / "predictions.parquet"


class PredictionError(RuntimeError):
    """Some forecasts were skipped or made from older data; the rest were saved."""


def latest_inputs(feats: pd.DataFrame, cols: list[str], label: str, issues: list[str]) -> pd.DataFrame:
    """The newest row with every input the model needs. Missing columns are an error (retrain needed);
    an incomplete newest day falls back to the last complete one and is reported."""
    if missing := [c for c in cols if c not in feats.columns]:
        raise PredictionError(
            f"{label}: model needs inputs that are no longer built ({', '.join(missing[:5])}"
            f"{'...' if len(missing) > 5 else ''}); retrain with `smp train`"
        )
    complete = feats[cols].dropna()
    if complete.empty:
        raise PredictionError(f"{label}: no day has all the inputs the model needs")
    newest, used = feats.index.max(), complete.index.max()
    if used < newest:
        gaps = [c for c in cols if pd.isna(feats.at[newest, c])]
        issues.append(f"{label}: inputs incomplete for {newest.date()} ({', '.join(gaps[:3])}); used {used.date()}")
    return complete.tail(1)


def load_model(path):
    try:
        return joblib.load(path)
    except Exception as exc:
        raise PredictionError(f"model file {path.name} in {path.parent} can't be loaded ({exc}); retrain") from exc


def predict_ticker(cfg: Config, ticker: str, horizon: int = 1, issues: list[str] | None = None) -> dict | None:
    mdir = model_dir(cfg, ticker, horizon)
    if not (mdir / "metrics.json").exists() or not features_path(cfg, ticker).exists():
        log.warning("%s %dd: missing model or features, skipping prediction", ticker, horizon)
        return None
    meta = json.loads((mdir / "metrics.json").read_text())
    feats = pd.read_parquet(features_path(cfg, ticker))
    latest = latest_inputs(feats, meta["features"], f"{ticker} {horizon}d", issues if issues is not None else [])

    clf = load_model(mdir / "classifier.joblib")
    reg = load_model(mdir / "regressor.joblib")
    p_up = float(clf.predict_proba(latest)[0, 1])
    return {
        "ticker": ticker,
        "horizon": horizon,
        "as_of": latest.index[0],
        "p_up": p_up,
        "direction": "up" if p_up > cfg.backtest.prob_threshold else "down",
        "pred_return": float(reg.predict(latest)[0]),
        "classifier": meta["classifier"]["selected"],
        "regressor": meta["regressor"]["selected"],
    }


def predict_all(cfg: Config) -> pd.DataFrame:
    """Forecast every ticker and horizon independently; save what works, then report anything skipped."""
    rows, issues = [], []
    for t in cfg.data.tickers:
        for h in cfg.train.horizons:
            try:
                if r := predict_ticker(cfg, t, h, issues):
                    rows.append(r)
            except PredictionError as exc:
                issues.append(str(exc))
            except Exception as exc:  # never let one ticker stop the others
                issues.append(f"{t} {h}d: {type(exc).__name__}: {exc}")
    df = pd.DataFrame(rows)
    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(df, predictions_path(cfg), index=False)
    update_history(cfg, df)
    from smp.risk import predict_risk_all

    risk = predict_risk_all(cfg, issues)
    if not risk.empty:
        from smp.risk_history import update_risk_history

        try:
            update_risk_history(cfg, risk)
        except Exception as exc:  # the forecasts are saved; report the record problem
            issues.append(f"risk track record not updated: {type(exc).__name__}: {exc}")
    for r in rows:
        log.info(
            "%s %dd (as of %s): P(up)=%.3f, predicted return=%+.4f%%",
            r["ticker"],
            r["horizon"],
            r["as_of"].date(),
            r["p_up"],
            r["pred_return"] * 100,
        )
    if issues:
        for i in issues:
            log.error("prediction problem: %s", i)
        raise PredictionError(f"{len(issues)} problem(s): " + "; ".join(issues))
    return df
