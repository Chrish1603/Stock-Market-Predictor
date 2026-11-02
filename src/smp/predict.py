"""Generate the latest next-day predictions from saved models."""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from smp.config import Config
from smp.features import features_path
from smp.train import model_dir

log = logging.getLogger(__name__)


def predictions_path(cfg: Config):
    return cfg.paths.models / "predictions.parquet"


def predict_ticker(cfg: Config, ticker: str) -> dict | None:
    mdir = model_dir(cfg, ticker)
    if not (mdir / "metrics.json").exists() or not features_path(cfg, ticker).exists():
        log.warning("%s: missing model or features, skipping prediction", ticker)
        return None
    meta = json.loads((mdir / "metrics.json").read_text())
    cols = meta["features"]
    feats = pd.read_parquet(features_path(cfg, ticker))
    latest = feats[cols].dropna().tail(1)
    if latest.empty:
        log.warning("%s: no complete feature row to predict from", ticker)
        return None

    clf = joblib.load(mdir / "classifier.joblib")
    reg = joblib.load(mdir / "regressor.joblib")
    p_up = float(clf.predict_proba(latest)[0, 1])
    return {
        "ticker": ticker,
        "as_of": latest.index[0],
        "p_up": p_up,
        "direction": "up" if p_up > cfg.backtest.prob_threshold else "down",
        "pred_return": float(reg.predict(latest)[0]),
        "classifier": meta["classifier"]["selected"],
        "regressor": meta["regressor"]["selected"],
    }


def predict_all(cfg: Config) -> pd.DataFrame:
    rows = [r for t in cfg.data.tickers if (r := predict_ticker(cfg, t))]
    df = pd.DataFrame(rows)
    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    df.to_parquet(predictions_path(cfg), index=False)
    for r in rows:
        log.info(
            "%s (as of %s): P(up)=%.3f, predicted return=%+.4f%%",
            r["ticker"],
            r["as_of"].date(),
            r["p_up"],
            r["pred_return"] * 100,
        )
    return df
