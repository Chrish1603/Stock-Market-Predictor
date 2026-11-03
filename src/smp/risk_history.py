"""Keep every risk forecast and score it once its horizon has passed.

The live counterpart of the walk-forward risk tests: each day's volatility forecast and chance of a
big drop are saved before the outcome exists, then compared with what happened and with the simple
alternatives (last month's volatility, the historical drop rate). Outcomes use the same definitions
as the training targets in `smp.features`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from smp.clean import clean_path
from smp.config import Config
from smp.features import TRADING_DAYS
from smp.history import KEY, load_record, save_record

log = logging.getLogger(__name__)

RECORD_COLS = [
    "ticker",
    "horizon",
    "as_of",
    "forecast_vol",
    "trailing_vol",
    "p_drop",
    "base_rate",
    "drop_threshold",
    "vol_model",
    "drop_model",
]
OUTCOME_COLS = ["realised_vol", "dropped"]


def risk_history_path(cfg: Config):
    return cfg.paths.history / "risk_predictions.parquet"


def risk_backup_path(cfg: Config):
    return cfg.paths.history / "risk_predictions.backup.parquet"


def load_risk_history(cfg: Config) -> pd.DataFrame:
    return load_record(risk_history_path(cfg), risk_backup_path(cfg))


def save_risk_history(cfg: Config, hist: pd.DataFrame) -> None:
    save_record(risk_history_path(cfg), risk_backup_path(cfg), hist)


def append_risk(hist: pd.DataFrame, preds: pd.DataFrame, made_at: datetime | None = None) -> pd.DataFrame:
    """Add new forecasts. The first per (ticker, horizon, day) is kept, so history is never rewritten."""
    if preds.empty:
        return hist
    new = preds[[c for c in RECORD_COLS if c in preds]].copy()
    new["as_of"] = pd.to_datetime(new["as_of"])
    if "base_rate" not in new:
        new["base_rate"] = np.nan
    new["made_at"] = pd.Timestamp(made_at or datetime.now(timezone.utc)).tz_localize(None)
    new["target_date"] = pd.NaT
    for col in OUTCOME_COLS:
        new[col] = np.nan
    combined = pd.concat([hist, new], ignore_index=True) if not hist.empty else new
    return combined.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)


def outcome(px: pd.DataFrame, as_of: pd.Timestamp, horizon: int, threshold: float) -> dict | None:
    """Realised annualised volatility and whether the close fell `threshold` below as_of's, over the
    next `horizon` trading days; None until all of them have closed."""
    close = px["Close"]
    if as_of not in close.index:
        return None
    i = close.index.get_loc(as_of)
    window = close.iloc[i : i + horizon + 1]  # as_of and the next `horizon` closes
    if len(window) < horizon + 1:
        return None
    rets = np.log(window / window.shift(1)).dropna()
    return {
        "target_date": window.index[-1],
        "realised_vol": float(np.sqrt((rets**2).sum() * TRADING_DAYS / horizon)),
        "dropped": float(window.iloc[1:].min() / window.iloc[0] - 1 <= -threshold),
    }


def score_risk(hist: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fill in outcomes for forecasts whose horizon has fully closed."""
    if hist.empty:
        return hist
    hist = hist.copy()
    for i in hist.index[hist["realised_vol"].isna()]:
        px = prices.get(hist.at[i, "ticker"])
        if px is None:
            continue
        res = outcome(px, hist.at[i, "as_of"], int(hist.at[i, "horizon"]), float(hist.at[i, "drop_threshold"]))
        if res:
            for col, v in res.items():
                hist.at[i, col] = v
    return hist


SUMMARY_COLS = [
    "ticker",
    "horizon",
    "predictions",
    "scored",
    "vol_error",
    "trailing_vol_error",
    "closer_share",
    "vol_bias",
    "drops",
    "avg_p_drop",
    "brier",
    "base_rate_brier",
    "first",
    "last",
]


def summarize_risk(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker, per-horizon live risk results.

    vol_error / trailing_vol_error: typical size of the miss, as mean |log(forecast / realised)|
    (0.20 ≈ off by 20%), for the forecast and for "same as last month". closer_share: how often the
    forecast was closer than last month's figure. vol_bias > 0 means forecasts ran high.
    brier / base_rate_brier: the drop probability vs quoting the historical drop rate (lower is better).
    """
    if hist.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    rows = []
    for (ticker, horizon), g in hist.groupby(["ticker", "horizon"]):
        s = g.dropna(subset=["realised_vol"])
        s = s[s["realised_vol"] > 0]
        err = np.log(s["forecast_vol"] / s["realised_vol"])
        trail_err = np.log(s["trailing_vol"] / s["realised_vol"])
        b = s.dropna(subset=["base_rate"])
        any_ = len(s) > 0
        rows.append(
            {
                "ticker": ticker,
                "horizon": int(horizon),
                "predictions": len(g),
                "scored": len(s),
                "vol_error": err.abs().mean() if any_ else np.nan,
                "trailing_vol_error": trail_err.abs().mean() if any_ else np.nan,
                "closer_share": (err.abs() < trail_err.abs()).mean() if any_ else np.nan,
                "vol_bias": err.mean() if any_ else np.nan,
                "drops": int(s["dropped"].sum()),
                "avg_p_drop": s["p_drop"].mean() if any_ else np.nan,
                "brier": ((s["p_drop"] - s["dropped"]) ** 2).mean() if any_ else np.nan,
                "base_rate_brier": ((b["base_rate"] - b["dropped"]) ** 2).mean() if len(b) else np.nan,
                "first": g["as_of"].min(),
                "last": g["as_of"].max(),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def update_risk_history(cfg: Config, preds: pd.DataFrame) -> pd.DataFrame:
    """Score pending risk forecasts with the latest prices, then record today's."""
    prices = {t: pd.read_parquet(p) for t in cfg.data.tickers if (p := clean_path(cfg, t)).exists()}
    hist = score_risk(load_risk_history(cfg), prices)
    hist = append_risk(hist, preds)
    save_risk_history(cfg, hist)
    for r in summarize_risk(hist).itertuples():
        if r.scored:
            log.info(
                "%s risk %dd live record: %d scored, vol off by %.0f%% (last month's vol: %.0f%%), "
                "%d drop(s), Brier %.3f (historical rate %.3f)",
                r.ticker,
                r.horizon,
                r.scored,
                r.vol_error * 100,
                r.trailing_vol_error * 100,
                r.drops,
                r.brier,
                r.base_rate_brier,
            )
        else:
            log.info("%s risk %dd live record: %d forecast(s) waiting", r.ticker, r.horizon, r.predictions)
    return hist
