"""Keep every prediction and score it once the next day's close is known.

This is the live track record: unlike the hold-out test, these predictions were made
before the outcome existed, so they can't have been tuned to it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from smp.clean import clean_path
from smp.config import Config

log = logging.getLogger(__name__)

KEY = ["ticker", "as_of"]
OUTCOME_COLS = ["target_date", "actual_return", "actual_direction", "correct"]


def history_path(cfg: Config):
    return cfg.paths.history / "predictions.parquet"


def load_history(cfg: Config) -> pd.DataFrame:
    p = history_path(cfg)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def save_history(cfg: Config, hist: pd.DataFrame) -> None:
    cfg.paths.history.mkdir(parents=True, exist_ok=True)
    hist.sort_values(KEY).reset_index(drop=True).to_parquet(history_path(cfg), index=False)


def append_predictions(hist: pd.DataFrame, preds: pd.DataFrame, made_at: datetime | None = None) -> pd.DataFrame:
    """Add new predictions. The first prediction for a (ticker, day) is kept, so history is never rewritten."""
    if preds.empty:
        return hist
    new = preds.copy()
    new["as_of"] = pd.to_datetime(new["as_of"])
    new["made_at"] = pd.Timestamp(made_at or datetime.now(timezone.utc)).tz_localize(None)
    for col in OUTCOME_COLS:
        new[col] = np.nan
    new["target_date"] = pd.NaT
    combined = pd.concat([hist, new], ignore_index=True) if not hist.empty else new
    return combined.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)


def score(hist: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fill in outcomes for predictions whose next trading day has closed."""
    if hist.empty:
        return hist
    hist = hist.copy()
    pending = hist["actual_return"].isna()
    for i in hist.index[pending]:
        px = prices.get(hist.at[i, "ticker"])
        if px is None:
            continue
        as_of = hist.at[i, "as_of"]
        later = px.index[px.index > as_of]
        if as_of not in px.index or later.empty:
            continue
        nxt = later[0]
        ret = float(np.log(px.at[nxt, "Close"] / px.at[as_of, "Close"]))
        up = float(ret > 0)
        hist.at[i, "target_date"] = nxt
        hist.at[i, "actual_return"] = ret
        hist.at[i, "actual_direction"] = up
        hist.at[i, "correct"] = float((hist.at[i, "direction"] == "up") == bool(up))
    return hist


def summarize(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker live results, compared with always guessing 'up' over the same days."""
    cols = [
        "ticker",
        "predictions",
        "scored",
        "hit_rate",
        "always_up_hit_rate",
        "brier",
        "avg_return_when_up_call",
        "avg_return_when_down_call",
        "first",
        "last",
    ]
    if hist.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for ticker, g in hist.groupby("ticker"):
        s = g.dropna(subset=["actual_return"])
        up_calls, down_calls = s[s["direction"] == "up"], s[s["direction"] == "down"]
        rows.append(
            {
                "ticker": ticker,
                "predictions": len(g),
                "scored": len(s),
                "hit_rate": s["correct"].mean() if len(s) else np.nan,
                "always_up_hit_rate": s["actual_direction"].mean() if len(s) else np.nan,
                "brier": ((s["p_up"] - s["actual_direction"]) ** 2).mean() if len(s) else np.nan,
                "avg_return_when_up_call": up_calls["actual_return"].mean() if len(up_calls) else np.nan,
                "avg_return_when_down_call": down_calls["actual_return"].mean() if len(down_calls) else np.nan,
                "first": g["as_of"].min(),
                "last": g["as_of"].max(),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def update_history(cfg: Config, preds: pd.DataFrame) -> pd.DataFrame:
    """Score pending predictions with the latest prices, then record today's predictions."""
    prices = {t: pd.read_parquet(p) for t in cfg.data.tickers if (p := clean_path(cfg, t)).exists()}
    hist = score(load_history(cfg), prices)
    hist = append_predictions(hist, preds)
    save_history(cfg, hist)
    for r in summarize(hist).itertuples():
        if r.scored:
            log.info(
                "%s live record: %d scored, hit rate %.1f%% (always-up %.1f%%)",
                r.ticker,
                r.scored,
                r.hit_rate * 100,
                r.always_up_hit_rate * 100,
            )
        else:
            log.info("%s live record: %d prediction(s) waiting for outcomes", r.ticker, r.predictions)
    return hist
