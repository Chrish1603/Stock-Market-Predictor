"""Market regime: calm / normal / stressed, for the market ticker (SPY by default).

Inputs, all known at each day's close:
- vol percentile: the 20-day vol forecast ranked against all realised volatility seen so far
- breadth: share of stocks (tickers + universe) trading above their 200-day average
- trend: whether the market is above its own 200-day average

The thresholds below were fixed up front, not tuned on results. The history uses out-of-sample
vol forecasts, so the per-regime statistics are a fair check of whether the labels mean anything.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from smp.allocation import out_of_sample_forecasts
from smp.config import Config
from smp.features import TRADING_DAYS, features_path, target_drop, target_logvol, target_return

log = logging.getLogger(__name__)

REGIMES = ["calm", "normal", "stressed"]
STRESSED_VOL_PCT = 0.80
CALM_VOL_PCT = 0.40
WEAK_BREADTH, STRONG_BREADTH = 0.40, 0.60
MIN_HISTORY = 252
MIN_BREADTH_TICKERS = 10
HORIZON = 20


def classify(vol_pct: float, breadth: float | None, trend_up: bool) -> str:
    """Rules: stressed if vol is in its top 20% or breadth is weak in a downtrend; calm if vol is in
    its bottom 40% with healthy breadth in an uptrend; otherwise normal. Missing breadth is ignored."""
    has_breadth = breadth is not None and not np.isnan(breadth)
    if vol_pct >= STRESSED_VOL_PCT or (has_breadth and breadth < WEAK_BREADTH and not trend_up):
        return "stressed"
    if vol_pct <= CALM_VOL_PCT and trend_up and (not has_breadth or breadth >= STRONG_BREADTH):
        return "calm"
    return "normal"


def breadth_series(cfg: Config) -> pd.Series:
    """Share of stocks above their 200-day average each day (funds like SPY excluded)."""
    cols = {}
    for t in cfg.data.feature_tickers:
        p = features_path(cfg, t)
        if t == cfg.risk.market_ticker or not p.exists():
            continue
        f = pd.read_parquet(p, columns=["sma_ratio_200", "has_earnings"] if cfg.data.earnings else ["sma_ratio_200"])
        if "has_earnings" in f and not f["has_earnings"].any():
            continue  # skip funds
        cols[t] = f["sma_ratio_200"]
    if len(cols) < MIN_BREADTH_TICKERS:
        return pd.Series(dtype=float)
    wide = pd.DataFrame(cols)
    above = (wide > 0).where(wide.notna())
    return above.mean(axis=1).where(above.count(axis=1) >= MIN_BREADTH_TICKERS)


def expanding_percentile(values: pd.Series, history: pd.Series) -> pd.Series:
    """Rank each value against the history strictly before its date."""
    hist = history.dropna()
    h_idx, h_vals = hist.index.to_numpy(), hist.to_numpy()
    out = []
    for date, v in values.items():
        past = h_vals[h_idx < date.to_datetime64()]
        out.append((past < v).mean() if len(past) >= MIN_HISTORY else np.nan)
    return pd.Series(out, index=values.index)


def regime_history(cfg: Config) -> pd.DataFrame:
    """Out-of-sample regime label for each day, with what happened over the next 20 days."""
    mkt = cfg.risk.market_ticker
    feats = pd.read_parquet(features_path(cfg, mkt))
    realised = feats["vol_20"] * np.sqrt(TRADING_DAYS)
    fc = out_of_sample_forecasts(cfg, mkt, HORIZON)
    df = pd.DataFrame(index=fc.index)
    df["forecast_vol"] = np.exp(fc["pred_logvol"])
    df["vol_pct"] = expanding_percentile(df["forecast_vol"], realised)
    df["breadth"] = breadth_series(cfg).reindex(df.index)
    df["trend_up"] = feats["sma_ratio_200"].reindex(df.index) > 0
    df["regime"] = [classify(v, b, t) for v, b, t in zip(df["vol_pct"], df["breadth"], df["trend_up"], strict=True)]
    df.loc[df["vol_pct"].isna(), "regime"] = None
    # outcomes (for validation only; never used to label)
    df["next_vol"] = np.exp(feats[target_logvol(HORIZON)].reindex(df.index))
    df["next_return"] = feats[target_return(HORIZON)].reindex(df.index) if target_return(HORIZON) in feats else np.nan
    df["next_drop"] = feats[target_drop(HORIZON)].reindex(df.index)
    return df


def validate(hist: pd.DataFrame) -> dict:
    """Average next-20-day outcomes per regime."""
    out = {}
    h = hist.dropna(subset=["regime", "next_vol"])
    for r in REGIMES:
        g = h[h["regime"] == r]
        out[r] = {
            "share_of_days": float(len(g) / len(h)) if len(h) else 0.0,
            "next_20d_vol": float(g["next_vol"].mean()) if len(g) else float("nan"),
            "next_20d_return": float(np.expm1(g["next_return"]).mean()) if len(g) else float("nan"),
            "next_20d_drop_rate": float(g["next_drop"].mean()) if len(g) else float("nan"),
        }
    return out


def market_dir(cfg: Config):
    return cfg.paths.models / "_market"


def build_regime(cfg: Config) -> dict:
    """Save the regime history and its validation table (run after risk training)."""
    if not cfg.risk.enabled:
        return {}
    try:
        hist = regime_history(cfg)
    except FileNotFoundError:
        log.warning("no risk forecasts for %s; skipping market regime", cfg.risk.market_ticker)
        return {}
    out = market_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(out / "regime_history.parquet")
    stats = validate(hist)
    (out / "regime_validation.json").write_text(json.dumps(stats, indent=2))
    for r, s in stats.items():
        log.info(
            "regime %-8s %4.0f%% of days | next 20d: vol %.0f%%, return %+.1f%%, 5%% drop %.0f%%",
            r,
            s["share_of_days"] * 100,
            s["next_20d_vol"] * 100,
            s["next_20d_return"] * 100,
            s["next_20d_drop_rate"] * 100,
        )
    return stats


def current_regime(cfg: Config, forecast_vol: float) -> dict:
    """Today's regime from today's 20-day vol forecast for the market ticker."""
    feats = pd.read_parquet(features_path(cfg, cfg.risk.market_ticker))
    realised = (feats["vol_20"] * np.sqrt(TRADING_DAYS)).dropna()
    as_of = feats["sma_ratio_200"].dropna().index[-1]
    vol_pct = float((realised.loc[:as_of] < forecast_vol).mean())
    breadth = breadth_series(cfg)
    b = float(breadth.loc[:as_of].dropna().iloc[-1]) if not breadth.dropna().empty else None
    trend_up = bool(feats["sma_ratio_200"].loc[as_of] > 0)
    info = {
        "as_of": str(as_of.date()),
        "ticker": cfg.risk.market_ticker,
        "regime": classify(vol_pct, b, trend_up),
        "forecast_vol": forecast_vol,
        "vol_percentile": vol_pct,
        "breadth": b,
        "trend_up": trend_up,
    }
    out = market_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regime.json").write_text(json.dumps(info, indent=2))
    log.info(
        "market regime (%s, %s): %s | vol forecast %.0f%% (percentile %.0f), breadth %s, trend %s",
        info["ticker"],
        info["as_of"],
        info["regime"].upper(),
        forecast_vol * 100,
        vol_pct * 100,
        f"{b:.0%}" if b is not None else "n/a",
        "up" if trend_up else "down",
    )
    return info
