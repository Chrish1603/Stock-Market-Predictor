"""Technical-indicator features and next-day targets.

Every feature at row t uses only data up to and including day t's close.
Targets describe day t -> t+h (h trading days ahead) and are the only columns that look forward.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from smp.clean import clean_path
from smp.config import Config, ticker_to_filename

log = logging.getLogger(__name__)

TARGET_PREFIX = "target_"


def target_return(h: int = 1) -> str:
    """Column name of the h-day-ahead log return."""
    return f"target_return_{h}"


def target_direction(h: int = 1) -> str:
    """Column name of the h-day-ahead up (1) / down (0) label."""
    return f"target_direction_{h}"


# --- indicator helpers (also used by the dashboard) -------------------------------------------


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's relative strength index, 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


# --- feature construction ---------------------------------------------------------------------


def build_features(
    df: pd.DataFrame, market: dict[str, pd.DataFrame] | None = None, horizons: list[int] | tuple[int, ...] = (1,)
) -> pd.DataFrame:
    """Return a frame of features plus targets for each horizon, indexed like `df`.

    The 1-day target is always included; backtests use it as the daily return.
    """
    c, v = df["Close"], df["Volume"]
    log_ret = np.log(c).diff()
    out = pd.DataFrame(index=df.index)

    # Returns
    for n in (1, 5, 10, 20):
        out[f"ret_{n}"] = np.log(c / c.shift(n))
    for lag in (1, 2, 3, 5):
        out[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # Trend
    for n in (10, 20, 50, 200):
        out[f"sma_ratio_{n}"] = c / c.rolling(n).mean() - 1
    for n in (12, 26):
        out[f"ema_ratio_{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
    m = macd(c)
    out["macd"] = m["macd"] / c
    out["macd_hist"] = m["hist"] / c

    # Momentum
    out["rsi_14"] = rsi(c, 14)
    low14, high14 = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    out["stoch_k"] = (c - low14) / (high14 - low14)

    # Volatility
    out["vol_20"] = log_ret.rolling(20).std()
    out["vol_60"] = log_ret.rolling(60).std()
    out["atr_14"] = atr(df, 14) / c
    bb = bollinger(c)
    out["bb_pctb"] = (c - bb["lower"]) / (bb["upper"] - bb["lower"])
    out["hl_range"] = (df["High"] - df["Low"]) / c

    # Volume
    vol_mean, vol_std = v.rolling(20).mean(), v.rolling(20).std()
    out["volume_z"] = (v - vol_mean) / vol_std
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["obv_change_10"] = obv.diff(10) / v.rolling(10).sum()

    # Calendar
    out["day_of_week"] = df.index.dayofweek
    out["month"] = df.index.month

    # Market context (aligned to this ticker's dates using only past values)
    for name, mdf in (market or {}).items():
        mc = mdf["Close"].reindex(df.index, method="ffill")
        key = ticker_to_filename(name).lower()
        if key == "vix":
            out["vix_level"] = mc
            out["vix_change_5"] = np.log(mc / mc.shift(5))
        else:
            out[f"{key}_ret_1"] = np.log(mc / mc.shift(1))
            out[f"{key}_ret_5"] = np.log(mc / mc.shift(5))

    out = out.replace([np.inf, -np.inf], np.nan)

    # Targets: next-day log return and its direction
    for h in sorted({1, *horizons}):
        ret = np.log(c.shift(-h) / c)
        out[target_return(h)] = ret
        out[target_direction(h)] = (ret > 0).astype(float).where(ret.notna())
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not col.startswith(TARGET_PREFIX)]


def features_path(cfg: Config, ticker: str):
    return cfg.paths.features / f"{ticker_to_filename(ticker)}.parquet"


def build_all(cfg: Config) -> dict[str, int]:
    """Build feature files for every modelled ticker from the cleaned data."""
    cfg.paths.features.mkdir(parents=True, exist_ok=True)
    market = {}
    for t in cfg.data.market_tickers:
        p = clean_path(cfg, t)
        if p.exists():
            market[t] = pd.read_parquet(p)
        else:
            log.warning("market ticker %s has no clean data; its features are skipped", t)

    counts: dict[str, int] = {}
    for ticker in cfg.data.tickers:
        p = clean_path(cfg, ticker)
        if not p.exists():
            log.warning("%s: no clean data, skipping features", ticker)
            continue
        feats = build_features(pd.read_parquet(p), market, cfg.train.horizons)
        feats.to_parquet(features_path(cfg, ticker))
        counts[ticker] = len(feats)
        log.info("%s: %d feature rows, %d features", ticker, len(feats), len(feature_columns(feats)))
    return counts
