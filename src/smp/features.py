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
from smp.ingest import earnings_path

log = logging.getLogger(__name__)

TARGET_PREFIX = "target_"


def target_return(h: int = 1) -> str:
    """Column name of the h-day-ahead log return."""
    return f"target_return_{h}"


def target_logvol(h: int) -> str:
    """Column name of the log annualised realised volatility over the next h days."""
    return f"target_logvol_{h}"


def target_drop(h: int) -> str:
    """Column name of the 'close falls drop_threshold below today's within h days' label."""
    return f"target_drop_{h}"


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


TRADING_DAYS = 252
RATE_TICKERS = {"tnx", "irx"}  # Treasury yields, quoted in percent
MAX_DAYS_TO_EARNINGS = 20  # dates are usually announced 2-4 weeks ahead; don't assume more
MAX_DAYS_SINCE_EARNINGS = 90


def earnings_features(index: pd.DatetimeIndex, earnings: pd.DataFrame) -> pd.DataFrame:
    """Business days to the next / since the last earnings reaction day, and the last EPS surprise.

    The surprise only becomes known on its reaction day, so day t sees reports with reaction_date <= t.
    """
    e = earnings.dropna(subset=["reaction_date"]).sort_values("reaction_date")
    rd = pd.DatetimeIndex(e["reaction_date"]).values.astype("datetime64[D]")
    surprise = e["surprise_pct"].ffill().fillna(0.0).to_numpy()
    days = index.values.astype("datetime64[D]")
    pos = np.searchsorted(rd, days, side="right")  # reaction days <= t

    has_next, has_prev = pos < len(rd), pos > 0
    to_next = np.full(len(days), MAX_DAYS_TO_EARNINGS, dtype=float)
    to_next[has_next] = np.busday_count(days[has_next], rd[pos[has_next]])
    since = np.full(len(days), MAX_DAYS_SINCE_EARNINGS, dtype=float)
    since[has_prev] = np.busday_count(rd[pos[has_prev] - 1], days[has_prev])
    last_surprise = np.zeros(len(days))
    last_surprise[has_prev] = surprise[pos[has_prev] - 1]

    return pd.DataFrame(
        {
            "days_to_earnings": np.clip(to_next, 0, MAX_DAYS_TO_EARNINGS),
            "days_since_earnings": np.clip(since, 0, MAX_DAYS_SINCE_EARNINGS),
            "last_eps_surprise": np.clip(last_surprise, -100, 100),
        },
        index=index,
    )


def build_features(
    df: pd.DataFrame,
    market: dict[str, pd.DataFrame] | None = None,
    horizons: list[int] | tuple[int, ...] = (1,),
    sector: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    risk_horizons: list[int] | tuple[int, ...] = (),
    drop_threshold: float = 0.05,
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

    # Volatility (daily units)
    for n in (5, 10, 20, 60):
        out[f"vol_{n}"] = log_ret.rolling(n).std()
    out["ewm_vol_20"] = log_ret.ewm(span=20, min_periods=20).std()
    out["downside_vol_20"] = np.sqrt((log_ret.clip(upper=0) ** 2).rolling(20).mean())
    hl = np.log(df["High"] / df["Low"]) ** 2
    out["parkinson_vol_20"] = np.sqrt(hl.rolling(20).mean() / (4 * np.log(2)))  # uses the daily range
    out["vol_ratio_5_60"] = out["vol_5"] / out["vol_60"]
    out["drawdown_60"] = c / c.rolling(60).max() - 1
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
    rates: dict[str, pd.Series] = {}
    for name, mdf in (market or {}).items():
        mc = mdf["Close"].reindex(df.index, method="ffill")
        key = ticker_to_filename(name).lower()
        if key == "vix":
            out["vix_level"] = mc
            out["vix_change_5"] = np.log(mc / mc.shift(5))
        elif key in RATE_TICKERS:
            # Changes only: rate levels drift for years, so trees would use them to recognise eras.
            rates[key] = mc
            out[f"{key}_change_5"] = mc - mc.shift(5)  # percentage points
            out[f"{key}_change_20"] = mc - mc.shift(20)
        else:
            out[f"{key}_ret_1"] = np.log(mc / mc.shift(1))
            out[f"{key}_ret_5"] = np.log(mc / mc.shift(5))
    if RATE_TICKERS <= set(rates):
        curve = rates["tnx"] - rates["irx"]  # 10-year minus 3-month
        out["yield_curve_change_20"] = curve - curve.shift(20)
        out["yield_curve_inverted"] = (curve < 0).astype(float).where(curve.notna())

    # Sector: how the stock's sector fund is doing, and the stock relative to it
    if sector is not None:
        sc = sector["Close"].reindex(df.index, method="ffill")
        out["sector_ret_5"] = np.log(sc / sc.shift(5))
        out["sector_ret_20"] = np.log(sc / sc.shift(20))
        out["sector_sma_ratio_50"] = sc / sc.rolling(50).mean() - 1
        out["rel_strength_20"] = out["ret_20"] - out["sector_ret_20"]
        out["rel_strength_60"] = np.log(c / c.shift(60)) - np.log(sc / sc.shift(60))

    # Earnings calendar
    if earnings is not None and not earnings.empty:
        out = out.join(earnings_features(df.index, earnings))

    out = out.replace([np.inf, -np.inf], np.nan)

    # Targets: next-day log return and its direction
    for h in sorted({1, *horizons}):
        ret = np.log(c.shift(-h) / c)
        out[target_return(h)] = ret
        out[target_direction(h)] = (ret > 0).astype(float).where(ret.notna())

    # Risk targets: realised volatility and "big drop" over the next h days
    for h in risk_horizons:
        fwd_sq = (log_ret**2).rolling(h).sum().shift(-h)  # sum of squared returns t+1 .. t+h
        out[target_logvol(h)] = 0.5 * np.log(fwd_sq * TRADING_DAYS / h)  # log annualised vol
        fwd_min = c.shift(-1).rolling(h).min().shift(-(h - 1))  # lowest close t+1 .. t+h
        out[target_drop(h)] = (fwd_min / c - 1 <= -drop_threshold).astype(float).where(fwd_min.notna())
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if not col.startswith(TARGET_PREFIX)]


def features_path(cfg: Config, ticker: str):
    return cfg.paths.features / f"{ticker_to_filename(ticker)}.parquet"


EARNINGS_DEFAULTS = {
    "days_to_earnings": MAX_DAYS_TO_EARNINGS,
    "days_since_earnings": MAX_DAYS_SINCE_EARNINGS,
    "last_eps_surprise": 0.0,
}
RANKED = ["ret_5", "ret_20", "sma_ratio_50", "vol_20", "rsi_14", "volume_z", "atr_14"]
MIN_TICKERS_FOR_RANKS = 10


def fill_earnings(feats: pd.DataFrame) -> pd.DataFrame:
    """Give tickers without earnings (funds) neutral values, so every ticker has the same columns."""
    feats = feats.copy()
    feats["has_earnings"] = float("days_to_earnings" in feats)
    for col, default in EARNINGS_DEFAULTS.items():
        feats[col] = feats[col].fillna(default) if col in feats else float(default)
    return feats


def add_cross_sectional_ranks(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add each ticker's percentile rank among all tickers on the same day (0 = lowest, 1 = highest).

    Uses only same-day values, so it adds no lookahead.
    """
    out = {t: f.copy() for t, f in frames.items()}
    for col in RANKED:
        wide = pd.DataFrame({t: f[col] for t, f in frames.items() if col in f})
        ranks = wide.rank(axis=1, pct=True)
        for t in wide.columns:
            out[t][f"xs_rank_{col}"] = ranks[t].reindex(out[t].index)
    return out


def build_all(cfg: Config) -> dict[str, int]:
    """Build feature files for the predicted tickers and the training universe."""
    cfg.paths.features.mkdir(parents=True, exist_ok=True)
    market = {}
    for t in cfg.data.market_tickers:
        p = clean_path(cfg, t)
        if p.exists():
            market[t] = pd.read_parquet(p)
        else:
            log.warning("market ticker %s has no clean data; its features are skipped", t)

    frames: dict[str, pd.DataFrame] = {}
    for ticker in cfg.data.feature_tickers:
        p = clean_path(cfg, ticker)
        if not p.exists():
            log.warning("%s: no clean data, skipping features", ticker)
            continue
        sector = None
        if (fund := cfg.data.sectors.get(ticker)) and (sp := clean_path(cfg, fund)).exists():
            sector = pd.read_parquet(sp)
        ep = earnings_path(cfg, ticker)
        earnings = pd.read_parquet(ep) if cfg.data.earnings and ep.exists() else None
        risk_h = cfg.risk.horizons if cfg.risk.enabled else []
        feats = build_features(
            pd.read_parquet(p), market, cfg.train.horizons, sector, earnings, risk_h, cfg.risk.drop_threshold
        )
        frames[ticker] = fill_earnings(feats) if cfg.data.earnings else feats

    if cfg.data.cross_sectional and len(frames) >= MIN_TICKERS_FOR_RANKS:
        frames = add_cross_sectional_ranks(frames)

    counts: dict[str, int] = {}
    for ticker, feats in frames.items():
        feats.to_parquet(features_path(cfg, ticker))
        counts[ticker] = len(feats)
    n_cols = {len(feature_columns(f)) for f in frames.values()}
    log.info("features for %d tickers, %s features each", len(frames), "/".join(map(str, sorted(n_cols))))
    return counts
