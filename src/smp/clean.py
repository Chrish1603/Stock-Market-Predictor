"""Validate and clean raw OHLCV data, producing a data-quality report."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from smp.config import Config, ticker_to_filename
from smp.ingest import OHLCV, raw_path

log = logging.getLogger(__name__)

PRICES = ["Open", "High", "Low", "Close"]
LARGE_GAP_DAYS = 5  # calendar days between rows; longer than a holiday weekend


def clean_ohlcv(df: pd.DataFrame, ffill_limit: int = 2, outlier_z: float = 6.0) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned copy of `df` and a report of what was changed or flagged."""
    missing = set(OHLCV) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    report: dict = {"rows_in": int(len(df))}
    df = df[OHLCV].copy()

    # Schema: naive datetime index, sorted, unique, numeric floats.
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    df = df.sort_index()
    dupes = df.index.duplicated(keep="last")
    report["duplicates_removed"] = int(dupes.sum())
    df = df[~dupes]
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)

    # Non-positive prices are invalid; treat as missing.
    df[PRICES] = df[PRICES].where(df[PRICES] > 0)
    missing_before = int(df[PRICES].isna().any(axis=1).sum())
    df[PRICES] = df[PRICES].ffill(limit=ffill_limit)
    still_missing = df[PRICES].isna().any(axis=1)
    report["missing_values_filled"] = missing_before - int(still_missing.sum())
    report["rows_dropped"] = int(still_missing.sum())
    df = df[~still_missing]
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)

    # OHLC consistency: High must be the max and Low the min of the bar.
    bar_max = df[PRICES].max(axis=1)
    bar_min = df[PRICES].min(axis=1)
    bad = (df["High"] < bar_max) | (df["Low"] > bar_min)
    report["ohlc_fixed"] = int(bad.sum())
    df["High"], df["Low"] = bar_max, bar_min

    # Flag (don't drop) extreme moves; they may be real events.
    log_ret = np.log(df["Close"]).diff()
    z = (log_ret - log_ret.mean()) / log_ret.std()
    outliers = z[z.abs() > outlier_z]
    report["outlier_dates"] = [d.strftime("%Y-%m-%d") for d in outliers.index]

    gaps = df.index.to_series().diff().dt.days
    report["large_gaps"] = [
        {"date": d.strftime("%Y-%m-%d"), "days": int(g)} for d, g in gaps[gaps > LARGE_GAP_DAYS].items()
    ]
    report["rows_out"] = int(len(df))
    report["start"] = df.index.min().strftime("%Y-%m-%d") if len(df) else None
    report["end"] = df.index.max().strftime("%Y-%m-%d") if len(df) else None
    return df, report


def clean_path(cfg: Config, ticker: str):
    return cfg.paths.clean / f"{ticker_to_filename(ticker)}.parquet"


def report_path(cfg: Config):
    return cfg.paths.clean / "quality_report.json"


def clean_all(cfg: Config) -> dict[str, dict]:
    """Clean every raw file that exists and write the combined quality report."""
    cfg.paths.clean.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    for ticker in cfg.data.all_tickers:
        src = raw_path(cfg, ticker)
        if not src.exists():
            log.warning("%s: no raw data, skipping clean", ticker)
            continue
        df, rep = clean_ohlcv(pd.read_parquet(src), cfg.data.ffill_limit, cfg.data.outlier_z)
        df.to_parquet(clean_path(cfg, ticker))
        reports[ticker] = rep
        log.info(
            "%s: %d rows clean (dropped %d, filled %d, ohlc fixed %d, outliers %d)",
            ticker, rep["rows_out"], rep["rows_dropped"], rep["missing_values_filled"],
            rep["ohlc_fixed"], len(rep["outlier_dates"]),
        )
    report_path(cfg).write_text(json.dumps(reports, indent=2))
    return reports
