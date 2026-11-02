"""Download daily OHLCV data from Yahoo Finance, with incremental updates."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from smp.config import Config, ticker_to_filename

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
OVERLAP_DAYS = 5  # re-fetch a few recent days to catch late revisions
ADJUSTMENT_TOLERANCE = 0.005  # >0.5% change in stored closes => prices were re-adjusted
MARKET_TZ = ZoneInfo("America/New_York")
SETTLED_AFTER = dtime(16, 30)  # US close is 16:00; allow time for final prices


def drop_incomplete_bar(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop today's bar while the US session is still open, so only final closes are stored."""
    now = (now or datetime.now(MARKET_TZ)).astimezone(MARKET_TZ)
    if df.empty or now.time() >= SETTLED_AFTER:
        return df
    return df[df.index < pd.Timestamp(now.date())]


def download(ticker: str, start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    """Fetch split/dividend-adjusted daily bars. Returns an empty frame when there's no data."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, interval="1d")
            if df is None or df.empty:
                return pd.DataFrame(columns=OHLCV)
            df = df[OHLCV].copy()
            idx = pd.to_datetime(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            df.index = idx.normalize()
            df.index.name = "Date"
            return drop_incomplete_bar(df)
        except Exception as exc:  # network / API errors
            log.warning("Download %s failed (attempt %d/%d): %s", ticker, attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def raw_path(cfg: Config, ticker: str) -> Path:
    return cfg.paths.raw / f"{ticker_to_filename(ticker)}.parquet"


def merge_incremental(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append new rows, letting fresh data win where dates overlap."""
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def prices_were_readjusted(existing: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Adjusted prices shift after dividends/splits; if overlapping closes moved, refetch all."""
    overlap = existing.index.intersection(new.index)
    if overlap.empty:
        return False
    old, fresh = existing.loc[overlap, "Close"], new.loc[overlap, "Close"]
    return bool(((fresh - old).abs() / old).max() > ADJUSTMENT_TOLERANCE)


def ingest_ticker(cfg: Config, ticker: str) -> int:
    """Download (or update) one ticker; returns the number of stored rows."""
    path = raw_path(cfg, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        start = (existing.index.max() - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
        new = download(ticker, start, cfg.data.end)
        if prices_were_readjusted(existing, new):
            log.info("%s: adjusted prices changed, re-downloading full history", ticker)
            data = download(ticker, cfg.data.start, cfg.data.end)
        else:
            data = merge_incremental(existing, new)
        added = len(data) - len(existing)
        log.info("%s: %d rows stored (%+d)", ticker, len(data), added)
    else:
        data = download(ticker, cfg.data.start, cfg.data.end)
        log.info("%s: downloaded %d rows", ticker, len(data))

    if data.empty:
        log.warning("%s: no data returned", ticker)
        return 0
    data.to_parquet(path)
    return len(data)


def download_earnings(ticker: str, limit: int = 100) -> pd.DataFrame:
    """Past and scheduled earnings dates with EPS surprise.

    `reaction_date` is the first trading day whose close reflects the report: the same day for
    reports before noon, otherwise the next business day (most reports come after the close).
    """
    cols = ["earnings_date", "reaction_date", "surprise_pct"]
    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    stamps = pd.DatetimeIndex(raw.index)
    if stamps.tz is not None:
        stamps = stamps.tz_convert(MARKET_TZ).tz_localize(None)
    day = stamps.normalize()
    reaction = day.where(stamps.hour < 12, day + pd.offsets.BDay(1))
    out = pd.DataFrame(
        {
            "earnings_date": day,
            "reaction_date": reaction,
            "surprise_pct": pd.to_numeric(raw.get("Surprise(%)"), errors="coerce").to_numpy(),
        }
    )
    return out.drop_duplicates("reaction_date", keep="first").sort_values("reaction_date").reset_index(drop=True)


def earnings_path(cfg: Config, ticker: str) -> Path:
    return cfg.paths.raw / "earnings" / f"{ticker_to_filename(ticker)}.parquet"


def ingest_earnings(cfg: Config, ticker: str) -> int:
    """Refresh earnings dates (cheap: one request). Keeps the old file if the download fails."""
    yf_log = logging.getLogger("yfinance")
    level = yf_log.level
    yf_log.setLevel(logging.CRITICAL)  # yfinance logs an error for funds, which have no earnings
    try:
        e = download_earnings(ticker)
    except Exception as exc:
        log.info("%s: no earnings dates (%s)", ticker, exc)
        return 0
    finally:
        yf_log.setLevel(level)
    if e.empty:
        log.info("%s: no earnings dates (likely an ETF)", ticker)
        return 0
    path = earnings_path(cfg, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    e.to_parquet(path, index=False)
    log.info("%s: %d earnings dates through %s", ticker, len(e), e["earnings_date"].max().date())
    return len(e)


def ingest_all(cfg: Config) -> dict[str, int]:
    """Ingest every configured ticker; one failure doesn't stop the others."""
    results: dict[str, int] = {}
    for ticker in cfg.data.all_tickers:
        try:
            results[ticker] = ingest_ticker(cfg, ticker)
        except Exception as exc:
            log.error("%s: ingest failed: %s", ticker, exc)
            results[ticker] = -1
    if cfg.data.earnings:
        for ticker in cfg.data.feature_tickers:
            ingest_earnings(cfg, ticker)
    return results
