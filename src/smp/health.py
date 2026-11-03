"""Data freshness checks, so a pipeline that silently stopped running is noticed.

The expected latest close is today's (New York time) once prices have settled after 16:30 on a trading
day, otherwise the previous trading day's. Trading days follow the NYSE's regular holiday rules; rare
one-off closures aren't modelled.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

from smp.config import Config

MARKET_TZ = ZoneInfo("America/New_York")
SETTLED_AFTER = time(16, 30)
STALE_AFTER_BUSINESS_DAYS = 0


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_HOLIDAYS = NYSEHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31").date.astype("datetime64[D]")


def is_trading_day(d: date) -> bool:
    return bool(np.is_busday(np.datetime64(d, "D"), holidays=_HOLIDAYS))


def _utc(now: datetime | None) -> datetime:
    """`now` as an aware datetime; naive values are taken as UTC."""
    now = now or datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def expected_latest_close(now: datetime | None = None) -> date:
    """The most recent weekday whose close should be available by `now`."""
    ny = _utc(now).astimezone(MARKET_TZ)
    day = ny.date()
    if is_trading_day(day) and ny.time() >= SETTLED_AFTER:
        return day
    d = day - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def business_days_behind(as_of: date, now: datetime | None = None) -> int:
    """How many trading days of closes are missing after `as_of`."""
    expected = np.datetime64(expected_latest_close(now), "D")
    start = np.datetime64(as_of, "D") + 1
    return max(0, int(np.busday_count(start, expected + 1, holidays=_HOLIDAYS)))


def is_stale(as_of: date | None, now: datetime | None = None) -> bool:
    return as_of is None or business_days_behind(as_of, now) > STALE_AFTER_BUSINESS_DAYS


def latest_price_dates(cfg: Config) -> dict[str, date | None]:
    """Last cleaned price date for each predicted and market ticker (None if missing)."""
    from smp.clean import clean_path

    out: dict[str, date | None] = {}
    for t in dict.fromkeys(cfg.data.tickers + cfg.data.market_tickers):
        p = clean_path(cfg, t)
        out[t] = pd.read_parquet(p, columns=["Close"]).index.max().date() if p.exists() else None
    return out


def data_problems(cfg: Config, now: datetime | None = None) -> list[str]:
    """Plain-English list of freshness problems; empty when everything is current.

    Tickers that end on the same stale date are reported together.
    """
    problems, stale_by_date = [], {}
    for t, d in latest_price_dates(cfg).items():
        if d is None:
            problems.append(f"no price data for {t}")
        elif is_stale(d, now):
            stale_by_date.setdefault(d, []).append(t)
    for d, tickers in sorted(stale_by_date.items()):
        n = business_days_behind(d, now)
        problems.append(f"prices for {', '.join(tickers)} end {d} ({n} trading day{'s' if n != 1 else ''} behind)")
    return problems


def snapshot_age_hours(manifest_path: Path, now: datetime | None = None) -> float | None:
    """Hours since a published snapshot was built (None if unknown)."""
    if not manifest_path.exists():
        return None
    built = _utc(datetime.fromisoformat(json.loads(manifest_path.read_text())["generated_at"]))
    return (_utc(now) - built).total_seconds() / 3600
