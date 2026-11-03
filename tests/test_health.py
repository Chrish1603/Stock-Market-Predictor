from datetime import date, datetime, timezone

import pandas as pd

from smp.config import Config, DataConfig, PathsConfig
from smp.health import business_days_behind, data_problems, expected_latest_close, is_stale, is_trading_day


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_trading_calendar():
    assert not is_trading_day(date(2026, 11, 26))  # Thanksgiving
    assert not is_trading_day(date(2026, 4, 3))  # Good Friday
    assert not is_trading_day(date(2026, 11, 1))  # Sunday
    assert is_trading_day(date(2026, 11, 2))


def test_expected_latest_close():
    # Mon 2 Nov 2026, 15:00 New York: Monday not settled yet -> Friday's close
    assert expected_latest_close(utc(2026, 11, 2, 20, 0)) == date(2026, 10, 30)
    # Mon 2 Nov, 17:00 New York -> Monday's close
    assert expected_latest_close(utc(2026, 11, 2, 22, 0)) == date(2026, 11, 2)
    # Fri 27 Nov (after Thanksgiving), early morning -> Wednesday 25th
    assert expected_latest_close(utc(2026, 11, 27, 12, 0)) == date(2026, 11, 25)


def test_staleness():
    now = utc(2026, 11, 3, 2, 30)  # Mon evening New York
    assert business_days_behind(date(2026, 11, 2), now) == 0
    assert business_days_behind(date(2026, 10, 30), now) == 1
    assert not is_stale(date(2026, 11, 2), now) and is_stale(date(2026, 10, 30), now) and is_stale(None, now)
    # over Thanksgiving, Wednesday's close is still current on Thursday night
    assert not is_stale(date(2026, 11, 25), utc(2026, 11, 27, 3, 0))


def test_data_problems_groups_stale_tickers(tmp_path):
    from smp.clean import clean_path

    cfg = Config(
        data=DataConfig(tickers=["AAA", "BBB", "CCC"], market_tickers=[]), paths=PathsConfig().resolve(tmp_path)
    )
    cfg.paths.clean.mkdir(parents=True)
    for t, last in (("AAA", "2026-10-30"), ("BBB", "2026-10-30")):
        idx = pd.bdate_range(end=last, periods=5, name="Date")
        pd.DataFrame({"Close": range(5)}, index=idx).to_parquet(clean_path(cfg, t))
    problems = data_problems(cfg, utc(2026, 11, 3, 2, 30))
    assert problems == ["no price data for CCC", "prices for AAA, BBB end 2026-10-30 (1 trading day behind)"]
    assert data_problems(cfg, utc(2026, 11, 2, 12, 0))[1:] == []  # Friday's close is current on Monday morning
