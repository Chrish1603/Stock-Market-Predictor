import numpy as np
import pandas as pd
import pytest

from smp.config import load_config
from smp.long_history import START, count_spells, crisis_table, long_config, run_dir, strategy_runs, window_stats


def _curves(start="2007-01-01", end="2010-12-31"):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    return pd.DataFrame(
        {
            "buy_hold_weight": 1.0,
            "buy_hold_return": r,
            "buy_hold_equity": (1 + r).cumprod(),
            "half_weight": 0.5,
            "half_return": r / 2,
            "half_equity": (1 + r / 2).cumprod(),
        }
    )


def test_long_config_is_isolated_and_starts_early():
    base = load_config()
    cfg = long_config(base)
    assert cfg.data.start == START and cfg.data.tickers == [base.risk.market_ticker]
    assert cfg.data.universe == [] and not cfg.data.earnings and not cfg.data.cross_sectional
    for p in (cfg.paths.raw, cfg.paths.clean, cfg.paths.features, cfg.paths.models, cfg.paths.history):
        assert run_dir() in p.parents  # never the live data or models
    assert base.data.start != START  # the live config is unchanged


def test_window_stats_restart_equity():
    runs = strategy_runs(_curves())
    assert set(runs) == {"buy_hold", "half"}
    s = window_stats(runs["buy_hold"], "2008-01-01", "2008-12-31")
    r = runs["buy_hold"]["return"].loc["2008"]
    assert s["total_return"] == pytest.approx((1 + r).prod() - 1)
    assert s["start"] == "2008-01-01" and s["end"] == "2008-12-31"
    assert window_stats(runs["buy_hold"], "1990-01-01", "1990-12-31") is None


def test_crisis_table_skips_uncovered_crises():
    rows = crisis_table(strategy_runs(_curves()))
    assert [r["crisis"] for r in rows] == ["Financial crisis"]  # only 2007-10 to 2009-03 is inside
    fc = rows[0]
    assert fc["half"]["avg_weight"] == 0.5
    assert abs(fc["half"]["max_drawdown"]) < abs(fc["buy_hold"]["max_drawdown"])


def test_count_spells():
    labels = pd.Series(["calm", "stressed", "stressed", "normal", "stressed", None, "stressed"])
    assert count_spells(labels, "stressed") == 3
    assert count_spells(labels, "calm") == 1
