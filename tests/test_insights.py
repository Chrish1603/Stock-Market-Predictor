import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from smp.insights import calibration_bins, earnings_days, price_range, range_coverage, rolling_vol


def test_price_range_widens_with_time():
    r = price_range(100.0, 0.20, "2026-11-02", days=20)
    assert len(r) == 21 and r.index[0] == pd.Timestamp("2026-11-02")
    assert r.iloc[0].tolist() == pytest.approx([100.0] * 4)  # today: no spread
    end = r.iloc[-1]
    s = 0.20 * np.sqrt(20 / 252)
    assert end["high_typical"] == pytest.approx(100 * np.exp(s))
    assert end["low_wide"] < end["low_typical"] < 100 < end["high_typical"] < end["high_wide"]
    assert r["high_wide"].is_monotonic_increasing


def test_range_coverage_matches_normal_moves():
    """With returns that really have the forecast volatility, coverage should be near 68% and 95%."""
    rng = np.random.default_rng(1)
    n, vol = 6000, 0.25
    idx = pd.bdate_range("2000-01-03", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol / np.sqrt(252), n))), index=idx)
    pred = pd.Series(np.log(vol), index=idx)
    c = range_coverage(pred, close, horizon=20)
    assert c["n"] == n - 20
    assert c["typical"] == pytest.approx(0.68, abs=0.05)
    assert c["wide"] == pytest.approx(0.95, abs=0.03)


def test_calibration_bins_on_calibrated_forecasts():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.6, 20000)
    y = rng.uniform(size=p.size) < p  # outcomes happen with exactly the forecast probability
    b = calibration_bins(p, y, n_bins=8)
    assert len(b) == 8 and b["n"].sum() == 20000
    assert (b["predicted"] - b["actual"]).abs().max() < 0.03
    assert b["predicted"].is_monotonic_increasing


def test_calibration_bins_with_constant_forecast_and_empty():
    b = calibration_bins([0.2] * 50, [0, 1] * 25, n_bins=10)
    assert b["predicted"].eq(0.2).all() and b["n"].sum() == 50
    assert calibration_bins([], []).empty


def test_rolling_vol():
    r = make_ohlcv(n=400, seed=3)["Close"].pct_change()
    v = rolling_vol(r, 252).dropna()
    assert v.iloc[-1] == pytest.approx(0.015 * np.sqrt(252), rel=0.15)


def test_earnings_days():
    idx = pd.bdate_range("2026-01-05", periods=12)
    d = pd.Series([3, 2, 1, 20, 20, 20, 5, 4, 3, 2, 1, 4], index=idx, dtype=float)
    past, nxt = earnings_days(d)
    assert list(past) == [idx[3], idx[11]]  # the day after each "1"
    assert nxt == idx[-1] + pd.offsets.BDay(4)
    assert earnings_days(d.iloc[:6])[1] is None  # 20 means not known yet
