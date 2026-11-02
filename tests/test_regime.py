import numpy as np
import pandas as pd

from smp.regime import classify, expanding_percentile


def test_classify_rules():
    assert classify(0.85, 0.7, trend_up=True) == "stressed"  # high vol alone
    assert classify(0.50, 0.3, trend_up=False) == "stressed"  # weak breadth in a downtrend
    assert classify(0.30, 0.7, trend_up=True) == "calm"
    assert classify(0.30, 0.5, trend_up=True) == "normal"  # breadth not strong enough for calm
    assert classify(0.30, 0.7, trend_up=False) == "normal"
    assert classify(0.30, None, trend_up=True) == "calm"  # no universe: breadth ignored
    assert classify(0.30, float("nan"), trend_up=True) == "calm"


def test_expanding_percentile_uses_only_the_past():
    idx = pd.bdate_range("2015-01-01", periods=600)
    hist = pd.Series(np.arange(600, dtype=float), index=idx)
    values = pd.Series(300.5, index=idx[[100, 400, 599]])
    pct = expanding_percentile(values, hist)
    assert np.isnan(pct.iloc[0])  # not enough history yet
    assert pct.iloc[1] == 301 / 400  # past values 0..399: 301 of them are below 300.5
    assert pct.iloc[2] == 301 / 599  # future values never count
