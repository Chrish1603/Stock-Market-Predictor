import numpy as np
import pandas as pd

from conftest import make_ohlcv
from smp.features import build_features, feature_columns, rsi, target_direction, target_return


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 50, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert np.isclose(rsi(up).iloc[-1], 100.0)
    assert np.isclose(rsi(down).iloc[-1], 0.0)
    assert rsi(up).iloc[:13].isna().all()


def test_rsi_in_range(ohlcv):
    r = rsi(ohlcv["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_targets_are_next_day(ohlcv):
    f = build_features(ohlcv)
    c = ohlcv["Close"]
    assert np.isclose(f[target_return(1)].iloc[10], np.log(c.iloc[11] / c.iloc[10]))
    assert np.isnan(f[target_return(1)].iloc[-1])
    assert np.isnan(f[target_direction(1)].iloc[-1])
    assert set(f[target_direction(1)].dropna().unique()) <= {0.0, 1.0}


def test_multi_horizon_targets(ohlcv):
    f = build_features(ohlcv, horizons=[5, 20])
    c = ohlcv["Close"]
    assert target_return(1) in f  # always present for backtests
    assert np.isclose(f[target_return(5)].iloc[10], np.log(c.iloc[15] / c.iloc[10]))
    assert f[target_return(20)].iloc[-20:].isna().all()
    assert f[target_return(20)].iloc[:-20].notna().all()
    assert not any(col.startswith("target_") for col in feature_columns(f))


def test_no_lookahead(ohlcv):
    """Changing future prices must not change any feature value up to day t."""
    market = {"SPY": make_ohlcv(seed=1), "^VIX": make_ohlcv(seed=2, base=20)}
    base = build_features(ohlcv, market)

    t = 400
    tampered = ohlcv.copy()
    tampered.iloc[t + 1 :] *= 1.5
    tampered_market = {k: v.copy() for k, v in market.items()}
    for v in tampered_market.values():
        v.iloc[t + 1 :] *= 0.5
    after = build_features(tampered, tampered_market)

    cols = feature_columns(base)
    pd.testing.assert_frame_equal(base[cols].iloc[: t + 1], after[cols].iloc[: t + 1])


def test_features_include_market_context(ohlcv):
    f = build_features(ohlcv, {"SPY": make_ohlcv(seed=1), "^VIX": make_ohlcv(seed=2, base=20)})
    for col in ("spy_ret_1", "vix_level", "vix_change_5", "rsi_14", "sma_ratio_200"):
        assert col in f.columns
    assert not np.isinf(f[feature_columns(f)].to_numpy(dtype=float)).any()
