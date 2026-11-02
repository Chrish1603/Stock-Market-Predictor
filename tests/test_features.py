import numpy as np
import pandas as pd

from conftest import make_ohlcv
from smp.features import TARGET_DIRECTION, TARGET_RETURN, build_features, feature_columns, rsi


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
    assert np.isclose(f[TARGET_RETURN].iloc[10], np.log(c.iloc[11] / c.iloc[10]))
    assert np.isnan(f[TARGET_RETURN].iloc[-1])
    assert np.isnan(f[TARGET_DIRECTION].iloc[-1])
    assert set(f[TARGET_DIRECTION].dropna().unique()) <= {0.0, 1.0}


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
