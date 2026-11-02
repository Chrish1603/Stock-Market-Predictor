import numpy as np
import pandas as pd

from smp.clean import clean_ohlcv
from smp.ingest import merge_incremental, prices_were_readjusted


def test_merge_incremental_dedupes_and_prefers_new(ohlcv):
    existing = ohlcv.iloc[:100]
    new = ohlcv.iloc[95:120].copy()
    new.loc[new.index[0], "Close"] = 999.0
    merged = merge_incremental(existing, new)
    assert len(merged) == 120
    assert merged.index.is_unique and merged.index.is_monotonic_increasing
    assert merged.loc[new.index[0], "Close"] == 999.0


def test_readjustment_detection(ohlcv):
    existing = ohlcv.iloc[:100]
    same = ohlcv.iloc[95:110]
    assert not prices_were_readjusted(existing, same)
    shifted = same.copy()
    shifted["Close"] *= 0.98  # e.g. after a dividend adjustment
    assert prices_were_readjusted(existing, shifted)


def test_clean_removes_duplicates_and_sorts(ohlcv):
    messy = pd.concat([ohlcv.iloc[10:20], ohlcv.iloc[:15]])  # unsorted with duplicates
    out, rep = clean_ohlcv(messy)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert rep["duplicates_removed"] == 5
    assert len(out) == 20


def test_clean_fills_small_gaps_and_drops_large(ohlcv):
    df = ohlcv.iloc[:50].copy()
    df.iloc[10, df.columns.get_loc("Close")] = np.nan  # single gap -> filled
    df.iloc[20:25, df.columns.get_loc("Close")] = np.nan  # 5-row gap -> 3 rows dropped
    df.iloc[30, df.columns.get_loc("Open")] = -1.0  # invalid price -> filled
    out, rep = clean_ohlcv(df, ffill_limit=2)
    assert out[["Open", "High", "Low", "Close"]].notna().all().all()
    assert rep["rows_dropped"] == 3
    assert rep["missing_values_filled"] == 4
    assert (out[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_clean_fixes_ohlc_and_flags_outliers(ohlcv):
    df = ohlcv.copy()
    df.iloc[5, df.columns.get_loc("High")] = df["Low"].iloc[5] * 0.5  # High below Low
    df.iloc[300:, :4] *= 3.0  # huge one-day jump
    out, rep = clean_ohlcv(df)
    assert (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)).all()
    assert (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1)).all()
    assert rep["ohlc_fixed"] >= 1
    assert df.index[300].strftime("%Y-%m-%d") in rep["outlier_dates"]
    assert len(out) == len(df)  # outliers are flagged, not dropped


def test_clean_requires_columns(ohlcv):
    import pytest

    with pytest.raises(ValueError):
        clean_ohlcv(ohlcv.drop(columns=["Volume"]))
