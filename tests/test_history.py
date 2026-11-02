from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from smp.history import append_predictions, score, summarize
from smp.ingest import MARKET_TZ, drop_incomplete_bar


def _preds(as_of, p_up=0.6, ticker="AAA"):
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "as_of": pd.Timestamp(as_of),
                "p_up": p_up,
                "direction": "up" if p_up > 0.5 else "down",
                "pred_return": 0.001,
                "classifier": "lightgbm",
                "regressor": "ridge",
            }
        ]
    )


def test_append_keeps_first_prediction_per_day():
    hist = append_predictions(pd.DataFrame(), _preds("2020-01-02", 0.6))
    hist = append_predictions(hist, _preds("2020-01-02", 0.2))  # re-run same day: ignored
    hist = append_predictions(hist, _preds("2020-01-03", 0.4))
    assert len(hist) == 2
    assert hist.loc[hist["as_of"] == "2020-01-02", "p_up"].item() == 0.6


def test_score_uses_next_trading_day():
    px = make_ohlcv(n=10, start="2020-01-02")
    d0, d1 = px.index[0], px.index[1]
    hist = append_predictions(pd.DataFrame(), _preds(d0, 0.7))
    hist = append_predictions(hist, _preds(px.index[-1], 0.7))  # outcome not known yet
    scored = score(hist, {"AAA": px})

    first = scored.iloc[0]
    expected = np.log(px.at[d1, "Close"] / px.at[d0, "Close"])
    assert first["target_date"] == d1
    assert first["actual_return"] == pytest.approx(expected)
    assert first["correct"] == float(expected > 0)
    assert np.isnan(scored.iloc[1]["actual_return"])


def test_summarize_hit_rates():
    px = make_ohlcv(n=60, start="2020-01-02", seed=3)
    hist = pd.DataFrame()
    for d in px.index[:-1]:
        hist = append_predictions(hist, _preds(d, 0.9))  # always calls "up"
    s = summarize(score(hist, {"AAA": px})).iloc[0]
    assert s["scored"] == 59
    assert s["hit_rate"] == pytest.approx(s["always_up_hit_rate"])  # same thing when always up


def test_drop_incomplete_bar():
    df = make_ohlcv(n=5, start="2024-03-04")  # Mon-Fri
    friday = df.index[-1].date()
    during = datetime(friday.year, friday.month, friday.day, 11, 0, tzinfo=MARKET_TZ)
    after = datetime(friday.year, friday.month, friday.day, 17, 0, tzinfo=MARKET_TZ)
    assert len(drop_incomplete_bar(df, during)) == 4
    assert len(drop_incomplete_bar(df, after)) == 5
