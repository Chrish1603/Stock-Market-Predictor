import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from smp.config import Config, PathsConfig
from smp.features import build_features, target_drop, target_logvol
from smp.history import HistoryError
from smp.risk_history import (
    append_risk,
    load_risk_history,
    outcome,
    risk_backup_path,
    save_risk_history,
    score_risk,
    summarize_risk,
)


def _fc(as_of, horizon=5, vol=0.25, trailing=0.20, p_drop=0.3, base_rate=0.2, ticker="AAA"):
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "horizon": horizon,
                "as_of": pd.Timestamp(as_of),
                "forecast_vol": vol,
                "trailing_vol": trailing,
                "p_drop": p_drop,
                "base_rate": base_rate,
                "drop_threshold": 0.05,
                "vol_model": "ridge",
                "drop_model": "vol_implied",
                "suggested_weight": 0.8,  # not part of the record
            }
        ]
    )


def test_outcome_matches_training_targets():
    """Live scoring must use the same definitions as the targets the models were trained on."""
    px = make_ohlcv(n=80, seed=5)
    feats = build_features(px, risk_horizons=[5, 20])
    for h in (5, 20):
        for d in px.index[30:40]:
            res = outcome(px, d, h, 0.05)
            assert res["target_date"] == px.index[px.index.get_loc(d) + h]
            assert np.log(res["realised_vol"]) == pytest.approx(feats.at[d, target_logvol(h)])
            assert res["dropped"] == feats.at[d, target_drop(h)]
    assert outcome(px, px.index[-3], 5, 0.05) is None  # horizon not over yet


def test_outcome_detects_drop():
    px = make_ohlcv(n=10)
    px.loc[px.index[3], "Close"] = px["Close"].iloc[0] * 0.94  # 6% below day 0, then recovers
    assert outcome(px, px.index[0], 5, 0.05)["dropped"] == 1.0
    assert outcome(px, px.index[0], 2, 0.05)["dropped"] == 0.0  # the drop is after day 2


def test_append_keeps_first_forecast_and_only_record_columns():
    hist = append_risk(pd.DataFrame(), _fc("2020-01-02", vol=0.25))
    hist = append_risk(hist, _fc("2020-01-02", vol=0.40))  # re-run same day: ignored
    hist = append_risk(hist, _fc("2020-01-02", horizon=20))
    assert len(hist) == 2
    assert hist.loc[hist["horizon"] == 5, "forecast_vol"].item() == 0.25
    assert "suggested_weight" not in hist


def test_score_and_summarize():
    px = make_ohlcv(n=40, seed=2)
    hist = pd.DataFrame()
    for d in px.index:
        hist = append_risk(hist, _fc(d, vol=0.24, trailing=0.04, p_drop=0.2, base_rate=0.5))
    scored = score_risk(hist, {"AAA": px})
    assert scored["realised_vol"].notna().sum() == 35  # the last 5 days are still pending
    s = summarize_risk(scored).iloc[0]
    assert s["predictions"] == 40 and s["scored"] == 35
    realised = scored["realised_vol"].dropna()
    assert s["vol_error"] == pytest.approx(np.abs(np.log(0.24 / realised)).mean())
    # realised vol here is 10-38% a year, so 24% beats a 4% "last month" figure
    assert s["vol_error"] < s["trailing_vol_error"]
    assert s["closer_share"] > 0.5
    assert s["base_rate_brier"] == pytest.approx(((0.5 - scored["dropped"].dropna()) ** 2).mean())


def test_summarize_empty_and_unscored():
    assert summarize_risk(pd.DataFrame()).empty
    s = summarize_risk(append_risk(pd.DataFrame(), _fc("2020-01-02"))).iloc[0]
    assert s["scored"] == 0 and np.isnan(s["vol_error"])


def test_risk_record_is_append_only_and_backed_up(tmp_path):
    cfg = Config(paths=PathsConfig().resolve(tmp_path))
    h1 = append_risk(pd.DataFrame(), _fc("2020-01-02"))
    save_risk_history(cfg, h1)
    h2 = append_risk(h1, _fc("2020-01-03"))
    save_risk_history(cfg, h2)
    assert len(load_risk_history(cfg)) == 2 and risk_backup_path(cfg).exists()
    with pytest.raises(HistoryError, match="would be lost"):
        save_risk_history(cfg, h1)
