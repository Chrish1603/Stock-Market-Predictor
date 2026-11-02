import numpy as np
import pandas as pd
import pytest

from smp.allocation import apply_band, performance, run_weights, suggested_weight, target_weights
from smp.config import Config, RiskConfig


def test_target_weights_cap_and_scale():
    w = target_weights(pd.Series([0.075, 0.15, 0.30, 0.60]), target_vol=0.15, max_weight=1.0)
    assert w.tolist() == [1.0, 1.0, 0.5, 0.25]
    assert target_weights(pd.Series([0.075]), 0.15, 1.5).item() == 1.5


def test_rebalance_band_skips_small_changes():
    w = apply_band(pd.Series([0.50, 0.55, 0.58, 0.65, 0.66]), band=0.10)
    assert w.tolist() == [0.50, 0.50, 0.50, 0.65, 0.65]


def test_run_weights_returns_and_costs():
    idx = pd.bdate_range("2021-01-01", periods=3)
    r = pd.Series(np.log([1.10, 1.10, 1.10]), index=idx)
    run = run_weights(r, pd.Series([0.5, 0.5, 1.0], index=idx), cost=0.01)
    assert run["return"].tolist() == pytest.approx([0.05 - 0.005, 0.05, 0.10 - 0.005])
    perf = performance(run)
    assert perf["max_drawdown"] == 0 and perf["avg_weight"] == pytest.approx(2 / 3)


def test_vol_targeting_cuts_drawdown_in_a_volatile_crash():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    vol = np.where((np.arange(400) > 200) & (np.arange(400) < 260), 0.06, 0.01)
    drift = np.where((np.arange(400) > 200) & (np.arange(400) < 260), -0.01, 0.0005)
    r = pd.Series(rng.normal(drift, vol), index=idx)
    trailing = r.rolling(20).std().shift(1).bfill() * np.sqrt(252)  # known before each day
    w = target_weights(trailing, 0.15, 1.0)
    assert (
        performance(run_weights(r, w, 0.0))["max_drawdown"]
        > performance(run_weights(r, w * 0 + 1, 0.0))["max_drawdown"]
    )


def test_suggested_weight_uses_configured_vol():
    cfg = Config(risk=RiskConfig(target_vol=0.15, sizing_vol="trailing"))
    assert suggested_weight(cfg, trailing_annual_vol=0.30, model_annual_vol=0.15) == 0.5
    cfg.risk.sizing_vol = "model"
    assert suggested_weight(cfg, trailing_annual_vol=0.30, model_annual_vol=0.15) == 1.0
