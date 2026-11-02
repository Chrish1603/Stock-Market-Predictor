import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from smp.config import Config, PathsConfig, RiskConfig, TrainConfig
from smp.features import build_features, target_drop, target_logvol
from smp.risk import Persistence, drop_probability, train_risk_ticker, vol_metrics


def test_drop_probability_matches_simulation():
    rng = np.random.default_rng(0)
    vol, h, thr, steps = 0.30, 20, 0.05, 50
    paths = np.cumsum(rng.normal(0, vol / np.sqrt(252 * steps), (20000, h * steps)), axis=1)
    barrier = np.log(1 - thr)
    near_continuous = (paths.min(axis=1) <= barrier).mean()
    daily_closes = (paths[:, steps - 1 :: steps].min(axis=1) <= barrier).mean()
    assert drop_probability(vol, h, thr) == pytest.approx(near_continuous, abs=0.03)
    assert daily_closes < near_continuous  # checking only closes misses some dips; calibration corrects this
    assert drop_probability(0.1, 20, thr) < drop_probability(0.4, 20, thr)


def test_persistence_baseline_and_metrics():
    X = pd.DataFrame({"vol_20": [0.01, 0.02]})
    pred = Persistence("vol_20").fit(X).predict(X)
    assert np.allclose(np.exp(pred), [0.01 * np.sqrt(252), 0.02 * np.sqrt(252)])
    m = vol_metrics(np.log([0.2, 0.3, 0.25]), np.log([0.2, 0.3, 0.25]))
    assert m["r2"] == pytest.approx(1.0) and m["rmse"] == 0


def test_risk_targets_have_no_future_in_features(ohlcv):
    f = build_features(ohlcv, risk_horizons=[5])
    tampered = ohlcv.copy()
    tampered.iloc[401:] *= 1.5
    g = build_features(tampered, risk_horizons=[5])
    cols = [c for c in f if not c.startswith("target_")]
    pd.testing.assert_frame_equal(f[cols].iloc[:401], g[cols].iloc[:401])
    assert f[target_logvol(5)].iloc[-5:].isna().all() and f[target_drop(5)].iloc[-5:].isna().all()


def test_train_risk_ticker(tmp_path):
    rng = np.random.default_rng(1)
    df = make_ohlcv(n=1200)
    # make volatility clustered so there is something to forecast
    regime = np.repeat(rng.choice([0.6, 1.8], size=24), 50)
    r = np.log(df["Close"]).diff().fillna(0).to_numpy() * regime
    scale = np.exp(np.cumsum(r)) / df["Close"].to_numpy() * df["Close"].iloc[0]
    df = df.mul(scale, axis=0).assign(Volume=df["Volume"])
    feats = build_features(df, risk_horizons=[20])
    cfg = Config(
        train=TrainConfig(n_splits=3, holdout_days=150),
        risk=RiskConfig(horizons=[20]),
        paths=PathsConfig().resolve(tmp_path),
    )
    m = train_risk_ticker(cfg, "TEST", feats, 20)
    assert m["vol"]["cv"]["har"]["rmse"] < m["vol"]["cv"]["persistence"]["rmse"]
    assert m["vol"]["beats_baseline_cv"]
    out = tmp_path / "models" / "TEST" / "risk_h20"
    oos = pd.read_parquet(out / "oos_predictions.parquet")
    assert oos.index.is_monotonic_increasing and oos["p_drop"].between(0, 1).all()
    assert {"pred_logvol", "har_logvol", "trailing_logvol", "actual_logvol"} <= set(oos.columns)
    assert json.loads((out / "metrics.json").read_text())["drop"]["selection_metric"] == "brier"
