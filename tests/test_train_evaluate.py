import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from smp.config import Config, PathsConfig, TrainConfig
from smp.evaluate import backtest, backtest_stats, classification_metrics, regression_metrics
from smp.features import build_features
from smp.train import train_ticker, walk_forward_splits


def test_walk_forward_splits_are_time_ordered():
    splits = walk_forward_splits(1000, n_splits=5, gap=1)
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() + 1 < te.min()  # gap respected, no overlap
        assert np.all(np.diff(te) == 1)
    assert splits[0][0].size < splits[-1][0].size  # expanding window


def test_metrics_basic():
    y = np.array([0, 1, 1, 0, 1])
    m = classification_metrics(y, [0.2, 0.8, 0.7, 0.4, 0.9])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0
    r = regression_metrics([0.01, -0.02, 0.03], [0.02, -0.01, -0.01])
    assert r["directional_accuracy"] == pytest.approx(2 / 3)


def test_backtest_costs_and_buy_hold():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series(np.log([1.01, 0.98, 1.02, 1.01]), index=idx)
    always = backtest(r, pd.Series(1.0, index=idx), cost=0.0)
    assert np.allclose(always["strategy_equity"], always["buy_hold_equity"])
    toggling = backtest(r, pd.Series([1, 0, 1, 0], index=idx, dtype=float), cost=0.01)
    assert toggling["strategy_return"].iloc[1] == pytest.approx(-0.01)  # exit cost, no exposure
    stats = backtest_stats(toggling)
    assert stats["n_trades"] == 4 and stats["exposure"] == 0.5


def test_train_ticker_end_to_end(tmp_path):
    feats = build_features(make_ohlcv(n=900), {"SPY": make_ohlcv(n=900, seed=1)})
    cfg = Config(
        train=TrainConfig(n_splits=3, holdout_days=150),
        paths=PathsConfig().resolve(tmp_path),
    )
    m = train_ticker(cfg, "TEST", feats)
    for task in ("classifier", "regressor"):
        assert m[task]["selected"] != "baseline"
        assert set(m[task]["cv"]) == {
            "baseline",
            "logistic" if task == "classifier" else "ridge",
            "lightgbm",
            "xgboost",
        }
    out = tmp_path / "models" / "TEST"
    for f in (
        "classifier.joblib",
        "regressor.joblib",
        "metrics.json",
        "holdout_predictions.parquet",
        "backtest_classifier.parquet",
        "backtest_regressor.parquet",
    ):
        assert (out / f).exists()
    # Random-walk data is unpredictable: a much-better-than-chance AUC would mean leakage.
    assert m["classifier"]["holdout"][m["classifier"]["selected"]]["roc_auc"] < 0.65
