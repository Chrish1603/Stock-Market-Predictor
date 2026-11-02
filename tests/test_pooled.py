import json

import numpy as np
import pandas as pd

from conftest import make_ohlcv
from smp.config import Config, DataConfig, PathsConfig, TrainConfig
from smp.features import add_cross_sectional_ranks, build_features, features_path, fill_earnings
from smp.pooled import date_splits, load_panel, split_dev_holdout_by_date, train_pooled


def _setup(tmp_path, n_universe=10, n=700):
    tickers, universe = ["AAA", "BBB"], [f"U{i}" for i in range(n_universe)]
    cfg = Config(
        data=DataConfig(tickers=tickers, universe=universe, market_tickers=[]),
        train=TrainConfig(horizons=[1, 5], n_splits=3, holdout_days=100, scope="pooled"),
        paths=PathsConfig().resolve(tmp_path),
    )
    frames = {
        t: fill_earnings(build_features(make_ohlcv(n=n - 10 * (i % 3), seed=i), horizons=[1, 5]))
        for i, t in enumerate(tickers + universe)
    }
    cfg.paths.features.mkdir(parents=True)
    for t, f in add_cross_sectional_ranks(frames).items():
        f.to_parquet(features_path(cfg, t))
    return cfg


def test_date_splits_keep_each_day_on_one_side():
    dates = pd.Series(np.repeat(pd.bdate_range("2020-01-01", periods=200), 5))
    for tr, te in date_splits(dates, n_splits=4, gap=5):
        tr_d, te_d = dates.iloc[tr], dates.iloc[te]
        assert tr_d.max() < te_d.min()
        assert set(tr_d).isdisjoint(set(te_d))
        assert len(pd.bdate_range(tr_d.max(), te_d.min())) - 2 >= 5  # at least `gap` days between


def test_holdout_purge_by_date(tmp_path):
    cfg = _setup(tmp_path)
    panel = load_panel(cfg)
    dev, hold = split_dev_holdout_by_date(panel, holdout_days=100, horizon=5)
    days = pd.DatetimeIndex(np.sort(panel["Date"].unique()))
    assert hold["Date"].nunique() == 100
    gap = days.get_loc(hold["Date"].min()) - days.get_loc(dev["Date"].max())
    assert gap > 5


def test_train_pooled_writes_per_ticker_artifacts(tmp_path):
    cfg = _setup(tmp_path)
    summary = train_pooled(cfg, horizon=5)
    assert summary["n_tickers"] == 12
    for t in ("AAA", "BBB"):
        out = tmp_path / "models" / t / "h5"
        m = json.loads((out / "metrics.json").read_text())
        assert m["scope"] == "pooled" and m["n_training_tickers"] == 12
        for task in ("classifier", "regressor"):
            assert m[task]["selected"] != "baseline"
            assert m[task]["holdout"]["baseline"]
        for f in (
            "classifier.joblib",
            "regressor.joblib",
            "holdout_predictions.parquet",
            "backtest_classifier.parquet",
        ):
            assert (out / f).exists()
        hold = pd.read_parquet(out / "holdout_predictions.parquet")
        assert 80 <= len(hold) <= 100 and hold["p_up"].between(0, 1).all()  # BBB data ends 10 days early
    # random-walk data: a pooled model must not look predictive (that would mean leakage)
    assert summary["classifier"]["holdout"][summary["classifier"]["selected"]]["roc_auc"] < 0.6


def test_pooled_models_work_with_predict(tmp_path):
    from smp.predict import predict_ticker

    cfg = _setup(tmp_path)
    train_pooled(cfg, horizon=1)
    r = predict_ticker(cfg, "AAA", 1)
    assert r is not None and 0 <= r["p_up"] <= 1
