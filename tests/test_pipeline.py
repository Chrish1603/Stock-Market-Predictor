"""End-to-end pipeline run on synthetic data (no network)."""

import json

import pandas as pd

from conftest import make_ohlcv
from smp import pipeline
from smp.config import Config, DataConfig, PathsConfig, TrainConfig
from smp.ingest import raw_path
from smp.predict import predictions_path


def test_pipeline_without_network(tmp_path):
    cfg = Config(
        data=DataConfig(tickers=["AAA", "SPY"], market_tickers=["SPY", "^VIX"]),
        train=TrainConfig(n_splits=3, holdout_days=120),
        paths=PathsConfig().resolve(tmp_path),
    )
    cfg.paths.raw.mkdir(parents=True)
    for i, t in enumerate(["AAA", "SPY", "^VIX"]):
        make_ohlcv(n=800, seed=i, base=20 if t == "^VIX" else 100).to_parquet(raw_path(cfg, t))

    pipeline.setup_logging(cfg.paths.logs)
    pipeline.run_all(cfg, skip_ingest=True)

    report = json.loads((cfg.paths.clean / "quality_report.json").read_text())
    assert set(report) == {"AAA", "SPY", "^VIX"}
    preds = pd.read_parquet(predictions_path(cfg))
    assert set(preds["ticker"]) == {"AAA", "SPY"}
    assert sorted(preds["horizon"].unique()) == [1, 5, 20]
    assert preds["p_up"].between(0, 1).all()
    assert (cfg.paths.logs / "pipeline.log").exists()

    from smp.history import load_history

    hist = load_history(cfg)
    assert len(hist) == 6 and hist["actual_return"].isna().all()  # 2 tickers x 3 horizons, not yet scorable
    pipeline.run(cfg, ["predict"])  # re-running the same day doesn't duplicate history
    assert len(load_history(cfg)) == 6
