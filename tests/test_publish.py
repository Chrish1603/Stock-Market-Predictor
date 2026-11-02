import json

import pandas as pd

from smp import pipeline
from smp.config import Config, DataConfig, PathsConfig, TrainConfig, data_root, load_config
from smp.publish import DASHBOARD_FEATURES, build_snapshot


def test_data_root_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SMP_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path
    assert load_config().paths.models == tmp_path / "models"
    monkeypatch.delenv("SMP_DATA_ROOT")
    assert load_config().paths.models != tmp_path / "models"


def test_snapshot_has_what_the_dashboard_needs(tmp_path):
    from conftest import make_ohlcv
    from smp.ingest import raw_path

    root = tmp_path / "project"
    cfg = Config(
        data=DataConfig(tickers=["AAA", "SPY"], market_tickers=["SPY"], earnings=False),
        train=TrainConfig(n_splits=3, holdout_days=120, horizons=[1]),
        paths=PathsConfig().resolve(root),
    )
    cfg.paths.raw.mkdir(parents=True)
    for i, t in enumerate(["AAA", "SPY"]):
        make_ohlcv(n=800, seed=i).to_parquet(raw_path(cfg, t))
    pipeline.run_all(cfg, skip_ingest=True)

    out = tmp_path / "snap"
    manifest = build_snapshot(cfg, out, root)
    assert json.loads((out / "manifest.json").read_text())["as_of"] == manifest["as_of"]
    assert not list(out.rglob("*.joblib"))  # no trained models
    assert (out / "models" / "risk_predictions.parquet").exists()
    assert (out / "data" / "history" / "predictions.parquet").exists()
    slim = pd.read_parquet(out / "data" / "processed" / "features" / "AAA.parquet")
    assert set(slim.columns) <= set(DASHBOARD_FEATURES) and "vol_20" in slim
