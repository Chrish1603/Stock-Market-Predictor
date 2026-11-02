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


def test_summary_changes_and_render():
    from smp.notify import changes, render

    before = {"regime": "normal", "stocks": [{"ticker": "AAA", "weight": 0.80}]}
    now = {
        "as_of": "2026-01-02",
        "regime": "stressed",
        "market_vol": 0.28,
        "stocks": [{"ticker": "AAA", "weight": 0.50, "vol": 0.30, "vol_pct": 0.9, "p_drop": 0.4, "earnings_in": 2}],
    }
    diff = changes(now, before)
    assert diff == ["Market regime changed: normal → stressed", "AAA holding 80% → 50%"]
    title, text, md = render(now, diff)
    assert title.startswith("Market regime changed")
    assert "AAA: hold 50%, vol 30% (well above normal), 40% chance of a 5% drop, earnings in 2d" in text
    assert "### " in md and "not financial advice" in md
    assert changes(now, None) == []
