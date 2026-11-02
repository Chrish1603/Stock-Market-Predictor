"""Smoke test: the dashboard script runs without exceptions on pipeline outputs."""

import pytest

from smp.config import load_config


def test_dashboard_renders():
    cfg = load_config()
    if not (cfg.paths.models / "predictions.parquet").exists():
        pytest.skip("run `smp run-all` first")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../app/dashboard.py", default_timeout=60).run()
    assert not at.exception, at.exception
    horizon_radio, model_radio = at.sidebar.radio[0], at.sidebar.radio[1]
    for h in cfg.train.horizons:
        horizon_radio.set_value(h).run()
        for task in ("classifier", "regressor"):
            model_radio.set_value(task).run()
            assert not at.exception, (h, task, at.exception)


def test_risk_tab_renders():
    cfg = load_config()
    if not (cfg.paths.models / "risk_predictions.parquet").exists():
        pytest.skip("run `smp run-all` first")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../app/dashboard.py", default_timeout=90).run()
    risk = next(t for t in at.tabs if t.label == "Risk & allocation")
    for h in cfg.risk.horizons:
        risk.radio[0].set_value(h).run()
        assert not at.exception, (h, at.exception)
    assert len(risk.metric) == len(cfg.data.tickers)
