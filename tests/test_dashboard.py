"""Smoke tests: the dashboard runs without exceptions on pipeline outputs, for every control value."""

import pytest

from smp.config import load_config

APP = "../app/dashboard.py"


def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=90).run()


def _control(at, key):
    return next(c for c in at.segmented_control if c.key == key)


def test_dashboard_renders():
    cfg = load_config()
    if not (cfg.paths.models / "predictions.parquet").exists():
        pytest.skip("run `smp run-all` first")
    at = _app()
    assert not at.exception, at.exception
    for h in cfg.train.horizons:
        _control(at, "dir_h").set_value(h).run()
        for task in ("classifier", "regressor"):
            _control(at, "dir_task").set_value(task).run()
            assert not at.exception, (h, task, at.exception)
        _control(at, "track_h").set_value(h).run()
        assert not at.exception, (h, at.exception)
    for r in ("3M", "All"):
        _control(at, "price_range").set_value(r).run()
        assert not at.exception, (r, at.exception)


def test_risk_tab_renders():
    cfg = load_config()
    if not (cfg.paths.models / "risk_predictions.parquet").exists():
        pytest.skip("run `smp run-all` first")
    at = _app()
    for h in cfg.risk.horizons:
        _control(at, "risk_h").set_value(h).run()
        assert not at.exception, (h, at.exception)
    show = _control(at, "alloc_show")
    for opt in show.options:
        show.set_value(opt).run()
        assert not at.exception, (opt, at.exception)
    today = next(t for t in at.tabs if t.label == "Today")
    assert "In plain English" in " ".join(m.value for m in today.markdown)
