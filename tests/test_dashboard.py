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
    at.sidebar.radio[0].set_value("regressor").run()
    assert not at.exception, at.exception
