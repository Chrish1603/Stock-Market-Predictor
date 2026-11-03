"""Smoke tests: the dashboard runs without exceptions on pipeline outputs, for every control value."""

import pytest

from smp.config import load_config

APP = "../app/dashboard.py"


@pytest.fixture(autouse=True)
def keep_smp_modules():
    """The dashboard may reload smp modules; put the originals back so later tests share one copy."""
    import sys

    saved = {k: v for k, v in sys.modules.items() if k == "smp" or k.startswith("smp.")}
    yield
    for k in [k for k in sys.modules if k == "smp" or k.startswith("smp.")]:
        del sys.modules[k]
    sys.modules.update(saved)


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
    for h in cfg.risk.horizons:
        _control(at, "track_risk_h").set_value(h).run()
        assert not at.exception, (h, at.exception)
    for r in ("3M", "All"):
        _control(at, "price_range").set_value(r).run()
        assert not at.exception, (r, at.exception)
    next(t for t in at.toggle if t.key == "show_bb").set_value(True).run()
    assert not at.exception, at.exception
    for t in cfg.data.tickers:  # every stock's price tab, including funds without earnings dates
        at.sidebar.selectbox[0].set_value(t).run()
        assert not at.exception, (t, at.exception)


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
    assert "card-text" in " ".join(m.value for m in today.markdown)


def test_dashboard_reloads_outdated_smp_modules(monkeypatch):
    """Streamlit Cloud pulls new code without restarting; an old smp module still in memory must not
    break a new one that imports from it (seen as ImportError: load_record from smp.history)."""
    import sys

    import smp.history

    cfg = load_config()
    if not (cfg.paths.models / "risk_predictions.parquet").exists():
        pytest.skip("run `smp run-all` first")
    monkeypatch.delattr(smp.history, "load_record")  # as if loaded before load_record existed
    monkeypatch.setattr(sys.modules["smp"], "_dashboard_loaded_at", 0.0, raising=False)  # source is newer
    sys.modules.pop("smp.risk_history", None)
    at = _app()
    assert not at.exception, at.exception
