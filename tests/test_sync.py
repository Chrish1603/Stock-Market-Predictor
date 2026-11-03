import tarfile

import pandas as pd
import pytest

from smp.config import Config, DashboardConfig, PathsConfig
from smp.history import append_predictions, load_history, save_history
from smp.publish import fetch_snapshot
from smp.sync import merge_official, skip_reason, sync_track_record


def _preds(as_of, p_up, ticker="AAA", horizon=1):
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "horizon": horizon,
                "as_of": pd.Timestamp(as_of),
                "p_up": p_up,
                "direction": "up" if p_up > 0.5 else "down",
                "pred_return": 0.0,
                "classifier": "lightgbm",
                "regressor": "ridge",
            }
        ]
    )


def _record(*rows):
    hist = pd.DataFrame()
    for as_of, p in rows:
        hist = append_predictions(hist, _preds(as_of, p))
    return hist


def test_official_rows_win_and_nothing_is_dropped():
    official = _record(("2026-01-02", 0.6), ("2026-01-05", 0.7))
    local = _record(("2026-01-05", 0.2), ("2026-01-06", 0.4))  # 01-05 differs; 01-06 is local only
    merged, counts = merge_official(local, official)
    assert counts == {"adopted": 1, "in_both": 1, "local_only": 1}
    by_day = merged.set_index("as_of")["p_up"]
    assert by_day[pd.Timestamp("2026-01-05")] == 0.7  # the official prediction
    assert by_day[pd.Timestamp("2026-01-06")] == 0.4  # kept
    assert len(merged) == 3


def test_merge_with_empty_sides():
    rec = _record(("2026-01-02", 0.6))
    assert merge_official(pd.DataFrame(), rec)[1]["adopted"] == 1
    assert merge_official(rec, pd.DataFrame())[1]["local_only"] == 1


def test_skip_reasons(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert skip_reason(Config()) == "no dashboard.data_url to sync from"
    on = Config(dashboard=DashboardConfig(data_url="https://example.com/s.tar.gz"))
    assert skip_reason(on) is None
    off = Config(dashboard=DashboardConfig(data_url="https://example.com/s.tar.gz", sync_track_record=False))
    assert "off" in skip_reason(off)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert "GitHub" in skip_reason(on)


def _snapshot(tmp_path, official: pd.DataFrame):
    """A published snapshot as GitHub serves it: a .tar.gz with one top-level folder."""
    snap = tmp_path / "publisher" / "Stock-Market-Predictor-dashboard-data"
    publisher = Config(paths=PathsConfig().resolve(snap))
    save_history(publisher, official)
    (snap / "manifest.json").write_text("{}")
    archive = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(snap, arcname=snap.name)
    return archive


def test_sync_from_published_snapshot(tmp_path):
    official = _record(("2026-01-02", 0.6), ("2026-01-05", 0.7))
    archive = _snapshot(tmp_path, official)
    local_cfg = Config(paths=PathsConfig().resolve(tmp_path / "local"))
    save_history(local_cfg, _record(("2026-01-05", 0.2), ("2026-01-06", 0.4)))

    root = fetch_snapshot(archive.as_uri(), tmp_path / "unpacked")
    out = sync_track_record(local_cfg, snapshot_root=root)
    assert out["direction"] == {"adopted": 1, "in_both": 1, "local_only": 1}
    assert "risk" not in out  # the snapshot has no risk record yet
    hist = load_history(local_cfg).set_index("as_of")["p_up"]
    assert list(hist) == [0.6, 0.7, 0.4]

    again = sync_track_record(local_cfg, snapshot_root=root)  # idempotent
    assert again["direction"] == {"adopted": 0, "in_both": 2, "local_only": 1}
    assert len(load_history(local_cfg)) == 3


def test_fetch_snapshot_rejects_other_archives(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "a.txt").write_text("hi")
    archive = tmp_path / "other.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "x", arcname="x")
    with pytest.raises(FileNotFoundError, match="not a dashboard snapshot"):
        fetch_snapshot(archive.as_uri(), tmp_path / "out")
