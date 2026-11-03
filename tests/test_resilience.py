import json
import os

import pandas as pd
import pytest

from smp import pipeline
from smp.config import Config, PathsConfig
from smp.history import HistoryError, append_predictions, backup_path, history_path, load_history, save_history
from smp.safeio import PipelineBusy, atomic_write_parquet, atomic_write_text, run_lock


def test_atomic_write_keeps_old_file_when_interrupted(tmp_path, monkeypatch):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "old")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]  # no temp files left behind


def test_atomic_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    atomic_write_parquet(df, tmp_path / "x" / "d.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "x" / "d.parquet"), df)


def test_run_lock_blocks_overlapping_runs(tmp_path):
    with run_lock(tmp_path), pytest.raises(PipelineBusy, match=str(os.getpid())), run_lock(tmp_path):
        pass
    with run_lock(tmp_path):  # released afterwards
        pass


def _preds(day, ticker="AAA"):
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "horizon": 1,
                "as_of": pd.Timestamp(day),
                "p_up": 0.6,
                "direction": "up",
                "pred_return": 0.0,
                "classifier": "x",
                "regressor": "y",
            }
        ]
    )


def test_history_is_append_only_backed_up_and_recoverable(tmp_path):
    cfg = Config(paths=PathsConfig().resolve(tmp_path))
    h1 = append_predictions(pd.DataFrame(), _preds("2026-01-02"))
    save_history(cfg, h1)
    h2 = append_predictions(h1, _preds("2026-01-05"))
    save_history(cfg, h2)
    assert len(pd.read_parquet(backup_path(cfg))) == 1  # previous version kept

    with pytest.raises(HistoryError, match="would be lost"):
        save_history(cfg, h1)  # dropping a recorded prediction is refused
    assert len(load_history(cfg)) == 2  # and nothing was written

    history_path(cfg).write_bytes(b"not a parquet file")  # corrupted main file
    assert len(load_history(cfg)) == 1  # falls back to the backup


def test_pipeline_keeps_going_and_reports_failures(tmp_path, monkeypatch):
    cfg = Config(paths=PathsConfig().resolve(tmp_path))
    ran = []

    def ok(name):
        return lambda c: ran.append(name)

    def broken(c):
        raise ValueError("Yahoo is down")

    def fake_notify(c):
        ran.append("notify")
        return json.loads((c.paths.logs / "last_run.json").read_text())

    monkeypatch.setattr(pipeline, "STAGES", {"ingest": broken, "clean": ok("clean"), "notify": fake_notify})
    with pytest.raises(pipeline.PipelineError, match="Yahoo is down"):
        pipeline.run_all(cfg)
    assert ran == ["clean", "notify"]  # later stages, including the summary, still ran
    status = json.loads((cfg.paths.logs / "last_run.json").read_text())
    assert status["failures"][0]["stage"] == "ingest" and status["finished"]

    with pytest.raises(ValueError):  # single-stage commands stop at the first failure
        pipeline.run(cfg, ["ingest", "clean"])
