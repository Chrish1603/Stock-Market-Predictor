"""One track record: local runs adopt the official one before adding their own predictions.

The daily GitHub Actions run keeps the official record and publishes it in the dashboard snapshot.
A local run (launchd, the dashboard's refresh button) downloads that snapshot and merges its records
into the local ones. For a (ticker, horizon, day) both have, the official row wins; rows only the
local record has (e.g. a day the GitHub run missed) are kept, so nothing is ever dropped and a bad
snapshot can't shrink the local record.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

from smp.config import Config, PathsConfig
from smp.history import KEY, backup_path, history_path, load_record, save_record
from smp.risk_history import risk_backup_path, risk_history_path

log = logging.getLogger(__name__)


def records(cfg: Config) -> dict[str, tuple[Path, Path]]:
    """Track record name -> (file, backup)."""
    return {
        "direction": (history_path(cfg), backup_path(cfg)),
        "risk": (risk_history_path(cfg), risk_backup_path(cfg)),
    }


def merge_official(local: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Union of both records; the official row wins where both have the same key."""
    if official.empty:
        return local, {"adopted": 0, "in_both": 0, "local_only": len(local)}
    if local.empty:
        return official.reset_index(drop=True), {"adopted": len(official), "in_both": 0, "local_only": 0}
    official = official.assign(as_of=pd.to_datetime(official["as_of"]))
    local = local.assign(as_of=pd.to_datetime(local["as_of"]))
    keys = lambda df: set(map(tuple, df[KEY].astype(str).to_numpy()))  # noqa: E731
    o, loc = keys(official), keys(local)
    merged = pd.concat([official, local], ignore_index=True).drop_duplicates(subset=KEY, keep="first")
    return merged.reset_index(drop=True), {
        "adopted": len(o - loc),
        "in_both": len(o & loc),
        "local_only": len(loc - o),
    }


def skip_reason(cfg: Config) -> str | None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "this is the GitHub run, which keeps the official record"
    if not cfg.dashboard.sync_track_record:
        return "dashboard.sync_track_record is off"
    if not cfg.dashboard.data_url:
        return "no dashboard.data_url to sync from"
    return None


def sync_track_record(cfg: Config, snapshot_root: Path | None = None) -> dict:
    """Merge the published (official) track records into the local ones. Returns counts per record.
    `snapshot_root` uses an already unpacked snapshot instead of downloading one."""
    if snapshot_root is None and (reason := skip_reason(cfg)):
        log.info("track record sync skipped: %s", reason)
        return {"skipped": reason}
    with tempfile.TemporaryDirectory(prefix="smp-sync-") as tmp:
        if snapshot_root is None:
            from smp.publish import fetch_snapshot

            snapshot_root = fetch_snapshot(cfg.dashboard.data_url, Path(tmp))
        history_dir = snapshot_root / PathsConfig().history  # the snapshot mirrors the default layout
        out = {}
        for name, (path, backup) in records(cfg).items():
            src = history_dir / path.name
            if not src.exists():
                log.info("track record sync: the snapshot has no %s record yet", name)
                continue
            official = pd.read_parquet(src)
            merged, counts = merge_official(load_record(path, backup), official)
            save_record(path, backup, merged)
            out[name] = counts
            log.info(
                "track record sync (%s): %d official rows added, %d recorded in both (official kept), "
                "%d local-only rows kept",
                name,
                counts["adopted"],
                counts["in_both"],
                counts["local_only"],
            )
    return out
