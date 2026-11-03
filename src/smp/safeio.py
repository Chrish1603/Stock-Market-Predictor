"""Crash-safe file writes and a lock against overlapping pipeline runs.

Writes go to a temporary file in the same folder and are then renamed into place, which is atomic on the
same filesystem: a crash or full disk leaves the old file intact rather than a half-written one.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path: Path, obj, **kwargs) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str, **kwargs))


def atomic_write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class PipelineBusy(RuntimeError):
    """Another pipeline run holds the lock."""


@contextmanager
def run_lock(lock_dir: Path, name: str = "pipeline") -> Iterator[None]:
    """Hold an exclusive lock for the duration of a run; fail fast if another run holds it.

    The OS releases the lock automatically if the process dies, so a crash never leaves it stuck.
    """
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f".{name}.lock"
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            holder = f.read().strip() or "unknown"
            raise PipelineBusy(f"another {name} run is in progress (pid {holder})") from None
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield
        finally:
            f.seek(0)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
