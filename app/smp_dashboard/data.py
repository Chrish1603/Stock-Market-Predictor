"""Cached loaders for pipeline outputs.

Loaders are keyed by ticker and horizon, not by file path, so `refresh_if_changed` clears them all when a new
snapshot arrives or the pipeline writes new forecasts (otherwise new forecasts could show beside old prices).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from smp.clean import clean_path, report_path
from smp.history import load_history
from smp.insights import range_coverage
from smp.predict import predictions_path
from smp.risk import risk_dir, risk_predictions_path
from smp.risk_history import load_risk_history
from smp.train import model_dir

_state: dict = {}


def configure(cfg) -> None:
    """Set the config the loaders read from (call once per run, before loading anything)."""
    _state["cfg"] = cfg


def _cfg():
    return _state["cfg"]


@st.cache_resource
def _loaded_version() -> dict:
    return {"version": None}


def refresh_if_changed() -> None:
    """Clear cached data when a new snapshot is downloaded or the pipeline writes new forecasts."""
    p = risk_predictions_path(_cfg())
    now = (os.environ.get("SMP_DATA_ROOT"), p.stat().st_mtime if p.exists() else None)
    seen = _loaded_version()
    if seen["version"] != now:
        st.cache_data.clear()
        seen["version"] = now


@st.cache_data
def load_prices(ticker: str) -> pd.DataFrame | None:
    p = clean_path(_cfg(), ticker)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    p = predictions_path(_cfg())
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_metrics(ticker: str, horizon: int) -> dict | None:
    p = model_dir(_cfg(), ticker, horizon) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_artifact(ticker: str, horizon: int, name: str) -> pd.DataFrame | None:
    p = model_dir(_cfg(), ticker, horizon) / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_track_record() -> pd.DataFrame:
    return load_history(_cfg())


@st.cache_data
def load_risk_track_record() -> pd.DataFrame:
    return load_risk_history(_cfg())


@st.cache_data
def load_oos(ticker: str, horizon: int) -> pd.DataFrame | None:
    """Out-of-sample risk forecasts: walk-forward folds, then the hold-out year."""
    parts = [
        load_parquet(str(risk_dir(_cfg(), ticker, horizon) / f))
        for f in ("oos_predictions.parquet", "holdout_predictions.parquet")
    ]
    if any(x is None for x in parts):
        return None
    oos = pd.concat(parts).sort_index()
    return oos[~oos.index.duplicated(keep="last")]


@st.cache_data
def coverage(ticker: str, horizon: int) -> dict:
    """How often past moves stayed inside the forecast price ranges."""
    oos, px_ = load_oos(ticker, horizon), load_prices(ticker)
    if oos is None or px_ is None:
        return {"typical": np.nan, "wide": np.nan, "n": 0}
    return range_coverage(oos["pred_logvol"], px_["Close"], horizon)


@st.cache_data
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_quality() -> dict:
    p = report_path(_cfg())
    return json.loads(p.read_text()) if p.exists() else {}
