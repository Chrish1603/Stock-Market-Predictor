"""Streamlit dashboard: risk forecasts, allocation, prices, direction models, track record, data quality.

Run with:  streamlit run app/dashboard.py
Reads pipeline outputs only; training happens in the pipeline (`smp run-all`). Each tab lives in its own
module in app/smp_dashboard/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
# The repo root is the folder above app/. Set it before importing smp, which reads it at import time
# (needed when smp is installed as a regular package, as on Streamlit Cloud).
os.environ.setdefault("SMP_PROJECT_ROOT", str(APP_DIR.parent))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

OWN_PACKAGES = ("smp", "smp_dashboard")


def drop_stale_modules() -> None:
    """Streamlit keeps imported modules between reruns, and Streamlit Cloud pulls code updates without
    restarting. A new module importing from an old one already loaded then fails, so when any of this
    project's source files is newer than the import, forget all of its modules and import them afresh."""
    stale = False
    for name in OWN_PACKAGES:
        if (pkg := sys.modules.get(name)) is not None:
            loaded_at = getattr(pkg, "_dashboard_loaded_at", 0.0)
            newest = max((p.stat().st_mtime for p in Path(pkg.__file__).parent.glob("*.py")), default=0.0)
            stale = stale or newest > loaded_at
    if stale:
        for name in [m for m in sys.modules if m.split(".")[0] in OWN_PACKAGES]:
            del sys.modules[name]
    for name in OWN_PACKAGES:
        pkg = __import__(name)
        if not hasattr(pkg, "_dashboard_loaded_at"):
            pkg._dashboard_loaded_at = time.time()


drop_stale_modules()

from smp import pipeline  # noqa: E402
from smp.config import load_config  # noqa: E402
from smp.risk import risk_predictions_path  # noqa: E402
from smp_dashboard import data, direction, price, quality, risk, today, track  # noqa: E402
from smp_dashboard.data import load_json, load_parquet, load_prices  # noqa: E402
from smp_dashboard.theme import inject_css  # noqa: E402

st.set_page_config(page_title="Stock Market Predictor", layout="wide")


@st.cache_resource(ttl=3600, show_spinner="Downloading the latest data snapshot...")
def download_snapshot(url: str) -> str:
    """Fetch and unpack a published snapshot (see `smp publish`); refreshed at most hourly."""
    from smp.publish import fetch_snapshot

    return str(fetch_snapshot(url))


def choose_data_root() -> bool:
    """Use local pipeline output if present; otherwise the published snapshot. True means snapshot mode."""
    if os.environ.get("SMP_DATA_ROOT") and not os.environ.get("SMP_SNAPSHOT"):
        return False  # set explicitly by the user
    os.environ.pop("SMP_DATA_ROOT", None)
    local = load_config()
    forced = os.environ.get("SMP_SNAPSHOT_URL")  # always use this snapshot (hosting, testing)
    url = forced or local.dashboard.data_url
    if not url or (not forced and (local.paths.models / "risk_predictions.parquet").exists()):
        return False
    os.environ["SMP_DATA_ROOT"] = download_snapshot(url)
    os.environ["SMP_SNAPSHOT"] = "1"
    return True


try:
    SNAPSHOT = choose_data_root()
except Exception as exc:  # network or GitHub problem: explain instead of showing a traceback
    st.error(
        "Couldn't download the latest data snapshot, so there's nothing to show yet. "
        "This is usually a temporary network or GitHub problem."
    )
    st.caption(f"Details: {type(exc).__name__}: {exc}")
    if st.button("Try again"):
        download_snapshot.clear()
        st.rerun()
    st.stop()

CFG = load_config()
data.configure(CFG)
data.refresh_if_changed()
inject_css()

# --- sidebar -----------------------------------------------------------------------------------

st.sidebar.title("Stock Market Predictor")
st.sidebar.caption("Risk forecasts and position sizing for US stocks, with direction models for comparison.")
ticker = st.sidebar.selectbox("Stock", CFG.data.tickers, help="Used by the Price, Direction and Track record tabs")
_latest = load_prices(CFG.data.tickers[0])
if _latest is not None:
    st.sidebar.markdown(f"**Data as of** {_latest.index[-1]:%a %d %b %Y}")
if SNAPSHOT:
    built = (load_json(str(Path(os.environ["SMP_DATA_ROOT"]) / "manifest.json")) or {}).get("generated_at", "?")
    st.sidebar.caption(f"Hosted snapshot, rebuilt each US trading day (last {built[:16].replace('T', ' ')} UTC).")
elif st.sidebar.button(
    "Refresh data", help="Downloads new prices and updates forecasts. Does not retrain.", width="stretch"
):
    from smp.safeio import PipelineBusy

    with st.spinner("Downloading prices and updating forecasts..."):
        pipeline.setup_logging(CFG.paths.logs)
        try:
            pipeline.refresh(CFG)
            st.sidebar.success("Updated")
        except PipelineBusy:
            st.sidebar.info("The scheduled run is in progress; try again in a few minutes.")
        except pipeline.PipelineError as exc:
            st.sidebar.warning(f"Partly updated. Failed: {', '.join(f['stage'] for f in exc.failures)}.")
    st.cache_data.clear()
st.sidebar.caption("Models retrain in the daily scheduled run (`smp run-all`).")
st.sidebar.divider()
st.sidebar.caption("Educational project, not financial advice.")

prices = load_prices(ticker)
if prices is None:
    st.warning("No data yet. Run `smp run-all` first.")
    st.stop()

# Warn visitors when the daily update has stopped
_rp = load_parquet(str(risk_predictions_path(CFG)))
if _rp is not None and len(_rp):
    from smp.health import business_days_behind, is_stale

    _as_of = pd.to_datetime(_rp["as_of"]).max().date()
    if is_stale(_as_of):
        _n = business_days_behind(_as_of)
        st.warning(
            f"⚠ Data is {_n} trading day{'s' if _n != 1 else ''} out of date (last close {_as_of:%a %d %b}). "
            "The daily update may have failed; forecasts below are from that close.",
        )

# --- tabs --------------------------------------------------------------------------------------

TABS = {
    "Today": today,
    "Risk & allocation": risk,
    "Price & indicators": price,
    "Direction models": direction,
    "Track record": track,
    "Data quality": quality,
}
for tab, view in zip(st.tabs(list(TABS)), TABS.values(), strict=True):
    with tab:
        view.render(CFG, ticker, prices)
