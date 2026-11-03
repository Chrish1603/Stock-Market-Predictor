"""Data quality tab: what the cleaning step fixed or flagged."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smp_dashboard.data import (
    load_quality,
)


def render(cfg, ticker: str, prices: pd.DataFrame) -> None:
    q = load_quality()
    if not q:
        st.info("No quality report yet. Run `smp clean`.")
    else:
        summary = pd.DataFrame(
            {
                t: {
                    "Rows in": r["rows_in"],
                    "Rows out": r["rows_out"],
                    "Start": r["start"],
                    "End": r["end"],
                    "Duplicates removed": r["duplicates_removed"],
                    "Values filled": r["missing_values_filled"],
                    "Rows dropped": r["rows_dropped"],
                    "OHLC fixed": r["ohlc_fixed"],
                    "Extreme days": len(r["outlier_dates"]),
                    "Gaps over 5 days": len(r["large_gaps"]),
                }
                for t, r in q.items()
            }
        ).T
        st.subheader("Cleaning report")
        k = st.columns(4)
        k[0].metric("Stocks and indices", len(q))
        k[1].metric("Daily rows checked", f"{int(summary['Rows in'].sum()):,}")
        k[2].metric("Rows dropped", int(summary["Rows dropped"].sum()))
        k[3].metric("Extreme days flagged", int(summary["Extreme days"].sum()), help="Kept in the data, not removed")
        st.caption(
            "Every run validates prices: duplicates removed, short gaps filled, impossible bars (high below low) "
            "fixed, and extreme one-day moves flagged but kept, since they are usually real events."
        )
        st.dataframe(summary, width="stretch")
        if ticker in q:
            r = q[ticker]
            days = ", ".join(r["outlier_dates"]) or "none"
            st.markdown(f"**{ticker}: extreme-move days** (flagged, kept in the data): {days}")
            if r["large_gaps"]:
                st.write(f"**{ticker} gaps longer than 5 days:**")
                st.dataframe(pd.DataFrame(r["large_gaps"]), hide_index=True)
