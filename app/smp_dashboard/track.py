"""Track record tab: every live forecast and how it turned out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smp.history import summarize
from smp.risk_history import summarize_risk
from smp_dashboard.charts import show, style
from smp_dashboard.data import (
    load_risk_track_record,
    load_track_record,
)
from smp_dashboard.labels import (
    as_text,
    horizon_label,
)
from smp_dashboard.theme import palette, roles


def render(cfg, ticker: str, prices: pd.DataFrame) -> None:
    C = palette()  # looked up on every run: the theme can change between runs
    OURS, ALT, _, ACTUAL = roles(C)
    st.subheader("Live track record")
    st.caption(
        "Every forecast is saved when it's made and scored once its horizon has passed. Unlike the backtests, "
        "these were made before the outcome existed, so they can't have been tuned to it."
    )

    st.markdown("#### Risk forecasts")
    st.caption(
        "Volatility forecasts are compared with what volatility turned out to be, and with simply quoting last "
        "month's volatility. Drop chances are compared with what happened, and with quoting the historical "
        "drop rate. Lower errors and Brier scores are better."
    )
    rhist = load_risk_track_record()
    risk_track_h = (
        st.segmented_control(
            "Risk horizon",
            cfg.risk.horizons,
            default=cfg.risk.horizons[0],
            format_func=horizon_label,
            key="track_risk_h",
        )
        or cfg.risk.horizons[0]
    )
    if rhist.empty:
        st.info("No risk forecasts recorded yet. Run `smp predict` (or `smp run-all`).")
    else:
        rsum = summarize_risk(rhist)
        rsum = rsum[rsum["horizon"] == risk_track_h].set_index("ticker")
        rsum = rsum.assign(first=rsum["first"].dt.date, last=rsum["last"].dt.date)
        rsum = rsum[
            [
                "predictions",
                "scored",
                "vol_error",
                "trailing_vol_error",
                "closer_share",
                "drops",
                "brier",
                "base_rate_brier",
                "first",
                "last",
            ]
        ]
        rsum.columns = [
            "Forecasts",
            "Scored",
            "Vol off by",
            "Last month's vol off by",
            "Forecast closer",
            "Drops seen",
            "Drop Brier",
            "Historical-rate Brier",
            "First",
            "Latest",
        ]
        rsum.index.name = "Stock"
        st.dataframe(
            as_text(
                rsum,
                {
                    "Vol off by": "{:.0%}",
                    "Last month's vol off by": "{:.0%}",
                    "Forecast closer": "{:.0%}",
                    "Drop Brier": "{:.3f}",
                    "Historical-rate Brier": "{:.3f}",
                },
            ),
            width="stretch",
        )
        rt = rhist[(rhist["ticker"] == ticker) & (rhist["horizon"] == risk_track_h)].sort_values("as_of")
        rscored = rt.dropna(subset=["realised_vol"])
        if len(rscored) < 2:
            pending = rt[rt["realised_vol"].isna()]
            first_due = (
                f" The first is scored after the close on about "
                f"{(pending['as_of'].min() + pd.offsets.BDay(risk_track_h)):%a %d %b}."
                if len(pending) and rscored.empty
                else ""
            )
            st.info(
                f"{ticker}: {len(rt)} {risk_track_h}-day forecast{'s' if len(rt) != 1 else ''} recorded, "
                f"{len(rscored)} scored so far.{first_due} The chart appears once a few have been scored."
            )
        else:
            fig = go.Figure()
            for col, name, color, width in (
                ("realised_vol", "What happened", ACTUAL, 1.5),
                ("trailing_vol", "Last month's vol", ALT, 1.75),
                ("forecast_vol", "Forecast", OURS, 2),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=rscored["as_of"],
                        y=rscored[col],
                        name=name,
                        line=dict(color=color, width=width),
                        hovertemplate="%{y:.0%}",
                    )
                )
            fig.update_layout(title=f"{ticker} volatility over the next {horizon_label(risk_track_h)}, by forecast day")
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, height=360))
        if not rt.empty:
            recent = rt.sort_values("as_of", ascending=False).assign(
                as_of=lambda d: d["as_of"].dt.date,
                target_date=lambda d: pd.to_datetime(d["target_date"]).dt.date,
                dropped=lambda d: d["dropped"].map({1.0: "yes", 0.0: "no"}).fillna("pending"),
            )
            recent = recent[
                ["as_of", "forecast_vol", "trailing_vol", "realised_vol", "p_drop", "dropped", "target_date"]
            ]
            recent.columns = [
                "Made on",
                "Forecast vol",
                "Last month's vol",
                "What happened",
                "Drop chance",
                "Dropped?",
                "Scored on",
            ]
            st.markdown(f"**{ticker}: every {risk_track_h}-day risk forecast**")
            st.dataframe(
                as_text(
                    recent,
                    {
                        "Forecast vol": "{:.0%}",
                        "Last month's vol": "{:.0%}",
                        "What happened": "{:.0%}",
                        "Drop chance": "{:.0%}",
                    },
                ),
                hide_index=True,
                width="stretch",
            )

    st.markdown("#### Direction calls")
    st.caption("Up/down calls, scored after each horizon's close. Compare the hit rate with always guessing 'up'.")
    hist = load_track_record()
    track_h = (
        st.segmented_control(
            "Horizon", cfg.train.horizons, default=cfg.train.horizons[0], format_func=horizon_label, key="track_h"
        )
        or cfg.train.horizons[0]
    )
    if hist.empty:
        st.info("No predictions recorded yet. Run `smp predict` (or `smp run-all`).")
    else:
        summary = summarize(hist)
        summary = summary[summary["horizon"] == track_h].set_index("ticker")
        summary = summary.assign(first=summary["first"].dt.date, last=summary["last"].dt.date)
        summary = summary[["predictions", "scored", "hit_rate", "always_up_hit_rate", "brier", "first", "last"]]
        summary.columns = [
            "Predictions",
            "Scored",
            "Hit rate",
            "Always 'up' hit rate",
            "Brier score",
            "First",
            "Latest",
        ]
        summary.index.name = "Stock"
        st.dataframe(
            as_text(summary, {"Hit rate": "{:.0%}", "Always 'up' hit rate": "{:.0%}", "Brier score": "{:.3f}"}),
            width="stretch",
        )
        th = hist[(hist["ticker"] == ticker) & (hist["horizon"] == track_h)].sort_values("as_of")
        scored = th.dropna(subset=["actual_return"])
        if len(scored) < 2:
            st.info(
                f"{ticker}: {len(th)} prediction{'s' if len(th) != 1 else ''} recorded, {len(scored)} scored so far. "
                "The chart appears once a few days have been scored. Run the pipeline each trading day."
            )
        else:
            n = np.arange(1, len(scored) + 1)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=scored["as_of"],
                    y=scored["correct"].cumsum() / n,
                    name="Model",
                    line=dict(color=OURS, width=2),
                    hovertemplate="%{y:.0%}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=scored["as_of"],
                    y=scored["actual_direction"].cumsum() / n,
                    name="Always 'up'",
                    line=dict(color=ALT, width=2),
                    hovertemplate="%{y:.0%}",
                )
            )
            fig.add_hline(
                y=0.5,
                line=dict(color=C["axis"], width=1),
                annotation=dict(text="coin flip", font=dict(color=C["muted"], size=11)),
                annotation_position="bottom right",
            )
            fig.update_layout(title=f"{ticker} cumulative hit rate")
            fig.update_yaxes(tickformat=".0%")
            show(style(fig, height=360))
        if not th.empty:
            recent = th.sort_values("as_of", ascending=False).assign(
                as_of=lambda d: d["as_of"].dt.date,
                target_date=lambda d: pd.to_datetime(d["target_date"]).dt.date,
                result=lambda d: d["correct"].map({1.0: "✓ right", 0.0: "✗ wrong"}).fillna("pending"),
            )
            recent = recent[["as_of", "direction", "p_up", "target_date", "actual_return", "result"]]
            recent.columns = ["Made on", "Call", "Chance of up", "Scored on", "What happened", "Result"]
            st.markdown(f"**{ticker}: every {track_h}-day prediction**")
            st.dataframe(
                as_text(recent, {"Chance of up": "{:.0%}", "What happened": "{:+.2%}"}),
                hide_index=True,
                width="stretch",
            )
