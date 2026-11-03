"""Price tab: candlesticks, the expected price range, earnings days; technical indicators on request."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from smp.features import bollinger, features_path, macd, rsi
from smp.insights import earnings_days, price_range
from smp.risk import risk_predictions_path
from smp_dashboard.charts import key, show, style
from smp_dashboard.data import (
    coverage,
    load_parquet,
)
from smp_dashboard.theme import FONT, palette


def render(cfg, ticker: str, prices: pd.DataFrame) -> None:
    C = palette()  # looked up on every run: the theme can change between runs
    ranges = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "All": None}
    ctl_range, ctl_bb = st.columns([3, 2], vertical_alignment="bottom")
    choice = ctl_range.segmented_control("Range", list(ranges), default="1Y", key="price_range") or "1Y"
    show_bb = ctl_bb.toggle("Bollinger band", key="show_bb")

    c = prices["Close"]
    ind = pd.DataFrame(index=prices.index)
    ind["sma20"], ind["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    ind = ind.join(bollinger(c)).join(macd(c))
    ind["rsi"] = rsi(c)
    n = ranges[choice]
    px, ix = (prices.iloc[-n:], ind.iloc[-n:]) if n else (prices, ind)

    # Expected range over the risk horizon, from today's volatility forecast
    rp_all = load_parquet(str(risk_predictions_path(cfg)))
    cone_h = max(cfg.risk.horizons) if cfg.risk.enabled else None
    rr = rp_all[(rp_all["ticker"] == ticker) & (rp_all["horizon"] == cone_h)] if rp_all is not None else None
    cone = (
        price_range(float(c.iloc[-1]), float(rr["forecast_vol"].iloc[0]), c.index[-1], cone_h)
        if rr is not None and not rr.empty
        else None
    )
    feats_t = load_parquet(str(features_path(cfg, ticker)))
    past_e, next_e = (
        earnings_days(feats_t["days_to_earnings"])
        if feats_t is not None and "days_to_earnings" in feats_t and feats_t.get("has_earnings", pd.Series([0])).max()
        else (pd.DatetimeIndex([]), None)
    )

    keys = [
        key(C["up"], "up day", "area"),
        key(C["down"], "down day", "area"),
        key(C["s2"], "20-day average"),
        key(C["s3"], "50-day average"),
    ]
    if cone is not None:
        keys.append(key(C["band_key"], f"expected range, next {cone_h} days", "area"))
    if len(past_e) or next_e is not None:
        keys.append(key(C["muted"], "earnings", "triangle"))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.8, 0.2],
        subplot_titles=(f"<b>{ticker}</b>   " + "  ".join(keys), "<b>Volume</b>"),
    )
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=13, family=FONT))
    if show_bb:
        fig.add_trace(go.Scatter(x=ix.index, y=ix["upper"], line=dict(width=0), hoverinfo="skip"), 1, 1)
        fig.add_trace(
            go.Scatter(
                x=ix.index,
                y=ix["lower"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor=C["band2"],
                name="Bollinger band",
                hoverinfo="skip",
            ),
            1,
            1,
        )
    if cone is not None:
        for lo, hi, alpha_key in (("low_wide", "high_wide", "band"), ("low_typical", "high_typical", "band_key")):
            fig.add_trace(go.Scatter(x=cone.index, y=cone[hi], line=dict(width=0), hoverinfo="skip"), 1, 1)
            fig.add_trace(
                go.Scatter(
                    x=cone.index,
                    y=cone[lo],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=C[alpha_key],
                    customdata=cone[hi],
                    name="typical range" if lo == "low_typical" else "wide range",
                    hovertemplate="$%{y:,.0f} – $%{customdata:,.0f}",
                ),
                1,
                1,
            )
        end = cone.iloc[-1]
        for col_ in ("high_wide", "high_typical", "low_typical", "low_wide"):
            fig.add_annotation(
                x=cone.index[-1],
                y=end[col_],
                text=f"${end[col_]:,.0f}",
                showarrow=False,
                xanchor="left",
                xshift=4,
                font=dict(size=11),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Candlestick(
            x=px.index,
            open=px["Open"],
            high=px["High"],
            low=px["Low"],
            close=px["Close"],
            name=ticker,
            increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up"]),
            decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down"]),
        ),
        1,
        1,
    )
    line = dict(width=1.75)
    fig.add_trace(go.Scatter(x=ix.index, y=ix["sma20"], name="20-day avg", line=line | dict(color=C["s2"])), 1, 1)
    fig.add_trace(go.Scatter(x=ix.index, y=ix["sma50"], name="50-day avg", line=line | dict(color=C["s3"])), 1, 1)
    shown_e = past_e[past_e >= px.index[0]]
    if len(shown_e):
        fig.add_trace(
            go.Scatter(
                x=shown_e,
                y=prices["Low"].reindex(shown_e) * 0.97,
                mode="markers",
                name="Earnings",
                marker=dict(symbol="triangle-up", size=9, color=C["muted"]),
                hovertemplate="earnings reaction day",
            ),
            1,
            1,
        )
    if next_e is not None and cone is not None and next_e <= cone.index[-1]:
        fig.add_vline(x=next_e, line=dict(color=C["muted"], width=1), row=1, col=1)
        fig.add_annotation(
            x=next_e,
            y=1,
            yref="y domain",
            text=f"earnings {next_e:%d %b}",
            showarrow=False,
            xanchor="left",
            xshift=4,
            yanchor="top",
            font=dict(size=11, color=C["muted"]),
            row=1,
            col=1,
        )
    vol_colors = np.where(px["Close"] >= px["Open"], C["up"], C["down"])
    fig.add_trace(go.Bar(x=px.index, y=px["Volume"], marker=dict(color=vol_colors, opacity=0.55), name="Volume"), 2, 1)
    x_end = cone.index[-1] + pd.offsets.BDay(3) if cone is not None else px.index[-1]
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False, bargap=0.25)
    fig.update_xaxes(range=[px.index[0], x_end])
    fig.update_yaxes(tickformat="~s", row=2, col=1)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    style(fig, height=640)
    fig.update_layout(margin=dict(t=30, r=56), showlegend=False)
    show(fig)
    if cone is not None:
        end, vol_now = cone.iloc[-1], float(rr["forecast_vol"].iloc[0])
        cov = coverage(ticker, cone_h)
        st.caption(
            f"Over the next {cone_h} trading days (to about {cone.index[-1]:%a %d %b}), {ticker} typically stays "
            f"between **\\${end['low_typical']:,.0f} and \\${end['high_typical']:,.0f}** (about 2 in 3 chance) and "
            f"very likely between \\${end['low_wide']:,.0f} and \\${end['high_wide']:,.0f} (about 19 in 20). "
            f"From the {vol_now:.0%} volatility forecast, centred on today's price, with no view on direction. "
            + (
                f"In testing, {cov['typical']:.0%} and {cov['wide']:.0%} of {cone_h}-day moves stayed inside these "
                "ranges, so treat them as slightly too narrow: big surprises happen more often than a bell curve "
                "allows."
                if cov["n"]
                else ""
            )
        )

    with st.expander("Technical indicators: RSI and MACD"):
        st.caption(
            "Standard chart-reading indicators. In this project's tests, inputs like these didn't make up/down "
            "calls meaningfully better than a coin flip, so they're here for reference."
        )
        tf = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.16,
            subplot_titles=(
                "<b>RSI (14)</b>   above 70 stretched, below 30 washed out",
                f"<b>MACD</b>   {key(C['s1'], 'MACD')}  {key(C['s2'], 'signal')}  "
                f"{key(C['muted'], 'difference', 'area')}",
            ),
        )
        for ann in tf.layout.annotations:
            ann.update(x=0, xanchor="left", font=dict(size=13, family=FONT))
        tf.add_hrect(
            y0=30, y1=70, fillcolor=C["band"], line_width=0, layer="below", row=1, col=1, exclude_empty_subplots=False
        )
        tf.add_trace(go.Scatter(x=ix.index, y=ix["rsi"], name="RSI", line=dict(color=C["s1"], width=1.75)), 1, 1)
        tf.add_trace(
            go.Bar(x=ix.index, y=ix["hist"], marker=dict(color=C["muted"], opacity=0.45), name="Difference"), 2, 1
        )
        tf.add_trace(go.Scatter(x=ix.index, y=ix["macd"], name="MACD", line=line | dict(color=C["s1"])), 2, 1)
        tf.add_trace(go.Scatter(x=ix.index, y=ix["signal"], name="Signal", line=line | dict(color=C["s2"])), 2, 1)
        tf.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=1, col=1)
        tf.update_layout(showlegend=False, bargap=0.25)
        style(tf, height=420)
        tf.update_layout(margin=dict(t=30), showlegend=False)
        show(tf)
    with st.expander("Data table"):
        st.dataframe(px.join(ix).sort_index(ascending=False).round(3), width="stretch")
