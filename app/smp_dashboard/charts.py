"""House chart style for Plotly figures, and small chart helpers."""

from __future__ import annotations

import re

import plotly.graph_objects as go
import streamlit as st

from smp_dashboard.theme import FONT, NEUTRAL


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    """House style: transparent background, left-aligned title with the legend right under it, hairline grid,
    solid crosshair, tooltips on the chart surface, 4px rounded bar ends."""
    has_title = bool(fig.layout.title.text)
    named = [t for t in fig.data if t.name and t.showlegend is not False]
    legend = fig.layout.showlegend is not False and len(named) >= 2  # one series: the title names it
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=12, t=(34 if has_title else 10) + (26 if legend else 0), b=4),
        font=dict(family=FONT, size=12),
        title=dict(
            x=0,
            xref="paper",
            xanchor="left",
            y=1,
            yref="paper",
            yanchor="bottom",
            pad=dict(b=34 if legend else 12),
            font=dict(size=15, family=FONT),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode=fig.layout.hovermode or "x unified",
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        showlegend=legend,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1,
            yanchor="bottom",
            font=dict(size=12),
            itemwidth=30,
            title=None,
        ),
        barcornerradius=4,
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=NEUTRAL["axis"],
        title_font=dict(size=12),
        ticks="",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=NEUTRAL["axis"],
        spikedash="solid",
    )
    fig.update_yaxes(
        gridcolor=NEUTRAL["grid"],
        gridwidth=1,
        zeroline=False,
        title_font=dict(size=12),
    )
    return fig


def fade(rgba: str, start: float | None = None, stop: float | None = None, deep_at_bottom: bool = False) -> dict:
    """A fill that fades to transparent, instead of a flat block (flat fills turn into slabs on dark pages)."""
    clear = re.sub(r"[\d.]+\)$", "0)", rgba)
    scale = [(0, rgba), (1, clear)] if deep_at_bottom else [(0, clear), (1, rgba)]
    return dict(type="vertical", colorscale=scale, start=start, stop=stop)


def key(color: str, label: str, mark: str = "line") -> str:
    """An inline legend key: a coloured mark beside label text (text stays in ink colours)."""
    glyph = {"line": "━", "area": "■", "dot": "●", "triangle": "▲"}[mark]
    return f"<span style='color:{color}'>{glyph}</span> {label}"


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit")
