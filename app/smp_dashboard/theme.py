"""Colours and page styling.

Chart colours come from a validated palette, stepped separately for light and dark. The viewer can switch theme
without the script rerunning, so anything page-wide is written with CSS light-dark() (which follows the live
theme), greys inside charts read on both backgrounds, and chart text colours come from Streamlit's theme.
"""

from __future__ import annotations

import streamlit as st

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'

# Greys used inside charts: readable on light and dark backgrounds alike.
NEUTRAL = dict(muted="#868b93", actual="#868b93", grid="rgba(128,128,128,0.2)", axis="rgba(128,128,128,0.45)")
RING = "rgba(255,255,255,0.65)"  # separates markers from the lines they sit on

PALETTES = {
    "light": dict(
        s1="#2a78d6",
        s2="#eb6834",
        s3="#1baf7a",
        up="#2a78d6",
        down="#e34948",
        fill_top="rgba(42,120,214,0.22)",  # top of fading area fills
        fill_top2="rgba(235,104,52,0.20)",
        band="rgba(42,120,214,0.10)",
        band_key="rgba(42,120,214,0.35)",  # the band's wash, strong enough to read as a legend key
        band2="rgba(235,104,52,0.10)",
        calm_wash="rgba(12,163,12,0.10)",
        stressed_wash="rgba(208,59,59,0.16)",
        seq=["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"],
    ),
    "dark": dict(
        s1="#3987e5",
        s2="#d95926",
        s3="#199e70",
        up="#3987e5",
        down="#e66767",
        fill_top="rgba(57,135,229,0.30)",
        fill_top2="rgba(217,89,38,0.26)",
        band="rgba(57,135,229,0.16)",
        band_key="rgba(57,135,229,0.45)",
        band2="rgba(217,89,38,0.16)",
        calm_wash="rgba(25,158,112,0.07)",  # green washes read as murky on a dark page, so keep calm faint
        stressed_wash="rgba(230,103,103,0.16)",
        seq=["#104281", "#256abf", "#5598e7", "#b7d3f6"],
    ),
}
for _p in PALETTES.values():
    _p.update(muted=NEUTRAL["muted"], grid=NEUTRAL["grid"], axis=NEUTRAL["axis"], actual=NEUTRAL["actual"])

# Status colours are reserved for the regime and always shown with an icon and a label.
REGIME_STYLE = {
    "calm": ("#0ca30c", "●", "Calm"),
    "normal": (NEUTRAL["muted"], "◐", "Normal"),
    "stressed": ("#d03b3b", "▲", "Stressed"),
}


def palette() -> dict:
    """Chart colours for the theme the viewer had at the start of this run."""
    kind = getattr(getattr(st.context, "theme", None), "type", None)
    return PALETTES["dark" if kind == "dark" else "light"]


def roles(c: dict) -> tuple[str, str, str, str]:
    """Colour roles, the same in every chart: what this project built (blue), the simple alternative it's
    compared with (orange), a second variant of our own (aqua), and what actually happened (grey)."""
    return c["s1"], c["s2"], c["s3"], c["actual"]


CSS = """
<style>
  [data-testid="stSidebar"] h1 { font-size: 1.3rem; letter-spacing: -0.01em; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
  [data-testid="stHeader"] { background: transparent; }
  /* light-dark() follows the theme the viewer has now, even if they switch without a rerun */
  [data-testid="stApp"] {
    background:
      radial-gradient(1100px 540px at 86% -9%,
        light-dark(rgba(42,120,214,0.13), rgba(57,135,229,0.14)), transparent 70%),
      radial-gradient(900px 480px at 8% -12%,
        light-dark(rgba(27,175,122,0.07), rgba(0,0,0,0)), transparent 70%),
      linear-gradient(180deg, light-dark(#f2f6fc, #121720) 0%, light-dark(#f8f9fa, #10141a) 400px,
        light-dark(#fcfcfb, #0f1216) 900px);
    background-attachment: fixed;
  }
  [class*="st-key-card-"] { background: light-dark(rgba(255,255,255,0.72), rgba(255,255,255,0.028)); }
  .tone-up { color: light-dark(#006300, #0ca30c); }
  .tone-down { color: light-dark(#c03030, #e66767); }
  h1, h2, h3 { letter-spacing: -0.015em; }
  .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; }
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding-left: 0; padding-right: 0; }
  [data-testid="stMetricValue"] { letter-spacing: -0.02em; }
  .eyebrow { text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem; font-weight: 600; opacity: 0.6; }
  .muted { opacity: 0.65; }
  .hero { border-radius: 0.9rem; padding: 1.25rem 1.5rem; margin: 0.25rem 0 1.25rem 0; }
  .hero-title { font-size: 1.9rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.2; }
  .chips { display: flex; gap: 2.2rem; flex-wrap: wrap; margin-top: 0.9rem; }
  .chip-label { font-size: 0.75rem; opacity: 0.65; }
  .chip-value { font-size: 1.05rem; font-weight: 600; }
  .card-head { display: flex; justify-content: space-between; align-items: baseline; }
  .ticker { font-size: 1.15rem; font-weight: 650; letter-spacing: -0.01em; }
  .price { font-size: 0.95rem; opacity: 0.8; margin-left: 0.5rem; }
  .delta { font-size: 0.85rem; font-weight: 600; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.25rem; }
  .stat-label { font-size: 0.74rem; opacity: 0.65; }
  .stat-value { font-size: 1.35rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.3; }
  .stat-note { font-size: 0.74rem; opacity: 0.65; margin-top: 0.15rem; }
  .bar { height: 5px; border-radius: 3px; background: rgba(128,128,128,0.18); margin-top: 0.35rem; overflow: hidden; }
  .bar > span { display: block; height: 100%; border-radius: 3px; background: light-dark(#2a78d6, #3987e5); }
  .card-text { font-size: 0.86rem; line-height: 1.45; margin-top: 0.75rem; opacity: 0.85; }
  .card-foot { font-size: 0.74rem; opacity: 0.55; margin-top: 0.6rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
