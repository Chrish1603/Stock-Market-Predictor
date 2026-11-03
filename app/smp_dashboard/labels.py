"""Readable names and text formatting for the dashboard."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

FEATURE_NAMES = {
    "macd": "MACD",
    "macd_hist": "MACD histogram",
    "rsi_14": "RSI (14)",
    "stoch_k": "Stochastic %K",
    "atr_14": "Average daily range (ATR)",
    "bb_pctb": "Position in Bollinger band",
    "hl_range": "Today's high-low range",
    "volume_z": "Volume vs normal",
    "obv_change_10": "On-balance volume, 10d",
    "day_of_week": "Day of week",
    "month": "Month",
    "ewm_vol_20": "Volatility (EWMA)",
    "downside_vol_20": "Downside volatility",
    "parkinson_vol_20": "Range-based volatility",
    "vol_ratio_5_60": "Volatility, 5d vs 60d",
    "drawdown_60": "Drop from 60-day high",
    "days_to_earnings": "Days to earnings",
    "days_since_earnings": "Days since earnings",
    "last_eps_surprise": "Last earnings surprise",
    "has_earnings": "Has earnings",
    "vix_level": "VIX level",
    "vix_change_5": "VIX change, 5d",
    "yield_curve_change_20": "Yield curve change, 20d",
    "yield_curve_inverted": "Yield curve inverted",
    "sector_sma_ratio_50": "Sector vs its 50-day average",
}
METRIC_NAMES = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "log_loss": "Log loss",
    "up_rate": "Share of up days",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "directional_accuracy": "Direction right",
}


def feature_label(name: str) -> str:
    """Readable name for a model input, e.g. 'sma_ratio_50' -> 'Price vs 50-day average'."""
    if name in FEATURE_NAMES:
        return FEATURE_NAMES[name]
    if name.startswith("xs_rank_"):
        return f"Rank among stocks: {feature_label(name[8:]).lower()}"
    patterns = [
        (r"ret_lag_1", "Return yesterday"),
        (r"ret_lag_(\d+)", "Return {} days ago"),
        (r"ret_(\d+)", "Return, past {}d"),
        (r"sma_ratio_(\d+)", "Price vs {}-day average"),
        (r"ema_ratio_(\d+)", "Price vs {}-day EMA"),
        (r"vol_(\d+)", "Volatility, {}d"),
        (r"spy_ret_(\d+)", "S&P 500 return, {}d"),
        (r"(tnx|irx)_change_(\d+)", "{} yield change, {}d"),
        (r"sector_ret_(\d+)", "Sector return, {}d"),
        (r"rel_strength_(\d+)", "Strength vs sector, {}d"),
    ]
    for pattern, text in patterns:
        if m := re.fullmatch(pattern, name):
            parts = [{"tnx": "10-year", "irx": "3-month"}.get(g, g) for g in m.groups()]
            return text.format(*parts)
    return name


def as_text(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    """Format columns as strings with '–' for missing values (st.dataframe ignores Styler na_rep)."""
    out = df.copy().astype(object)
    for col in out.columns:
        fmt = formats.get(col)
        out[col] = [
            "–"
            if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v))
            else (fmt.format(v) if fmt else str(v))
            for v in df[col]
        ]
    return out


def horizon_label(h: int) -> str:
    return f"{h} day" if h == 1 else f"{h} days"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def risk_words(pct: float) -> str:
    if pct >= 0.8:
        return "well above normal"
    if pct >= 0.6:
        return "above normal"
    if pct > 0.4:
        return "normal"
    return "below normal" if pct > 0.2 else "low"


MODEL_NAMES = {
    "baseline": "Baseline",
    "logistic": "Logistic regression",
    "ridge": "Ridge regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "har": "HAR",
    "persistence": "Last month's vol",
    "vol_implied": "Volatility formula",
}


def model_label(name: str) -> str:
    """'lightgbm' -> 'LightGBM'."""
    return MODEL_NAMES.get(name, name)


STRATEGY_LABELS = {
    "buy_hold": "Buy & hold",
    "trailing_vol_target": "Vol target (trailing vol)",
    "model_vol_target": "Vol target (model forecast)",
    "model_vol_target_drop_filter": "Vol target + drop filter",
}
SHOWN_STRATEGIES = ["buy_hold", "trailing_vol_target", "model_vol_target"]
