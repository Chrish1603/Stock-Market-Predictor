"""Numbers behind the dashboard's explanatory charts.

- Expected price range: today's volatility forecast turned into a price cone (and a check of how
  often past moves stayed inside it).
- Calibration: do forecasts mean what they say? Predicted vs what happened, in groups.
- Vol targeting in action: rolling volatility, and when earnings fall.

All inputs are out-of-sample forecasts or plain prices, so these are fair checks, not fitted ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
Z_TYPICAL = 1.0  # about 2 in 3 moves, if returns were normal
Z_WIDE = 1.96  # about 19 in 20


def price_range(last_close: float, annual_vol: float, last_date, days: int = 20) -> pd.DataFrame:
    """Typical (±1σ) and wide (±1.96σ) price ranges for each of the next `days` trading days, centred on
    today's price (no view on direction). Row 0 is today."""
    t = np.arange(days + 1)
    s = annual_vol * np.sqrt(t / TRADING_DAYS)
    dates = pd.bdate_range(pd.Timestamp(last_date), periods=days + 1)
    return pd.DataFrame(
        {
            "low_typical": last_close * np.exp(-Z_TYPICAL * s),
            "high_typical": last_close * np.exp(Z_TYPICAL * s),
            "low_wide": last_close * np.exp(-Z_WIDE * s),
            "high_wide": last_close * np.exp(Z_WIDE * s),
        },
        index=dates,
    )


def range_coverage(pred_logvol: pd.Series, close: pd.Series, horizon: int) -> dict:
    """How often the actual move over `horizon` days stayed inside the ranges forecast that day.
    About 68% (typical) and 95% (wide) would mean the ranges are about right."""
    close = close.dropna()
    future = np.log(close.shift(-horizon) / close)
    df = pd.DataFrame({"vol": np.exp(pred_logvol), "move": future.reindex(pred_logvol.index)}).dropna()
    if df.empty:
        return {"typical": np.nan, "wide": np.nan, "n": 0}
    s = df["vol"] * np.sqrt(horizon / TRADING_DAYS)
    return {
        "typical": float((df["move"].abs() <= Z_TYPICAL * s).mean()),
        "wide": float((df["move"].abs() <= Z_WIDE * s).mean()),
        "n": int(len(df)),
    }


def calibration_bins(predicted, actual, n_bins: int = 10) -> pd.DataFrame:
    """Group forecasts into `n_bins` equal-sized groups by forecast; each group's average forecast vs average
    outcome. Points on the diagonal mean the forecasts meant what they said."""
    df = pd.DataFrame({"p": np.asarray(predicted, float), "y": np.asarray(actual, float)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["predicted", "actual", "n"])
    bins = pd.qcut(df["p"].rank(method="first"), min(n_bins, len(df)), labels=False)
    out = df.groupby(bins).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)


def rolling_vol(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Annualised volatility over a trailing window (simple daily returns)."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def earnings_days(days_to_earnings: pd.Series) -> tuple[pd.DatetimeIndex, pd.Timestamp | None]:
    """Past earnings reaction days, and the next one if it's known.

    `days_to_earnings` counts business days to the next reaction day (capped at 20 when unknown), so a
    reaction day is the day after a reading of 1, and the next one is that many business days ahead."""
    d = days_to_earnings.dropna()
    if d.empty:
        return pd.DatetimeIndex([]), None
    past = d.index[(d.shift(1) == 1).to_numpy()]
    last = float(d.iloc[-1])
    nxt = d.index[-1] + pd.offsets.BDay(int(last)) if 0 < last < 20 else None
    return past, nxt
