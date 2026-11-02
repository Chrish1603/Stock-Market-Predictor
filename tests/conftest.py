import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 600, seed: int = 0, start: str = "2018-01-01", base: float = 100.0) -> pd.DataFrame:
    """Synthetic random-walk daily bars on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="Date")
    close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv()
