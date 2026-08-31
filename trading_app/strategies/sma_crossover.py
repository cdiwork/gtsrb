from __future__ import annotations

import pandas as pd

from .base import Strategy


class SMACrossoverStrategy(Strategy):
    """Classic trend-following strategy: go long while the fast SMA is above
    the slow SMA, flat otherwise."""

    name = "sma_crossover"

    def __init__(self, fast_window: int = 10, slow_window: int = 30):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        fast = bars["close"].rolling(self.fast_window).mean()
        slow = bars["close"].rolling(self.slow_window).mean()
        signal = (fast > slow).astype(int)
        signal[fast.isna() | slow.isna()] = 0
        signal.name = "signal"
        return signal
