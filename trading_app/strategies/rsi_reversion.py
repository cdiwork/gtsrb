from __future__ import annotations

import pandas as pd

from .base import Strategy


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


class RSIReversionStrategy(Strategy):
    """Mean-reversion strategy: buy when RSI drops below `oversold`, exit once
    it climbs back above `exit_level`."""

    name = "rsi_reversion"

    def __init__(self, window: int = 14, oversold: float = 30, exit_level: float = 50):
        self.window = window
        self.oversold = oversold
        self.exit_level = exit_level

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        rsi = _rsi(bars["close"], self.window)

        signal = pd.Series(0, index=bars.index, dtype=int)
        in_position = False
        for i, value in enumerate(rsi):
            if pd.isna(value):
                signal.iloc[i] = 0
                continue
            if not in_position and value < self.oversold:
                in_position = True
            elif in_position and value > self.exit_level:
                in_position = False
            signal.iloc[i] = 1 if in_position else 0

        signal.name = "signal"
        return signal
