from __future__ import annotations

import pandas as pd

from .base import Strategy


class BreakoutStrategy(Strategy):
    """Donchian-channel breakout: go long when price closes above the highest
    close of the prior `lookback` bars, exit when it closes below the lowest
    close of the prior `lookback` bars."""

    name = "breakout"

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        upper = bars["close"].rolling(self.lookback).max().shift(1)
        lower = bars["close"].rolling(self.lookback).min().shift(1)

        signal = pd.Series(0, index=bars.index, dtype=int)
        in_position = False
        for i in range(len(bars)):
            close = bars["close"].iloc[i]
            hi, lo = upper.iloc[i], lower.iloc[i]
            if pd.isna(hi) or pd.isna(lo):
                signal.iloc[i] = 0
                continue
            if not in_position and close > hi:
                in_position = True
            elif in_position and close < lo:
                in_position = False
            signal.iloc[i] = 1 if in_position else 0

        signal.name = "signal"
        return signal
