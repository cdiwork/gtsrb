"""Strategy interface shared by all strategies.

A strategy turns a price DataFrame into a signal series: for each date,
 1  = hold/enter long
 0  = flat / no position
-1  = hold/enter short (only used by strategies that support shorting)

The backtest engine reacts to *changes* in this series (a transition into 1
opens a long, a transition to 0 closes it, etc.), so strategies just describe
the desired position at each bar rather than individual buy/sell orders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.Series:
        """bars: DataFrame indexed by date with [open, high, low, close, volume].
        Returns a Series of the same index with values in {-1, 0, 1}."""
        raise NotImplementedError
