from .base import Strategy
from .breakout import BreakoutStrategy
from .rsi_reversion import RSIReversionStrategy
from .sma_crossover import SMACrossoverStrategy

STRATEGIES = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi_reversion": RSIReversionStrategy,
    "breakout": BreakoutStrategy,
}

__all__ = [
    "Strategy",
    "SMACrossoverStrategy",
    "RSIReversionStrategy",
    "BreakoutStrategy",
    "STRATEGIES",
]
