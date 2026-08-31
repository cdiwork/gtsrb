import numpy as np
import pandas as pd
import pytest

from trading_app.strategies import BreakoutStrategy, RSIReversionStrategy, SMACrossoverStrategy


def make_bars(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000] * len(closes),
        },
        index=index,
    )


def test_sma_crossover_goes_long_on_uptrend():
    closes = list(np.linspace(100, 200, 60))
    bars = make_bars(closes)
    strategy = SMACrossoverStrategy(fast_window=5, slow_window=20)
    signals = strategy.generate_signals(bars)

    assert signals.iloc[-1] == 1
    assert signals.iloc[:19].eq(0).all()  # slow SMA needs 20 points to be valid


def test_sma_crossover_rejects_invalid_windows():
    with pytest.raises(ValueError):
        SMACrossoverStrategy(fast_window=30, slow_window=10)


def test_rsi_reversion_enters_after_selloff():
    closes = [100] * 15 + list(np.linspace(100, 60, 15)) + list(np.linspace(60, 90, 15))
    bars = make_bars(closes)
    strategy = RSIReversionStrategy(window=14, oversold=30, exit_level=50)
    signals = strategy.generate_signals(bars)

    assert signals.max() == 1
    assert signals.iloc[-1] == 0  # should have exited by the time RSI recovered


def test_breakout_enters_on_new_high():
    closes = [100] * 25 + [150] + [150] * 10
    bars = make_bars(closes)
    strategy = BreakoutStrategy(lookback=20)
    signals = strategy.generate_signals(bars)

    assert signals.iloc[26] == 1
