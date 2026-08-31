import pandas as pd

from trading_app.backtest import run_backtest


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


def test_always_long_matches_buy_and_hold_minus_commission():
    closes = [100, 110, 121, 133.1]
    bars = make_bars(closes)
    signals = pd.Series([1, 1, 1, 1], index=bars.index)

    result = run_backtest(bars, signals, initial_capital=1000, commission_bps=0)

    assert result.equity_curve.iloc[-1] == pytest_approx(1000 * 133.1 / 100)
    assert result.metrics["num_trades"] == 1


def test_flat_signal_produces_no_trades():
    closes = [100, 101, 99, 102]
    bars = make_bars(closes)
    signals = pd.Series([0, 0, 0, 0], index=bars.index)

    result = run_backtest(bars, signals, initial_capital=1000)

    assert result.metrics["num_trades"] == 0
    assert result.equity_curve.iloc[-1] == 1000


def test_commission_reduces_returns_on_position_changes():
    closes = [100, 110, 100, 110]
    bars = make_bars(closes)
    signals = pd.Series([1, 0, 1, 0], index=bars.index)

    no_cost = run_backtest(bars, signals, initial_capital=1000, commission_bps=0)
    with_cost = run_backtest(bars, signals, initial_capital=1000, commission_bps=50)

    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]


def pytest_approx(value, rel=1e-6):
    import pytest

    return pytest.approx(value, rel=rel)
