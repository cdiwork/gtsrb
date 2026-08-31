"""A minimal, dependency-light backtesting engine.

Assumptions (deliberately simple for a first pass):
- Signals are evaluated on each day's close; a position change executes at
  that same close price (no next-day fill delay, no slippage model beyond a
  flat commission).
- Single-symbol, long-only or long/flat position sizing: 100% of equity is
  allocated to the position whenever signal == 1, 0% when signal == 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .metrics import summarize


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trade_returns: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def report(self) -> str:
        m = self.metrics
        lines = [
            f"Total return:   {m['total_return']:.2%}",
            f"CAGR:           {m['cagr']:.2%}",
            f"Sharpe ratio:   {m['sharpe_ratio']:.2f}",
            f"Max drawdown:   {m['max_drawdown']:.2%}",
            f"Trades:         {m['num_trades']}",
            f"Win rate:       {m['win_rate']:.2%}",
        ]
        return "\n".join(lines)


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10_000.0,
    commission_bps: float = 5.0,
) -> BacktestResult:
    """commission_bps: round-trip cost charged on every position change,
    in basis points of the traded notional (5 bps = 0.05%)."""
    bars = bars.copy()
    signals = signals.reindex(bars.index).fillna(0)

    position = signals.shift(1).fillna(0)  # act on yesterday's signal at today's close
    daily_asset_returns = bars["close"].pct_change().fillna(0)
    gross_returns = position * daily_asset_returns

    position_changes = position.diff().abs().fillna(position.abs())
    commission_cost = position_changes * (commission_bps / 10_000)
    daily_returns = gross_returns - commission_cost

    equity_curve = initial_capital * (1 + daily_returns).cumprod()
    equity_curve.name = "equity"

    trades = _extract_trades(bars, position)
    trade_returns = trades["return"] if not trades.empty else pd.Series(dtype=float)

    result = BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        trade_returns=trade_returns,
        trades=trades,
    )
    result.metrics = summarize(equity_curve, daily_returns, trade_returns)
    return result


def _extract_trades(bars: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    trades = []
    entry_date = None
    entry_price = None

    for date, pos in position.items():
        price = bars.loc[date, "close"]
        if pos != 0 and entry_date is None:
            entry_date, entry_price = date, price
        elif pos == 0 and entry_date is not None:
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "return": price / entry_price - 1,
                }
            )
            entry_date, entry_price = None, None

    if entry_date is not None:
        last_date = bars.index[-1]
        last_price = bars.loc[last_date, "close"]
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": last_date,
                "entry_price": entry_price,
                "exit_price": last_price,
                "return": last_price / entry_price - 1,
            }
        )

    return pd.DataFrame(trades)
