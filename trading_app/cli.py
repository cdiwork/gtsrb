"""Command-line entry point.

Example:
    python -m trading_app.cli backtest --symbol AAPL --strategy sma_crossover \
        --start 2022-01-01 --end 2024-01-01
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from .backtest import run_backtest
from .data.providers import YFinanceProvider, get_bars_cached
from .strategies import STRATEGIES


def _strategy_kwargs(pairs: list[str]) -> dict:
    kwargs = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
        kwargs[key] = value
    return kwargs


def cmd_backtest(args: argparse.Namespace) -> None:
    strategy_cls = STRATEGIES[args.strategy]
    strategy = strategy_cls(**_strategy_kwargs(args.param or []))

    provider = YFinanceProvider()
    bars = get_bars_cached(provider, args.symbol, args.start, args.end)

    signals = strategy.generate_signals(bars)
    result = run_backtest(bars, signals, initial_capital=args.capital)

    print(f"Symbol:    {args.symbol}")
    print(f"Strategy:  {strategy.name} {_strategy_kwargs(args.param or [])}")
    print(f"Period:    {args.start} to {args.end}")
    print("-" * 40)
    print(result.report())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading_app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bt = subparsers.add_parser("backtest", help="Run a strategy backtest on historical data")
    bt.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    bt.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    bt.add_argument(
        "--start", default=str(date.today() - timedelta(days=365 * 2)), help="YYYY-MM-DD"
    )
    bt.add_argument("--end", default=str(date.today()), help="YYYY-MM-DD")
    bt.add_argument("--capital", type=float, default=10_000.0)
    bt.add_argument(
        "--param",
        action="append",
        help="Strategy constructor override, e.g. --param fast_window=5 (repeatable)",
    )
    bt.set_defaults(func=cmd_backtest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
