# trading_app

A small day-trading analysis & backtesting toolkit: pull historical price
data, run a strategy over it, and get back performance metrics. This is the
first slice of a bigger day trading app — analysis only, no real money moves.

## Setup

```bash
pip install -r trading_app/requirements.txt
```

## Run a backtest

```bash
python -m trading_app.cli backtest --symbol AAPL --strategy sma_crossover \
    --start 2022-01-01 --end 2024-01-01
```

Available strategies: `sma_crossover`, `rsi_reversion`, `breakout` (see
`trading_app/strategies/`). Override a strategy's parameters with repeatable
`--param key=value` flags, e.g.:

```bash
python -m trading_app.cli backtest --symbol MSFT --strategy rsi_reversion \
    --param oversold=25 --param exit_level=55
```

## Data

> **Note:** this project was scaffolded inside a sandboxed Claude Code Remote
> session whose network egress policy blocks general internet hosts
> (including Yahoo Finance and Alpaca's API, confirmed via direct testing).
> So the data-fetching path was verified with synthetic price data + unit
> tests rather than a live download. Run `pip install -r requirements.txt`
> and the CLI command below on your own machine (or any environment with
> normal internet access) to pull real data — nothing else needs to change.

Historical bars come from `yfinance` by default (free, no account needed)
and are cached as CSVs under `trading_app/data/cache/` so repeated backtests
don't re-hit the network. An `AlpacaProvider` is also included in
`trading_app/data/providers.py` for when you set up a free Alpaca paper
trading account and want the same interface backed by their API (needed
later for actually placing paper trades, not just backtesting).

## Adding a new strategy

Subclass `trading_app.strategies.base.Strategy` and implement
`generate_signals(bars) -> pd.Series` returning 1 (long) / 0 (flat) per bar.
Register it in `trading_app/strategies/__init__.py`'s `STRATEGIES` dict so
the CLI picks it up.

## Tests

```bash
pytest trading_app/tests
```

## Roadmap (not built yet)

- Paper trading execution via Alpaca (place real simulated orders from a
  strategy's live signal instead of just backtesting historical data).
- A results dashboard/report beyond the terminal summary.
- Multi-symbol / portfolio-level backtests.
- Video transcripts and research notes that motivated this project live in
  `../docs/transcripts/` and `../docs/research-notes.md`.
