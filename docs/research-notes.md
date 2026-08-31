# Research notes: day trading app

Context gathered while scaffolding `trading_app/`, since the two source
YouTube videos' transcripts weren't fetchable from this environment (see
`transcripts/`).

## Reference project: tradermonty/claude-trading-skills

https://github.com/tradermonty/claude-trading-skills

An open-source toolkit of Claude Code skills for equity investors combining
long-term holds with swing trading. Closely matches what video 2
("Claude Tested Over 9,000 Trading Strategies") appears to describe.

- **Structure**: `skills/` (individual skills), `workflows/` (YAML manifests
  chaining skills into routines like daily market review or trade
  postmortems), `skillsets/` (bundles), `docs/`.
- **Skill categories** (60+ skills): market regime detection, core portfolio
  management (via Alpaca), swing-opportunity screeners (VCP, CANSLIM,
  Stockbee momentum), trade planning (position sizing, pre-trade checklists,
  circuit breakers), trade journaling/postmortems, and **strategy
  research** (backtesting frameworks, edge-detection pipelines).
- **Data sources**: free tier of Financial Modeling Prep (FMP, 250
  req/day), public CSVs on GitHub, optional FINVIZ Elite, Alpaca for
  paper/live trading.
- **Backtesting**: a Rust engine (`manifoldbt`) plus MetaTrader 5 local
  testing and custom scenario analyzers.
- **Philosophy**: structure the human's decision-making (market review, risk
  management, journaling) rather than fully automate trades; circuit
  breakers block trades that violate preset risk rules.

## What this means for `trading_app/`

The first slice built here (`trading_app/`) mirrors the "strategy research /
backtesting" piece of that toolkit in plain Python: pull historical bars,
run a strategy's signal generator over them, and score the result (return,
Sharpe, drawdown, win rate) — without needing Rust or a paid data plan.
Natural next steps, following the reference project's shape:
- Wire up Alpaca for paper trading execution (provider stub already in
  `trading_app/data/providers.py`).
- Add a market-regime/screener layer before the per-symbol backtest.
- Add a trade journal that logs backtest and (later) live paper trades.
