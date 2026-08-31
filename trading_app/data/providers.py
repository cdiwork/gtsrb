"""Historical market data providers.

Default provider is yfinance (no account/API key needed). An optional Alpaca
provider is included for when the user has a (free) paper trading account and
wants the same interface backed by Alpaca's market data API.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DataProvider(ABC):
    """Common interface: fetch daily OHLCV bars for a symbol as a DataFrame
    indexed by date with columns [open, high, low, close, volume]."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start: str | date,
        end: str | date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    """Free historical data via yfinance. No API key required."""

    def get_bars(
        self,
        symbol: str,
        start: str | date,
        end: str | date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=str(start),
            end=str(end),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for {symbol} between {start} and {end}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.lower)
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]]


class AlpacaProvider(DataProvider):
    """Historical + (future) live data via Alpaca's market data API.

    Requires a free paper trading account: https://alpaca.markets/
    Reads credentials from ALPACA_API_KEY / ALPACA_SECRET_KEY env vars.
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY, or pass them explicitly."
            )

    def get_bars(
        self,
        symbol: str,
        start: str | date,
        end: str | date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        timeframe_map = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour, "1m": TimeFrame.Minute}
        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_map.get(interval, TimeFrame.Day),
            start=datetime.fromisoformat(str(start)),
            end=datetime.fromisoformat(str(end)),
        )
        bars = client.get_stock_bars(request).df
        bars = bars.reset_index(level="symbol", drop=True)
        bars.index.name = "date"
        return bars[["open", "high", "low", "close", "volume"]]


def get_bars_cached(
    provider: DataProvider,
    symbol: str,
    start: str | date,
    end: str | date,
    interval: str = "1d",
) -> pd.DataFrame:
    """Wraps a provider with an on-disk CSV cache so repeated backtests over
    the same symbol/range don't re-hit the network every run."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{start}_{end}_{interval}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col="date", parse_dates=True)

    df = provider.get_bars(symbol, start, end, interval)
    df.to_csv(cache_file)
    return df
