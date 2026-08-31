from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def cagr(equity_curve: pd.Series) -> float:
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = n_days / 365.25
    if years <= 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def win_rate(trade_returns: pd.Series) -> float:
    if len(trade_returns) == 0:
        return 0.0
    return (trade_returns > 0).mean()


def summarize(equity_curve: pd.Series, daily_returns: pd.Series, trade_returns: pd.Series) -> dict:
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "num_trades": len(trade_returns),
        "win_rate": win_rate(trade_returns),
    }
