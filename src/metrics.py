import numpy as np
import pandas as pd

def adjusted_sharpe(market_returns, risk_free_rate, positions):
    """
    Competition-aligned adjusted Sharpe ratio.

    Includes penalties for:
    - underperforming the market
    - excessive volatility
    """
    market_returns = pd.Series(market_returns)
    risk_free_rate = pd.Series(risk_free_rate)
    positions = pd.Series(positions)

    market_excess = market_returns - risk_free_rate
    strategy_excess = market_excess * positions
    strategy_returns = strategy_excess + risk_free_rate

    market_exm = (1 + market_excess).prod() ** (1 / len(positions)) - 1
    strategy_exm = (1 + strategy_excess).prod() ** (1 / len(positions)) - 1

    market_std = market_returns.std()
    strategy_std = strategy_returns.std()

    if strategy_std == 0:
        sharpe = 0.0
    else:
        sharpe = strategy_exm / strategy_std * np.sqrt(252)

    return_penalty = 1 + max(0, (market_exm - strategy_exm) * 252) ** 2 * 100
    vol_penalty = 1 + max(0, strategy_std / market_std - 1.2)

    return sharpe / (return_penalty * vol_penalty)
