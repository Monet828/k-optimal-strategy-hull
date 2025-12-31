import numpy as np
from scipy.optimize import minimize
from metrics import adjusted_sharpe
from strategy import make_positions

def optimize_k(market_returns, risk_free_rate, x0=0.0007):
    """
    Optimize constant k via BFGS to maximize adjusted Sharpe ratio.
    """
    def objective(k):
        positions = make_positions(k, market_returns, risk_free_rate)
        return -adjusted_sharpe(market_returns, risk_free_rate, positions)

    res = minimize(objective, x0=[x0], method="BFGS")
    return float(res.x[0]), -res.fun
