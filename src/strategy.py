import numpy as np

def make_positions(k, market_returns, risk_free_rate):
    """
    Construct position sizes from a single parameter k.
    Positions are clipped to [0, 2].
    """
    excess = market_returns - risk_free_rate
    positions = (k - risk_free_rate) / excess
    return np.clip(positions, 0, 2)
