# k-optimal-strategy-hull
This repository presents a simple yet powerful optimization-based trading strategy  developed for the Kaggle competition "Hull Tactical Market Prediction".  The core idea is to directly optimize a constant parameter-k to maximize an  adjusted Sharpe ratio under realistic constraints.

## Key Results
- Public Leaderboard Sharpe: ~17.5
- Ranked in the top ~4% (approx. 125th out of 3,700+ participants, public leaderboard)

## Strategy Overview
We consider a simple constant-position strategy defined by a single parameter k.
Positions are determined as:

position_t = clip((k - r_f) / (r_t - r_f), 0, 2)

where r_t is the market return and r_f is the risk-free rate.

## Objective Function
Instead of maximizing a proxy objective, we directly optimize the competition's
Adjusted Sharpe Ratio, including penalties for:
- underperforming the market
- excessive volatility

This ensures alignment between optimization and evaluation.

## Optimization
- Method: BFGS (scipy.optimize)
- Parameter space: 1D (constant k)
- Rationale: simplicity, interpretability, robustness

## Why Not Machine Learning?
- The market is largely efficient
- Overfitting risk dominates
- A single well-optimized parameter outperforms complex models

## Code Structure
- src/: research-oriented and reusable implementation
- kaggle_submission/: competition submission code

