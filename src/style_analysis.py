"""Static style analysis -- Phase 2 (not yet implemented).

Will implement:
  - Constrained (Sharpe, 1992) returns-based style analysis via
    scipy.optimize: factor weights >= 0 and sum to 1.
  - Unconstrained Fama-French OLS via statsmodels with Newey-West (HAC)
    standard errors, for comparison against the constrained result.
"""
