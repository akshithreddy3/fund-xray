"""Time-varying factor exposure -- Phase 3 (not yet implemented).

Will implement:
  - Rolling-window OLS (252-day, 126-day) as the naive baseline.
  - A Kalman filter state-space model treating factor betas as latent
    states following a random walk, to reduce the lag inherent to
    rolling windows.
"""
