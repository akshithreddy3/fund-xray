"""Time-varying factor exposure: rolling-window OLS vs. a Kalman filter.

Both estimators fit the same model as the unconstrained regression in
`style_analysis.py` -- an intercept plus the six Fama-French/Carhart
factors -- but let the coefficients move through time instead of fixing
them over the whole sample. The intercept ("alpha") is estimated jointly
with the betas so it doesn't bias them via omitted-variable effects, but
it is not itself a "style" exposure -- `drift_score.py` and the
dashboard only ever look at the factor columns, never `const`.

1. **Rolling-window OLS** (the naive baseline): refit an ordinary least
   squares regression on the trailing `window` observations at every
   date. Simple and assumption-light, but it has two well-known
   weaknesses that motivate the Kalman alternative below:
     - *Lag*: a genuine, sudden shift in exposure only fully enters the
       window average after `window` days, so the estimate reacts to
       regime changes with a built-in delay of up to `window` days.
     - *Ghosting / step artifacts*: an old, no-longer-representative
       observation still gets full weight until it falls out of the
       window, then disappears abruptly -- producing a visible jump in
       the beta series that has nothing to do with new information.

2. **Kalman filter (linear-Gaussian state-space model)**: betas follow
   a random walk, `beta_t = beta_{t-1} + w_t`, observed through
   `y_t = x_t' beta_t + v_t`. Unlike the rolling window, every past
   observation contributes to today's estimate with *exponentially
   decaying* weight (governed by `delta`) rather than a hard cutoff, so
   there's no ghosting artifact, and a real shift is incorporated
   immediately rather than only once it dominates a fixed window average.
   The tradeoff is a distributional assumption (Gaussian state
   innovations) and one tuning parameter (`delta`) that the rolling
   approach doesn't require.

The state-transition uncertainty is set via the discount-factor trick
from Bayesian dynamic linear models (West & Harrison, 1997,
*Bayesian Forecasting and Dynamic Models*): rather than specifying an
absolute process-noise covariance Q, we inflate the posterior
covariance by `1 / (1 - delta)` at every step
(`Q_t = delta/(1-delta) * P_{t-1}` is algebraically equivalent to
`P_pred = P_{t-1} / (1 - delta)`). This makes the single tuning knob
`delta` scale-free and interpretable: `delta -> 0` gives a nearly static
(very smooth) beta; `delta -> 1` gives a filter that reacts almost
entirely to the most recent observation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

STATE_COLUMNS = ["const", *config.FACTOR_COLUMNS]


@dataclass
class TimeVaryingBetaResult:
    """Common result shape for rolling-OLS and Kalman estimators."""

    method: str
    betas: pd.DataFrame  # index=dates, columns=["const", *factor names]
    r_squared: pd.Series  # local (windowed, for rolling) or one-step-ahead (for Kalman) R^2
    n_burn_in: int  # observations at the start with no estimate (logged, not filled)


def _design_matrix(excess_returns: pd.Series, factor_returns: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    y, X = excess_returns.align(factor_returns, join="inner", axis=0)
    X_with_const = X.copy()
    X_with_const.insert(0, "const", 1.0)
    return X_with_const, X_with_const.to_numpy(), y.to_numpy()


def rolling_ols_betas(
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int,
) -> TimeVaryingBetaResult:
    """Refit OLS on the trailing `window` observations at every date."""
    design, X, y = _design_matrix(excess_returns, factor_returns)
    n, k = X.shape

    if n <= window:
        raise ValueError(f"Need more than {window} observations, got {n}.")

    betas = np.full((n, k), np.nan)
    r_squared = np.full(n, np.nan)

    for t in range(window - 1, n):
        X_win = X[t - window + 1 : t + 1]
        y_win = y[t - window + 1 : t + 1]
        params, *_ = np.linalg.lstsq(X_win, y_win, rcond=None)
        betas[t] = params

        fitted = X_win @ params
        ss_res = float(np.sum((y_win - fitted) ** 2))
        ss_tot = float(np.sum((y_win - y_win.mean()) ** 2))
        r_squared[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    n_burn_in = window - 1
    logger.info(
        "Rolling OLS (window=%d): first %d observations have no estimate "
        "(insufficient history for a full window).",
        window,
        n_burn_in,
    )

    return TimeVaryingBetaResult(
        method=f"rolling_{window}",
        betas=pd.DataFrame(betas, index=design.index, columns=design.columns),
        r_squared=pd.Series(r_squared, index=design.index, name="r_squared"),
        n_burn_in=n_burn_in,
    )


def _static_warmup_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS fit on the warm-up window: initial state mean, covariance, and residual variance.

    `P0 = sigma^2 * (X'X)^-1` is the classical OLS parameter-covariance
    estimate -- a reasonable diffuse-ish prior for the filter's initial
    uncertainty, since it's exactly the sampling uncertainty an analyst
    with only that data would have.
    """
    params, residuals_ss, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ params
    resid = y - fitted
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    P0 = sigma2 * xtx_inv
    return params, P0, sigma2


def kalman_filter_betas(
    excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    delta: float = config.KALMAN_DELTA,
    warmup_window: int = min(config.ROLLING_WINDOWS),
) -> TimeVaryingBetaResult:
    """Random-walk Kalman filter for time-varying (alpha, betas).

    The first `warmup_window` observations are used only to seed the
    filter's initial state mean/covariance and to estimate a fixed
    observation-noise variance R (the fund's idiosyncratic variance,
    assumed constant -- a simplification noted in LIMITATIONS.md); no
    beta estimate is reported for that period, matching the "no silent
    backfill" policy used throughout this project.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    design, X, y = _design_matrix(excess_returns, factor_returns)
    n, k = X.shape
    if n <= warmup_window:
        raise ValueError(f"Need more than {warmup_window} observations, got {n}.")

    beta, P, sigma2 = _static_warmup_fit(X[:warmup_window], y[:warmup_window])
    R = sigma2
    logger.info(
        "Kalman filter warm-up: %d observations used to seed initial state "
        "(estimated observation variance R=%.3e); no estimate reported for "
        "this period.",
        warmup_window,
        R,
    )

    n_filter = n - warmup_window
    betas = np.full((n_filter, k), np.nan)
    innovations = np.full(n_filter, np.nan)

    inflate = 1.0 / (1.0 - delta)

    for i, t in enumerate(range(warmup_window, n)):
        x_t = X[t]
        y_t = y[t]

        # Predict: random-walk transition, discount-factor covariance inflation.
        beta_pred = beta
        P_pred = P * inflate

        # Update.
        innovation = float(y_t - x_t @ beta_pred)
        S = float(x_t @ P_pred @ x_t + R)
        K = (P_pred @ x_t) / S
        beta = beta_pred + K * innovation
        P = P_pred - np.outer(K, x_t) @ P_pred

        betas[i] = beta
        innovations[i] = innovation

    result_index = design.index[warmup_window:]
    r_squared = _rolling_r_squared_from_residuals(
        residuals=pd.Series(innovations, index=result_index),
        actual=pd.Series(y[warmup_window:], index=result_index),
        window=min(config.ROLLING_WINDOWS),
    )

    return TimeVaryingBetaResult(
        method="kalman",
        betas=pd.DataFrame(betas, index=result_index, columns=design.columns),
        r_squared=r_squared,
        n_burn_in=warmup_window,
    )


def _rolling_r_squared_from_residuals(
    residuals: pd.Series, actual: pd.Series, window: int
) -> pd.Series:
    """Trailing local R^2 from a one-step-ahead residual series.

    Mirrors the R^2 computed inside `rolling_ols_betas` (same window, same
    formula) so the two methods' local fit quality is directly comparable
    rather than one being a whole-sample cumulative figure and the other
    a windowed one.
    """
    ss_res = residuals.pow(2).rolling(window).sum()
    ss_tot = actual.rolling(window).apply(lambda w: float(np.sum((w - w.mean()) ** 2)), raw=True)
    return (1.0 - ss_res / ss_tot).rename("r_squared")


def compare_lag_around_date(
    rolling_result: TimeVaryingBetaResult,
    kalman_result: TimeVaryingBetaResult,
    event_date: pd.Timestamp,
    factor: str,
    window_days: int = 40,
) -> pd.DataFrame:
    """Beta path for one factor around a known event, both methods aligned.

    Useful for visually/numerically confirming the Kalman filter reacts
    faster than the rolling window around a real, dateable shift (e.g. a
    known regime transition from Phase 4).
    """
    start = event_date - pd.Timedelta(days=window_days)
    end = event_date + pd.Timedelta(days=window_days)

    rolling_slice = rolling_result.betas.loc[start:end, factor].rename("rolling")
    kalman_slice = kalman_result.betas.loc[start:end, factor].rename("kalman")
    return pd.concat([rolling_slice, kalman_slice], axis=1)
