"""Central configuration for Fund X-Ray.

Every tunable constant used by more than one module lives here so that
`style_analysis.py`, `kalman_beta.py`, etc. never redefine a magic number
locally. Streamlit sidebar inputs populate an `AppSettings` instance;
everything else is a module-level constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Paths / caching
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Price data goes stale within a day; the Fama-French library updates
# roughly monthly, so its cache can live longer.
PRICE_CACHE_MAX_AGE_DAYS = 1
FACTOR_CACHE_MAX_AGE_DAYS = 7

# --------------------------------------------------------------------------
# Date range
# --------------------------------------------------------------------------
DEFAULT_START_DATE = date(2010, 1, 1)
DEFAULT_END_DATE = date.today()

# --------------------------------------------------------------------------
# Tickers
# --------------------------------------------------------------------------
# An actively managed, stock-picking fund is a more interesting subject
# than an index ETF for style analysis, since its true exposures are
# expected to move over time.
DEFAULT_FUND_TICKER = "FMAGX"  # Fidelity Magellan Fund

# Used only as a phase-2 sanity check: constrained style analysis on an
# S&P 500 index ETF should load almost entirely onto Mkt-RF with a
# weight close to 1 and near-zero on every other factor.
VALIDATION_TICKER = "SPY"

# Default peer group for the crowding score: large-cap growth funds/ETFs
# that plausibly compete for similar factor exposure.
DEFAULT_PEER_GROUP: list[str] = ["FCNTX", "AGTHX", "ANCFX", "VUG"]

# Regime detection deliberately runs on the broad market / vol, never on
# the target fund itself, so the "stress" label isn't contaminated by
# the fund's own idiosyncratic moves.
REGIME_MARKET_PROXY = "SPY"
REGIME_VOL_PROXY = "^VIX"

# --------------------------------------------------------------------------
# Factor model
# --------------------------------------------------------------------------
# Fama-French 5 factors (Fama & French, 2015) plus momentum (Carhart,
# 1997 / Kenneth French's library). The constrained (Sharpe, 1992) and
# unconstrained regressions in style_analysis.py both use this exact set
# so their coefficients are directly comparable.
FF_5_FACTOR_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"
FF_MOMENTUM_DATASET = "F-F_Momentum_Factor_daily"
FACTOR_COLUMNS: list[str] = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
RISK_FREE_COLUMN = "RF"

TRADING_DAYS_PER_YEAR = 252

# --------------------------------------------------------------------------
# Time-varying exposure
# --------------------------------------------------------------------------
ROLLING_WINDOWS: list[int] = [252, 126]  # ~1 trading year, ~6 trading months

# Kalman filter state-space model: factor betas follow a random walk
#   beta_t = beta_{t-1} + w_t,   w_t ~ N(0, Q)
#   r_t    = x_t' beta_t + v_t,  v_t ~ N(0, R)
# Q is set via the standard "discount factor" parameterization used in
# adaptive/Bayesian regression (West & Harrison, "Bayesian Forecasting
# and Dynamic Models"): Q = delta / (1 - delta) * P_{t-1}. A smaller
# delta means slower-moving, smoother betas; delta -> 1 approaches an
# unstable, highly reactive filter.
KALMAN_DELTA = 1e-4
KALMAN_OBSERVATION_COV_PRIOR = 1e-3  # prior on residual (idiosyncratic) variance

# --------------------------------------------------------------------------
# Regime detection
# --------------------------------------------------------------------------
HMM_N_REGIMES_DEFAULT = 2
HMM_N_REGIMES_OPTIONS: list[int] = [2, 3]
HMM_RANDOM_STATE = 42
HMM_N_ITER = 1000

# Known stress windows used ONLY to sanity-check HMM regime labels after
# fitting -- never as training input. Using them as a fitting target
# would be look-ahead bias / circular validation.
KNOWN_STRESS_PERIODS: dict[str, tuple[date, date]] = {
    "COVID crash": (date(2020, 2, 19), date(2020, 4, 7)),
    "2022 rate-hike selloff": (date(2022, 1, 3), date(2022, 10, 14)),
}

# --------------------------------------------------------------------------
# Risk metrics
# --------------------------------------------------------------------------
VAR_CONFIDENCE = 0.95
CVAR_CONFIDENCE = 0.95

# --------------------------------------------------------------------------
# Style drift
# --------------------------------------------------------------------------
# Number of initial months used to build the "baseline" exposure vector
# when the user does not supply a stated-mandate vector.
STYLE_DRIFT_BASELINE_MONTHS = 12
# Flag drift when it exceeds the baseline period's own
# mean + N * std of the drift series, rather than a fixed absolute
# cutoff -- this adapts to each fund's natural exposure variability
# instead of assuming one universal threshold fits every mandate.
STYLE_DRIFT_THRESHOLD_STD = 2.0


@dataclass
class AppSettings:
    """User-adjustable settings; one instance per Streamlit session."""

    fund_ticker: str = DEFAULT_FUND_TICKER
    start_date: date = DEFAULT_START_DATE
    end_date: date = DEFAULT_END_DATE
    peer_group: list[str] = field(default_factory=lambda: list(DEFAULT_PEER_GROUP))
    n_regimes: int = HMM_N_REGIMES_DEFAULT
    rolling_window: int = ROLLING_WINDOWS[0]
