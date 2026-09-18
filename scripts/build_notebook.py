"""One-off script that builds notebooks/01_methodology_walkthrough.ipynb.

Not part of the package or the dashboard -- run manually whenever the
notebook needs regenerating (e.g. after a src/ change that would alter
its printed numbers or plots):

    python scripts/build_notebook.py

Building the notebook as code (rather than hand-editing the .ipynb
JSON, or building it once by hand in Jupyter and never touching it
again) means the narrative text and the analysis code it describes
can't silently drift apart.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "01_methodology_walkthrough.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md(
    """
# Fund X-Ray: Methodology Walkthrough

This notebook narrates and validates each modeling decision in this project,
end to end, on one real fund (`FMAGX`, Fidelity Magellan). It's meant to be
read alongside `README.md` and `LIMITATIONS.md` -- this notebook shows *why*
each result looks the way it does; the README states the findings, and
LIMITATIONS.md states what each method can't tell you.

Every number below comes from `src/`, not from analysis duplicated in the
notebook -- if a src/ module changes, rerunning this notebook is how you'd
notice.
"""
)

code(
    """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_loader import DataLoader
from src.drift_score import (
    build_peer_exposure_vectors,
    compute_crowding_score,
    compute_style_drift,
)
from src.kalman_beta import compare_lag_around_date, kalman_filter_betas, rolling_ols_betas
from src.regime_detection import (
    fit_regime_hmm,
    regime_episodes,
    validate_against_known_stress_periods,
)
from src.risk_metrics import compute_regime_risk_metrics, regime_comparison_table
from src.style_analysis import (
    compare_style_results,
    run_constrained_style_analysis,
    run_unconstrained_ols,
)

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

FUND_TICKER = "FMAGX"
START, END = "2015-01-01", "2024-01-01"
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 1. Data layer

`DataLoader` fetches prices (`yfinance`) and Fama-French 5-factor + momentum
data (`pandas_datareader`), caches both as parquet, and aligns them onto a
single trading calendar -- logging, not silently filling, any non-overlapping
dates.
"""
)

code(
    """
loader = DataLoader()
dataset = loader.build_fund_dataset(FUND_TICKER, START, END)

print(f"{FUND_TICKER}: {len(dataset.fund_returns):,} trading days, "
      f"{dataset.start.date()} to {dataset.end.date()}")
dataset.factor_returns.head()
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 2. Static style analysis: constrained vs. unconstrained

Sharpe's (1992) constrained ("returns-based") style regression restricts
factor weights to a simplex (`w_j >= 0`, `sum(w_j) == 1`), solved via
`scipy.optimize` since that constraint isn't expressible as ordinary least
squares. We compare it against an unconstrained Fama-French OLS regression
with Newey-West (1987, 1994) HAC standard errors.

**Validation**: on `SPY`, a market-tracking ETF, the constrained model should
load almost entirely onto `Mkt-RF`.
"""
)

code(
    """
spy_dataset = loader.build_fund_dataset(config.VALIDATION_TICKER, START, END)
spy_constrained = run_constrained_style_analysis(spy_dataset.excess_returns, spy_dataset.factor_returns)

print("SPY constrained weights (should load almost entirely onto Mkt-RF):")
print(spy_constrained.weights.round(4))
print(f"R^2 = {spy_constrained.r_squared:.4f}")
assert spy_constrained.weights["Mkt-RF"] > 0.9, "Validation failed: SPY should load >90% onto Mkt-RF"
print("Validation passed.")
"""
)

md(
    """
Now the fund we actually care about, **FMAGX**. The interesting part isn't
either estimate alone -- it's where they disagree.
"""
)

code(
    """
constrained = run_constrained_style_analysis(dataset.excess_returns, dataset.factor_returns)
unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
comparison = compare_style_results(constrained, unconstrained)

print(f"Constrained R^2:   {constrained.r_squared:.1%}")
print(f"Unconstrained R^2: {unconstrained.r_squared:.1%}")
print(f"Unconstrained annualized alpha: {unconstrained.alpha * config.TRADING_DAYS_PER_YEAR:+.2%}")
comparison
"""
)

code(
    """
fig, ax = plt.subplots()
x = np.arange(len(comparison))
width = 0.35
ax.bar(x - width / 2, comparison["constrained_weight"], width, label="Constrained")
ax.bar(x + width / 2, comparison["unconstrained_beta"], width, label="Unconstrained")
ax.set_xticks(x)
ax.set_xticklabels(comparison.index)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Weight / beta")
ax.set_title(f"{FUND_TICKER}: constrained vs. unconstrained factor exposure")
ax.legend()
plt.show()
"""
)

md(
    """
**Reading this chart**: every factor where the constrained model shows exactly
0 while the unconstrained model shows something clearly negative (`SMB`,
`HML`, `CMA` here) is a case where the simplex constraint is *hiding* a real,
economically meaningful negative tilt -- not just adding noise. See
`LIMITATIONS.md` for why Sharpe's simplex constraint, built for long-only
asset-class indices, is a stylized adaptation when applied to the
Fama-French/Carhart long-short factors used here.
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 3. Time-varying exposure: rolling OLS vs. Kalman filter

A single whole-sample beta hides how exposure moves through time. We compare
two estimators of the same time-varying model:

- **Rolling-window OLS**: refit on the trailing window at every date. Simple,
  but reacts to a real shift only gradually (up to a full window's lag), and
  drops old observations abruptly once they exit the window ("ghosting").
- **Kalman filter**: factor betas follow a random walk; the process-noise
  covariance uses the discount-factor trick from Bayesian dynamic linear
  models (West & Harrison, 1997), so `delta` has a scale-free half-life
  interpretation (`half_life_days = ln(0.5) / ln(1 - delta)`).

**Calibration note**: the first default tried for `delta` (`1e-4`, a
~19-year half-life) made the filter essentially frozen. `config.KALMAN_DELTA
= 0.02` (~34-day half-life) was chosen empirically by checking which value
made the filter visibly lead the rolling window around a real, dateable
event -- the Feb-Apr 2020 COVID crash -- shown below.
"""
)

code(
    """
rolling = rolling_ols_betas(dataset.excess_returns, dataset.factor_returns, window=126)
kalman = kalman_filter_betas(dataset.excess_returns, dataset.factor_returns)  # delta from config

fig, ax = plt.subplots()
ax.plot(rolling.betas.index, rolling.betas["Mom"], label="Rolling (126d)")
ax.plot(kalman.betas.index, kalman.betas["Mom"], label="Kalman filter")
ax.axvspan(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-04-07"), color="red", alpha=0.1, label="COVID crash (known)")
ax.set_ylabel("Momentum factor beta")
ax.set_title(f"{FUND_TICKER}: Momentum beta, rolling vs. Kalman")
ax.legend()
plt.show()
"""
)

code(
    """
lag_table = compare_lag_around_date(rolling, kalman, pd.Timestamp("2020-03-23"), factor="Mom", window_days=45)
print("Beta path around the COVID crash (every 5th trading day):")
lag_table.iloc[::5]
"""
)

md(
    """
By late April 2020, the Kalman estimate has already dropped closer to its
post-crash level than the 126-day rolling estimate, which is still catching
up -- a direct, dateable demonstration of reduced lag, not just a visually
smoother line.
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 4. Regime detection: Gaussian HMM on SPY + VIX

A 2-state Gaussian HMM (`hmmlearn`) fit on **market-wide** signals only (SPY
returns, VIX level) -- deliberately never on the target fund's own returns,
since that would make every downstream regime-conditional number circular.
`hmmlearn` assigns state indices arbitrarily, so states are relabeled after
fitting by ascending volatility-feature mean.

Two standard HMM assumptions worth naming: the **Markov property** (regime
persistence is entirely a function of the fitted transition matrix, no
richer memory), and **Gaussian, state-conditional-i.i.d. emissions** (daily
equity returns are known to have fatter tails than this implies).

**Validation**: check the fitted regimes against known historical stress
windows -- used only as a post-fit sanity check, never as fitting input
(that would be circular).
"""
)

code(
    """
spy_prices = loader.get_prices(config.REGIME_MARKET_PROXY, START, END)
spy_returns = loader.to_log_returns(spy_prices)[config.REGIME_MARKET_PROXY]
vix = loader.get_vix(START, END)

regime_result = fit_regime_hmm(spy_returns, vix=vix, n_regimes=2)
print("State means (original units):")
print(regime_result.state_means)
print()
validate_against_known_stress_periods(regime_result)
"""
)

code(
    """
cumulative_return = np.exp(dataset.fund_returns.cumsum()) * 100
episodes = regime_episodes(regime_result.regime_labels)

fig, ax = plt.subplots()
colors = {"calm": "#0ca30c", "stressed": "#d03b3b"}
for _, ep in episodes.iterrows():
    ax.axvspan(ep["start"], ep["end"], color=colors.get(ep["label"], "gray"), alpha=0.12)
ax.plot(cumulative_return.index, cumulative_return, color="#2a78d6", linewidth=1.2)
ax.set_ylabel("Cumulative return (indexed to 100)")
ax.set_title(f"{FUND_TICKER} cumulative return, shaded by market regime")
plt.show()
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 5. Regime-conditional risk attribution

Within each regime (plus "overall" as a baseline): historical (empirical
-quantile, not parametric) VaR(95%)/CVaR(95%) of the fund's actual returns,
factor betas and R^2 refit on that regime's observations, and a per-factor
Euler variance decomposition (`beta_j * (Sigma_f @ beta)_j`) that sums
exactly to the factor-driven share of total variance.
"""
)

code(
    """
risk_metrics = compute_regime_risk_metrics(
    dataset.fund_returns, dataset.excess_returns, dataset.factor_returns, regime_result.regime_labels
)
regime_comparison_table(risk_metrics)
"""
)

md(
    """
**The finding that motivates this whole project**: `Mkt-RF` beta barely
moves between regimes, but idiosyncratic variance share collapses in the
stressed regime -- the fund's *market* exposure is stable, but its
*diversification* quietly disappears exactly when it matters most. A
whole-sample static style analysis (section 2) would never surface this.
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 6. Style Drift Score

Distance (cosine or Euclidean) between the fund's current exposure vector
and a baseline -- its own first 12 months by default, or a stated-mandate
vector. A drift event is flagged when the score exceeds
`baseline-period mean + 2*std`, calibrated to each fund's own baseline-period
noise.
"""
)

code(
    """
drift = compute_style_drift(rolling.betas, metric="cosine")

fig, ax = plt.subplots()
for _, ev in drift.flagged_events.iterrows():
    ax.axvspan(ev["start"], ev["end"], color="#d03b3b", alpha=0.12)
ax.plot(drift.drift.index, drift.drift, color="#2a78d6")
ax.axhline(drift.threshold, color="#d03b3b", linestyle="--", label="threshold (baseline mean + 2 std)")
ax.set_ylabel("Drift score (cosine)")
ax.set_title(f"{FUND_TICKER}: Style Drift Score")
ax.legend()
plt.show()

drift.flagged_events
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 7. Crowding Score (stretch goal)

For a peer group of similarly-categorized funds/ETFs, fit each one's own
rolling-OLS exposure vector, then average pairwise cosine *similarity*
across the group at each date -- a high score means the group's managers are
converging on the same factor tilts.
"""
)

code(
    """
peer_betas, skipped = build_peer_exposure_vectors(loader, config.DEFAULT_PEER_GROUP, START, END)
if skipped:
    print(f"Skipped peers (couldn't fetch): {skipped}")

crowding = compute_crowding_score(peer_betas, skipped_tickers=skipped)

fig, ax = plt.subplots()
ax.plot(crowding.crowding.index, crowding.crowding, color="#2a78d6")
ax.set_ylim(0, 1)
ax.set_ylabel("Avg. pairwise cosine similarity")
ax.set_title(f"Crowding score: {', '.join(crowding.peer_tickers)}")
plt.show()

print(f"Mean crowding: {crowding.crowding.mean():.3f}  "
      f"(range {crowding.crowding.min():.3f}-{crowding.crowding.max():.3f})")
"""
)

# ---------------------------------------------------------------------------
md(
    """
## Summary

This walkthrough validated every modeling choice against either a known
ground truth (SPY's near-100% `Mkt-RF` loading) or a known historical event
(the COVID crash, the 2022 rate-hike selloff), rather than asserting each
method works. See `README.md` for the full write-up of each phase and
`LIMITATIONS.md` for what none of these methods can tell you -- most
importantly, that everything here is inferred from returns alone and can't
see intra-period trading, hedges, or leverage changes that unwind before
the return series is observed.
"""
)

nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")

print("Executing notebook...")
client = NotebookClient(nb, timeout=600, kernel_name="fundxray", resources={"metadata": {"path": str(REPO_ROOT)}})
client.execute()
nbf.write(nb, NOTEBOOK_PATH)
print("Executed and saved with outputs.")
