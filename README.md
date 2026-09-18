# Fund X-Ray — Returns-Based Style Analysis & Regime-Conditional Risk Attribution

**Status: Phase 3 of 10 complete (time-varying exposure: rolling OLS vs. Kalman filter).** See
Build Order below.

## Problem statement

Given only a fund's public return series (no holdings disclosure), estimate its true factor
exposures, track how those exposures drift over time, and answer a harder question than a
single backtest can: what does this fund's risk actually look like *during a stress regime*,
as opposed to on average? This project builds that returns-based analysis pipeline end to end
— constrained style analysis, Kalman-filtered time-varying betas, HMM regime detection, and a
style-drift score — as a Streamlit dashboard.

## Methodology (summary; full detail added as each phase lands)

- **Static style analysis**: Sharpe's (1992) constrained returns-based style analysis (factor
  weights bounded [0,1], summing to 1, solved via `scipy.optimize`), compared against an
  unconstrained Fama-French OLS regression with Newey-West (HAC) standard errors.
- **Time-varying exposure**: rolling-window OLS (naive baseline) vs. a Kalman filter treating
  factor betas as latent random-walk states — the Kalman approach should show materially less
  lag around known regime transitions.
- **Regime detection**: a Gaussian Hidden Markov Model (`hmmlearn`) fit on SPY/VIX, independent
  of the target fund, validated against known stress windows (2020 COVID crash, 2022 rate-hike
  selloff).
- **Regime-conditional risk**: VaR/CVaR, factor R², and idiosyncratic-vs-factor variance share,
  recomputed within each detected regime.
- **Style Drift Score**: a distance metric between a fund's current exposure vector and its own
  historical baseline (or a stated mandate), flagged when it exceeds baseline mean + 2·std.

## Repo structure

```
src/
  config.py            # all tunable parameters (dates, tickers, factor set, thresholds)
  data_loader.py        # DataLoader: fetches/caches/aligns prices + Fama-French factors
  style_analysis.py     # constrained + unconstrained regression (Phase 2)
  kalman_beta.py         # rolling OLS + Kalman filter time-varying beta (Phase 3)
  regime_detection.py    # HMM regime labeling (Phase 4)
  risk_metrics.py         # regime-conditional VaR/CVaR/R^2 (Phase 5)
  drift_score.py          # style drift + crowding score (Phase 6/7)
tests/                    # unit tests per module
notebooks/                # 01_methodology_walkthrough.ipynb (narrated validation)
app.py                    # Streamlit dashboard (Phase 8)
```

## Data sources (all free)

- Fund/ETF/stock returns: `yfinance` (daily adjusted close → log returns)
- Factors: Fama-French 5 factors + momentum, daily, via `pandas_datareader.famafrench`
- Regime signal: `^VIX` and/or realized SPY volatility via `yfinance`
- Peer group: configurable ticker list in `config.py` for the crowding score

## Data layer (Phase 1 — done)

`DataLoader` (`src/data_loader.py`) fetches prices and Fama-French factors, caches raw pulls
as parquet under `data/cache/` (skipped if the cache is fresh — 1 day for prices, 7 days for
factors), and returns a `FundDataset` with fund returns, excess returns, and factor returns
all aligned to a single inner-joined trading calendar.

**Missing-data policy**: dates that don't overlap across sources (e.g. a fund's inception date
newer than the factor history) are dropped, and every drop is logged with a row count and the
affected date range — there is no silent forward-fill.

```python
from src.data_loader import DataLoader

loader = DataLoader()
dataset = loader.build_fund_dataset("FMAGX", start="2015-01-01", end="2024-01-01")
dataset.fund_returns      # daily log returns
dataset.excess_returns    # fund_returns - risk_free
dataset.factor_returns    # Mkt-RF, SMB, HML, RMW, CMA, Mom
```

## Static style analysis (Phase 2 — done)

`src/style_analysis.py` implements the same linear factor model two ways, so the disagreement
between them is itself informative:

- **Constrained ("Sharpe") style regression** — Sharpe (1992): factor loadings are solved via
  `scipy.optimize` (SLSQP) subject to `beta_j >= 0` and `sum(beta_j) == 1`, no intercept. Note
  that Sharpe's simplex constraint was designed for fully-invested, long-only asset-class
  indices; applied here to the Fama-French/Carhart long-short factors it has no literal
  portfolio-allocation meaning, so read the constrained weights as *relative factor tilts*, not
  holdings. See [LIMITATIONS.md](LIMITATIONS.md).
- **Unconstrained Fama-French OLS** — estimated with an intercept (alpha) and Newey-West (1987,
  1994) HAC standard errors (automatic lag selection: `floor(4*(T/100)^(2/9))`), which corrects
  for the serial correlation typical of daily return regressions.

**Validation**: running constrained style analysis on `SPY` (2015–2024) loads 95.5% onto
`Mkt-RF` with R² = 0.99 — exactly the sanity check a market-tracking ETF should pass.

**Concrete finding**: on `FMAGX` (Fidelity Magellan, 2015–2024), the unconstrained regression
finds a clearly negative HML loading (-0.16, a growth tilt) and negative SMB (-0.14, a
large-cap tilt) — both real and economically sensible. The constrained model, unable to
represent negative exposures, forces both to exactly 0 and instead over-loads onto Mkt-RF
(0.97 vs. 1.04 unconstrained). This is precisely why both estimators are reported side by side:
the simplex constraint doesn't just add noise, it can hide a fund's actual style tilts.

```python
from src.style_analysis import run_constrained_style_analysis, run_unconstrained_ols, compare_style_results

constrained = run_constrained_style_analysis(dataset.excess_returns, dataset.factor_returns)
unconstrained = run_unconstrained_ols(dataset.excess_returns, dataset.factor_returns)
compare_style_results(constrained, unconstrained)  # side-by-side table, flags boundary weights
```

## Time-varying exposure (Phase 3 — done)

`src/kalman_beta.py` implements both a naive baseline and an improved estimator for the same
question — how does a fund's factor exposure move through time?

- **Rolling-window OLS** (252-day and 126-day): refit an ordinary least squares regression on
  the trailing window at every date. Simple, but has two known failure modes: it takes up to a
  full window to fully absorb a real shift (lag), and an old observation gets full weight right
  up until it drops out of the window, then vanishes abruptly (a "ghosting" step artifact).
- **Kalman filter**: factor betas follow a random walk (`beta_t = beta_{t-1} + w_t`), so every
  past observation contributes with exponentially decaying weight instead of a hard cutoff. The
  process-noise covariance uses the discount-factor trick from Bayesian dynamic linear models
  (West & Harrison, 1997): the posterior covariance is inflated by `1/(1-delta)` each step,
  which turns `delta` into a single, scale-free tuning knob expressible as a half-life
  (`half_life_days = ln(0.5)/ln(1-delta)`).

**Calibrating delta**: the first default I tried (`1e-4`, implying a ~19-year half-life) made
the filter essentially frozen — worse than useless for tracking anything. Sweeping delta and
checking the reaction speed around a real, dateable event (FMAGX's momentum-factor loading
around the Feb–Apr 2020 COVID crash) showed `delta=0.02` (~34-day half-life) gives a filter that
visibly leads the 126-day rolling estimate down during the crash, without amplifying day-to-day
noise. That sweep — not a formula — is what sets the default in `config.py`, and it's recorded
there so the choice isn't a mystery constant.

**Concrete finding**: for FMAGX's Momentum factor loading around March–April 2020, the Kalman
estimate reaches roughly the post-crash level (~0.11–0.13) by late April, while the 126-day
rolling estimate is still elevated (~0.15–0.17) at the same date — a real, measurable lag
reduction, not just a smoother-looking line.

```python
from src.kalman_beta import rolling_ols_betas, kalman_filter_betas, compare_lag_around_date

rolling = rolling_ols_betas(dataset.excess_returns, dataset.factor_returns, window=126)
kalman = kalman_filter_betas(dataset.excess_returns, dataset.factor_returns)  # delta from config
compare_lag_around_date(rolling, kalman, event_date=pd.Timestamp("2020-03-23"), factor="Mom")
```

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Build order

1. ✅ Scaffold repo structure + config + data loader, verify data pulls correctly
2. ✅ Static style analysis (constrained + unconstrained), validated against SPY
3. ✅ Rolling OLS and Kalman filter time-varying betas, compared visually
4. HMM regime detection, validated against known stress periods
5. Regime-conditional risk metrics
6. Style drift score
7. (Stretch) Crowding score
8. Streamlit dashboard wiring all modules together
9. README finding, LIMITATIONS.md, unit tests
10. Streamlit Cloud deployment

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) (added once the modeling phases land).
