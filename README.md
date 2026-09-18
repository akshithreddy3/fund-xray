# Fund X-Ray — Returns-Based Style Analysis & Regime-Conditional Risk Attribution

**Status: Phase 1 of 10 complete (repo scaffold + data layer).** See Build Order below.

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

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Build order

1. ✅ Scaffold repo structure + config + data loader, verify data pulls correctly
2. Static style analysis (constrained + unconstrained), validated against SPY
3. Rolling OLS and Kalman filter time-varying betas, compared visually
4. HMM regime detection, validated against known stress periods
5. Regime-conditional risk metrics
6. Style drift score
7. (Stretch) Crowding score
8. Streamlit dashboard wiring all modules together
9. README finding, LIMITATIONS.md, unit tests
10. Streamlit Cloud deployment

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) (added once the modeling phases land).
