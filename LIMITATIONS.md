# Limitations

This file is built up incrementally as each phase lands, not written retroactively — each
entry reflects a real caveat that surfaced while implementing that module.

## Returns-based analysis, generally

- All conclusions here are inferred from a fund's *return series*, never its actual holdings.
  Anything that changes intra-period and mean-reverts by the time returns are observed —
  temporary hedges, an option overlay, a leverage change that's unwound before period-end,
  short-term tactical trades — is invisible to every model in this project.
- Every estimate assumes the fund's exposures are reasonably stable *within* the estimation
  window. A fund that changes its factor exposure mid-window will show up as a blend of the two
  regimes, not either one cleanly (this is exactly the motivation for Phase 3's time-varying
  approach, but even that has a reaction lag — see below once implemented).

## Style analysis (`src/style_analysis.py`)

- **The Sharpe (1992) simplex constraint (`beta >= 0`, `sum(beta) == 1`) is a stylized
  adaptation, not a textbook application.** Sharpe's method was designed for a spanning set of
  fully-invested, long-only asset-class indices, where "weights sum to 1" literally describes a
  portfolio allocation. The Fama-French/Carhart factors used here (Mkt-RF, SMB, HML, RMW, CMA,
  Mom) are long-short, zero-net-investment portfolios, so the constrained weights should be read
  as *relative factor tilts* forced into an interpretable, bounded shape — not as an implied
  holdings breakdown.
- **The constraint can hide real, economically meaningful exposures.** Concretely: on FMAGX
  (2015–2024) the unconstrained regression finds a real negative HML loading (growth tilt) and
  negative SMB loading (large-cap tilt); the constrained model, unable to go negative, zeroes
  both out and compensates by over-loading Mkt-RF. Never treat the constrained result alone as
  ground truth — always check it against the unconstrained comparison table.
- Newey-West HAC standard errors correct for serial correlation/heteroskedasticity in the
  *unconstrained* regression's inference, but the constrained model reports no standard errors
  at all (there's no closed-form sampling distribution for a constrained-optimization estimator
  without bootstrapping, which this project doesn't implement) — its weights should be treated
  as point estimates only, not tested for statistical significance.

## Time-varying exposure (`src/kalman_beta.py`)

- **The Kalman filter assumes constant observation (idiosyncratic) variance R for the whole
  sample**, estimated once from the warm-up window. A real fund's residual volatility is not
  actually constant — it's typically higher during stress regimes (see Phase 4/5) — so the
  filter's uncertainty band, and to a lesser extent its gain, is mis-specified precisely during
  the periods most interesting to a risk analysis. A more complete model would let R vary too
  (e.g. via a separate volatility filter or regime-conditional R), which this project doesn't
  implement.
- **`delta` (and therefore the filter's responsiveness) is set by one empirical sweep against
  one event** (FMAGX's momentum beta around the 2020 COVID crash), not estimated by maximum
  likelihood. A formal treatment would estimate `delta` (or Q directly) via MLE over the
  state-space model's likelihood, which is possible with
  `statsmodels.tsa.statespace.MLEModel` but adds meaningful implementation complexity; the
  hand-picked default here is a reasonable, documented starting point, not a fitted parameter.
  Users comparing a different fund should re-check the default against a known event for that
  fund's return history rather than assuming `delta=0.02` transfers unchanged.
- **The initial state (mean and covariance) is seeded from a single static OLS fit on the first
  `warmup_window` observations**, and that warm-up period itself gets no beta estimate (logged,
  not backfilled). A fund with an unusual first few months (e.g. a fund's actual inception
  during an unusually volatile period) will get a less representative warm start than one with
  a typical early period.
- Rolling-window OLS and the Kalman filter are compared on the same beta series, but only
  informally (visual/point-in-time lag comparison in the README, not a formal statistical test
  of which estimator is "better" out-of-sample).

## Regime detection (`src/regime_detection.py`)

- **The HMM assumes Gaussian, state-conditional-i.i.d. emissions.** Daily equity returns have
  fatter tails than a Gaussian, so a single extreme day can be absorbed as "regime noise"
  rather than triggering (or ending) a regime transition, and the model has no way to represent
  that mismatch.
- **The Markov property means regime persistence is entirely a function of the fitted
  transition matrix's self-transition probabilities** — there's no explicit "minimum regime
  duration" or richer memory of how long a regime has already lasted. In practice this can
  produce brief, single-day flickers between labels around a regime boundary, which a human
  reading the same chart might smooth over.
- **Regime *count* is a modeling choice, not something estimated from the data.** The 2- vs.
  3-regime comparison above shows this concretely: the 2022 selloff is called "stressed" 99% of
  the time in a 2-state model but splits into distinct "elevated"/"stressed" phases in a
  3-state model — neither is more "correct"; they're different resolutions of the same
  underlying volatility path. Selecting `n_regimes` by, e.g., BIC would give a formal answer
  but was not implemented here in favor of letting the user choose and compare (Streamlit
  sidebar `n_regimes` control).
- **Regimes are fit once over the full available history and are not truly point-in-time.**
  Because `model.fit()` sees the whole sample at once, a day early in the sample is labeled
  using information about volatility regimes that hadn't happened yet from that day's
  perspective. This is standard practice for *retrospective* regime characterization (which is
  what this project does), but it means the regime labels should not be read as what a
  point-in-time analyst could have known on that date, and the pipeline as built is not
  suitable for real-time regime nowcasting without re-fitting only on data available up to
  each date.
