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
