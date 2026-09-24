# Fund X-Ray

**See what's really inside a fund — and how its risk changes when markets get rough.**

**Live app:** [fund-xray-aduyuzapdmiwtghrcdxrus.streamlit.app](https://fund-xray-aduyuzapdmiwtghrcdxrus.streamlit.app)

![Fund X-Ray dashboard](docs/dashboard_screenshot.png)

## The problem

When you buy a mutual fund or ETF, you get a name, a category label, a performance chart, and
— if you're lucky — a holdings report that's already a few months out of date. None of that
tells you what's actually driving the fund's returns, whether it's really investing the way it
claims to, or what happens to your risk once markets turn stressed. Two funds with the same
"growth" label can behave completely differently in a downturn, and there's normally no way to
see that coming until it's already happened to you.

## What Fund X-Ray does

Point it at a ticker and it reconstructs the fund's real risk profile from nothing but its
public daily returns — no holdings disclosure required. It answers questions a fund fact sheet
never will:

- **What am I actually exposed to?** The fund's real factor tilts — market, size,
  value/growth, profitability, investment style, and momentum — not just its category label.
- **Is it doing what it says?** Whether its investing style has quietly drifted since inception,
  and by how much.
- **What happens in a downturn?** How its volatility, worst-case losses, and diversification
  change between calm markets and stressed ones — and what specifically drives its risk in
  each.
- **Am I actually diversified?** Whether a group of funds — or your whole portfolio — gives you
  independent bets, or quietly the same one repeated under different names.

**A real example.** On Fidelity Magellan (FMAGX), the fund's exposure to the overall market
barely changes in a downturn (beta ≈ 1.04, calm or stressed) — but the part of its risk that
*isn't* tied to the market almost disappears, falling from about 10% of total risk in calm
periods to under 2% in stressed ones. Its diversification benefit is real most of the time, and
quietly vanishes exactly when you'd want it most. That's the kind of thing no performance chart
or holdings PDF will ever show you — and Fund X-Ray double-checks it: the same pattern still
shows up even when the calm-market exposure is applied, unrefit, to the stressed-market data, so
it isn't just an artifact of letting the model refit itself twice.

## How to use it

The app has one main page and two comparison pages, all reachable from the sidebar.

**Single fund (the main page).** Enter any ticker — try `FMAGX`, `VUG`, or one of your own
holdings — and land on **Overview**: one screen, plain English, covering what drives the fund,
how it holds up in a downturn, and whether its style is drifting. From there:

| Tab | What it tells you |
|---|---|
| **Overview** | The one-screen summary — what drives the fund, calm-vs-stress comparison, style stability, all in plain language |
| **What Drives This Fund?** | The full factor breakdown: which exposures dominate its risk, and how confident each estimate is |
| **Stress Behavior** | Volatility, worst-case loss (VaR/CVaR), and diversification, calm vs. stressed, with the fund's price chart shaded by market regime |
| **Style Drift** | Has the fund's investing style moved away from how it used to invest, and is that shift temporary or permanent? |
| **Diversification** | How closely the fund tracks a peer group — are they really different funds, or the same bet? |
| **Detailed Quant Analysis** | Every underlying number for a technical reader: regression diagnostics, time-varying beta paths, regime-model validation and stability checks |

**Own more than one fund?**
- **Portfolio X-Ray** — enter your holdings and their weights and see whether they actually
  diversify you or quietly concentrate you in one factor bet, with a blended exposure chart and
  a pairwise-similarity heatmap.
- **Fund vs. Fund** — put 2–3 tickers side by side on exposure, stress behavior, and drift.

## What it is — and isn't

Fund X-Ray is a **risk-transparency tool**. It does not predict returns, it is not a trading
system, and it will not tell you which fund is "best" or recommend buying, holding, or selling
anything. It exists to make a fund's real risk visible, so you — or your advisor — can make
that call with better information. Nothing here is investment advice.

## Try it yourself

```bash
git clone https://github.com/akshithreddy3/fund-xray.git
cd fund-xray
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

No signup, no API keys, no database — every data source (Yahoo Finance, the Kenneth French
data library) is free and public, and the app runs entirely on your machine.

## How it works, briefly

Under the hood, Fund X-Ray combines a few well-established techniques into one pipeline:

- **Returns-based style analysis** (Sharpe, 1992) to infer factor exposure without holdings
  data — checked two ways (a bounded, Sharpe-style fit and an unconstrained regression with
  honest statistical uncertainty) so the two can be compared against each other.
- **Time-varying exposure tracking**, via both a simple rolling window and a hand-rolled Kalman
  filter, to see how a fund's style moves over time rather than assuming it's fixed.
- **Regime detection** — a Hidden Markov Model that labels calm-vs-stressed market periods
  using only broad market data (never the fund's own returns, to avoid circular reasoning),
  validated against real historical crashes (the 2020 COVID crash, the 2022 rate-hike selloff)
  and checked for stability across random restarts and against a simple, transparent
  percentile-based rule.
- **Regime-conditional risk metrics** — volatility, VaR, CVaR, and exactly which factors drive
  each regime's risk — recomputed separately for calm and stressed markets instead of blended
  into one misleading average.
- **Style drift and diversification scoring**, and a plain-English interpretation layer that
  turns all of the above into a stated takeaway before showing the underlying numbers.

Every result comes with an honest accounting of what it assumes and where it can be wrong —
that accounting isn't an afterthought, it's a first-class part of the project:

- **Full methodology and a worked example:**
  [`notebooks/01_methodology_walkthrough.ipynb`](notebooks/01_methodology_walkthrough.ipynb)
- **What every model assumes, and where its numbers should be read as directional rather than
  exact:** [LIMITATIONS.md](LIMITATIONS.md)
- **Tech stack, architecture, tests, and deployment, for developers:**
  [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## Contributing and license

No `LICENSE` file is included yet, and there's no formal contributing process — open an issue
or pull request against `main` on
[github.com/akshithreddy3/fund-xray](https://github.com/akshithreddy3/fund-xray) if you'd like
to suggest a change.
