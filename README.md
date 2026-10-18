**English** | [Українська](README.uk.md) | [Polski](README.pl.md)

# EIAL-FX · Explainable Information & Algorithmic FX Research

[![CI](https://github.com/pavmarshyk0/EIAL_FX_V2/actions/workflows/ci.yml/badge.svg)](https://github.com/pavmarshyk0/EIAL_FX_V2/actions/workflows/ci.yml)

**Reproducible, falsification-first FX research** for EUR/USD, EUR/PLN and cross-derived USD/PLN. Combines observed macro/rate data, market microstructure, optional positioning and time-stamped information factors with a chronological validation pipeline and a separate bid/ask paper backtester.

> **CURRENT STATUS: `NO_EDGE_ESTABLISHED`.** This is a research prototype, not an approved trading strategy. No real-market profit, predictive advantage or live execution is claimed. Demonstration datasets are **synthetic**.

```text
PRICE_FORECAST_VALIDATION: IMPLEMENTED (no observed market results included)
QUOTE_BACKTEST: IMPLEMENTED (requires venue bid/ask dataset)
HISTORICAL_PIT_FEATURES: PARTIAL / USER-SUPPLIED
CBDC_AND_AI_LATENCY: NOT IDENTIFIABLE FROM PUBLIC DAILY DATA
LIVE_TRADING: DISABLED
```

## Quick start · Python 3.11+

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q

# Offline smoke tests: SYNTHETIC data ONLY
python eialfx.py demo --out results/demo
python quote_backtest.py demo --out results/quote_demo
python research_gate.py --forecast results/demo/report.json --quotes results/quote_demo/report.json

# Real daily ECB fixings (internet needed); forecast evaluation, NOT trading P&L
python eialfx.py fetch-ecb --start 2016-01-01 --out data/ecb_fx.csv
python eialfx.py train --rates data/ecb_fx.csv --out results/ecb_forecasts

# Optional FRED policy series (FRED_API_KEY environment variable required)
python eialfx.py fetch-fred-policy --start 2016-01-01 --out data/policy_events.csv
python eialfx.py train --rates data/ecb_fx.csv --events data/policy_events.csv --out results/policy_forecasts
```

### Real quote-level paper backtest (requires **externally obtained** venue quotes)

```bash
python dukascopy_to_quotes.py --input raw_eurusd.csv --pair eurusd --out data/eurusd_quotes.csv
# Combine quotes from desired pairs into one timestamp,pair,bid,ask CSV, if needed.
python quote_backtest.py run --predictions results/ecb_forecasts/oos_predictions.csv --rates data/ecb_fx.csv --quotes data/eurusd_quotes.csv --out results/quote_real
python research_gate.py --forecast results/ecb_forecasts/report.json --quotes results/quote_real/report.json
```

`quote_backtest.py` enters only at the first recorded venue quote at/after the signal time plus configurable execution delay, and exits at/after the following fixing day's 16:30 decision timestamp. It charges bid/ask, configurable slippage and fees, but **not financing, carry, rejects, market impact or collateral**. It reports diagnostic bps per trade — **not portfolio returns**. The target is the next ECB fixing, while the trade is at executable quotes; evaluation targets and trade horizons are therefore not identical.

## Repository structure

```text
eialfx.py                 daily-factor model / ECB-FRED readers / expanding walk-forward
quote_backtest.py          quote-aware single-position-per-pair paper simulation
dukascopy_to_quotes.py     tick CSV normalization
research_gate.py           conservative diagnostics / explicit no-go conditions
make_catalog.py            variable-source catalog generator
VARIABLES_AND_SOURCES.csv  42 documented feature candidates + access caveats
requirements.txt           runtime and test dependencies
tests/                     unit and time-leakage smoke tests
.github/workflows/ci.yml  GitHub Actions (Python 3.11 / 3.12)
docs/                      hypotheses, data and backtest methodology
README.uk.md / README.pl.md  localized docs
LICENSE                    short proprietary source-available terms
```

## Model and falsification logic

1. The dependent variable is explicitly a **next ECB fixing log-return**, not a directly observed worldwide FX "demand" series. Signed executed order flow is optional and scored as an **independent task** only with actual labels.
2. Features must exist before **16:30 Europe/Brussels** on prediction date, with controlled freshness. Revised macroeconomic series still require true publication vintages; a guessed lag is not enough.
3. Ridge regression, fixed regularization, in-fold imputation/scaling, expanding chronological splits and an embargo produce inspectable out-of-sample predictions.
4. Measure both **zero-return and price-history baselines**, remove each factor family (ablation), and reject claims when improvement disappears out of sample.
5. Model EUR/USD and EUR/PLN; derive USD/PLN algebraically to avoid inconsistent triangle forecasts.
6. Only proceed to cost-aware paper testing after a positive forecast baseline comparison. A synthetic demonstration does **not** imply an edge.

Read [research protocol](docs/RESEARCH_PROTOCOL.md), [data sourcing and strict point-in-time rules](docs/HISTORICAL_DATA_AND_BACKTEST.md), and [source catalog](VARIABLES_AND_SOURCES.csv).

## Core data caveats

- ECB reference prices are **not tradable fills**; the ECB data pipeline validates prediction quality only.
- A broker's bid/ask and displayed sizes are **not** consolidated global OTC FX order flow.
- CFTC positioning is delayed; releases need accurate publication timestamps.
- GDELT/news text must be processed into reproducible currency-specific sentiment with publication availability.
- Wholesale CBDC-AMM experiments are not evidence of substantial historical spot-FX flows.
- HFT-versus-LLM internal latency cannot be inferred from the difference between public price ticks.
- No market tick or broker-derived confidential data, credentials, large research output or backtest screenshots are tracked.

## Independent project

This model is a separate research hypothesis from [Varrior Labs FX Analytics](https://github.com/pavmarshyk0/varrior-labs-fx-analytics). Compatible CSV conventions and rigorous validation ideas are intentional; **no historical results, private ticks or claims of edge are inherited from Varrior Labs**.

## License and scope

© 2026 Paweł Maruszczyk. [Proprietary source-available license](LICENSE): viewing/evaluation allowed; reuse or distribution requires written permission. Research only; no investment advice or live-execution authorization.
