# EIAL-FX experimental viability assessment — 2026-10-19

**Judgment: worthwhile to run a low-cost, falsification-first research backtest; not justified to deploy or purchase expensive niche data yet.** These are methodological judgments, **not predictive probabilities**.

## What we can and cannot infer before a real backtest

- A mental/qualitative backtest cannot establish win rate, excess returns, p-values, causal identification or positive net expectancy.
- A well-defined experiment is worthwhile because price/rate history is inexpensive and permits benchmarking; one can reject unsupported claims rapidly.
- CBDC-AMM real-world historic spot FX flows, global executed dealer orders and private algorithm-class reaction latency are not independently observable using public ECB daily time series. Exclude them rather than replacing them with fabricated values.
- The dominant anticipated failure modes are reverse causation, timestamp errors, short-lived signal decay, higher execution costs than signal size, and overfitting across factor families/horizons.

## Reproducible synthetic smoke test

With code bundled here (`python eialfx.py demo` and `python quote_backtest.py demo`):

| Metric | EUR/USD | EUR/PLN |
|:--|--:|--:|
| Out-of-sample synthetic prediction rows | 878 | 878 |
| Full-model OOS R² vs zero return | -2.91% | +0.09% |
| Full-model direction match | 51.25% | 53.99% |
| Mean synthetic quote-backtest bps per trade | -1.11 | +3.09 |

This table **must not be used as real-market evidence**. Data-generation choices introduce artificial relationships. The trade simulator's synthetic quotes are explicitly fiction and omit financing, dealer rejections and inventory risks. On synthetic data, the fully observed model performs worse than the price-only baseline for **both** targets in RMSE, so even the diagnostic gate does not promote it.

## Efficient real-data decision sequence

1. **Price baseline:** reproduce 2016–2026 ECB daily returns and out-of-sample price-only Ridge; track OOS RMSE against zero-return, including an untouched holdout.
2. **Point-in-time rate events:** compare only matched OOS dates against price baseline. Ensure historical publication times and macro revisions are documented; current FRED API observation values are not automatically vintage-safe.
3. **Institutional timing:** add CFTC report data using actual Friday release availability; check delayed-vs-prior-week contribution.
4. **News:** add time-stamped GDELT news processed using a frozen NLP pipeline. Apply latency and timestamp error audits.
5. **Quote backtest:** only if incremental forecasts pass basic OOS gates. Import legally obtained synchronized bid/ask after signal and test realistic spread, slip, fees, and 2x/3x sensitivity.
6. **Expensive features:** attempt private FX flow or venue order book after a baseline result is already positive and stable. CBDC/HFT latency remain separate future research hypotheses.

## Confirmation requirements

Passing `research_gate.py` is **not** a proof of alpha. Before any live-deployment proposal require locked-holdout replication; uncertainty intervals robust to autocorrelation; no multiple-testing leakage; regime stability; net P&L under legitimate venue cost assumptions; portfolio/risk constraints; and documented independence from Varrior Labs results.

**Conclusion:** `GO_FOR_CHEAP_DATA_VALIDATION`, `NO_GO_FOR_LIVE_TRADING`, `NO_EDGE_ESTABLISHED`.