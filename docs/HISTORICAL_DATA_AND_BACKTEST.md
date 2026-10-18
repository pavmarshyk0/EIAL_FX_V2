# EIAL-FX v2.1 — historical backtesting protocol

Status: **research software with unit tests; no validated trading edge**. This module adds a PAPER quote-level simulator to the existing ECB-fixing OOS forecast engine. The environment can run synthetic demonstrations but cannot obtain the complete original public-market datasets here. Never present synthetic results as real evidence.

## Two tests that MUST NOT be conflated

1. **Predictive accuracy**: `eialfx.py train`: fit only on past samples, compare actual **next ECB reference fixing** against historical out-of-sample forecasts. Report zero-return baseline, RMSE, OOS R², direction accuracy and factor ablation. Does not imply real executable P&L.
2. **Trading quote simulation**: `quote_backtest.py run`: take the OOS forecasts at t=**16:30 Europe/Brussels**, enter at first observed quote at/after t + execution delay and exit at a quote at/after **16:30 on the next published ECB-fixing date**. It uses actual bid/ask rather than ECB fixings. The prediction target at ~16:00 next day and exit at 16:30 next day differ; inspect this intentional mismatch, since this is a rough executable check, not aligned target P&L.

## 0. Dependencies

```bash
pip install -r requirements.txt
pytest -q
```

## 1. End-to-end SMOKE TEST (ALL FICTIONAL)

```bash
python quote_backtest.py demo --out results/quote_demo
```

`results/quote_demo/report.json` and `trades.csv` are examples only. Fictional quotes are synthesized from the same synthetic daily rates. Any P&L is **not market evidence**.

## 2. Download REAL ECB reference-rate history (forecast labels and macro-scale baseline)

```bash
python eialfx.py fetch-ecb --start 2016-01-01 --out data/ecb_fx.csv
python eialfx.py train --rates data/ecb_fx.csv --out results/real_forecasts
```

**Never open or close a trade at ECB reference rates.** For strict point-in-time features, pass genuine `--events` with event timestamps corresponding to **first availability**, not observation/reference dates; inspect the existing README for the input schema.

## 3. Download REAL venue quotes for executable quote simulation

Sample `dukascopy-node` CLI from vendor documentation (requires installed Node/npm and internet):

```bash
npx dukascopy-node -i eurusd -from 2024-01-02 -to 2024-01-05 -t tick -f csv
```

The CLI's output file name depends on its arguments and version. Take the actual resulting CSV path and normalize:

```bash
python dukascopy_to_quotes.py --input REAL_DUKASCOPY_CSV.csv --pair eurusd --out data/eurusd_quotes.csv
```

For multiple pairs, normalize each and concatenate rows, keeping one `timestamp,pair,bid,ask` header:

```python
import pandas as pd
files = ['data/eurusd_quotes.csv', 'data/eurpln_quotes.csv']
pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv('data/all_quotes.csv', index=False)
```

Only use instruments actually supported by the data vendor. Verify `eurpln` coverage independently; if unavailable, test `eurusd` only. For meaningful quote-P&L results, historical quote timestamps **must overlap the OOS forecast period**; do not feed a few days of quotes into a 10-year prediction history and expect meaningful performance.

```bash
python quote_backtest.py run \
  --predictions results/real_forecasts/oos_predictions.csv \
  --rates data/ecb_fx.csv \
  --quotes data/all_quotes.csv \
  --out results/real_quote_backtest \
  --threshold-bps 2 \
  --execution-delay-ms 100 \
  --max-quote-wait-ms 120000 \
  --slippage-bps-per-side 0.25 \
  --fee-bps-per-side 0
```

Use nonzero realistic venue-specific fees and slippage. Broker-level quotes may have hidden rejections, spreads during news, liquidity limitations, and operational constraints; quote backtest is **not** an execution guarantee. The script excludes swap/carry, interest, financing, size, leverage and portfolio drawdown. A normalized bps trade P&L is **not** an account or portfolio percent return.

## 4. Start with smallest credible data stack

| Research feature | Historical source | Availability | PIT rule | Fidelity |
|---|---|---|---|---|
| EUR/USD + EUR/PLN daily fixings | ECB EXR API | free; daily history | as of publication (~16:00 local) | labels only; no execution |
| Bid/ask FX ticks | Dukascopy historical quotes | historical available | time-stamp and venue; milliseconds when possible | real quotes; not trades |
| Fed/ECB/NBP policy rates | FRED, ECB, NBP | free | exact decision/effective dates + release time | directly useful |
| Historical CPI/GDP levels | ALFRED vintages, Eurostat release archives | free with processing | vintage available at **forecast time** | revision-sensitive |
| CFTC futures positions | CFTC historical COT | free weekly | Tuesday position snapshot is generally released Friday 15:30 ET | futures proxy, not OTC spot |
| News article event/tone | GDELT GKG/Event historical archives (from 2015) | public files | first-seen/ingestion time; dedupe articles | requires currency-specific NLP |
| FX option implied volatility or futures OI | CME, exchanges / data vendors | mixed | actual time of dissemination | venue-specific proxy |
| FX dealer inventory/client flow | dealer or institutional data licensing | mainly non-public | original transaction timestamp | unavailable to most researchers |
| Cross-venue latency | multi-venue order-level streams | commercial, costly | clock synchronization, venue latency | do not infer from one stream |
| Live production CBDC-AMM flows in EUR/USD/PLN | not a reliable general historical market series | experimental | cannot backtest as actual historical factor | keep OFF |

### Verified data-source details

- ECB Data API: https://data.ecb.europa.eu/help/api/data-examples
- NBP API: https://api.nbp.pl/en.html — historical FX back to 2002 and **93-day maximum interval per request**.
- ALFRED vintage semantics: https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- CFTC release schedule: https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
- Dukascopy tick documentation: https://www.dukascopy-node.app/downloading-tick-data
- GDELT Event/GKG archived releases: https://gdeltproject.org/data.html
- BIS Mariana experimental CBDC: https://www.bis.org/project/mariana

## 5. Suggested rigorous validation protocol

- Define a frozen feature/target definition and candidate hypothesis **before** viewing holdout results.
- Separate **information available at t** from historical reference periods using fields `observed_at`, `released_at`, `available_at`, `revision_vintage`, and `source` (even though the existing v2 importer only requires `available_at`).
- Suggested split: **2016–2021** development, **2022–2023** tuning, **2024–2026** untouched holdout (subject to actual sources having coverage). Require re-evaluation on unusual regimes, e.g. 2020/2022.
- Compare: zero-return; lagged-price Ridge baseline; macro-only; price+microstructure; price+macro+sentiment. Use **same dates for factor comparisons** to avoid sample-composition bias.
- Report OOS R², RMSE, sign accuracy, by-year and by-volatility-regime statistics; also net quote P&L and robustness to stronger spreads, 1s/10s/60s delays and varying venues.
- Use block bootstrap or Diebold-Mariano with assumptions appropriate to serial dependence; test significance and confidence intervals, not just one headline accuracy number.
- Apply correction for trying many features/thresholds; avoid selecting the best among dozens of trials on one holdout.
- A measured conditional predictive association is **not** causal evidence about why banks demand currency.

## Data engineering note

`askVolume` and `bidVolume` in a Dukascopy tick record do **not** automatically measure signed **executed order flow** or the entire FX market. A venue quote feed cannot identify hidden bank inventory or whether a trade was buyer-initiated.

## Important scope limitation

The `quote_backtest.py` simulator is deliberately a **simple single-position-per-pair prototype**. It does not simulate portfolio exposure, balance-sheet usage, position sizing, funding, interest carry, margin calls or market impact. For tick/HFT prediction itself, train a **separate intraday-target model** on time-synchronized venue data. Daily ECB signals cannot validate a sub-second latency hypothesis.
