# EIAL-FX: falsification-first research protocol

**2026-10-19 | v2.2 proposal | no trading edge established**

## Prederegister before data inspection

1. Pick exactly one primary target: next published ECB EUR/USD or EUR/PLN log return (daily, information-only). Record timestamp cutoff 16:30 Europe/Brussels, target horizon and matching news/vintage availability.
2. Start with `M0`: zero return; `M1`: lagged price model; add one family at a time (`policy`, `institutional`, `sentiment`, `microstructure`). Use identical OOS dates for all comparisons.
3. Do not infer actual dealer demand from spot returns or public bid/ask sizes. Only test `Q` when original signed executed flow labels exist.
4. Keep a locked final period untouched after feature definitions/hyperparameters and research gates are frozen. Backtest performance on any slice used for iteration is *development*, not final confirmation.
5. Record acquisition URL, licensing status, download date, event timestamp, first-publication timestamp, provider-vintage identifier, timezone, clock-resolution and missingness for every factor.

## Evaluations

- Price task: OOS RMSE / MAE, R² against zero-return, correlation, directional accuracy, regime stability, block-bootstrap uncertainty, benchmark delta and ablations.
- Quote paper task: next tradable ask/bid only after signal+delay, round-trip spread, realistic slippage and fees, sample count, net bps per trade, skipped signals, volatility/volume strata, shock and ordinary sessions separately.
- Robustness: cost sensitivity at realistic 1x / 2x / 3x estimates, different time periods and holdouts, failure analysis on large moves, currency cross-consistency.
- Diagnostics in `research_gate.py` are **screening heuristics**, not p-values, claims of profitability or confirmation.

## Stop / Go

`NO_GO` for any synthetic-only result, inconsistent publication timestamp, look-ahead, below-baseline forecast, materially low trade coverage or negative net bps after realistic costs. Passing the diagnostic gate yields `FURTHER_VALIDATION_REQUIRED`, **never** `EDGE_PROVEN`.

## Scientific limits

- Estimating predictive association is not proving causality.
- CFTC / BIS position proxies are incomplete and delayed.
- A quoted trade size/FX broker book cannot represent the entire OTC FX market.
- GDELT event timestamps and LLM sentiment model changes can introduce silent look-ahead and selection bias.
- Public CBDC PoCs do not supply a market-wide historical CBDC-spot causal series.
- Other market makers may internalize flows without showing an exchange-book footprint.

## Independent study

The Varrior FX Analytics repo is a methodological precedent (negative results, strict holdout), but EIAL-FX is deliberately a separate empirical hypothesis. Do not pool evaluation results or transfer any proprietary tick dataset into this public repository.
