# AI Alpha Discovery v0.1 — research protocol

> Status: implemented prototype; **no proven financial alpha**. Author: Paweł Maruszczyk. License: repository root proprietary source-available license.

## Architecture

1. **Data**: load real ECB `date,eurusd,eurpln` fixes, derive `usdpln = eurpln / eurusd`, and compute signals using only data at/before each information cutoff.
2. **LLM scientist (optional)**: local Ollama (`gemma3:4b` or equivalent) suggests structured JSON factor recipes. The output is **untrusted**; a strict allowlist validates each recipe. Never execute generated Python or shell commands.
3. **Factor laboratory**: deterministic catalog of momentum, price z-score, volatility compression/expansion, return acceleration, and cross-pair relative momentum. Each hypothesis has a stable ID and thesis.
4. **Screening**: compare `Ridge(price-only + factor)` vs `Ridge(price-only)` on the **same expanding-window chronological validation folds**, with a horizon-length label embargo and 21-observation refits.
5. **Ablation and log**: preserve **all** candidates (including rejected factors), incremental MSE, direction accuracy, OOS R² against zero return, and an illustrative block-bootstrap interval. Select a candidate **using development-only data**.
6. **Optional neural network**: an 8-neuron MLP on the preselected factor is tested **separately**, not selected as part of the exploratory factor search. Its statistical value is not presumed.
7. **Later stage**: optional post-development run. For this project, 2026 is already seen: any 2026 result is an *exploratory retest*, not a blind confirmatory test. Genuine confirmation needs a sealed future period or timestamped paper-trading forecasts.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m alpha_discovery --demo --out results/alpha_demo --max-candidates 24 --explore-2026
```

`--demo` produces synthetic 2021–2025 observations. It is NOT a market performance test.

### Real rates: development selection only

```bash
python eialfx.py fetch-ecb --start 2017-01-01 --out data/ecb_fx.csv
python -m alpha_discovery --rates data/ecb_fx.csv --dev-end 2025-12-31 \
  --out results/alpha_real --pair eurusd --horizon 1 --max-candidates 36
```

### Local LLM hypothesis proposals

Run Ollama yourself (e.g. model `gemma3:4b`) on your computer. Then:

```bash
python -m alpha_discovery --rates data/ecb_fx.csv --dev-end 2025-12-31 \
  --llm-model gemma3:4b --max-candidates 24 --out results/alpha_ollama
```

This requires a running **local** Ollama endpoint at `http://127.0.0.1:11434`; the software does not download models. Proposals are safe structured recipes, not arbitrary generated Python code.

### Explicitly unsealed exploratory 2026 + MLP comparison

```bash
python -m alpha_discovery --rates data/ecb_fx.csv --dev-end 2025-12-31 \
  --explore-2026 --nonlinear --out results/alpha_2026_exploratory
```

The above will write `exploratory_predictions.csv`, and **2026 is NOT a pristine test set** since its outcomes have already been inspected in this project.

## Outputs

- `candidates.json` — complete hypothesis register and all rejections
- `leaderboard.csv` — development-period metrics for the screened factors
- `report.json` — sources and status; no alpha-confirmation claim
- `exploratory_predictions.csv` — only when post-development evaluation is explicitly enabled

## Correctness limits

- Daily ECB reference rates are **not executable bid/ask**. No realized net P&L or fill claims.
- Historical evidence cannot be established without genuine historical rates. `--demo` only verifies software behavior.
- CIs calculated on the *selection* sample are biased by multiple comparisons. A high factor rank is not statistically significant proof of discovery.
- Institutional flows, true order-book imbalance, private HFT/LLM latencies and CBDC AMM transaction histories are **not fabricated** in this release.
- The same economic signal with a different lookback can count as multiple tests. No factor discovery here should be marketed as profitable.
- Stage 2 will need asynchronous available-at event joins, purged/CPCV for appropriate horizons, spread-aware executable data, paper-trading logs, and data provenance.

## Success / fail decision

1. A factor must beat a matched price-only model on **development OOS** and improve against the zero-change benchmark.
2. Additional validation must be held in the future and independent of any prompts/selection tuning.
3. A tradable alpha also requires positive expected net return after actual bid/ask, slippage, financing and execution uncertainty.
4. If the discovery result is null, preserve the failure; do not re-run selection on the nominal holdout and claim success.
