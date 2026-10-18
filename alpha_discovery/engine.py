"""Time-aware, research-only factor discovery on daily ECB reference FX rates.

NO EXECUTABLE PNL. NO claim of tradable alpha or causal effect.
Protocol: select candidates exclusively within development data; 2026 is
exploratory because it has already been inspected in this project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from eialfx import read_rates, asof_events, make_demo
from .factors import default_hypotheses, validated, factor_series, base_features
from .llm import propose_ollama

SEED = 20261019


def model(kind: str = "ridge"):
    if kind == "ridge":
        estimator = Ridge(alpha=30.0)
    elif kind == "mlp":
        estimator = MLPRegressor(hidden_layer_sizes=(8,), alpha=1.0, max_iter=250,
                                 learning_rate_init=0.002, random_state=SEED)
    else:
        raise ValueError("unsupported model")
    reg = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()), ("model", estimator)])
    if kind == "mlp":
        # FX log-returns have standard deviation ~0.003. Scale y as well as X;
        # without this, MLP output magnitudes can dwarf daily returns.
        return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
    return reg


def frame_from_rates(rates: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if horizon not in (1, 5):
        raise ValueError("Only prespecified 1 and 5 reference-observation horizons")
    lp = np.log(rates[["eurusd", "eurpln", "usdpln"]])
    x = base_features(lp)
    y = pd.DataFrame({p: lp[p].shift(-horizon) - lp[p] for p in ("eurusd", "eurpln")})
    return lp, x, y


def rolling_predictions(x: pd.DataFrame, target: pd.Series, idx: pd.DatetimeIndex,
                        horizon: int, block: int = 21, kind: str = "ridge",
                        min_train: int = 120) -> pd.Series:
    """Train positions i < start-horizon, so every training label is known at start."""
    if not idx.is_monotonic_increasing or not x.index.is_monotonic_increasing or not x.index.is_unique:
        raise ValueError("Indices must be strictly chronological with unique timestamps")
    result = pd.Series(index=idx, dtype=float)
    pos = x.index.get_indexer(idx)
    if (pos < 0).any():
        raise ValueError("test indices must exist in feature matrix")
    for k in range(0, len(pos), block):
        test_positions = pos[k:k + block]
        first = int(test_positions[0])
        train_stop = first - horizon
        if train_stop < min_train:
            raise ValueError("Not enough historical training samples for embargo")
        xt, yt = x.iloc[:train_stop], target.iloc[:train_stop]
        complete = yt.notna() & (xt.notna().mean(axis=1) > 0)
        xt, yt = xt.loc[complete], yt.loc[complete]
        if len(xt) < min_train:
            raise ValueError("Missing or insufficient past target observations")
        active = [c for c in xt if xt[c].notna().mean() >= .75 and xt[c].std(skipna=True) > 1e-12]
        if not active:
            # Constant factor is not a meaningful discovery; baseline should be well formed.
            raise ValueError("No valid training predictors")
        reg = model(kind)
        reg.fit(xt[active], yt)
        test_x = x.iloc[test_positions][active]
        result.iloc[k:k + len(test_positions)] = reg.predict(test_x)
    return result


def metrics(p: pd.Series, y: pd.Series, baseline: pd.Series) -> dict:
    if not (p.index.equals(y.index) and baseline.index.equals(y.index)):
        raise ValueError("misaligned predictions")
    a, b, z = y.to_numpy(), p.to_numpy(), baseline.to_numpy()
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(z).all()):
        raise ValueError("nonfinite evaluation input")
    mse = float(np.mean((a - b)**2))
    base_mse = float(np.mean((a - z)**2))
    zero_mse = float(np.mean(a**2))
    return {"n": len(y), "rmse": float(np.sqrt(mse)),
            "baseline_rmse": float(np.sqrt(base_mse)),
            "zero_rmse": float(np.sqrt(zero_mse)),
            "mse_improvement_vs_price_only_pct": 100 * (1 - mse/base_mse) if base_mse > 0 else None,
            "oos_r2_vs_zero": 1 - mse/zero_mse if zero_mse > 0 else None,
            "direction_accuracy": float(np.mean(np.sign(a) == np.sign(b))),
            "price_only_direction_accuracy": float(np.mean(np.sign(a) == np.sign(z))),
            "correlation": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None}


def block_bootstrap_improvement(y: pd.Series, p: pd.Series, baseline: pd.Series,
                                horizon: int, draws: int = 350) -> list[float]:
    """Exploratory CI (selection bias remains); resample contiguous paired loss blocks."""
    delta = (y.to_numpy() - baseline.to_numpy())**2 - (y.to_numpy() - p.to_numpy())**2
    n = len(delta)
    block = max(5, 2 * horizon)
    if n < 3 * block:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        ids = np.concatenate([(np.arange(s, s+block) % n) for s in starts])[:n]
        boot.append(float(np.mean(delta[ids])))
    return [float(v) for v in np.quantile(boot, [.025, .975])]


def research(rates: pd.DataFrame, specs: list[dict], out: Path, dev_end: str,
             horizon: int = 1, block: int = 21, min_train: int = 150,
             target_pair: str = "eurusd", explore_after_dev: bool = False,
             nonlinear: bool = False, synthetic: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if target_pair not in ("eurusd", "eurpln"):
        raise ValueError("Target pair is EUR/USD or EUR/PLN; USD/PLN is derived")
    lp, x_base, targets = frame_from_rates(rates, horizon)
    target = targets[target_pair]
    # The development regime is immutable during candidate comparison.
    dev_dates = x_base.index[(x_base.index <= pd.Timestamp(dev_end)) & target.notna()]
    if len(dev_dates) < min_train + horizon + 90:
        raise ValueError("Insufficient development history: need more observations before dev-end")
    val_pos = max(min_train + horizon, int(len(dev_dates) * .70))
    validation = dev_dates[val_pos:]
    # Baseline is evaluated on identical timestamps for every candidate.
    base_preds = rolling_predictions(x_base, target, validation, horizon, block, "ridge", min_train)
    y_val = target.loc[validation]
    baseline_record = metrics(base_preds, y_val, pd.Series(0.0, index=validation))
    candidates, used_ids = [], set()
    for raw in specs:
        try:
            spec = validated(raw)
        except (ValueError, TypeError, KeyError) as ex:
            candidates.append({"status": "invalid", "reason": str(ex), "submitted": str(raw)[:200]})
            continue
        if spec["id"] in used_ids:
            continue
        used_ids.add(spec["id"])
        fac = factor_series(lp, spec)
        # Screening may inspect only development period; never 2026.
        dev_factor = fac.loc[dev_dates]
        if dev_factor.notna().mean() < .90 or dev_factor.std(skipna=True) < 1e-10:
            candidates.append({"id": spec["id"], "status": "rejected_data_quality", "spec": spec})
            continue
        x = x_base.assign(**{spec["id"]: fac})
        # Correlation gate on development only.
        corr = x.loc[dev_dates].corr(numeric_only=True)[spec["id"]].drop(spec["id"]).abs().max()
        if pd.notna(corr) and corr > .995:
            candidates.append({"id": spec["id"], "status": "rejected_redundant", "spec": spec, "max_abs_corr": float(corr)})
            continue
        try:
            predictions = rolling_predictions(x, target, validation, horizon, block, "ridge", min_train)
        except (ValueError, TypeError) as ex:
            candidates.append({"id": spec["id"], "status": "rejected_fit", "spec": spec, "reason": str(ex)})
            continue
        m = metrics(predictions, y_val, base_preds)
        lo, hi = block_bootstrap_improvement(y_val, predictions, base_preds, horizon)
        candidates.append({"id": spec["id"], "spec": spec, "status": "screened",
                           "max_abs_corr": float(corr) if pd.notna(corr) else None,
                           "development_validation": m,
                           "exploratory_block_bootstrap_mse_difference_ci95": [lo, hi]})
    scored = [c for c in candidates if c.get("status") == "screened"]
    scored.sort(key=lambda c: c["development_validation"]["mse_improvement_vs_price_only_pct"], reverse=True)
    champion = scored[0] if scored else None
    # A promising dev candidate is NOT a confirmed alpha. Multiple testing biases CIs.
    for c in scored:
        m = c["development_validation"]
        c["status"] = "promising_dev_ONLY" if (m["mse_improvement_vs_price_only_pct"] > 0 and
                             m["oos_r2_vs_zero"] is not None and m["oos_r2_vs_zero"] > 0) else "no_dev_edge"
    champion = scored[0] if scored else None
    report = {"label": "SYNTHETIC EXERCISE - NO MARKET EVIDENCE" if synthetic else
              "HISTORICAL ECB REFERENCE RATE EXPERIMENT - NON-EXECUTABLE",
              "pair": target_pair, "horizon_reference_observations": horizon,
              "n_candidates_submitted": len(specs), "n_candidates_unique_valid": len(used_ids),
              "development_start": str(dev_dates[0].date()), "development_end": str(dev_dates[-1].date()),
              "validation_start": str(validation[0].date()), "validation_end": str(validation[-1].date()),
              "development_price_baseline": baseline_record, "champion": champion,
              "candidate_register": "candidates.json", "evaluation_2026_status": "LOCKED unless --explore-2026",
              "research_status": "EXPLORATORY FACTOR SEARCH ONLY — NO CONFIRMED TRADABLE ALPHA",
              "protocol": {"selection_data": "<= dev_end ONLY", "label_horizon_embargo": horizon,
                           "retrain_block": block, "candidate_model": "Ridge(alpha=30)",
                           "baseline": "price-only Ridge, same folds", "bootstrap": "selection-biased exploratory intervals",
                           "price_source": "ECB reference only; no executable bid/ask"}}
    if explore_after_dev:
        eval_dates = x_base.index[(x_base.index > pd.Timestamp(dev_end)) & target.notna()]
        if len(eval_dates) == 0:
            report["evaluation_2026_status"] = "NO DATA POST dev_end"
        elif champion is None:
            report["evaluation_2026_status"] = "SKIPPED: no valid champion"
        else:
            sp = champion["spec"]
            xs = x_base.assign(**{sp["id"]: factor_series(lp, sp)})
            pred_base = rolling_predictions(x_base, target, eval_dates, horizon, block, "ridge", min_train)
            pred_full = rolling_predictions(xs, target, eval_dates, horizon, block, "ridge", min_train)
            tr = target.loc[eval_dates]
            report["evaluation_2026_status"] = "POST-DEVELOPMENT EXPLORATORY RETEST, NOT BLIND / NOT CONFIRMED"
            report["evaluation_2026_metrics"] = metrics(pred_full, tr, pred_base)
            report["evaluation_price_baseline"] = metrics(pred_base, tr, pd.Series(0., index=eval_dates))
            tab = pd.DataFrame({"actual_logret": tr, "price_only_pred_logret": pred_base,
                                "candidate_pred_logret": pred_full})
            if nonlinear:
                mlp_pred = rolling_predictions(xs, target, eval_dates, horizon, block, "mlp", min_train)
                tab["mlp_pred_logret"] = mlp_pred
                report["evaluation_mlp_metrics"] = metrics(mlp_pred, tr, pred_base)
            tab.to_csv(out / "exploratory_predictions.csv", index_label="date")
    (out / "candidates.json").write_text(json.dumps(candidates, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if scored:
        pd.DataFrame([{"id": c["id"], "status": c["status"], "family": c["spec"]["family"],
                       "source": c["spec"]["source"], **c["development_validation"]}
                      for c in scored]).to_csv(out / "leaderboard.csv", index=False)
    return report


def main():
    p = argparse.ArgumentParser(description="AI Alpha Discovery: restricted LLM factor proposals + leak-aware screening")
    p.add_argument("--rates", help="date,eurusd,eurpln CSV")
    p.add_argument("--demo", action="store_true", help="explicit SYNTHETIC exercise")
    p.add_argument("--out", default="results/alpha_discovery")
    p.add_argument("--dev-end", default="2025-12-31")
    p.add_argument("--pair", choices=["eurusd", "eurpln"], default="eurusd")
    p.add_argument("--horizon", type=int, choices=[1,5], default=1)
    p.add_argument("--max-candidates", type=int, default=24)
    p.add_argument("--llm-model", help="Optional local Ollama name e.g. gemma3:4b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--explore-2026", action="store_true", help="Explicitly evaluate post-dev; NOT blind")
    p.add_argument("--nonlinear", action="store_true", help="Also evaluate MLP on the selected candidate (exploratory)")
    p.add_argument("--min-train", type=int, default=150)
    args = p.parse_args()
    if args.demo == bool(args.rates):
        p.error("choose exactly one of --demo or --rates")
    if not 1 <= args.max_candidates <= 200:
        p.error("--max-candidates must be 1..200")
    out = Path(args.out)
    if args.demo:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "SYNTHETIC_RATES.csv"
        # Synthetic demo starts in 2021 and runs into 2025. Use a development split within it.
        make_demo(str(path), str(out / "SYNTHETIC_EVENTS.csv"), n=1150)
        rates = read_rates(path)
        dev_end = "2024-03-29"
    else:
        rates = read_rates(args.rates)
        dev_end = args.dev_end
    candidates = default_hypotheses(args.max_candidates)
    if args.llm_model:
        raw = propose_ollama(args.llm_model, n=min(args.max_candidates, 12), endpoint=args.ollama_url)
        # Give LLM proposals first priority; otherwise the default list can
        # exhaust max_candidates and silently discard all LLM ideas.
        selected = {}
        for proposal in raw:
            try:
                v = validated(proposal)
            except (ValueError, TypeError, KeyError):
                continue
            selected[v["id"]] = v
        for c in candidates:
            selected.setdefault(c["id"], c)
        candidates = list(selected.values())[:args.max_candidates]
    result = research(rates, candidates, out, dev_end=dev_end, horizon=args.horizon,
                      target_pair=args.pair, explore_after_dev=args.explore_2026,
                      nonlinear=args.nonlinear, min_train=args.min_train,
                      synthetic=args.demo)
    print(json.dumps({k: v for k,v in result.items() if k not in ("champion",)}, indent=2))
    print("CHAMPION (development only):", result.get("champion", {}).get("id") if result.get("champion") else "none")

if __name__ == "__main__":
    main()
