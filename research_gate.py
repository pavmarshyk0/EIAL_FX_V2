#!/usr/bin/env python3
"""Conservative screening of EIAL-FX reports; NEVER declares an alpha proven.

Inputs are JSON files emitted by eialfx.py and quote_backtest.py. Gates are
heuristic design decisions, not statistical confidence intervals or a backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(forecast: dict, quotes: dict | None = None, min_trades: int = 100) -> dict:
    if min_trades < 1:
        raise ValueError("min_trades must be >= 1")
    reasons = []
    label = str(forecast.get("label", ""))
    if "SYNTHETIC" in label.upper():
        reasons.append("synthetic_forecast_data")
    metrics = forecast.get("metrics", {})
    base = metrics.get("price_baseline", {})
    full = metrics.get("all_observed_factors", {})
    tested = []
    for target in ("eurusd_next_logret", "eurpln_next_logret"):
        m = full.get(target)
        b = base.get(target)
        if m is None or b is None:
            reasons.append(f"{target}:missing_comparable_metrics")
            continue
        n = m.get("n_test", 0)
        r2 = m.get("oos_r2_vs_zero")
        improvement = float(b.get("rmse", float("inf"))) - float(m.get("rmse", float("inf")))
        passed = n >= 300 and r2 is not None and r2 > 0 and improvement > 0
        tested.append({"target": target, "n_test": n, "oos_r2_vs_zero": r2,
                       "rmse_improvement_over_price_baseline": improvement,
                       "exploratory_screen_pass": bool(passed)})
        if not passed:
            reasons.append(f"{target}:no_demonstrated_forecast_improvement")
    if quotes is None:
        reasons.append("real_quote_backtest_not_provided")
    else:
        if "SYNTHETIC" in str(quotes.get("label", "")).upper():
            reasons.append("synthetic_quote_data")
        q = quotes.get("pairs", {})
        if not q:
            reasons.append("no_executed_trades")
        for pair, row in q.items():
            if row.get("n_trades", 0) < min_trades:
                reasons.append(f"{pair}:trade_count_below_{min_trades}")
            if row.get("mean_net_bps_per_trade", 0) <= 0:
                reasons.append(f"{pair}:nonpositive_avg_net_bps")
    # Even zero reasons is only a hypothesis worth stronger independent confirmation.
    return {
        "status": "NO_GO_SCREEN" if reasons else "FURTHER_VALIDATION_REQUIRED",
        "diagnostic_only": True,
        "edge_proven": False,
        "reason_codes": reasons,
        "targets": tested,
        "caveats": ["heuristic thresholds are not statistical significance tests",
                    "no locked-holdout or confidence interval proven by these reports",
                    "cross-pair book P&L and real-world venue execution remain unverified"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast", required=True)
    p.add_argument("--quotes")
    p.add_argument("--min-trades", type=int, default=100)
    p.add_argument("--out")
    args = p.parse_args()
    forecast = json.loads(Path(args.forecast).read_text(encoding="utf-8"))
    quotes = json.loads(Path(args.quotes).read_text(encoding="utf-8")) if args.quotes else None
    result = evaluate(forecast, quotes, min_trades=args.min_trades)
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
