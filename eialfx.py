#!/usr/bin/env python3
"""EIAL-FX v2.0: reproducible, time-aware FX research baseline.

IMPORTANT: ECB fixing data are NOT executable prices. Outputs are research
forecasts and not an executable trading backtest or causal evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261019
TZ = "Europe/Brussels"
BASE_FEATURES = [
    "eurusd_ret1", "eurusd_ret2", "eurusd_ret5", "eurusd_vol10",
    "eurpln_ret1", "eurpln_ret2", "eurpln_ret5", "eurpln_vol10",
    "usdpln_ret1", "usdpln_ret5", "usdpln_vol10",
]
SOURCE_ECB = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD+PLN.EUR.SP00.A"
SOURCE_FRED = "https://api.stlouisfed.org/fred/series/observations"


def _fix_time(days: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """One fixed observation decision time, AFTER the ECB ~16:00 release."""
    d = pd.to_datetime(days)
    return pd.DatetimeIndex(d).tz_localize(None).normalize().tz_localize(TZ) + pd.Timedelta(hours=16, minutes=30)


def read_rates(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "eurusd", "eurpln"}
    if not required.issubset(df.columns):
        raise ValueError(f"rates CSV must contain {sorted(required)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if df["date"].duplicated().any():
        raise ValueError("Duplicated fixing dates; cannot align point-in-time observations")
    df = df.sort_values("date").set_index("date")
    df = df[["eurusd", "eurpln"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 80 or (df <= 0).any().any():
        raise ValueError("Need at least 80 positive simultaneous EURUSD/EURPLN fixings")
    df["usdpln"] = df["eurpln"] / df["eurusd"]
    return df


def fetch_ecb(out_path: str | Path, start: str, end: str | None = None) -> pd.DataFrame:
    params = {"format": "csvdata", "startPeriod": start}
    if end:
        params["endPeriod"] = end
    response = requests.get(SOURCE_ECB, params=params, timeout=45,
                            headers={"Accept": "text/csv", "User-Agent": "eialfx-research/2.0"})
    response.raise_for_status()
    from io import StringIO
    raw = pd.read_csv(StringIO(response.text))
    expected = {"CURRENCY", "TIME_PERIOD", "OBS_VALUE"}
    if not expected.issubset(raw.columns):
        raise RuntimeError(f"ECB response schema changed: {list(raw.columns)}")
    wide = (raw.loc[raw["CURRENCY"].isin(["USD", "PLN"])]
            .pivot_table(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE", aggfunc="last")
            .dropna(subset=["USD", "PLN"])
            .rename(columns={"USD": "eurusd", "PLN": "eurpln"})
            .reset_index().rename(columns={"TIME_PERIOD": "date"}))
    wide = wide[["date", "eurusd", "eurpln"]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    return read_rates(out_path)


def fetch_fred_policy(out_path: str | Path, start: str, end: str | None = None) -> pd.DataFrame:
    """Conservative lag of effective date; NOT an audited vintage-level feed.

    The 2-business-day lag prevents most effective-date lookahead, but research
    requiring historical real-time data must use release timestamps and ALFRED vintages.
    """
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY from https://fred.stlouisfed.org/docs/api/api_key.html")
    frames = []
    for series, feature in [("ECBDFR", "policy_eur"), ("DFEDTARL", "policy_usd")]:
        params = {"series_id": series, "api_key": key, "file_type": "json",
                  "observation_start": start}
        if end:
            params["observation_end"] = end
        response = requests.get(SOURCE_FRED, params=params, timeout=45)
        response.raise_for_status()
        data = response.json().get("observations", [])
        rows = []
        for obs in data:
            try:
                val = float(obs["value"])
            except (ValueError, KeyError, TypeError):
                continue
            # Conservative publication proxy; 2 US/EU weekdays, not actual releases.
            effective = pd.Timestamp(obs["date"]) + pd.offsets.BDay(2)
            available_at = effective.strftime("%Y-%m-%d") + "T12:00:00+00:00"
            rows.append({"available_at": available_at, "feature": feature, "value": val})
        frames.append(pd.DataFrame(rows, columns=["available_at", "feature", "value"]))
    events = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_path, index=False)
    return events


def _age_limit(feature: str) -> pd.Timedelta:
    """Only carry recent observations; policy levels persist longer."""
    f = feature.lower()
    if f.startswith("policy_"):
        return pd.Timedelta(days=120)
    if any(x in f for x in ("m3_", "gdp_", "cpi_", "bank_positions")):
        return pd.Timedelta(days=100)
    if "cot_" in f:
        return pd.Timedelta(days=9)
    if "_signed_flow" in f:
        # Never carry yesterday's completed trade-flow label to today.
        return pd.Timedelta(hours=12)
    if any(x in f for x in ("sentiment", "mood_", "news_", "latency", "_bid_size", "_ask_size", "_signed_flow", "_depth", "_spread", "_amm_mid")):
        return pd.Timedelta(days=1)
    return pd.Timedelta(days=5)


def asof_events(rates: pd.DataFrame, events_path: str | Path | None) -> pd.DataFrame:
    """Strict as-of join: NEVER use an observation before it became available.

    events CSV: available_at ISO-8601 with UTC offset, feature string, value.
    Rate observation information cutoff: 16:30 Europe/Brussels on fixing day.
    """
    res = pd.DataFrame(index=rates.index)
    if events_path is None:
        return res
    ev = pd.read_csv(events_path)
    if not {"available_at", "feature", "value"}.issubset(ev.columns):
        raise ValueError("events must contain available_at,feature,value")
    # Reject timezone-naive strings: silently assuming UTC would invalidate causal ordering.
    ts_strings = ev["available_at"].astype(str)
    if not ts_strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, case=False).all():
        raise ValueError("Every available_at must include a timezone offset or Z (UTC)")
    ev["available_at"] = pd.to_datetime(ev["available_at"], utc=True, errors="raise")
    ev["feature"] = ev["feature"].astype(str)
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    ev = ev.dropna(subset=["value"])
    cutoffs = _fix_time(rates.index).tz_convert("UTC")
    left = pd.DataFrame({"cutoff": cutoffs}).sort_values("cutoff")
    for feature, part in ev.groupby("feature"):
        if not feature.replace("_", "").isalnum():
            raise ValueError(f"Unsafe feature name: {feature!r} (use only letters, digits, underscores)")
        part = (part[["available_at", "value"]]
                .sort_values("available_at")
                .drop_duplicates("available_at", keep="last"))
        joined = pd.merge_asof(left, part, left_on="cutoff", right_on="available_at",
                               direction="backward", tolerance=_age_limit(feature))
        res[feature] = joined["value"].to_numpy()
    return res


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.div(b.where(b.abs() > 1e-10))


def feature_matrix(rates: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict]:
    """Features at t; targets from t to next fixing t+1.

    Prediction is for the NEXT ECB reference observation, NOT a trade at fixing t.
    """
    logp = np.log(rates[["eurusd", "eurpln", "usdpln"]])
    ret = logp.diff()
    x = pd.DataFrame(index=rates.index)
    for pair in ["eurusd", "eurpln", "usdpln"]:
        x[f"{pair}_ret1"] = ret[pair]
        x[f"{pair}_ret2"] = ret[pair].rolling(2).sum()
        x[f"{pair}_ret5"] = ret[pair].rolling(5).sum()
        x[f"{pair}_vol10"] = ret[pair].rolling(10).std()

    used = {"price_lags": list(x.columns)}

    # Policy rates / macro levels as of cutoff; no assumption that rate differences = order flows.
    for a, b, key in [("eur", "usd", "eurusd"), ("eur", "pln", "eurpln")]:
        ca, cb = f"policy_{a}", f"policy_{b}"
        if ca in events and cb in events:
            x[f"{key}_rate_diff_pp"] = events[ca] - events[cb]
            x[f"{key}_rate_diff_change"] = x[f"{key}_rate_diff_pp"].diff()

    # Directional cross-currency news sentiment, only if both currencies exist.
    for a, b, key in [("eur", "usd", "eurusd"), ("eur", "pln", "eurpln")]:
        if f"mood_{a}" in events and f"mood_{b}" in events:
            s = events[f"mood_{a}"] - events[f"mood_{b}"]
            x[f"{key}_sent_level"] = s
            x[f"{key}_sent_velocity"] = s.diff()
            x[f"{key}_sent_acceleration"] = s.diff().diff()
            x[f"{key}_sent_nonlinear"] = s.diff() * s.diff().abs()

    for key in ("eurusd", "eurpln"):
        prefix = f"{key}__"
        a, b = prefix + "ask_size", prefix + "bid_size"
        if a in events and b in events:
            # ORDER BOOK imbalance is not executed trade flow.
            x[f"{key}_book_imbalance"] = _safe_div(events[b] - events[a], events[b] + events[a])
            logratio = np.log(_safe_div(events[a], events[b]).where(lambda z: z > 0))
            x[f"{key}_book_ratio_change"] = logratio.diff()
        depth = prefix + "depth"
        if depth in events and f"{key}_book_imbalance" in x:
            x[f"{key}_imbalance_per_depth"] = _safe_div(x[f"{key}_book_imbalance"], events[depth])
        amm = prefix + "amm_mid"
        if amm in events:
            x[f"{key}_amm_basis"] = np.log(_safe_div(events[amm], rates[key]).where(lambda z: z > 0))

    # Incorporate other measured factors WITHOUT fabricating missing inputs.
    skip = {"policy_eur", "policy_usd", "policy_pln", "mood_eur", "mood_usd", "mood_pln"}
    skip.update(f"{key}__{col}" for key in ("eurusd", "eurpln")
                for col in ("ask_size", "bid_size", "amm_mid", "depth"))
    for col in events.columns:
        if col not in skip:
            x[f"observed__{col}"] = events[col]

    # Liquidity interaction: observed book imbalance * spread (if available).
    for pair in ("eurusd", "eurpln"):
        spread = f"observed__{pair}__spread_bps"
        imb = f"{pair}_book_imbalance"
        if spread in x and imb in x:
            x[f"{pair}_imbalance_x_spread"] = x[imb] * x[spread]

    # Target is a strictly future close-to-next-close fixing return.
    y = ret[["eurusd", "eurpln"]].shift(-1).rename(columns={
        "eurusd": "eurusd_next_logret", "eurpln": "eurpln_next_logret"})

    # A separate observable future signed-flow target, ONLY if true data supplied.
    qtarget = None
    flow_columns = [f"{key}__signed_flow" for key in ("eurusd", "eurpln")]
    if all(c in events.columns for c in flow_columns):
        qtarget = events[flow_columns].shift(-1).copy()
        qtarget.columns = ["eurusd_next_signed_flow", "eurpln_next_signed_flow"]

    meta = {"features": x.columns.tolist(), "base_features": [c for c in BASE_FEATURES if c in x]}
    return x, y, qtarget, meta


def _feature_group(name: str) -> str:
    if name in BASE_FEATURES:
        return "price_baseline"
    if "sent_" in name or "mood_" in name or "news_" in name:
        return "sentiment"
    if "amm_" in name or "cbdc" in name:
        return "cbdc_experimental"
    if "latency" in name or "lag_" in name:
        return "latency_proxy"
    if any(t in name for t in ("imbalance", "depth", "spread", "signed_flow", "book_ratio", "volume")):
        return "microstructure"
    if any(t in name for t in ("rate_diff", "policy", "m3_", "cpi_", "gdp_")):
        return "macro"
    return "institutional_or_other"


def _make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=30.0)),
    ])


def _predict_cv(x: pd.DataFrame, y: pd.DataFrame, features: list[str],
                min_train: int, block: int, embargo: int = 1):
    # Complete target rows; preserve chronological index. Require strict t+1 < test start.
    valid = y.notna().all(axis=1)
    X = x.loc[valid, features].copy()
    Y = y.loc[valid].copy()
    if len(X) < min_train + block:
        raise ValueError(f"Only {len(X)} samples; need at least {min_train + block}")
    reports = []
    all_pred = []
    for start in range(min_train + embargo, len(X), block):
        stop = min(start + block, len(X))
        # Last train sample index <= start-embargo-1.
        train_x = X.iloc[:start - embargo]
        train_y = Y.iloc[:start - embargo]
        test_x = X.iloc[start:stop]
        test_y = Y.iloc[start:stop]
        # Feature selection only with training data; no fold-specific test peeking.
        active = [c for c in features if train_x[c].notna().mean() >= .60
                  and train_x[c].std(skipna=True) > 1e-12]
        if not active:
            raise ValueError("No useful predictors in training fold")
        model = _make_model()
        model.fit(train_x[active], train_y)
        predicted = model.predict(test_x[active])
        pp = pd.DataFrame(predicted, index=test_y.index, columns=test_y.columns)
        pp["fold"] = len(reports)
        all_pred.append(pp)
        reports.append({"train_start": str(train_x.index[0].date()),
                        "train_end": str(train_x.index[-1].date()),
                        "test_start": str(test_x.index[0].date()),
                        "test_end": str(test_x.index[-1].date()),
                        "active_features": len(active)})
    forecast = pd.concat(all_pred)
    truth = Y.loc[forecast.index]
    metrics = {}
    for col in y.columns:
        p = forecast[col].to_numpy()
        target = truth[col].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(target, p)))
        naive_rmse = float(np.sqrt(np.mean(target**2)))  # next return = 0 baseline
        metrics[col] = {
            "n_test": len(target),
            "rmse": rmse, "mae": float(mean_absolute_error(target, p)),
            "zero_return_baseline_rmse": naive_rmse,
            "oos_r2_vs_zero": float(1 - (rmse / naive_rmse)**2) if naive_rmse else None,
            "direction_accuracy": float(np.mean(np.sign(target) == np.sign(p))),
            "pearson_corr": float(np.corrcoef(p, target)[0, 1]) if np.std(p) > 0 and np.std(target) > 0 else None,
        }
    result = truth.add_suffix("_actual").join(forecast.add_suffix("_pred"))
    return result, metrics, reports


def _latest_forecast(x: pd.DataFrame, y: pd.DataFrame, available: list[str]):
    valid = y.notna().all(axis=1)
    train_x, train_y = x.loc[valid], y.loc[valid]
    active = [c for c in available if train_x[c].notna().mean() >= .60 and train_x[c].std() > 1e-12]
    model = _make_model()
    model.fit(train_x[active], train_y)
    current = x.iloc[[-1]]
    prediction = model.predict(current[active])[0]
    weights = pd.DataFrame(model.named_steps["ridge"].coef_, columns=active, index=y.columns)
    # Standardized coefficients of predictive association, NOT causal effects.
    return dict(zip(y.columns, map(float, prediction))), weights.reset_index().rename(columns={"index": "target"})


def train(rates_path: str, events_path: str | None, out_dir: str,
          min_train: int = 260, block: int = 63, synthetic: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rates = read_rates(rates_path)
    ev = asof_events(rates, events_path)
    x, y, q, metadata = feature_matrix(rates, ev)
    valid_x = x.dropna(subset=metadata["base_features"])
    y = y.loc[valid_x.index]
    x = valid_x
    # Every benchmark uses the same OUT-OF-SAMPLE chronological folds.
    experiments = {"price_baseline": metadata["base_features"], "all_observed_factors": x.columns.tolist()}
    groups = sorted(set(map(_feature_group, x.columns)) - {"price_baseline"})
    for group in groups:
        experiments[f"without__{group}"] = [c for c in x if _feature_group(c) != group]
    experiments = {k: v for k, v in experiments.items() if v}

    metrics = {}
    predictions = None
    folds = None
    for name, cols in experiments.items():
        p, m, f = _predict_cv(x, y, cols, min_train=min_train, block=block)
        metrics[name] = m
        if name == "all_observed_factors":
            predictions, folds = p, f
    assert predictions is not None and folds is not None
    # Cross-derived USDPLN, never fit as a third unconstrained response.
    predictions["usdpln_next_logret_actual"] = (
        predictions["eurpln_next_logret_actual"] - predictions["eurusd_next_logret_actual"])
    predictions["usdpln_next_logret_pred"] = (
        predictions["eurpln_next_logret_pred"] - predictions["eurusd_next_logret_pred"])
    price_latest, weights = _latest_forecast(x, y, x.columns.tolist())
    price_latest["usdpln_next_logret"] = (price_latest["eurpln_next_logret"] -
                                          price_latest["eurusd_next_logret"])

    flow_status = "NOT ESTIMATED: no observed signed order-flow labels"
    if q is not None:
        q = q.loc[x.index]
        # Flow labels are truly observed, not inferred from price; only report if sufficiently complete.
        valid = q.notna().all(axis=1)
        if int(valid.sum()) >= min_train + block:
            # Use rows where true flow exists; no synthetic interpolation.
            _, flow_metrics, flow_folds = _predict_cv(x.loc[valid], q.loc[valid], x.columns.tolist(),
                                                      min_train, block)
            metrics["signed_flow_separate_task"] = flow_metrics
            flow_status = f"Estimated as SEPARATE predictive task using {int(valid.sum())} flow observations"

    predictions.to_csv(out / "oos_predictions.csv", index_label="date")
    weights.to_csv(out / "standardized_coefficients.csv", index=False)
    x.to_csv(out / "feature_matrix.csv", index_label="date")
    report = {
        "label": "SYNTHETIC DEMONSTRATION - NOT MARKET EVIDENCE" if synthetic else
                 "HISTORICAL REFERENCE-RATE RESEARCH - NOT EXECUTABLE PNL",
        "units": "daily close-to-next ECB fixing log-return",
        "asof_cutoff": "16:30 Europe/Brussels on ECB fixing date",
        "data_start": str(x.index[0].date()), "data_end": str(x.index[-1].date()),
        "events_used": list(ev.columns), "prediction_feature_count": len(x.columns),
        "flow_status": flow_status, "walk_forward_folds": folds,
        "cv_embargo_observations": 1, "model": "Ridge(alpha=30) with in-fold scaling + median imputation",
        "metrics": metrics, "next_fixing_forecast_log_return": price_latest,
        "warnings": [
            "No proof of causal factor impact from Ridge coefficients",
            "ECB fixings are information-only; price at decision cutoff is NOT an executable fixing",
            "Reference model needs executable timestamped bid/ask data for any trading PnL claims",
            "FRED latest vintage is not a historical point-in-time dataset; publication timestamps required",
            "No signed flow is fabricated when flow labels are missing",
            "No intraday/HFT forecast can be tested from daily ECB reference rates",
        ]}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def make_demo(rates_path: str, events_path: str, n: int = 1150) -> None:
    """Artificial data explicitly labeled. Demonstrates code path, not edge."""
    if n < 400:
        raise ValueError("demo needs >= 400 observations")
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2021-01-04", periods=n)
    news = rng.standard_normal(n)
    news_pln = rng.standard_normal(n)
    eur, pln = np.empty(n), np.empty(n)
    eur[0], pln[0] = np.log(1.12), np.log(4.45)
    r_eur, r_pln = 0., 0.
    for i in range(1, n):
        r_eur = .10 * r_eur + .00035 * news[i-1] + rng.normal(0, .0033)
        r_pln = .11 * r_pln + .00050 * news_pln[i-1] + rng.normal(0, .0037)
        eur[i], pln[i] = eur[i-1] + r_eur, pln[i-1] + r_pln
    rates = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                          "eurusd": np.exp(eur), "eurpln": np.exp(pln)})
    rate_p = Path(rates_path)
    event_p = Path(events_path)
    rate_p.parent.mkdir(parents=True, exist_ok=True)
    event_p.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(rate_p, index=False)
    ts = (idx.tz_localize(TZ) + pd.Timedelta(hours=14)).tz_convert("UTC")
    rows = []
    for i in range(n):
        t = ts[i].isoformat()
        rows.extend([
            {"available_at": t, "feature": "mood_eur", "value": news[i]},
            {"available_at": t, "feature": "mood_usd", "value": 0.},
            {"available_at": t, "feature": "mood_pln", "value": news_pln[i]},
            {"available_at": t, "feature": "eurusd__spread_bps", "value": 1.5 + abs(news[i])},
            {"available_at": t, "feature": "eurusd__bid_size", "value": 5 + abs(news[i])},
            {"available_at": t, "feature": "eurusd__ask_size", "value": 5 + abs(news_pln[i])},
            {"available_at": t, "feature": "eurusd__depth", "value": 100 + abs(news[i])*10},
            {"available_at": t, "feature": "eurpln__spread_bps", "value": 4.},
        ])
    pd.DataFrame(rows).to_csv(event_p, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Offline synthetic end-to-end smoke test (NO market edge)")
    d.add_argument("--out", default="results/demo")
    e = sub.add_parser("fetch-ecb", help="Fetch real ECB USD/PLN reference FX time series")
    e.add_argument("--start", default="2015-01-01")
    e.add_argument("--end")
    e.add_argument("--out", default="data/ecb_fx.csv")
    f = sub.add_parser("fetch-fred-policy", help="Download ECB and Fed policy rates (FRED_API_KEY required)")
    f.add_argument("--start", default="2015-01-01")
    f.add_argument("--end")
    f.add_argument("--out", default="data/fred_policy_events.csv")
    merge = sub.add_parser("merge-events", help="Concatenate several time-stamped feature event CSVs")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    t = sub.add_parser("train", help="Chronological no-trade forecast validation")
    t.add_argument("--rates", required=True)
    t.add_argument("--events")
    t.add_argument("--out", default="results/real")
    t.add_argument("--min-train", type=int, default=260)
    t.add_argument("--test-block", type=int, default=63)
    args = ap.parse_args()
    if args.command == "demo":
        out = Path(args.out)
        make_demo(str(out / "synthetic_rates.csv"), str(out / "synthetic_events.csv"))
        report = train(str(out / "synthetic_rates.csv"), str(out / "synthetic_events.csv"), str(out),
                       synthetic=True)
        print(json.dumps({"label": report["label"], "metrics": report["metrics"],
                          "forecast": report["next_fixing_forecast_log_return"]}, indent=2))
    elif args.command == "fetch-ecb":
        rates = fetch_ecb(args.out, args.start, args.end)
        print(f"Saved {len(rates)} actual ECB fixing observations: {args.out}")
    elif args.command == "fetch-fred-policy":
        ev = fetch_fred_policy(args.out, args.start, args.end)
        print(f"Saved {len(ev)} FRED events with conservative 2-business-day delay: {args.out}")
    elif args.command == "merge-events":
        combined = pd.concat([pd.read_csv(p) for p in args.inputs], ignore_index=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out, index=False)
        print(f"Saved {len(combined)} raw events into {args.out}")
    elif args.command == "train":
        report = train(args.rates, args.events, args.out, args.min_train, args.test_block)
        print(json.dumps({"label": report["label"], "metrics": report["metrics"],
                          "forecast": report["next_fixing_forecast_log_return"]}, indent=2))


if __name__ == "__main__":
    main()
