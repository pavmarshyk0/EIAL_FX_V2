"""A strictly allowlisted numerical factor DSL. Every operation uses data <= t."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd

PAIRS = ("eurusd", "eurpln", "usdpln")
FAMILIES = ("momentum", "zscore", "vol_ratio", "acceleration", "cross_difference")
LOOKBACKS = (3, 5, 10, 20, 40)


def validated(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("candidate must be object")
    family, source = spec.get("family"), spec.get("source")
    if family not in FAMILIES or source not in PAIRS:
        raise ValueError("unsupported factor family or source")
    clean = {"family": family, "source": source}
    if family == "vol_ratio":
        short, long = spec.get("short"), spec.get("long")
        if type(short) is not int or type(long) is not int or short not in (5, 10) or long not in (20, 40) or short >= long:
            raise ValueError("invalid vol window")
        clean.update(short=short, long=long)
    else:
        window = spec.get("lookback")
        if type(window) is not int or window not in LOOKBACKS:
            raise ValueError("invalid lookback")
        clean["lookback"] = window
    if family == "cross_difference":
        other = spec.get("other")
        if other not in PAIRS or other == source:
            raise ValueError("invalid cross pair")
        clean["other"] = other
    # Never accept arbitrary Python, expressions, or labels from an LLM.
    thesis = str(spec.get("thesis", ""))[:240]
    clean["thesis"] = thesis
    stable = {k: v for k, v in clean.items() if k != "thesis"}
    clean["id"] = "H_" + hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:10]
    return clean


def default_hypotheses(limit: int = 36) -> list[dict]:
    proposals = []
    for source in PAIRS:
        for family in ("momentum", "zscore", "acceleration"):
            for lookback in LOOKBACKS:
                proposals.append({"family": family, "source": source, "lookback": lookback,
                                  "thesis": "Historical returns may reveal regime-dependent FX serial structure."})
        for short, long in ((5, 20), (10, 20), (5, 40), (10, 40)):
            proposals.append({"family": "vol_ratio", "source": source, "short": short, "long": long,
                              "thesis": "Volatility compression or expansion may precede repricing."})
        for other in PAIRS:
            if other != source:
                for lookback in (3, 5, 10, 20):
                    proposals.append({"family": "cross_difference", "source": source, "other": other,
                                      "lookback": lookback,
                                      "thesis": "Cross-pair asynchronous returns may encode lagged information."})
    unique = {}
    for p in proposals:
        v = validated(p)
        unique[v["id"]] = v
    return list(unique.values())[:limit]


def factor_series(log_prices: pd.DataFrame, spec: dict) -> pd.Series:
    spec = validated(spec)
    r = log_prices.diff()
    source, family = spec["source"], spec["family"]
    if family == "momentum":
        out = log_prices[source].diff(spec["lookback"])
    elif family == "zscore":
        w = spec["lookback"]
        mean = log_prices[source].rolling(w, min_periods=w).mean()
        std = log_prices[source].rolling(w, min_periods=w).std().replace(0, np.nan)
        out = (log_prices[source] - mean) / std
    elif family == "acceleration":
        w = spec["lookback"]
        out = r[source].rolling(w, min_periods=w).mean().diff()
    elif family == "vol_ratio":
        s, l = spec["short"], spec["long"]
        volshort = r[source].rolling(s, min_periods=s).std()
        vollong = r[source].rolling(l, min_periods=l).std().replace(0, np.nan)
        out = volshort / vollong
    elif family == "cross_difference":
        w = spec["lookback"]
        out = r[source].rolling(w, min_periods=w).sum() - r[spec["other"]].rolling(w, min_periods=w).sum()
    else:
        raise ValueError("unexpected factor")
    return out.replace([np.inf, -np.inf], np.nan).rename(spec["id"])


def base_features(logp: pd.DataFrame) -> pd.DataFrame:
    ret = logp.diff()
    x = pd.DataFrame(index=logp.index)
    for p in ("eurusd", "eurpln"):  # baseline only direct pairs; USD/PLN derived
        x[f"{p}_ret1"] = ret[p]
        x[f"{p}_ret5"] = ret[p].rolling(5).sum()
        x[f"{p}_vol20"] = ret[p].rolling(20).std()
    return x
