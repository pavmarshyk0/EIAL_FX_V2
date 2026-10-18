import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eialfx import (asof_events, feature_matrix, make_demo, read_rates,
                    train, fetch_ecb, _fix_time)


def _rates(tmp_path):
    days = pd.bdate_range("2022-01-03", periods=500)
    x = pd.DataFrame({"date": days,
                      "eurusd": 1.1 + np.arange(500)*0.0001,
                      "eurpln": 4.4 + np.arange(500)*0.0002})
    p = tmp_path / "rates.csv"
    x.to_csv(p, index=False)
    return read_rates(p)


def test_triangle_identity(tmp_path):
    r = _rates(tmp_path)
    assert np.allclose(r.eurpln.to_numpy(), (r.eurusd * r.usdpln).to_numpy())
    x, y, q, meta = feature_matrix(r, pd.DataFrame(index=r.index))
    assert q is None
    assert np.allclose((np.log(r.usdpln).diff().shift(-1)).iloc[20:-1],
                       (y.eurpln_next_logret - y.eurusd_next_logret).iloc[20:-1])
    assert not any("signed_flow" in c for c in x)


def test_timezones_and_future_events_do_not_leak(tmp_path):
    r = _rates(tmp_path)
    target_day = r.index[20]
    event_date = target_day.strftime("%Y-%m-%d")
    events = pd.DataFrame([
        {"available_at": event_date + "T15:00:00+01:00", "feature": "mood_eur", "value": 1},
        {"available_at": event_date + "T17:00:00+01:00", "feature": "mood_eur", "value": 999},
    ])
    p = tmp_path / "events.csv"
    events.to_csv(p, index=False)
    observed = asof_events(r, p)
    assert observed.loc[target_day, "mood_eur"] == 1
    assert observed.loc[r.index[21], "mood_eur"] == 999
    events.loc[0, "available_at"] = event_date + "T15:00:00"
    events.to_csv(p, index=False)
    with pytest.raises(ValueError, match="timezone"):
        asof_events(r, p)


def test_future_target_not_current(tmp_path):
    r = _rates(tmp_path)
    x, y, *_ = feature_matrix(r, pd.DataFrame(index=r.index))
    target = float(np.log(r.eurusd.iloc[101] / r.eurusd.iloc[100]))
    assert y.iloc[100].eurusd_next_logret == pytest.approx(target)
    assert x.iloc[100].eurusd_ret1 != pytest.approx(target, abs=1e-9)


def test_full_offline_demo(tmp_path):
    data = tmp_path / "input"
    make_demo(str(data / "rates.csv"), str(data / "events.csv"), n=475)
    report = train(str(data / "rates.csv"), str(data / "events.csv"),
                   str(tmp_path / "output"), min_train=260, block=63, synthetic=True)
    assert "SYNTHETIC" in report["label"]
    assert "price_baseline" in report["metrics"]
    assert "all_observed_factors" in report["metrics"]
    preds = pd.read_csv(tmp_path / "output" / "oos_predictions.csv")
    assert np.allclose(preds.usdpln_next_logret_pred,
                       preds.eurpln_next_logret_pred - preds.eurusd_next_logret_pred)
    assert Path(tmp_path / "output" / "report.json").exists()


def test_ecb_api_parser_with_mock(monkeypatch, tmp_path):
    import eialfx
    class FakeResponse:
        text = ("CURRENCY,TIME_PERIOD,OBS_VALUE\n"
                "USD,2024-04-10,1.074\nPLN,2024-04-10,4.26\n"
                "USD,2024-04-11,1.076\nPLN,2024-04-11,4.27\n")
        def raise_for_status(self): pass
    monkeypatch.setattr(eialfx.requests, "get", lambda *args, **kwargs: FakeResponse())
    # Need >=80 days for normal read_rates sanity check.
    more = pd.bdate_range('2023-01-01', periods=100)
    FakeResponse.text = "CURRENCY,TIME_PERIOD,OBS_VALUE\n" + "".join(
        f"{curr},{d:%Y-%m-%d},{v}\n"
        for d in more for curr, v in [('USD',1.074),('PLN',4.26)])
    r = fetch_ecb(tmp_path / "ecb.csv", "2023-01-01")
    assert len(r) == 100
    assert r.usdpln.iloc[0] == pytest.approx(4.26/1.074)
