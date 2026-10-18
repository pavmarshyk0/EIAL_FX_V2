from pathlib import Path
import pandas as pd
import pytest
from quote_backtest import run_backtest, load_quotes, fixing_cutoff


def fixture_files(tmp_path,signal=0.02, quote_after_only=True):
    dates=pd.to_datetime(['2024-07-01','2024-07-02'])
    rates=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'eurusd':[1.0,1.03],'eurpln':[4.1,4.11]})
    pred=pd.DataFrame({'date':['2024-07-01'], 'eurusd_next_logret_pred':[signal],
                       'eurpln_next_logret_pred':[0.]})
    quotes=[]
    for d,bid,ask in [('2024-07-01',1.0000,1.0002),('2024-07-02',1.0300,1.0302)]:
        t=fixing_cutoff(pd.Timestamp(d)).tz_convert('UTC') + pd.Timedelta(seconds=2)
        quotes.append({'timestamp':t.isoformat(),'pair':'eurusd','bid':bid,'ask':ask})
    if quote_after_only:
        # Extreme stale price BEFORE signal should be ignored.
        before=fixing_cutoff(pd.Timestamp('2024-07-01')).tz_convert('UTC')-pd.Timedelta(seconds=2)
        quotes.insert(0,{'timestamp':before.isoformat(),'pair':'eurusd','bid':100.,'ask':101.})
    rp=tmp_path/'rates.csv'; pp=tmp_path/'pred.csv'; qp=tmp_path/'quote.csv'
    rates.to_csv(rp,index=False); pred.to_csv(pp,index=False); pd.DataFrame(quotes).to_csv(qp,index=False)
    return rp,pp,qp


def test_asof_execution_never_fills_before_signal(tmp_path):
    rp,pp,qp=fixture_files(tmp_path)
    r=run_backtest(str(pp),str(rp),str(qp),str(tmp_path/'out'),execution_delay_ms=100,slippage_bps_per_side=0)
    assert r['total_executed_trades']==1
    t=pd.read_csv(tmp_path/'out'/'trades.csv').iloc[0]
    assert t.entry_bid==pytest.approx(1.)
    assert t.net_normalized_bps>0


def test_short_sign(tmp_path):
    rp,pp,qp=fixture_files(tmp_path,signal=-0.02)
    r=run_backtest(str(pp),str(rp),str(qp),str(tmp_path/'out'),slippage_bps_per_side=0)
    assert r['total_executed_trades']==1
    t=pd.read_csv(tmp_path/'out'/'trades.csv').iloc[0]
    assert t.side=='SHORT_BASE'
    assert t.net_normalized_bps<0


def test_quote_wait_limit(tmp_path):
    rp,pp,qp=fixture_files(tmp_path)
    r=run_backtest(str(pp),str(rp),str(qp),str(tmp_path/'out'),execution_delay_ms=100,max_quote_wait_ms=10)
    assert r['total_executed_trades']==0
    assert r['skipped']['eurusd:missing_executable_quote']==1


def test_invalid_quote_rejected(tmp_path):
    f=tmp_path/'bad.csv'
    pd.DataFrame([{'timestamp':'2024-07-01T12:00:00Z','pair':'eurusd','bid':2,'ask':1}]).to_csv(f,index=False)
    with pytest.raises(ValueError,match='invalid quote'):
        load_quotes(str(f))
