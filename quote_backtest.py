#!/usr/bin/env python3
"""Quote-executable paper backtester for EIAL-FX next-ECB-fixing OOS forecasts.

Does NOT trade ECB fixings. Each signal generated at 16:30 Europe/Brussels
enters at a venue quote AFTER the cutoff, exits at/after 16:30 on next ECB
fixing date. FX quote bps are a normalized single-unit-base trade payoff,
NOT a funded trading portfolio; carry, borrowing costs and margins not modeled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

TZ = 'Europe/Brussels'
PAIRS = ('eurusd', 'eurpln', 'usdpln')


def fixing_cutoff(day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(None).normalize().tz_localize(TZ) + pd.Timedelta(hours=16, minutes=30)


def load_quotes(path: str) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path)
    rename = {'askPrice': 'ask', 'bidPrice': 'bid', 'time': 'timestamp'}
    raw = raw.rename(columns=rename)
    if not {'timestamp','bid','ask','pair'} <= set(raw.columns):
        raise ValueError('quotes require timestamp,pair,bid,ask (UTC timestamps or with timezone offset)')
    t = raw['timestamp']
    if pd.api.types.is_numeric_dtype(t):
        # Dukascopy-node epoch milliseconds, never implicit seconds.
        stamp = pd.to_datetime(t, unit='ms', utc=True, errors='raise')
    else:
        # Do not silently accept timezone-naive input; DST would be ambiguous.
        s = t.astype(str)
        if not s.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', regex=True, case=False).all():
            raise ValueError('quotes timestamp must have UTC offset, Z, or epoch milliseconds')
        stamp = pd.to_datetime(s, utc=True, errors='raise')
    raw['timestamp'] = stamp
    raw['pair'] = raw['pair'].astype(str).str.lower().str.replace('/', '', regex=False)
    for k in ('bid','ask'):
        raw[k] = pd.to_numeric(raw[k], errors='raise')
    if not (raw['bid'].gt(0) & raw['ask'].ge(raw['bid'])).all():
        raise ValueError('invalid quote: need 0 < bid <= ask')
    return {pair: g.sort_values('timestamp').drop_duplicates('timestamp',keep='last').reset_index(drop=True)
            for pair,g in raw.groupby('pair')}


def _first_quote_after(q: pd.DataFrame, moment: pd.Timestamp, max_wait_ms: int):
    ts = q['timestamp']
    i = int(ts.searchsorted(moment, side='left'))
    if i >= len(q):
        return None
    obs = q.iloc[i]
    if (obs['timestamp'] - moment) > pd.Timedelta(milliseconds=max_wait_ms):
        return None
    return obs


def run_backtest(predictions_path: str, rates_path: str, quotes_path: str, out: str,
                 threshold_bps: float=2.0, execution_delay_ms: int=100,
                 max_quote_wait_ms: int=120000, slippage_bps_per_side: float=0.25,
                 fee_bps_per_side: float=0.0, synthetic: bool=False) -> dict:
    if any(x < 0 for x in (threshold_bps,execution_delay_ms,max_quote_wait_ms,slippage_bps_per_side,fee_bps_per_side)):
        raise ValueError('all costs and delays must be non-negative')
    predictions = pd.read_csv(predictions_path)
    if 'date' not in predictions:
        raise ValueError('predictions require date and pair_next_logret_pred columns')
    predictions['date'] = pd.to_datetime(predictions['date'], errors='raise').dt.normalize()
    if predictions['date'].duplicated().any():
        raise ValueError('duplicate signal dates in predictions')
    rates = pd.read_csv(rates_path)
    if 'date' not in rates:
        raise ValueError('rates require date')
    dates = pd.DatetimeIndex(pd.to_datetime(rates['date'],errors='raise').dt.normalize()).sort_values().unique()
    quotes = load_quotes(quotes_path)
    trades, reasons = [], {}
    def skip(reason):
        reasons[reason] = reasons.get(reason,0) + 1
    previous_exit = {}
    for rec in predictions.sort_values('date').to_dict('records'):
        day = rec['date']
        ix = int(dates.searchsorted(day))
        if ix >= len(dates) or dates[ix] != day or ix+1 >= len(dates):
            skip('no_known_next_fixing_date'); continue
        decision = fixing_cutoff(day).tz_convert('UTC')
        next_decision = fixing_cutoff(dates[ix+1]).tz_convert('UTC')
        delayed_entry = decision + pd.Timedelta(milliseconds=execution_delay_ms)
        delayed_exit = next_decision + pd.Timedelta(milliseconds=execution_delay_ms)
        for pair in PAIRS:
            pred = rec.get(f'{pair}_next_logret_pred')
            if pd.isna(pred):
                continue
            strength = float(pred)*10000.0
            if not np.isfinite(strength) or abs(strength) < threshold_bps:
                skip('below_threshold'); continue
            side = 1 if strength > 0 else -1
            q = quotes.get(pair)
            if q is None or q.empty:
                skip(f'{pair}:no_quotes'); continue
            e = _first_quote_after(q, delayed_entry,max_quote_wait_ms)
            z = _first_quote_after(q, delayed_exit,max_quote_wait_ms)
            if e is None or z is None:
                skip(f'{pair}:missing_executable_quote'); continue
            if z['timestamp'] <= e['timestamp']:
                skip(f'{pair}:nonpositive_holding_time'); continue
            if pair in previous_exit and e['timestamp'] < previous_exit[pair]:
                skip(f'{pair}:overlapping_trade'); continue
            entry_mid = (float(e['ask'])+float(e['bid']))/2
            adverse = slippage_bps_per_side / 10000.0
            if side == 1:
                entry_fill = float(e['ask'])*(1+adverse)
                exit_fill = float(z['bid'])*(1-adverse)
                pnl_quote_per_base = exit_fill-entry_fill
            else:
                entry_fill = float(e['bid'])*(1-adverse)
                exit_fill = float(z['ask'])*(1+adverse)
                pnl_quote_per_base = entry_fill-exit_fill
            pnl_bps = pnl_quote_per_base/entry_mid*10000.0 - 2*fee_bps_per_side
            previous_exit[pair] = z['timestamp']
            trades.append({
                'pair': pair,'signal_date':str(day.date()),'signal_time_utc':decision.isoformat(),
                'entry_quote_utc':e['timestamp'].isoformat(),'exit_quote_utc':z['timestamp'].isoformat(),
                'side':'LONG_BASE' if side==1 else 'SHORT_BASE','prediction_bps':strength,
                'entry_bid':float(e['bid']),'entry_ask':float(e['ask']),
                'exit_bid':float(z['bid']),'exit_ask':float(z['ask']),
                'entry_fill':entry_fill,'exit_fill':exit_fill,
                'pnl_quote_per_one_base_unit_before_fee':pnl_quote_per_base,
                'net_normalized_bps':pnl_bps,
            })
    outp = Path(out)
    outp.mkdir(parents=True,exist_ok=True)
    df = pd.DataFrame(trades)
    df.to_csv(outp/'trades.csv',index=False)
    per_pair = {}
    for pair in PAIRS:
        if df.empty or pair not in df['pair'].values:
            continue
        v = df.loc[df.pair==pair,'net_normalized_bps'].to_numpy()
        per_pair[pair] = {
            'n_trades':len(v), 'win_rate_net':float(np.mean(v>0)),
            'mean_net_bps_per_trade':float(np.mean(v)),
            'median_net_bps_per_trade':float(np.median(v)),
            'sum_normalized_bps_NOT_portfolio_return':float(np.sum(v)),
            'min_bps':float(np.min(v)), 'max_bps':float(np.max(v)),
        }
    report = {
        'label': 'SYNTHETIC - NO MARKET EVIDENCE' if synthetic else 'EXECUTABLE QUOTE PAPER SIMULATION - VERIFY SOURCE',
        'forecast_horizon': 'forecast next ECB fixing; position held until AFTER next fixing 16:30 Brussels',
        'execution': 'first observed venue quote >= signal+delay, never at ECB fixing',
        'pairs':per_pair,'skipped':reasons,'total_executed_trades':int(len(df)),
        'settings':{'min_prediction_bps':threshold_bps,'execution_delay_ms':execution_delay_ms,
                    'max_quote_wait_ms':max_quote_wait_ms,'slippage_bps_per_side':slippage_bps_per_side,
                    'fee_bps_per_side':fee_bps_per_side},
        'limitations':["Forex swaps/carry, financing, queue depth, market impact, execution rejection and counterparty rules excluded",
                       "Venue quotes are not proof of guaranteed executable size or globally consolidated FX demand",
                       "Summed trade bps are not portfolio returns; mixed PLN/USD quote-currency PnL cannot be directly summed",
                       "Prediction target is next ECB 16:00 reference fixing, but trade exit is at >= 16:30 on next fixing day",
                       "OOS forecasts and factor vintages must be independently audited for true point-in-time availability",
                       "This script is a single-position-per-pair diagnostic; not a complete portfolio simulator"]
    }
    (outp/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def demo_quotes(rates_path: str, out_path: str, spread_bps: float=1.0):
    """Create FICTIONAL quotes strictly for a smoke test; NEVER real market data."""
    rates=pd.read_csv(rates_path)
    items=[]
    for _,r in rates.iterrows():
        cutoff = fixing_cutoff(pd.Timestamp(r['date'])).tz_convert('UTC') + pd.Timedelta(seconds=1)
        for pair,mid in [('eurusd',float(r['eurusd'])),('eurpln',float(r['eurpln']))]:
            items.append({'timestamp':cutoff.isoformat(),'pair':pair,
                'bid':mid*(1-spread_bps/20000),'ask':mid*(1+spread_bps/20000)})
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(items).to_csv(out_path,index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('run')
    b.add_argument('--predictions',required=True)
    b.add_argument('--rates',required=True)
    b.add_argument('--quotes',required=True)
    b.add_argument('--out',default='results/quote_backtest')
    b.add_argument('--threshold-bps',type=float,default=2.)
    b.add_argument('--execution-delay-ms',type=int,default=100)
    b.add_argument('--max-quote-wait-ms',type=int,default=120000)
    b.add_argument('--slippage-bps-per-side',type=float,default=0.25)
    b.add_argument('--fee-bps-per-side',type=float,default=0.0)
    d=sub.add_parser('demo')
    d.add_argument('--out',default='results/quote_demo')
    args=p.parse_args()
    if args.command=='demo':
        # The core demo command must have produced the synthetic signals already.
        from eialfx import make_demo, train
        out=Path(args.out)
        out.mkdir(parents=True,exist_ok=True)
        rates=out/'synthetic_rates.csv'; events=out/'synthetic_events.csv'
        make_demo(str(rates),str(events))
        train(str(rates),str(events),str(out),synthetic=True)
        quotes=out/'synthetic_quotes.csv'
        demo_quotes(str(rates),str(quotes))
        result=run_backtest(str(out/'oos_predictions.csv'),str(rates),str(quotes),str(out),synthetic=True)
    else:
        result=run_backtest(args.predictions,args.rates,args.quotes,args.out,
            args.threshold_bps,args.execution_delay_ms,args.max_quote_wait_ms,
            args.slippage_bps_per_side,args.fee_bps_per_side)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
