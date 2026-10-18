#!/usr/bin/env python3
"""Normalize externally downloaded Dukascopy-node tick CSV into timestamp,pair,bid,ask.

No historical market prices are bundled. Command is deterministic and offline.
Example: python dukascopy_to_quotes.py --input raw_eurusd.csv --pair eurusd --out data/eurusd_quotes.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def convert(path, pair, out):
    raw = pd.read_csv(path)
    names={c.lower(): c for c in raw.columns}
    def find(*opts):
        for key in opts:
            if key.lower() in names:
                return names[key.lower()]
        raise ValueError(f'Cannot find any {opts}; columns {list(raw.columns)}')
    t=find('timestamp','time','datetime')
    b=find('bidPrice','bid')
    a=find('askPrice','ask')
    # Numeric epoch milliseconds or explicitly timezone-aware timestamps.
    ts = raw[t]
    if pd.api.types.is_numeric_dtype(ts):
        dt=pd.to_datetime(ts,unit='ms',utc=True,errors='raise')
    else:
        s=ts.astype(str)
        if not s.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True,case=False).all():
            raise ValueError('Explicit UTC offset is required for non-numeric timestamps')
        dt=pd.to_datetime(s,utc=True,errors='raise')
    ans=pd.DataFrame({'timestamp':dt.dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                      'pair':pair.lower().replace('/',''),
                      'bid':pd.to_numeric(raw[b],errors='raise'),
                      'ask':pd.to_numeric(raw[a],errors='raise')})
    if not (ans.bid.gt(0)&ans.ask.ge(ans.bid)).all():
        raise ValueError('Invalid bid/ask or negative bid prices')
    outp=Path(out)
    outp.parent.mkdir(parents=True,exist_ok=True)
    ans.to_csv(outp,index=False)
    return len(ans)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--pair',required=True,choices=['eurusd','eurpln','usdpln'])
    p.add_argument('--out',required=True)
    arg=p.parse_args()
    print('Saved rows:',convert(arg.input,arg.pair,arg.out))
