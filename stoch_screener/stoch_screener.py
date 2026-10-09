"""Nifty 500 multi-timeframe Stochastic screener (14,3,3).

Lists stocks whose smoothed %K is below a threshold on every timeframe:
15m, 1h, daily, weekly, monthly. Data: NSE constituent list + Yahoo Finance.
"""
import io
import sys

import pandas as pd
import requests
import yfinance as yf

K_LEN, K_SMOOTH, D_SMOOTH = 14, 3, 3
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# timeframe label -> (yfinance interval, history period)
TIMEFRAMES = {
    "15m": ("15m", "30d"),
    "1H": ("60m", "90d"),
    "D": ("1d", "1y"),
    "W": ("1wk", "5y"),
    "M": ("1mo", "max"),
}


def stochastic(df):
    low_min = df["Low"].rolling(K_LEN).min()
    high_max = df["High"].rolling(K_LEN).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    k = raw_k.rolling(K_SMOOTH).mean()
    d = k.rolling(D_SMOOTH).mean()
    return k, d


def nifty500_symbols():
    resp = requests.get(NIFTY500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))[["Symbol", "Company Name", "Industry"]]


def scan(tickers, interval, period):
    data = yf.download(tickers, interval=interval, period=period, group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t].dropna(subset=["High", "Low", "Close"])
        except KeyError:
            continue
        if len(df) < K_LEN + K_SMOOTH + D_SMOOTH:
            continue
        k, d = stochastic(df)
        out[t] = (round(k.iloc[-1], 2), round(d.iloc[-1], 2), df.index[-1])
    return out


def main():
    meta = nifty500_symbols()
    tickers = [s + ".NS" for s in meta["Symbol"]]
    rows = {t: {} for t in tickers}
    for label, (interval, period) in TIMEFRAMES.items():
        print(f"Scanning {label} ...", file=sys.stderr)
        res = scan(tickers, interval, period)
        for t, (k, d, ts) in res.items():
            rows[t][f"%K {label}"] = k
            rows[t][f"%D {label}"] = d
            rows[t][f"bar {label}"] = ts

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = table.index.str.removesuffix(".NS")
    table = meta.set_index("Symbol").join(table)
    table.to_csv("stoch_all_nifty500.csv")

    k_cols = [f"%K {tf}" for tf in TIMEFRAMES]
    hits = table.dropna(subset=k_cols)
    hits = hits[(hits[k_cols] < THRESHOLD).all(axis=1)].sort_values("%K D")
    cols = ["Company Name", "Industry"] + [c for tf in TIMEFRAMES for c in (f"%K {tf}", f"%D {tf}")]
    hits[cols].to_csv("stoch_below_threshold.csv")

    missing = table[table[k_cols].isna().any(axis=1)]
    print(f"Universe: {len(table)}  |  incomplete data: {len(missing)}  |  "
          f"%K < {THRESHOLD} on all of {list(TIMEFRAMES)}: {len(hits)}")
    print(hits[["Company Name"] + k_cols].to_string())
    for tf in TIMEFRAMES:
        print(f"  latest {tf} bar: {table[f'bar {tf}'].dropna().max()}  "
              f"| %K<{THRESHOLD} on {tf} alone: {(table[f'%K {tf}'] < THRESHOLD).sum()}")


if __name__ == "__main__":
    main()
