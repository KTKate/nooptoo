"""Fetch Alpaca crypto bars (1Hour, 1Day) for all tradable USD pairs + a latest-quote spread sample.
Read-only market data. Uses the shared rate limiter in alpaca_data (RL.wait()).
    python src/fetch_crypto.py bars
    python src/fetch_crypto.py quotes [n_samples]
"""
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A  # noqa: E402

OUT = os.path.join(A.ROOT, "data", "local", "crypto")
BARS = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
QUOTES = "https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes"
STABLE = {"USDC/USD", "USDG/USD", "USDT/USD", "PAXG/USD"}


def symbols():
    A.RL.wait()
    r = requests.get("https://paper-api.alpaca.markets/v2/assets", params={"asset_class": "crypto"},
                     headers=A.HDR, timeout=30).json()
    return sorted(a["symbol"] for a in r if a["symbol"].endswith("/USD") and a["tradable"]
                  and a["symbol"] not in STABLE)


def get(url, params):
    for k in range(8):
        A.RL.wait()
        try:
            r = requests.get(url, params=params, headers=A.HDR, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError("request failed")


def fetch_bars(tf):
    syms = symbols()
    rows = []
    for i in range(0, len(syms), 6):
        grp = syms[i:i + 6]
        tok = None
        while True:
            p = {"symbols": ",".join(grp), "timeframe": tf, "start": "2015-01-01T00:00:00Z", "limit": 10000}
            if tok:
                p["page_token"] = tok
            j = get(BARS, p)
            for s, bl in (j.get("bars") or {}).items():
                for b in bl:
                    rows.append((s, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("n", 0), b.get("vw")))
            tok = j.get("next_page_token")
            if not tok:
                break
        print(tf, grp, len(rows), flush=True)
    df = pd.DataFrame(rows, columns=["symbol", "t", "o", "h", "l", "c", "v", "n", "vw"])
    df["t"] = pd.to_datetime(df["t"], utc=True)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, f"bars_{tf}.parquet"), index=False)


def sample_quotes(n=20, gap=30):
    syms = symbols()
    rows = []
    for k in range(n):
        j = get(QUOTES, {"symbols": ",".join(syms)})
        for s, q in (j.get("quotes") or {}).items():
            rows.append((s, q["t"], q["bp"], q["ap"], q.get("bs"), q.get("as")))
        time.sleep(gap)
    df = pd.DataFrame(rows, columns=["symbol", "t", "bp", "ap", "bs", "as"])
    df.to_parquet(os.path.join(OUT, "quotes_sample.parquet"), index=False)
    df["spr_bps"] = (df.ap - df.bp) / ((df.ap + df.bp) / 2) * 1e4
    print(df.groupby("symbol").spr_bps.median().sort_values().to_string())


if __name__ == "__main__":
    if sys.argv[1] == "bars":
        fetch_bars("1Day")
        fetch_bars("1Hour")
    else:
        sample_quotes(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
