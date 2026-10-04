"""Intraday bar fetches for studies 63 and 64 (read-only Alpaca market data, SIP, split-adjusted as of the fetch
date, bar START times in New York time; same helpers and progress files as src/s6162_common.py).

  python src/s6364_fetch.py m5s63    # study 63: 5-minute bars 10:25 and 10:30 for the 1000 most traded stocks of
                                     #   each day (lagged 20d median dollar volume, traded price > $5),
                                     #   2020-01 .. latest -> data/local/m5s63/
                                     #   price at 10:30 = close of the 10:25 bar; entry = open of the 10:30 bar
  python src/s6364_fetch.py m15s64   # study 64, 2020-23: 15-minute bars 14:45 .. 15:45 for the 500 most traded
                                     #   stocks of each day -> data/local/m15s64/
  python src/s6364_fetch.py m5s64    # study 64, 2024+: 5-minute bars 14:55, 15:00 and 15:25 for the 500 most traded
                                     #   stocks (15:30-16:00 5-minute bars come from the existing m5snap set)

  python src/s6364_fetch.py d1s63    # split-adjusted Alpaca daily bars (1Day) 2020-01 .. latest for every stock ever in
                                     #   the daily top 1000 -> data/local/d1s63.parquet; used to find tickers whose Yahoo
                                     #   close is on a different price scale (spin-off adjustments, e.g. T, GE, MMM)

Alpaca serves 500 symbols per request here (about 10,000 underlying 1-minute bars per page), so chunk = 500.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel
from s6162_common import fetch_daylists, liquid_masks, traded_price


def top_n(P, n, start, end):
    """Boolean day x ticker mask: the n most traded stocks (lagged 20-day median dollar volume) with a traded
    previous close above $5."""
    cols, adv, _ = liquid_masks(P)
    px = traded_price(P)[cols].shift(1)
    rk = adv.where(px > 5).rank(axis=1, ascending=False)
    return (rk <= n).loc[start:end]


def daylists(mask):
    return {d: [t.replace("-", ".") for t in row.index[row.values]] for d, row in mask.iterrows()}


def main(which):
    P = load_panel()
    last = P["c"].index[-1].strftime("%Y-%m-%d")
    if which == "m5s63":
        fetch_daylists("m5s63", daylists(top_n(P, 1000, "2020-01-01", last)), [("10:25", "10:30")], "5Min",
                       workers=2, chunk=500)
    elif which == "m15s64":
        fetch_daylists("m15s64", daylists(top_n(P, 500, "2020-01-01", "2023-12-31")), [("14:45", "15:45")], "15Min",
                       workers=2, chunk=500)
    elif which == "d1s63":
        import alpaca_data as A
        m = top_n(P, 1000, "2020-01-01", last)
        tk = sorted(t.replace("-", ".") for t in m.columns[m.any().values])
        parts = [A.bars(tk[i:i + 200], "1Day", "2020-01-01T00:00:00Z", f"{last}T23:59:00Z") for i in range(0, len(tk), 200)]
        d = pd.concat([p for p in parts if len(p)], ignore_index=True)
        d["ticker"] = d.ticker.str.replace(".", "-", regex=False)
        d["date"] = d.ts.dt.normalize()
        d[["date", "ticker", "o", "c", "v"]].to_parquet(os.path.join(A.LOCAL, "d1s63.parquet"), index=False)
        print("d1s63", len(d), d.ticker.nunique())
    elif which == "m5s64":
        fetch_daylists("m5s64", daylists(top_n(P, 500, "2024-01-01", last)), [("14:55", "15:00"), ("15:25", "15:25")],
                       "5Min", workers=2, chunk=500)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        main(w)
