"""Shared data fetch and cost helpers for studies 61 and 62 (read-only Alpaca market data, SIP, split-adjusted
as of the fetch date, bar START times in New York time).

  python src/s6162_common.py m30    # study 61: 30-minute bars for 09:30-10:00 and 15:30-16:00, the 500 most traded
                                    #   stocks of each day (lagged 20d median dollar volume, price > $5), 2020-01 ..
                                    #   2023-12 -> data/local/m30s61/  (2024+ comes from the existing m5snap set)
  python src/s6162_common.py gaps   # study 62: 5-minute bars 07:00-09:50, 10:25-10:30 and 11:55-12:00 for every
                                    #   (stock, day) with an opening gap >= 2% in the liquid universe (ADV > $20M,
                                    #   price > $5), 2020-01 .. latest -> data/local/m5s62/

Alpaca paginates multi-symbol bar requests by about 10,000 underlying 1-minute bars, so the cost of a fetch is
proportional to symbol-minutes; hence narrow windows and per-day ticker lists. Yahoo class-share tickers (BRK-B) are
requested in Alpaca form (BRK.B) and stored in Alpaca form; readers map back. Progress is tracked per day in
done_<ds>.parquet (column 'month' holds the day), so a fetch can be re-run after an interruption.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
from core import load_panel, stock_cols

A.RL = A.RateLimiter(150)          # leave headroom for any other process sharing the account limit


def traded_price(P):
    """Price as traded on each day: Yahoo's 'raw' close is adjusted for every later split (a later 1:50 reverse
    split makes 2020 prices 50x too high), so multiply back the splits dated after each day
    (data/local/events/splits_yf.csv, yfinance, ratio = new shares per old share)."""
    from core import DATA
    s = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
    days = P["rawc"].index
    fac = pd.DataFrame(1.0, index=days, columns=P["rawc"].columns)
    for t, g in s[s.ticker.isin(fac.columns)].groupby("ticker"):
        f = np.ones(len(days))
        for d, r in zip(g.date, g.ratio):
            f[days < d] *= r
        fac[t] = f
    return (P["rawc"] * fac).astype("float32")


def with_traded_price(P):
    """Copy of the panel dict whose 'rawc' is the traded price (dollar volume 'dv' is unchanged: it is already
    rawc x split-adjusted volume), so the core cost model sees real price levels."""
    Q = dict(P)
    Q["rawc"] = traded_price(P)
    return Q


def liquid_masks(P):
    cols = stock_cols(P)
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
    px = P["rawc"][cols].shift(1)
    return cols, adv, px


def fetch_daylists(ds, daylists, windows, timeframe, workers=3, chunk=200):
    """daylists: {day (Timestamp): [alpaca symbols]}. Fetch `windows` (list of (HH:MM, HH:MM)) for those symbols."""
    dn = A.done_pairs(ds)
    have = set(dn.month)
    todo = {d: v for d, v in daylists.items() if d.strftime("%Y-%m-%d") not in have and len(v)}
    print(ds, "days to fetch", len(todo), "symbol-days", sum(len(v) for v in todo.values()), flush=True)
    by_month = {}
    for d in sorted(todo):
        by_month.setdefault(d.strftime("%Y-%m"), []).append(d)
    for m, dl in by_month.items():
        t0 = time.time()
        jobs = []
        for d in dl:
            tk = sorted(todo[d])
            for wa, wb in windows:
                ta = pd.Timestamp(f"{d.date()} {wa}").tz_localize("America/New_York").tz_convert("UTC")
                tb = pd.Timestamp(f"{d.date()} {wb}").tz_localize("America/New_York").tz_convert("UTC")
                for i in range(0, len(tk), chunk):
                    jobs.append((tk[i:i + chunk], ta.strftime("%Y-%m-%dT%H:%M:%SZ"), tb.strftime("%Y-%m-%dT%H:%M:%SZ")))
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(lambda j: A.bars(j[0], timeframe, j[1], j[2]), jobs))
        parts = [p for p in parts if len(p)]
        os.makedirs(os.path.join(A.LOCAL, ds), exist_ok=True)
        fn = os.path.join(A.LOCAL, ds, f"{m}.parquet")
        if parts:
            new = pd.concat(parts, ignore_index=True)
            if os.path.exists(fn):
                new = pd.concat([pd.read_parquet(fn), new], ignore_index=True)
            new = new.drop_duplicates(["ts", "ticker"], keep="last")
            new.sort_values(["ticker", "ts"]).to_parquet(fn, compression="zstd", index=False)
        dn = pd.concat([dn, pd.DataFrame({"ticker": "*", "month": [d.strftime("%Y-%m-%d") for d in dl]})],
                       ignore_index=True)
        dn.to_parquet(A._done_fn(ds), index=False)
        print(ds, m, sum(len(p) for p in parts), f"{time.time() - t0:.0f}s", flush=True)


def m30():
    P = load_panel()
    cols, adv, px = liquid_masks(P)
    rk = adv.where(px > 5).rank(axis=1, ascending=False)
    top = (rk <= 500).loc["2020-01-01":"2023-12-31"]
    daylists = {d: [t.replace("-", ".") for t in row.index[row.values]] for d, row in top.iterrows()}
    fetch_daylists("m30s61", daylists, [("09:30", "10:00"), ("15:30", "16:00")], "30Min")


def gap_mask(P):
    cols, adv, px = liquid_masks(P)
    gap = P["o"][cols] / P["c"][cols].shift(1) - 1
    return ((adv > 2e7) & (px > 5) & (gap.abs() >= 0.02)).loc["2020-01-01":]


def gaps():
    P = load_panel()
    m = gap_mask(P)
    m = m[m.index <= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
    daylists = {d: [t.replace("-", ".") for t in row.index[row.values]] for d, row in m.iterrows()}
    fetch_daylists("m5s62", daylists, [("07:00", "09:50"), ("10:25", "10:30"), ("11:55", "12:00")], "5Min", workers=6)


# ------------------------------------------------------------------ costs at any time of day
# time-of-day coefficients of the quote-calibrated spread model (results/spread_model.json); 15:45 is the base.
# Times between the calibration points are interpolated linearly in minutes (10:00 and 10:30 are therefore
# charged more than a fitted curve that decays quickly after the open would give: conservative).
_TOD = [(9 * 60 + 31, "t09:31"), (9 * 60 + 35, "t09:35"), (12 * 60, "t12:00"), (15 * 60 + 45, None)]


def tod_offset(hm):
    import json
    from core import RES
    m = json.load(open(os.path.join(RES, "spread_model.json")))
    x = [a for a, _ in _TOD]
    y = [m[k] if k else 0.0 for _, k in _TOD]
    h, mi = map(int, hm.split(":"))
    return float(np.interp(h * 60 + mi, x, y))


def cont_cost_bps(P, hm, extra=2.5):
    """Per-side cost (bps) of a marketable order at time hm in continuous trading, known before the day:
    exec_cost_bps-style (quoted half-spread at hm + 2 bp slippage + 0.3 bp fees) + `extra` bp."""
    import core
    core.half_spread_model(1e8, 50.0, 0.02)                    # loads core._SPREAD_MODEL
    core._SPREAD_MODEL.setdefault(f"t{hm}", tod_offset(hm))    # interpolated time-of-day term
    hs = core.half_spread_panel(P, hm)
    return (hs + 2.0 + 0.3 + extra).astype("float32")


def auction_cost_bps(P, extra=2.5):
    from core import exec_cost_bps
    return exec_cost_bps(P, "auction") + extra


if __name__ == "__main__":
    {"m30": m30, "gaps": gaps}[sys.argv[1]]()
