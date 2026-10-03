"""Shared data fetch and cost helpers for studies 61 and 62.

Data (read-only Alpaca market data, SIP, split-adjusted as of the fetch date).

  python src/s6162_common.py m30     # study 61: 30-minute bars 09:30-16:00, liquid stocks (top 500 by ADV on
                                    #   any day 2020-01..latest), every trading day -> data/local/m30s61/
  python src/s6162_common.py gaps    # study 62: 5-minute bars 07:00-16:00 for every (stock, day) with an
                                    #   opening gap >= 2% in the liquid universe -> data/local/m5s62/

Yahoo class-share tickers (BRK-B) are requested in Alpaca form (BRK.B) and stored in Alpaca form; readers map back.
Both fetches are incremental (done_<ds>.parquet) and can be re-run after an interruption.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
from core import load_panel, stock_cols

A.RL = A.RateLimiter(150)          # leave headroom for any other process sharing the account limit


def liquid_masks(P):
    cols = stock_cols(P)
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
    px = P["rawc"][cols].shift(1)
    return cols, adv, px


def m30():
    P = load_panel()
    cols, adv, px = liquid_masks(P)
    rk = adv.where(px > 5).rank(axis=1, ascending=False)
    tick = sorted((rk.loc["2020-01-01":] <= 500).any().pipe(lambda s: s[s].index))
    tick = [t.replace("-", ".") for t in tick]
    print("m30 tickers", len(tick), flush=True)
    A.fetch_windows("m30s61", start="2020-01", end=P["c"].index[-1].strftime("%Y-%m"),
                    windows=(("09:30", "16:00"),), tickers=tick, timeframe="30Min", chunk=700, workers=3)


def gaps():
    P = load_panel()
    cols, adv, px = liquid_masks(P)
    gap = P["o"][cols] / P["c"][cols].shift(1) - 1
    m = (adv > 2e7) & (px > 5) & (gap.abs() >= 0.02)
    m = m.loc["2020-01-01":]
    s = m.stack()
    s = s[s]
    pairs = [(t.replace("-", "."), d) for d, t in s.index]
    print("gap pairs", len(pairs), flush=True)
    A.fetch_pairs(pairs, ds="m5s62", timeframe="5Min", workers=3, a="07:00", b="16:00")


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
