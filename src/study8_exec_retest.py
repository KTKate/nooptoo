"""Study 8: re-test the study-1 leads with executable timing, using 5-minute Alpaca snapshots
(09:30-10:00 and 15:30-16:00 bars for the whole liquid universe), 2024-01 .. 2026-09.

Daily-bar backtests assume two things that cannot be done:
  * "close:" rules compute the signal from the closing print and also trade in the closing auction;
    market-on-close orders must be sent by ~15:50. Here the signal uses the 15:45 price (close of the
    15:40-15:45 bar) and the day's data up to then; entry is the official close (Yahoo close = closing cross).
  * "open:" rules compute the gap from the opening print and trade at that print. Here entry is at 09:35
    (close of the first 5-minute bar) or 10:00, with the quoted spread at 09:35 charged.
Exits for overnight holds: opening auction (market-on-open; Yahoo open = opening cross in the median, +2.5 bp
per side for the measured average gap), the first trade with the opening spread charged, the 09:35 price,
or the 09:30-09:35 bar VWAP (a VWAP order over the first five minutes).
Universe tiers as in study 1 (M: $5-50M ADV, S: $1-5M ADV; L: > $50M). Top-10 long-only equal weight,
plus the short side for the gap-up fade (needs margin and borrow).
Output: results/study8_exec_retest.csv
"""
import os
import numpy as np
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols, exec_cost_bps, half_spread_panel, ann_stats, RES
import bt


def snap_panels():
    ndone = len(A.done_pairs("m5snap"))                      # cache key: number of fetched days
    fn = os.path.join(A.LOCAL, f"m5snap_panels_{ndone}.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    d = A.read("m5snap")
    d["date"] = d.ts.dt.normalize()
    d["hm"] = d.ts.dt.strftime("%H:%M")
    out = {}
    for hm in ["09:30", "09:35", "09:55", "15:40", "15:55"]:
        s = d[d.hm == hm]
        for k in ["o", "c", "vwap", "v"]:
            out[f"{k}{hm}"] = s.pivot(index="date", columns="ticker", values=k)
    # consistency with the Yahoo daily store: a few tickers are adjusted differently (splits, reused symbols);
    # drop stock-days where the Alpaca 09:30 open or 15:55 close is more than 15% away from Yahoo's open/close
    from core import load_panel
    P = load_panel()
    raw = P["rawc"].reindex(index=out["o09:30"].index, columns=out["o09:30"].columns)
    yo = (P["o"] / (P["c"] / P["rawc"])).reindex_like(raw)
    r1 = np.log(out["o09:30"].reindex_like(raw) / yo).abs()
    r2 = np.log(out["c15:55"].reindex_like(raw) / raw).abs()
    ok = (r1 < np.log(1.15)) & (r2 < np.log(1.15))
    # tickers whose Yahoo history carries spin-off / stock-dividend adjustments that Alpaca's split
    # adjustment lacks (e.g. CMCSA, SPGI, LEN, FNF, ILMN): the 15:55 close and the Yahoo close differ by a
    # constant factor for months. Mixing the two sources would create fake gaps, so drop those tickers.
    lr = np.log(out["c15:55"].reindex_like(raw) / raw)
    monthly = lr.groupby(lr.index.to_period("M")).median().abs()
    bad = monthly.columns[(monthly > 0.005).any()]
    print("snap consistency: tickers dropped for adjustment mismatch", len(bad))
    ok.loc[:, bad] = False
    print("snap consistency: dropped share of stock-days", float(1 - ok.values[raw.notna().values].mean()))
    for k in out:
        out[k] = out[k].reindex_like(raw).where(ok)
    pd.to_pickle(out, fn)
    return out


if __name__ == "__main__":
    P = load_panel()
    S = snap_panels()
    days = S["c09:30"].index
    tick = [t for t in S["c09:30"].columns if t in P["c"].columns and t in stock_cols(P)]
    rs = lambda x: x.reindex(index=days, columns=tick)
    raw = rs(P["rawc"])
    adjf = rs(P["c"]) / raw                                   # dividend factor to put raw prices on the adjusted scale
    c = raw                                                   # official close (raw, split-adjusted)
    o = rs(P["o"]) / adjf                                     # Yahoo open (first trade), raw scale
    p935 = rs(S["c09:30"]); vw930 = rs(S["vwap09:30"]); p1000 = rs(S["c09:55"])
    p1545 = rs(S["c15:40"])
    dv = rs(P["dv"])
    adv = rs(P["dv"].rolling(20, min_periods=10).median().shift(1))
    px = raw.shift(1)
    tiers = {"L": (px > 5) & (adv > 5e7), "M": (px > 5) & (adv > 5e6) & (adv <= 5e7),
             "S": (px > 2) & (adv > 1e6) & (adv <= 5e6)}
    hs_open = rs(half_spread_panel(P, "09:35"))
    hs_close = rs(half_spread_panel(P, "15:45"))
    c_auc = rs(exec_cost_bps(P, "auction"))
    fees = 0.3
    # overnight returns on the raw scale need the dividend (ex-date) adjustment: ratio of factors
    divstep = adjf.shift(-1) / adjf          # >1 when t+1 is an ex-date (price drops by the dividend)
    nightY = o / c.shift(1)                  # used only for the 20-day overnight momentum signal
    on_mom20 = np.log(nightY * (adjf / adjf.shift(1))).rolling(20, min_periods=15).mean()
    # signals at 15:45 (day t): only data up to 15:45
    sig = {
        "overnight_mom20": on_mom20,                                  # no day-t close needed
        "intraday_loser_overnight_1545": -(p1545 / o - 1),
        "intraday_loser_overnight_close": -(c / o - 1),               # the unexecutable daily-bar version
        "day_loser_overnight_1545": -(p1545 / c.shift(1) - 1),
        "day_loser_overnight_close": -(c / c.shift(1) - 1),
        "day_winner_overnight_1545": p1545 / c.shift(1) - 1,
        "day_winner_overnight_close": c / c.shift(1) - 1,
    }
    exits = {"open_auction": (o.shift(-1), c_auc.shift(-1) + 2.5),
             "yahoo_open_cont": (o.shift(-1), hs_open.shift(-1) + 2 + fees),
             "p0935": (p935.shift(-1), hs_open.shift(-1) + 2 + fees),
             "vwap0930_0935": (vw930.shift(-1), hs_open.shift(-1) + 1 + fees)}
    rows = []

    def rec(name, tier, exitname, r, extra={}):
        for per, a, b in [("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
            x = r.loc[a:b]
            st = ann_stats(x["net"])
            rows.append(dict(rule=name, tier=tier, exit=exitname, period=per, net_sharpe=st["sharpe"],
                             net_annret=st["ann_ret"], maxdd=st["maxdd"], gross_bps=1e4 * x.gross.mean(),
                             cost_bps=1e4 * x.cost.mean(), **extra))

    for tn, E in tiers.items():
        for sn, s in sig.items():
            W = bt.select_topk(s, E & s.notna() & p935.shift(-1).notna(), 10)
            for en, (xp, xc) in exits.items():
                R = xp * divstep / c - 1
                cost = (c_auc + xc) / 2
                rec(sn, tn, en, bt.run(W, R, cost, roundtrip=True))
        # open rules with delayed entry, exit at the close (auction)
        gap = o / c.shift(1) / (adjf.shift(1) / adjf) - 1
        vol20 = np.log(rs(P["c"]) / rs(P["c"]).shift(1)).rolling(20, min_periods=15).std().shift(1)
        g935 = p935 / c.shift(1) / (adjf.shift(1) / adjf) - 1       # gap measured at 09:35
        for sn, s in {"gap_down_fade_volnorm": -g935 / vol20, "gap_down_fade": -g935, "gap_up_fade_short": g935}.items():
            for en, ep, hs in [("p0935", p935, hs_open), ("p1000", p1000, hs_open)]:
                W = bt.select_topk(s, E & s.notna() & ep.notna(), 10)
                R = c / ep - 1
                cost = (hs + 2 + fees + c_auc) / 2
                side = -1 if sn.endswith("short") else 1
                rec(sn, tn, "entry_" + en, bt.run(W, R, cost, roundtrip=True, side=side))
        print(tn, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study8_exec_retest.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(df.pivot_table(index=["rule", "tier", "exit"], columns="period",
                         values=["gross_bps", "cost_bps", "net_sharpe"]).round(2))
