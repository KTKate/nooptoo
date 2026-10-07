"""Studies 85, 86 and 87: execution timing around the open for the overnight blend (study 33) and its day short
(study 68), 2024-01 .. 2026-09.

Picks: top 10 by (2 x within-day rank of study23 'ensemble' + rank of P(jump)-P(drop) from study33) / 3, exactly as
src/study68_day_short_picks.py. Picked at day t, bought in t's closing auction, sold in t+1's opening auction.
Day short (study 68): the easy-to-borrow picks (Alpaca's current list: flatters the past) shorted on t+1, 10% of
capital per name as in study 68 (the paper job uses 2.5%; bp-of-capital figures scale by 1/4, per-name bp do not).

Intraday data: 1-minute SIP bars 09:30-16:00 (plus the 16:00 minute, which holds the closing cross) for every pick on
its day t+1, fetched here into data/local/m1s85/ (adjustment=split, ts = bar start, New York time), about one
request per day. Price "at hh:mm" = open of the 1-minute bar starting hh:mm (if no trade in that minute: the first
trade after it, within 5 minutes). The opening/closing auction prices come from the daily panel (o, c); bar prices are
put on the same scale through the anchor first-trade-of-the-day = opening auction (R(t) = p_t / first trade, so the
open auction is 1 and the close auction is 1 + r_day, r_day = c/o - 1 from the panel). Pairs whose bars disagree
with the panel (|log(last trade before 16:00 / first trade) - log(1 + r_day)| > 5%) or have no bars fall back to the
auction-to-auction return for every variant (counted in the output as 'bars_missing').

Costs per side: auction trades core.exec_cost_bps(P, 'auction') + 2.5 bp; continuous trades
s6162_common.cont_cost_bps(P, hh:mm) (quoted half-spread at hh:mm, interpolated time-of-day model, + 2.3 + 2.5 bp).
No borrow fee (easy-to-borrow names), as in study 68.

85  Day-short entry at the opening auction (backtest) vs 09:31, 09:35, 09:45, 10:00, and the paper job's actual
    window (average of the 09:31-09:36 one-minute VWAPs, 'paper_0931_36'); cover at the closing auction vs 15:55.
86  Stops for the day short (entry at the opening auction and at 09:35): per-name stop when the price rises 3/5/8%
    above entry (a) checked on every 1-minute bar high, filled at the stop or at the bar open if it gapped through,
    (b) checked on 5-minute bar closes (a job polling every 5 minutes), filled at the next 1-minute bar open; stop
    covers pay the continuous cost at that time. Book stop: cover everything when the book's mark-to-market loss
    reaches 1/2/3% of capital (checked on 5-minute closes).
87  Selling the overnight longs at the opening auction vs at 09:31, 09:35, 09:45, 10:00 and the paper job's market
    orders filling 09:30-09:35 (average of the 09:30-09:34 one-minute VWAPs, 'paper_0930_34').
Periods: 2024-01..2025-06 and 2025-07..2026-09.
Output: results/study85_day_short_timing.csv, results/study86_day_short_stops.csv, results/study87_long_exit_timing.csv

    python src/study85_87_open_execution.py fetch   # bars (incremental)
    python src/study85_87_open_execution.py         # studies
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
from s6162_common import cont_cost_bps, with_traded_price
import bt

A.RL = A.RateLimiter(90)
DS = "m1s85"
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("all", "2024-01", "2026-09")]

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < days[-2]]
pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
W = bt.select_topk(S, S.notna(), 10)
W = W.loc["2024-01-01":]
nxt = pd.Series(days[1:], index=days[:-1])


def pick_list():
    s = W.stack()
    s = s[s > 0].reset_index()
    s.columns = ["pdate", "symbol", "w"]
    s["tdate"] = s.pdate.map(nxt)
    return s.dropna(subset=["tdate"])


# ------------------------------------------------------------------ fetch
def _fetch_day(d, tk):
    a = pd.Timestamp(f"{d.date()} 09:30").tz_localize("America/New_York").tz_convert("UTC")
    b = pd.Timestamp(f"{d.date()} 16:01").tz_localize("America/New_York").tz_convert("UTC")
    p = dict(symbols=",".join(tk), timeframe="1Min", start=a.strftime("%Y-%m-%dT%H:%M:%SZ"),
             end=b.strftime("%Y-%m-%dT%H:%M:%SZ"), limit=10000, adjustment="split", feed="sip", sort="asc")
    rows = []
    while True:
        j = A.get("bars", p)
        for s_, bl in (j.get("bars") or {}).items():
            for x in bl:
                x["ticker"] = s_
            rows.extend(bl)
        if not j.get("next_page_token"):
            break
        p["page_token"] = j["next_page_token"]
    if not rows:
        return pd.DataFrame()
    x = pd.DataFrame.from_records(rows)
    x["ts"] = pd.to_datetime(x.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    x = x.rename(columns={"vw": "vwap"})[["ts", "ticker", "o", "h", "l", "c", "v", "n", "vwap"]]
    for k in ["o", "h", "l", "c", "vwap"]:
        x[k] = x[k].astype("float32")
    return x


def fetch():
    pk = pick_list()
    pk = pk[pk.tdate <= pd.Timestamp.today().normalize() - pd.Timedelta(days=1)]
    dn = A.done_pairs(DS)
    have = set(dn.month)
    todo = {d: sorted(g.symbol.str.replace("-", ".", regex=False)) for d, g in pk.groupby("tdate")
            if d.strftime("%Y-%m-%d") not in have}
    print("days to fetch", len(todo), flush=True)
    by_month = {}
    for d in sorted(todo):
        by_month.setdefault(d.strftime("%Y-%m"), []).append(d)
    os.makedirs(os.path.join(A.LOCAL, DS), exist_ok=True)
    for m, dl in by_month.items():
        t0 = time.time()
        with ThreadPoolExecutor(2) as ex:
            parts = list(ex.map(lambda d: _fetch_day(d, todo[d]), dl))
        parts = [p for p in parts if len(p)]
        fn = os.path.join(A.LOCAL, DS, f"{m}.parquet")
        if parts:
            new = pd.concat(parts, ignore_index=True)
            if os.path.exists(fn):
                new = pd.concat([pd.read_parquet(fn), new], ignore_index=True)
            new.drop_duplicates(["ts", "ticker"], keep="last").sort_values(["ticker", "ts"]).to_parquet(
                fn, compression="zstd", index=False)
        dn = pd.concat([dn, pd.DataFrame({"ticker": "*", "month": [d.strftime("%Y-%m-%d") for d in dl]})],
                       ignore_index=True)
        dn.to_parquet(A._done_fn(DS), index=False)
        print(m, sum(len(p) for p in parts), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch()
    sys.exit()


# ------------------------------------------------------------------ pair table and minute grids
NM = 391                                            # 09:30 .. 16:00 one-minute bars
HM = [f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}" for i in range(NM)]
IX = {h: i for i, h in enumerate(HM)}


def build_pairs():
    pk = pick_list()
    o, c = P["o"][cols], P["c"][cols]
    oi, ci = o.index, o.columns
    ti, pi_ = oi.get_indexer(pk.tdate), oi.get_indexer(pk.pdate)
    jj = ci.get_indexer(pk.symbol)
    ov, cv = o.values, c.values
    pk["night"] = ov[ti, jj] / cv[pi_, jj] - 1
    pk["r_day"] = cv[ti, jj] / ov[ti, jj] - 1
    a = pd.read_parquet(f"{DATA}/local/alpaca_assets_active.parquet")
    etb = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)])
    pk["etb"] = pk.symbol.isin(etb)
    pk = pk[pk.tdate <= pd.Timestamp("2026-09-30")].reset_index(drop=True)
    # grids
    n = len(pk)
    G = {k: np.full((n, NM), np.nan, dtype="float64") for k in ["o", "h", "l", "c", "vw", "v"]}
    key = {(s.replace("-", "."), d): i for i, (s, d) in enumerate(zip(pk.symbol, pk.tdate))}
    for m_ in sorted(pk.tdate.dt.strftime("%Y-%m").unique()):
        fn = os.path.join(A.LOCAL, DS, f"{m_}.parquet")
        if not os.path.exists(fn):
            continue
        d = pd.read_parquet(fn)
        d["tdate"] = d.ts.dt.normalize()
        r = np.array([key.get(k, -1) for k in zip(d.ticker, d.tdate)])
        t = (d.ts.dt.hour.values * 60 + d.ts.dt.minute.values) - (9 * 60 + 30)
        m = (r >= 0) & (t >= 0) & (t < NM)
        for k, src in [("o", "o"), ("h", "h"), ("l", "l"), ("c", "c"), ("vw", "vwap"), ("v", "v")]:
            G[k][r[m], t[m]] = d[src].values[m]
    first = G["o"][:, :6]                                       # first trade within 09:30-09:35
    fidx = np.where(np.isfinite(first).any(1), np.argmax(np.isfinite(first), 1), -1)
    f = np.where(fidx >= 0, G["o"][np.arange(n), np.maximum(fidx, 0)], np.nan)
    cf = pd.DataFrame(G["c"][:, :390]).ffill(axis=1).values          # last trade by end of each minute (pre-16:00)
    last = cf[:, -1]
    pk["bars_ok"] = np.isfinite(f) & np.isfinite(last) & (np.abs(np.log(last / f) - np.log1p(pk.r_day)) < 0.05)
    # price "at" minute k: open of bar k if traded, else previous last trade (ffilled close), else NaN
    prev = np.concatenate([np.full((n, 1), np.nan), cf[:, :-1]], 1)
    prev = np.concatenate([prev, cf[:, -1:]], 1)                        # minute 390 (16:00)
    oo = np.where(np.isfinite(G["o"]), G["o"], prev)
    # if nothing traded yet at minute k (late first trade), use the first later open within 5 minutes
    nb = pd.DataFrame(G["o"]).bfill(axis=1, limit=5).values
    oo = np.where(np.isfinite(oo), oo, nb)
    N = {"o": oo / f[:, None], "h": G["h"] / f[:, None], "c": np.concatenate([cf, cf[:, -1:]], 1) / f[:, None],
         "vw": G["vw"] / f[:, None], "v": G["v"]}
    pk["c1600_vs_close"] = N["o"][:, 390] / (1 + pk.r_day) - 1            # 16:00 bar open (closing cross) vs anchor
    return pk, N


def cost_grid(pk):
    """Per-side costs (decimal) for each pair: auction (on tdate and pdate) and continuous at grid times."""
    PT = with_traded_price(P)
    sym = sorted(set(pk.symbol))
    Q = {k: PT[k].loc["2023-09-01":, sym] for k in ["o", "h", "l", "c", "rawc", "v", "dv"]}
    ti = Q["c"].index.get_indexer(pk.tdate)
    pi_ = Q["c"].index.get_indexer(pk.pdate)
    jj = Q["c"].columns.get_indexer(pk.symbol)
    auc = exec_cost_bps(Q, "auction").values + 2.5
    C = {"auc_t": auc[ti, jj] / 1e4, "auc_p": auc[pi_, jj] / 1e4}
    for hm in ["09:30", "09:31", "09:33", "09:35", "09:45", "10:00", "10:30", "11:00", "12:00", "13:00", "14:00",
               "15:00", "15:45", "15:55"]:
        C[hm] = cont_cost_bps(Q, hm).values[ti, jj] / 1e4
    return C


def cont_cost_at(C, k):
    """Continuous cost at minute index k (array): nearest earlier grid time."""
    grid = [h for h in C if ":" in h]
    gi = np.array([IX[h] for h in grid])
    M = np.stack([C[h] for h in grid], 1)
    pos = np.clip(np.searchsorted(gi, k, side="right") - 1, 0, len(gi) - 1)
    return M[np.arange(len(k)), pos]


def stats_rows(study, variant, per_pair, pk, mask, wname=0.1, extra=None):
    """per_pair: net return (decimal) per pair; book = wname x sum over masked pairs per pick day (cash otherwise)."""
    per_pair = np.asarray(per_pair, dtype="float64")
    mask = np.asarray(mask, dtype=bool) & np.isfinite(per_pair)
    s = pd.Series(np.where(mask, per_pair, 0.0) * wname, index=pk.index).groupby(pk.pdate).sum()
    cnt = pd.Series(mask.astype(float), index=pk.index).groupby(pk.pdate).sum()
    mon = pk.pdate.dt.strftime("%Y-%m").values
    rows = []
    for p, a_, b_ in PER:
        y = s.loc[a_:b_]
        st = ann_stats(y)
        mp = mask & (mon >= a_) & (mon <= b_)
        r = dict(study=study, variant=variant, period=p, days=len(y), names_per_day=cnt.loc[a_:b_].mean(),
                 net_bp_day=1e4 * y.mean(), per_name_net_bp=1e4 * per_pair[mp].mean(), sharpe=st["sharpe"],
                 tstat=st["tstat"], maxdd=st["maxdd"], worst_day_bp=1e4 * y.min(), hit=st["hit"])
        for k, v in (extra or {}).items():
            r[k] = 1e4 * np.nanmean(np.asarray(v, dtype="float64")[mp])
        rows.append(r)
    return rows


def first_hit(cond, start, end):
    """First minute index k in [start, end) with cond[:, k] True; -1 if none."""
    c = cond[:, start:end]
    any_ = c.any(1)
    return np.where(any_, start + np.argmax(c, 1), -1)


def short_with_stop(N, E, start, end, exit_px, exit_cost, s, mode, C):
    """Per-name stop at E x (1 + s). mode 'bar': bar highs, fill max(stop, bar open); 'poll5': 5-minute closes,
    fill at the next minute's open. Returns (gross short return, cost of exit leg, stopped flag)."""
    n = len(E)
    stop = E * (1 + s)
    if mode == "bar":
        k = first_hit(N["h"] >= stop[:, None], start, end)
        fill = np.where(k >= 0, np.maximum(stop, N["o"][np.arange(n), np.maximum(k, 0)]), np.nan)
    else:
        ks = np.arange(start + 4, end - 1, 5)
        hit = N["c"][:, ks] >= stop[:, None]
        any_ = hit.any(1)
        k = np.where(any_, ks[np.argmax(hit, 1)] + 1, -1)
        fill = np.where(k >= 0, N["o"][np.arange(n), np.maximum(k, 0)], np.nan)
    st = k >= 0
    X = np.where(st, fill, exit_px)
    cx = np.where(st, cont_cost_at(C, np.maximum(k, 0)), exit_cost)
    return 1 - X / E, cx, st


def book_stop(pk, N, E, start, end, exit_px, exit_cost, L, mask, C, w=0.1):
    """Cover the whole day-short book when its mark-to-market loss (5-minute closes) reaches L of capital."""
    n = len(E)
    X = exit_px.copy()
    cx = exit_cost.copy()
    st = np.zeros(n, bool)
    ks = np.arange(start + 4, end - 1, 5)
    pnl = w * (1 - N["c"][:, ks] / E[:, None])
    pnl = np.where(mask[:, None] & np.isfinite(pnl), pnl, 0.0)
    for d, idx in pk.groupby("pdate").indices.items():
        bk = pnl[idx].sum(0)
        h = np.nonzero(bk <= -L)[0]
        if len(h):
            k = ks[h[0]] + 1
            X[idx] = N["o"][idx, k]
            cx[idx] = cont_cost_at(C, np.full(len(idx), k))
            st[idx] = True
    return 1 - X / E, cx, st


def main():
    pk, N = build_pairs()
    C = cost_grid(pk)
    n = len(pk)
    ok = pk.bars_ok.values
    print("pairs", n, "bars ok", ok.mean().round(4), "etb share", pk.etb.mean().round(3))
    print("16:00-bar open vs close anchor: median |diff| bp", np.nanmedian(np.abs(pk.c1600_vs_close[ok])) * 1e4,
          "mean bp", np.nanmean(pk.c1600_vs_close[ok]) * 1e4)
    print("mean per-side cost bp (etb picks):", {k: round(1e4 * np.nanmean(v[pk.etb.values]), 1) for k, v in C.items()})
    rday = pk.r_day.values
    nan = np.full(n, np.nan)

    def pt(k):                     # price at minute index k, auction fallback for pairs without usable bars
        return np.where(ok, N["o"][:, k], nan)

    paper_short = np.nanmean(N["c"][:, 1:6], 1)              # closes of the 09:31..09:35 bars
    paper_long = np.nanmean(N["c"][:, 0:5], 1)               # closes of the 09:30..09:34 bars
    ENT = {"open_auction": (np.ones(n), C["auc_t"], 0), "09:31": (pt(1), C["09:31"], 1),
           "09:35": (pt(5), C["09:35"], 5), "09:45": (pt(15), C["09:45"], 15), "10:00": (pt(30), C["10:00"], 30),
           "paper_0931_36": (np.where(ok, paper_short, nan), C["09:33"], 6)}
    EXI = {"close_auction": (1 + rday, C["auc_t"], 390), "15:55": (pt(385), C["15:55"], 385)}
    etb = pk.etb.values
    full = etb & ok                     # every timing variant on the same pairs (bars usable)
    print("etb pairs", etb.sum(), "with bars", full.sum())
    # ------------------------------------------------ study 85
    rows = []
    for en, (E, ce, _) in ENT.items():
        for ex, (X, cx, _) in EXI.items():
            g = 1 - X / E
            net = g - ce - cx
            rows += stats_rows(85, f"short {en} -> {ex}", net, pk, full, extra={"gross_bp": g, "cost_bp": ce + cx})
            if en != "open_auction" or ex != "close_auction":    # timing alone: every leg charged the auction cost
                rows += stats_rows(85, f"short {en} -> {ex} [auction cost on all legs]", g - 2 * C["auc_t"], pk, full,
                                   extra={"gross_bp": g, "cost_bp": 2 * C["auc_t"]})
    # study-68 reproduction on all etb pairs (auction to auction, no bar requirement)
    g = 1 - (1 + rday)
    rows += stats_rows(85, "study68 repro: open_auction -> close_auction, all etb pairs", g - 2 * C["auc_t"], pk, etb,
                       extra={"gross_bp": g, "cost_bp": 2 * C["auc_t"]})
    # day-short P&L split: open -> 09:3x drift that the paper job misses
    for k, lab in [(1, "09:31"), (5, "09:35"), (15, "09:45"), (30, "10:00")]:
        rows += stats_rows(85, f"short drift open_auction -> {lab} (gross only)", 1 - pt(k), pk, full)
    d85 = pd.DataFrame(rows)
    d85.to_csv(f"{RES}/study85_day_short_timing.csv", index=False)
    show = ["variant", "period", "names_per_day", "gross_bp", "cost_bp", "per_name_net_bp", "net_bp_day", "sharpe",
            "maxdd"]
    pd.set_option("display.width", 250)
    print(d85[d85.period != "all"][[c for c in show if c in d85]].round(2).to_string(index=False))
    # ------------------------------------------------ study 86
    rows = []
    cfgs = {"open_auction -> close_auction": ("open_auction", "close_auction"),
            "09:35 -> close_auction": ("09:35", "close_auction"),
            "paper: 0931_36 -> 15:55": ("paper_0931_36", "15:55")}
    for cname, (en, ex) in cfgs.items():
        E, ce, start = ENT[en]
        X, cx, end = EXI[ex]
        g = 1 - X / E
        rows += stats_rows(86, f"{cname} | no stop", g - ce - cx, pk, full, extra={"gross_bp": g})
        for mode in ["bar", "poll5"]:
            for s in [0.03, 0.05, 0.08]:
                g2, cx2, st = short_with_stop(N, E, start, end, X, cx, s, mode, C)
                rows += stats_rows(86, f"{cname} | name stop {int(s * 100)}% {mode}", g2 - ce - cx2, pk, full,
                                   extra={"gross_bp": g2, "stopped_pct": st / 100.0})
        for L in [0.01, 0.02, 0.03]:
            g2, cx2, st = book_stop(pk, N, E, start, end, X, cx, L, full, C)
            rows += stats_rows(86, f"{cname} | book stop {int(L * 100)}% poll5", g2 - ce - cx2, pk, full,
                               extra={"gross_bp": g2, "stopped_pct": st / 100.0})
    d86 = pd.DataFrame(rows)
    d86.to_csv(f"{RES}/study86_day_short_stops.csv", index=False)
    show = ["variant", "period", "gross_bp", "stopped_pct", "per_name_net_bp", "net_bp_day", "sharpe", "maxdd",
            "worst_day_bp"]
    print(d86[[c for c in show if c in d86]].round(3).to_string(index=False))
    # ------------------------------------------------ study 87
    rows = []
    night = pk.night.values
    allp = ok.copy()
    spy = spy_points(pk)
    for lab, (X, cx) in {"open_auction": (np.ones(n), C["auc_t"]), "09:31": (pt(1), C["09:31"]),
                         "09:35": (pt(5), C["09:35"]), "09:45": (pt(15), C["09:45"]), "10:00": (pt(30), C["10:00"]),
                         "paper_0930_34": (np.where(ok, paper_long, nan), C["09:31"])}.items():
        g = (1 + night) * X - 1
        net = g - C["auc_p"] - cx
        ex = {"gross_bp": g, "cost_bp": C["auc_p"] + cx, "drift_after_open_bp": X - 1}
        if lab in spy:
            ex["spy_drift_bp"] = spy[lab]
        rows += stats_rows(87, f"long sell at {lab}", net, pk, allp, extra=ex)
        if lab != "open_auction":
            rows += stats_rows(87, f"long sell at {lab} [auction cost on all legs]", g - C["auc_p"] - C["auc_t"], pk,
                               allp, extra={"gross_bp": g})
    rows += stats_rows(87, "study33 repro: sell at open_auction, all pairs", night - C["auc_p"] - C["auc_t"], pk,
                       np.isfinite(night), extra={"gross_bp": night})
    # combined book as the paper job runs it (long sold 09:30-09:34, etb short 09:31-09:36 -> 15:55) vs backtest
    lg_bt = night - C["auc_p"] - C["auc_t"]
    lg_pp = (1 + night) * paper_long - 1 - C["auc_p"] - C["09:31"]
    sh_bt = -rday - 2 * C["auc_t"]
    sh_pp = 1 - EXI["15:55"][0] / paper_short - C["09:33"] - C["15:55"]
    for lab, lg, sh in [("backtest auctions", lg_bt, sh_bt), ("paper timing", lg_pp, sh_pp)]:
        comb = np.where(ok, lg, np.nan) + np.where(full, 0.25 * sh, 0.0)   # paper short = 1/4 size of a long slot
        rows += stats_rows(87, f"combined long + 2.5% etb day short, {lab}", comb, pk, ok)
    d87 = pd.DataFrame(rows)
    d87.to_csv(f"{RES}/study87_long_exit_timing.csv", index=False)
    show = ["variant", "period", "drift_after_open_bp", "spy_drift_bp", "gross_bp", "cost_bp", "per_name_net_bp",
            "net_bp_day", "sharpe", "maxdd"]
    print(d87[[c for c in show if c in d87]].round(2).to_string(index=False))


def spy_points(pk):
    """SPY open (first 09:30 trade) -> hh:mm on each pair's day, from data/local/m1 (1-minute ETF bars)."""
    try:
        d = A.read("m1", start="2024-01", end="2026-10", tickers=["SPY"], columns=["ts", "ticker", "o"])
    except Exception as e:
        print("no SPY minute bars", e)
        return {}
    d["tdate"] = d.ts.dt.normalize()
    hm = d.ts.dt.strftime("%H:%M")
    out = {}
    f = d[hm == "09:30"].set_index("tdate").o
    for lab in ["09:31", "09:35", "09:45", "10:00"]:
        r = d[hm == lab].set_index("tdate").o / f - 1
        out[lab] = r.reindex(pk.tdate).values
    return out


if __name__ == "__main__":
    main()
