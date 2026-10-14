"""Study 145: sell the overnight picks in the premarket instead of the opening auction?

Question: the overnight blend (study 33; live version with at most 3 names per industry) buys its picks in the
closing auction of t and sells them in the opening auction of t+1. Studies 85-88 found the picks fall after the
opening cross (10-11 bp by 09:31, 24-28 bp by 09:35) and that the first trade prints above the cross. Do the picks
also trade above the cross earlier, in the premarket (08:00-09:28, when the gappers may already have jumped), by
enough to pay the premarket spread? Variants: (a) sell everything at one premarket time (08:00, 08:30, 09:00, 09:15,
09:25); (b) sell at 09:25 only the picks up more than X% from the buy price (rest in the opening auction; X chosen
on 2024-01..2025-06); (c) sell half of every pick at 09:25 and half in the opening auction.

Data: picks exactly as study 69 lines 22-46 (top 10 of (2 rank(ensemble) + rank(p_jump - p_drop)) / 3, results/
study33_pred.parquet, study23_pred.parquet), and the live version capped at 3 names per industry
(paper_overnight.cap_by_industry, data/store/sectors.parquet). Official crosses (closing cross of t, opening cross
of t+1) and the panel-to-cross ratio method from study 94 (src/study94_cross_rescore.py: data/local/auctions94.parquet,
cross_ratios; night returns at the crosses, panel price where no usable cross). Premarket trades: 5-minute SIP bars
07:00-09:30 of study 25 (data/local/m5pre, src/fetch_premarket.py; split-adjusted as of the fetch date, put back on
the traded scale with the later splits in data/local/events/splits_yf.csv as in study 94). Premarket quotes: fetched
here from Alpaca SIP quotes (free plan, history older than 15 minutes) into data/local/quotes145.parquet: for a seeded
sample of pick-days (capped picks; per period 300 at random plus 100 among picks whose 09:25 premarket price was
more than 2% above the buy price and within 25% of the cross, so variant (b) has quote data) the last valid NBBO (bid > 0, ask > bid) since
04:00 before each of 08:00, 08:30, 09:00, 09:15, 09:25 (one request per pick-day and time, newest first).

Design: per pick, the sell price at premarket time T is compared with the opening cross of t+1 (raw traded prices on
the same day). Bar-based price for every pick: the last premarket trade before T (close of the last 5-minute bar
ending by T, at most 30 minutes old; otherwise no premarket sale is possible and the pick is sold at the cross)
minus half the quoted premarket spread. Half spread per pick: from the quote sample, a regression of log half
spread (bp) on log 20-day ADV, log price and the time slot (fit on all sampled pick-days, both periods). Quote-based
price on the sampled pick-days: the bid (a marketable limit sell), and bid + 25% / 50% of the spread (a passive limit
that may not fill). Passive limit variant (every pick with a premarket price): a limit sell from T at the last
trade (or 0.25% / 0.5% above it), counted as filled if a later premarket bar (ending by 09:25) has a high at least
1 cent above the limit (strict version: max(20 bp, 2 cents) above), otherwise cancelled before the 09:28 OPG cutoff and
sold in the opening auction. Edge per pick = sell price / cross - 1 in bp; cost: the opening-auction sell is charged the
study 69 auction cost (core.exec_cost_bps 'auction' + 2.5 bp), a premarket sell 0.3 bp fees + 2.5 bp on top of the
bid (the half spread is already in the price). Net gain vs the cross = edge - premarket cost + auction cost. Fill
risk: premarket dollar volume in [T, T+15 min) and the bid size in dollars vs a $1k and $2.5k order. Book: the
night returns at the crosses (study 94) with the sell leg replaced; equal weight over the night's picks; Sharpe,
mean net bp per night. Data errors: premarket prices more than 25% away from the cross (after the split scale fix)
are listed and treated as no premarket price; all edges are also reported with 1%/99% winsorizing and as medians.
Periods by entry date: 2024-01..2025-06 (P1, X chosen), 2025-07..2026-09 (P2).

Alpaca rules (docs.alpaca.markets/docs/orders-at-alpaca, read 2026-10-14): pre-market is 04:00-09:30 ET; only limit
orders with time_in_force day or gtc and extended_hours=true are extended-hours eligible (market, stop, stop-limit
are rejected); OPG (market/limit-on-open) orders are rejected between 09:28 and 19:00 ET and unfilled ones are
cancelled after the open. So a premarket exit is a limit sell with extended_hours=true; a share left unsold by 09:28
cannot be moved into the opening auction any more (it can only be sold in the regular session).

Output: results/study145_premarket_exit.csv (tables: data_errors, spread_model, edge_bars (per time, period, set),
edge_quotes (quote sample), fill (volume and bid size vs order), threshold (variant b grid on P1), book (Sharpe /
net bp per variant and period)).

    python src/study145_premarket_exit.py picks    # pick list with cross prices -> data/local/_study145_picks.parquet
    python src/study145_premarket_exit.py fetch    # premarket quote sample (resumable; ALPACA_PER_MIN, default 60)
    python src/study145_premarket_exit.py          # study
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A   # noqa: E402
from core import RES, DATA, ann_stats   # noqa: E402

A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
PICKS = os.path.join(A.LOCAL, "_study145_picks.parquet")
QFN = os.path.join(A.LOCAL, "quotes145.parquet")
OUT = os.path.join(RES, "study145_premarket_exit.csv")
TIMES = ["08:00", "08:30", "09:00", "09:15", "09:25"]
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]
START, END = "2024-01", "2026-09"
FETCH_DATE = "2026-09-29"     # m5pre files written 2026-09-29: bars adjusted for splits up to then


# ------------------------------------------------------------------ picks
def picks():
    import bt
    import study94_cross_rescore as S94
    from paper_overnight import cap_by_industry
    from core import load_panel, stock_cols, exec_cost_bps, traded_close
    A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    # blend, exactly as study 69 lines 22-46
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr
    W = bt.select_topk(S, S.notna(), 10)
    # live version: paper_overnight.cap_by_industry on the day's ranking
    Wc = pd.DataFrame(0.0, index=S.index, columns=cols)
    for d, row in S.loc[START:END].iterrows():
        r = row.dropna().sort_values(ascending=False, kind="stable")
        out = list(cap_by_industry(r, 10).index)
        if out:
            Wc.loc[d, out] = 1.0 / len(out)
    tc = traded_close(P)[cols]
    Q = S94.cross_ratios(dict(P=P, cols=cols, days=days, tc=tc))
    print("study 94 cross data errors:", Q["err"], flush=True)
    ro = Q["ro"].reindex(index=days, columns=cols)
    rc = Q["rc"].reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rn_p = (o.shift(-1) / c - 1).reindex_like(S)
    Rn_x = (1 + Rn_p) * ro.fillna(1.0).shift(-1).reindex_like(S) / rc.fillna(1.0).reindex_like(S) - 1
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    adv = P["dv"][cols].rolling(20, min_periods=10).mean().reindex_like(S)
    # raw traded-scale prices: sell reference (cross, else panel open in raw terms) and buy price (cross, else panel)
    XO = Q["XO"].reindex(index=days, columns=cols)
    XC = Q["XC"].reindex(index=days, columns=cols)
    PRO = Q["PRO"].reindex(index=days, columns=cols)
    PRC = Q["PRC"].reindex(index=days, columns=cols)
    sell_ref = XO.where(ro.notna(), PRO)
    buy_px = XC.where(rc.notna(), PRC)
    nxt = pd.Series(days[1:], index=days[:-1])
    rows = []
    for name, Wx in [("capped", Wc), ("blend", W)]:
        s = Wx.loc[START:END].stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        s["t1"] = s.t.map(nxt)
        s = s.dropna(subset=["t1"])
        mi = pd.MultiIndex.from_arrays([s.t, s.ticker])
        mi1 = pd.MultiIndex.from_arrays([s.t1, s.ticker])
        g = lambda X, m: X.stack(future_stack=True).reindex(m).values
        s = s.assign(set=name, R_cross=g(Rn_x, mi), cost=g(C, mi), adv=g(adv, mi), buy_px=g(buy_px, mi),
                     sell_ref=g(sell_ref, mi1), cross_ok=~np.isnan(g(ro, mi1)), xo=g(XO, mi1))
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out["ticker"] = out.ticker.astype(str)
    out.to_parquet(PICKS, index=False)
    print(out.groupby("set").size(), out.isna().mean().round(3).to_dict(), flush=True)


# ------------------------------------------------------------------ premarket bars
def split_back(tk, dates):
    """Factor that puts m5pre bars (adjusted for splits up to FETCH_DATE) back on the traded scale of each date."""
    sp = pd.read_csv(f"{DATA}/local/events/splits_yf.csv", parse_dates=["date"])
    sp = sp[sp.date <= FETCH_DATE]
    f = np.ones(len(tk))
    for t, g in sp[sp.ticker.isin(set(tk))].groupby("ticker"):
        m = tk == t
        for d, r in zip(g.date, g.ratio):
            f[m & (dates < d)] *= r
    return f


def bar_prices(pk):
    """Per (ticker, t1): last trade before each T (bar close, <= 30 min old), dollar volume in [T, T+15),
    on the traded scale. pk: unique (ticker, t1)."""
    want = pk[["ticker", "t1"]].drop_duplicates()
    want["tk_a"] = want.ticker.str.replace("-", ".", regex=False)
    res = []
    for m, wm in want.groupby(want.t1.dt.strftime("%Y-%m")):
        fn = os.path.join(DATA, "local", "m5pre", f"{m}.parquet")
        if not os.path.exists(fn):
            continue
        d = pd.read_parquet(fn, columns=["ts", "ticker", "h", "c", "v", "vwap"],
                            filters=[("ticker", "in", list(wm.tk_a.unique()))])
        d["t1"] = d.ts.dt.normalize()
        d = d.merge(wm[["tk_a", "t1"]].rename(columns={"tk_a": "ticker"}), on=["ticker", "t1"])
        d["end"] = d.ts + pd.Timedelta(minutes=5)
        d["dv"] = d.v * d.vwap.astype("float64")
        for T in TIMES:
            tt = d.t1 + pd.Timedelta(hours=int(T[:2]), minutes=int(T[3:]))
            last = d[(d.end <= tt) & (d.end > tt - pd.Timedelta(minutes=30))].groupby(["ticker", "t1"]).agg(
                px=("c", "last"), last_end=("end", "last"))
            t930 = d.t1 + pd.Timedelta(hours=9, minutes=30)
            t925 = d.t1 + pd.Timedelta(hours=9, minutes=25)
            vol = d[(d.ts >= tt) & (d.ts < np.minimum(tt + pd.Timedelta(minutes=15), t930))].groupby(
                ["ticker", "t1"]).agg(dv15=("dv", "sum"), vwap5=("vwap", "first"), ts5=("ts", "first"))
            later = d[(d.ts >= tt) & (d.end <= t925)].groupby(["ticker", "t1"]).agg(hi_after=("h", "max"))
            x = last.join(vol, how="outer").join(later, how="outer").reset_index()
            x["T"] = T
            res.append(x)
        print("bars", m, len(d), flush=True)
    b = pd.concat(res, ignore_index=True).rename(columns={"ticker": "tk_a"})
    b["ticker"] = b.tk_a.str.replace(".", "-", regex=False)
    f = split_back(b.ticker.values, b.t1.values)
    b["px"] = b.px * f
    b["vwap5"] = b.vwap5.astype("float64") * f
    b["hi_after"] = b.hi_after.astype("float64") * f
    return b.drop(columns="tk_a")


# ------------------------------------------------------------------ quotes
def sample(pk, bars):
    """Seeded sample of capped pick-days per period: 300 random + 100 among 09:25 gainers (> 2% over the buy)."""
    c = pk[pk.set == "capped"].drop_duplicates(["ticker", "t1"]).copy()
    b = bars[bars["T"] == "09:25"][["ticker", "t1", "px"]]
    c = c.merge(b, on=["ticker", "t1"], how="left")
    c["up"] = c.px / c.buy_px - 1
    rng = np.random.default_rng(145)
    out = []
    for lab, a, z in PER:
        x = c[(c.t >= a) & (c.t <= pd.Timestamp(z) + pd.offsets.MonthEnd(0))]
        r = x.iloc[np.sort(rng.choice(len(x), 300, replace=False))]
        rest = x.drop(r.index)
        gx = rest[(rest.up > 0.02) & (rest.px / rest.sell_ref - 1).abs().lt(0.25)]
        gx = gx.iloc[np.sort(rng.choice(len(gx), min(100, len(gx)), replace=False))]
        out += [r.assign(per=lab, stratum="random"), gx.assign(per=lab, stratum="gainer")]
    return pd.concat(out, ignore_index=True)[["ticker", "t1", "t", "per", "stratum"]]


def _utc(day, hms):
    return pd.Timestamp(f"{day} {hms}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def last_quote(sym, day, T):
    """Last valid NBBO (bid > 0, ask > bid) in [04:00, T) (the NBBO stands until it changes, so a thin name's last
    update can be long before T); the age of the quote is kept."""
    p = dict(symbols=sym, start=_utc(day, "04:00:00"), end=_utc(day, T + ":00"), limit=50, feed="sip", sort="desc")
    j = A.get("quotes", p)
    qs = (j.get("quotes") or {}).get(sym, [])
    for q in qs:
        if q.get("bp", 0) > 0 and q.get("ap", 0) > q.get("bp", 0):
            return dict(bid=q["bp"], ask=q["ap"], bsz=q.get("bs"), asz=q.get("as"), qts=q["t"], bx=q.get("bx"),
                        ax=q.get("ax"), n_q=len(qs))
    return dict(bid=np.nan, ask=np.nan, bsz=np.nan, asz=np.nan, qts=None, bx=None, ax=None, n_q=len(qs))


def fetch():
    pk = pd.read_parquet(PICKS)
    bars = bar_prices(pk[pk.set == "capped"])
    smp = sample(pk, bars)
    have = pd.read_parquet(QFN) if os.path.exists(QFN) else pd.DataFrame(columns=["ticker", "t1", "T"])
    done = set(zip(have.ticker, pd.to_datetime(have.t1), have["T"]))
    jobs = [(r.ticker, r.t1, T, r.per, r.stratum) for r in smp.itertuples() for T in TIMES
            if (r.ticker, r.t1, T) not in done]
    print("sample", len(smp), "quote requests to do", len(jobs), flush=True)
    rows = []
    for i, (tk, t1, T, per, st) in enumerate(jobs):
        try:
            q = last_quote(tk.replace("-", "."), t1.strftime("%Y-%m-%d"), T)
        except RuntimeError as e:
            print("err", tk, t1, T, e, flush=True)
            continue
        rows.append(dict(ticker=tk, t1=t1, T=T, per=per, stratum=st, **q))
        if len(rows) % 100 == 0 or i == len(jobs) - 1:
            new = pd.DataFrame(rows)
            for k in ["qts", "bx", "ax"]:
                new[k] = new[k].astype("string")
            out = pd.concat([have, new], ignore_index=True) if len(have) else new
            out.to_parquet(QFN, index=False)
            print("quotes", i + 1, "/", len(jobs), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "picks":
    picks()
    sys.exit()
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch()
    sys.exit()


# ------------------------------------------------------------------ study
def per_of(t):
    return np.where(t <= pd.Timestamp("2025-06-30"), "2024-25H1", "2025H2-26")


def summ(v, day=None):
    """mean, 1/99% winsorized mean, median, t (by day when day given), n of a bp series."""
    v = pd.Series(np.asarray(v, dtype=float))
    ok = v.notna().values
    v = v[ok]
    if len(v) < 3:
        return dict(n=len(v))
    lo, hi = v.quantile([0.01, 0.99])
    if day is not None:
        dm = v.groupby(np.asarray(day)[ok]).mean()
        t = dm.mean() / (dm.std() / np.sqrt(len(dm)))
    else:
        t = v.mean() / (v.std() / np.sqrt(len(v)))
    return dict(n=len(v), mean_bp=v.mean(), wins_mean_bp=v.clip(lo, hi).mean(), median_bp=v.median(),
                share_pos=(v > 0).mean(), t=t)


def spread_model(qs):
    """OLS of log half spread (bp) on log ADV, log price and time-slot dummies. Returns coef dict and rows."""
    q = qs.dropna(subset=["hs_bp", "adv", "buy_px"])
    q = q[q.hs_bp > 0]
    X = np.column_stack([np.ones(len(q)), np.log(q.adv), np.log(q.buy_px)] +
                        [(q["T"] == T).astype(float) for T in TIMES[1:]])
    y = np.log(q.hs_bp.values)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    smear = np.mean(np.exp(res))         # Duan smearing: mean, not median, half spread
    r2 = 1 - res.var() / y.var()
    names = ["const", "log_adv", "log_px"] + [f"T={T}" for T in TIMES[1:]]
    return dict(zip(names, b)), smear, r2, len(q)


def predict_hs(coef, smear, adv, px, T):
    z = coef["const"] + coef["log_adv"] * np.log(adv) + coef["log_px"] * np.log(px)
    z = z + sum(coef[f"T={t}"] * (T == t) for t in TIMES[1:])
    return np.exp(z) * smear


LIMS = {"l0": (0.0, 1e-4, 0.01), "l0s": (0.0, 0.002, 0.02), "l25": (0.0025, 1e-4, 0.01), "l50": (0.005, 1e-4, 0.01)}
LIM_LAB = {"l0": "limit at last trade (fill: later print >= limit + 1c)",
           "l0s": "limit at last trade (strict fill: later print >= limit + max(20 bp, 2c))",
           "l25": "limit 0.25% above last trade", "l50": "limit 0.5% above last trade"}
PM_COST = 0.3 + 2.5      # premarket limit sell: fees + the studies' 2.5 bp per side (spread is in the bid)


def main():
    pk = pd.read_parquet(PICKS)
    pk["per"] = per_of(pk.t)
    bars = bar_prices(pk)
    x = pk.merge(pd.DataFrame({"T": TIMES}), how="cross").merge(bars, on=["ticker", "t1", "T"], how="left")
    rows = []
    # ---- data errors: premarket price far from the cross
    x["dev"] = x.px / x.sell_ref - 1
    bad = x.dev.abs() > 0.25
    err = x[bad & (x.set == "capped")].drop_duplicates(["ticker", "t1"])
    rows.append(dict(table="data_errors", set="capped", period="all", n=int(x[x.set == "capped"].px.notna().sum()),
                     n_bad=int((bad & (x.set == "capped")).sum()), pick_days_bad=len(err),
                     tickers=" ".join(sorted(err.ticker.unique()))))
    for lim in [0.05, 0.10]:
        m = (x.set == "capped") & x.dev.abs().between(lim, 0.25)
        rows.append(dict(table="data_errors", set=f"capped, {lim:.0%}..25% from cross (kept)", period="all",
                         n_bad=int(m.sum())))
    print(err[["ticker", "t", "t1", "T", "px", "sell_ref", "dev"]].head(30).to_string(), flush=True)
    x.loc[bad, ["px", "vwap5", "hi_after"]] = np.nan
    # ---- quotes
    qs = pd.read_parquet(QFN)
    qs["t1"] = pd.to_datetime(qs.t1)
    cap = x[x.set == "capped"]
    qs = qs.merge(cap[["ticker", "t1", "T", "t", "adv", "buy_px", "sell_ref", "cross_ok", "cost", "R_cross", "px",
                       "dv15"]], on=["ticker", "t1", "T"], how="left")
    qs["mid"] = (qs.bid + qs.ask) / 2
    qs["hs_bp"] = 1e4 * (qs.ask - qs.bid) / (qs.ask + qs.bid)
    qbad = (qs.mid / qs.sell_ref - 1).abs() > 0.25
    rows.append(dict(table="data_errors", set="quote sample: NBBO mid > 25% from cross (dropped)", period="all",
                     n=int(qs.mid.notna().sum()), n_bad=int(qbad.sum())))
    wide = qs.hs_bp > 500
    rows.append(dict(table="data_errors", set="quote sample: quoted spread > 10% (kept, bid used as is)", period="all",
                     n_bad=int(wide.sum())))
    qs.loc[qbad, ["bid", "ask", "mid", "hs_bp"]] = np.nan
    coef, smear, r2, nq = spread_model(qs)
    rows.append(dict(table="spread_model", set="log half spread bp ~ log adv + log px + time", n=nq, r2=r2,
                     smear=smear, **{f"b_{k}": v for k, v in coef.items()}))
    print("spread model", coef, "r2", round(r2, 3), flush=True)
    # half spreads by time and period (random stratum)
    for (p, T), g in qs[qs.stratum == "random"].groupby(["per", "T"]):
        rows.append(dict(table="spread_obs", set="quote sample, random", period=p, T=T, n=int(g.hs_bp.notna().sum()),
                         quote_found=g.bid.notna().mean(), median_hs_bp=g.hs_bp.median(), mean_hs_bp=g.hs_bp.mean(),
                         p90_hs_bp=g.hs_bp.quantile(0.9)))
    # ---- bar-based sell price for every pick
    x["hs_hat"] = predict_hs(coef, smear, x.adv.values, x.buy_px.values, x["T"].values)
    x["sell_bar"] = x.px * (1 - x.hs_hat / 1e4)
    x["edge_last"] = 1e4 * (x.px / x.sell_ref - 1)                      # last trade vs cross (no spread)
    x["gain_bar"] = 1e4 * (x.sell_bar / x.sell_ref - 1) - PM_COST + x.cost    # net vs selling at the cross
    # passive limit sell from T at the last trade (or k above it); filled if a later premarket trade (bars ending by
    # 09:25) prints at least max(1 cent, margin) above the limit, otherwise cancelled at 09:27 and sold in the opening
    # auction (market-on-open accepted until 09:28). No limit variant at 09:25 (no time left to work the order).
    for key, (k, marg, cents) in LIMS.items():
        lim = x.px * (1 + k)
        fill = (x.hi_after >= np.maximum(lim * (1 + marg), lim + cents)) & (x["T"] != "09:25")
        x[f"fill_{key}"] = fill
        x[f"sell_{key}"] = lim.where(fill)
        g_ = np.where(fill, 1e4 * (lim / x.sell_ref - 1) - PM_COST + x.cost, 0.0)
        x[f"gain_{key}"] = np.where(x.px.isna() | (x["T"] == "09:25"), np.nan, g_)
    for (s, p, T), g in x.groupby(["set", "per", "T"]):
        rows.append(dict(table="edge_bars", set=s, period=p, T=T, what="last trade vs cross (bp)",
                         have_px=g.px.notna().mean(), **summ(g.edge_last, g.t)))
        rows.append(dict(table="edge_bars", set=s, period=p, T=T, what="net gain vs cross: last - modelled half spread",
                         have_px=g.px.notna().mean(), **summ(g.gain_bar, g.t)))
        if T != "09:25":
            for key, lab in LIM_LAB.items():
                rows.append(dict(table="edge_bars", set=s, period=p, T=T,
                                 what=f"net gain vs cross: {lab}, else cross (picks with a price)",
                                 have_px=g.px.notna().mean(), fill_rate=g[f"fill_{key}"][g.px.notna()].mean(),
                                 **summ(g[f"gain_{key}"], g.t)))
                f = g[g[f"fill_{key}"]]
                rows.append(dict(table="edge_bars", set=s, period=p, T=T, what=f"{lab}: filled picks only",
                                 **summ(f[f"gain_{key}"], f.t)))
    # ---- quote-based
    qs["gain_bid"] = 1e4 * (qs.bid / qs.sell_ref - 1) - PM_COST + qs.cost
    qs["gain_b25"] = 1e4 * ((qs.bid + 0.25 * (qs.ask - qs.bid)) / qs.sell_ref - 1) - PM_COST + qs.cost
    qs["gain_mid"] = 1e4 * (qs.mid / qs.sell_ref - 1) - PM_COST + qs.cost
    qs["hs_hat"] = predict_hs(coef, smear, qs.adv.values, qs.buy_px.values, qs["T"].values)
    qs["calib"] = 1e4 * (qs.px * (1 - qs.hs_hat / 1e4) / qs.bid - 1)       # bar estimate vs actual bid
    qs["up"] = qs.px / qs.buy_px - 1
    for (p, st, T), g in qs.groupby(["per", "stratum", "T"]):
        for k, lab in [("gain_bid", "net gain vs cross: sell at bid"),
                       ("gain_b25", "net gain vs cross: bid + 25% of spread (may not fill)"),
                       ("gain_mid", "net gain vs cross: mid (upper bound)"),
                       ("calib", "bar estimate (last - model half spread) minus actual bid")]:
            rows.append(dict(table="edge_quotes", set=f"capped sample, {st}", period=p, T=T, what=lab,
                             quote_found=g.bid.notna().mean(), **summ(g[k], g.t)))
    qs["pos"] = (qs.px - qs.bid) / (qs.ask - qs.bid)                    # last trade within the NBBO (0 bid, 1 ask)
    for (p, T), g in qs[qs.stratum == "random"].groupby(["per", "T"]):
        v = g.pos.dropna()
        rows.append(dict(table="last_vs_nbbo", set="capped sample, random", period=p, T=T, n=len(v),
                         share_at_or_below_bid=(v <= 0).mean(), share_inside=((v > 0) & (v < 1)).mean(),
                         share_at_or_above_ask=(v >= 1).mean(), median_pos=v.median(),
                         median_quote_age_min=(pd.to_datetime(g.t1).dt.tz_localize("America/New_York")
                                               + pd.to_timedelta(g["T"] + ":00") -
                                               pd.to_datetime(g.qts, format="ISO8601").dt.tz_convert("America/New_York")
                                               ).dt.total_seconds().median() / 60))
    # ---- fill risk
    qs["bid_usd"] = qs.bsz * 100 * qs.bid           # Alpaca SIP quote sizes are round lots (100 shares)
    for (p, T), g in x[x.set == "capped"].groupby(["per", "T"]):
        dv = g.dv15.fillna(0)
        rows.append(dict(table="fill", set="capped, all picks (bars)", period=p, T=T, n=len(g),
                         share_dv15_ge_1k=(dv >= 1e3).mean(), share_dv15_ge_2p5k=(dv >= 2.5e3).mean(),
                         share_dv15_ge_25k=(dv >= 2.5e4).mean(), median_dv15=dv.median(),
                         share_trade_last30=g.px.notna().mean()))
    for (p, T), g in qs[qs.stratum == "random"].groupby(["per", "T"]):
        b = g.bid_usd.fillna(0)
        rows.append(dict(table="fill", set="capped sample, random (quotes)", period=p, T=T, n=len(g),
                         share_bid_ge_1k=(b >= 1e3).mean(), share_bid_ge_2p5k=(b >= 2.5e3).mean(),
                         median_bid_usd=b.median(), quote_found=g.bid.notna().mean()))
    # ---- book
    base_net = lambda d: d.R_cross - 2 * d.cost / 1e4
    for s in ["capped", "blend"]:
        y = x[x.set == s]
        ref = y.drop_duplicates(["t", "ticker"])[["t", "ticker", "per", "R_cross", "cost", "buy_px", "sell_ref"]]
        W = y.pivot_table(index=["t", "ticker"], columns="T", values="sell_bar", aggfunc="first")
        L = y.pivot_table(index=["t", "ticker"], columns="T", values="px", aggfunc="first")
        LF = {key: y.pivot_table(index=["t", "ticker"], columns="T", values=f"sell_{key}", aggfunc="first")
              for key in LIMS}
        ref = ref.set_index(["t", "ticker"])
        n0 = base_net(ref)

        def pm_net(sell, frac=1.0):
            """frac of the pick sold at the premarket price sell (raw), the rest at the cross; NaN sell -> cross."""
            r_pm = (1 + ref.R_cross) * (sell / ref.sell_ref) - 1 - ref.cost / 1e4 - PM_COST / 1e4
            r_pm = r_pm.where(sell.notna(), n0)
            return frac * r_pm + (1 - frac) * n0

        variants = {"cross (opening auction, base)": n0}
        for T in TIMES:
            variants[f"a: sell all at {T} (last - half spread)"] = pm_net(W[T].reindex(ref.index))
        for key, lab in LIM_LAB.items():
            for T in TIMES[:-1]:
                variants[f"a2: {lab} from {T}, unfilled at the cross"] = pm_net(LF[key][T].reindex(ref.index))
        variants["c: half at 09:25, half at the cross"] = pm_net(W["09:25"].reindex(ref.index), 0.5)
        up = L["09:25"].reindex(ref.index) / ref.buy_px - 1
        thr = {}
        for X in [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]:
            sel = W["09:25"].reindex(ref.index).where(up > X)
            thr[X] = pm_net(sel)
            for p, a, z in PER:
                nd = thr[X].groupby(level="t").mean().loc[a:z]
                m = (ref.per == p)
                rows.append(dict(table="threshold", set=s, period=p, what=f"X={X:.3f}", share_sold_pm=(up[m] > X).mean(),
                                 sharpe=ann_stats(nd)["sharpe"], net_bp=1e4 * nd.mean(),
                                 gain_per_sold_bp=1e4 * (thr[X] - n0)[m & (up > X)].mean()))
        # X chosen on P1 by mean net bp per night
        p1 = {X: thr[X].groupby(level="t").mean().loc[PER[0][1]:PER[0][2]].mean() for X in thr}
        Xb = max(p1, key=p1.get)
        variants[f"b: sell at 09:25 if up > {Xb:.1%} (X chosen on P1), rest at the cross"] = thr[Xb]
        for name, v in variants.items():
            nd = v.groupby(level="t").mean()
            dlt = 1e4 * (v - n0)
            for p, a, z in PER:
                st = ann_stats(nd.loc[a:z])
                m = (ref.per == p).values
                rows.append(dict(table="book", set=s, period=p, what=name, sharpe=st["sharpe"],
                                 net_bp=1e4 * nd.loc[a:z].mean(), ann=st["ann_ret"], maxdd=st["maxdd"],
                                 delta_per_pick_bp=dlt[m].mean(), delta_per_pick_median_bp=dlt[m].median(),
                                 delta_t=summ(dlt[m], ref.index.get_level_values("t")[m])["t"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, g in df.groupby("table", sort=False):
        print("\n==", t)
        print(g.dropna(axis=1, how="all").drop(columns="table").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
