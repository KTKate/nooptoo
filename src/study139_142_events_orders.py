"""Studies 139 and 142 (docstring completed below after the runs).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")
import alpaca_data as A   # noqa: E402
A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
import study94_cross_rescore as S94   # noqa: E402
S94.A.RL = A.RL
import store   # noqa: E402
import bt   # noqa: E402
from core import exec_cost_bps, ann_stats, RES, DATA   # noqa: E402

LOC = os.path.join(DATA, "local")
AUC139 = os.path.join(LOC, "auctions139.parquet")
TR142 = os.path.join(LOC, "study142_trades.parquet")
OUT139 = os.path.join(RES, "study139_earnings_nights.csv")
OUT142 = os.path.join(RES, "study142_open_orders.csv")
SCR = os.environ.get("SCRATCH", "/tmp/claude-0/-home-user-nooptoo/582a0ce2-853b-5fb5-9a27-c0a484705892/scratchpad")
PER = S94.PER
START, END = "2024-01", "2026-09"
CACHES = ["auctions94", "auctions99", "auctions108", "auctions110", "auctions113", "auctions118", "auctions121",
          "auctions128", "auctions139"]


# ------------------------------------------------------------------ crosses
def all_crosses():
    """Every cached cross row (ticker in panel style). One row per (ticker, date, kind): rows with a price first,
    then the order of CACHES; study 88's opens (same rule) last, filtered to the listing venue as study 94 seeds."""
    parts = []
    for i, f in enumerate(CACHES):
        fn = os.path.join(LOC, f + ".parquet")
        if os.path.exists(fn):
            c = pd.read_parquet(fn)
            for k in ["first_px", "rule"]:
                if k not in c:
                    c[k] = np.nan if k == "first_px" else "venue"
            parts.append(c[["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x", "rule"]].assign(src=i))
    b = pd.read_parquet(os.path.join(LOC, "auctions88.parquet"))
    ven = S94.listing_venue()
    lv = b.ticker.map(ven)
    b = b[b.open_px.isna() | lv.isna() | (b.open_x == lv)]
    parts.append(pd.DataFrame({"ticker": b.ticker, "date": b.date, "kind": "open", "first_px": b.first_px,
                               "px": b.open_px, "sz": b.open_sz, "ts": b.open_ts, "x": b.open_x, "rule": "venue",
                               "src": len(CACHES)}))
    x = pd.concat(parts, ignore_index=True)
    x["date"] = pd.to_datetime(x.date)
    x["ticker"] = x.ticker.astype(str).str.replace(".", "-", regex=False)
    for k in ["ts", "x", "rule"]:
        x[k] = x[k].astype("string")
    x["rule"] = x.rule.fillna("venue")
    x = x.assign(nopx=x.px.isna()).sort_values(["nopx", "src"], kind="stable")
    return x.drop_duplicates(["ticker", "date", "kind"]).drop(columns=["nopx"]).reset_index(drop=True)


def ratios(K, X):
    """Study 94 cross_ratios on the merged cache."""
    fn = os.path.join(SCR, "auctions_merged139.parquet")
    X.drop(columns="src").to_parquet(fn, index=False)
    S94.AUC_FN = fn
    return S94.cross_ratios(K)


# ------------------------------------------------------------------ selections
IND = None


def capped(Sx, n=10, cap=3):
    """Top n per row with at most cap names per industry (paper_overnight.cap_by_industry rule), equal weight."""
    global IND
    if IND is None:
        IND = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index(
            "ticker").industry
    W = pd.DataFrame(0.0, index=Sx.index, columns=Sx.columns)
    for d, row in Sx.iterrows():
        r = row.dropna().sort_values(ascending=False, kind="stable")
        out, cnt = [], {}
        for t in r.index:
            g = IND.get(t)
            g = g if isinstance(g, str) and g else t
            if cnt.get(g, 0) >= cap:
                continue
            cnt[g] = cnt.get(g, 0) + 1
            out.append(t)
            if len(out) == n:
                break
        if out:
            W.loc[d, out] = 1.0 / len(out)
    return W


def topk(Sx):
    return bt.select_topk(Sx, Sx.notna(), 10)


# ------------------------------------------------------------------ earnings timing
EPS_RE = r"(?i)(?:beats?|miss(?:es)?|in-?line|in line|vs\.? .{0,20}est|estimate|up from|down from)"
NOT_RE = r"(?i)(?:\bsees\b|guidance|outlook|raises|lowers|affirms|reaffirms|forecast)"


def eps_headlines(syms):
    """First-party EPS-result headlines (study 74's rule: Benzinga, one symbol, 'EPS' + beats/misses/in line/vs est.,
    not guidance) as (symbol, New York time)."""
    src = os.path.join(LOC, "news_scored")
    out = []
    for f in sorted(os.listdir(src)):
        if not f.endswith(".parquet") or f < "2023-11":
            continue
        d = pd.read_parquet(os.path.join(src, f), columns=["ts", "symbols", "headline"])
        d = d[d.symbols.notna() & (d.symbols.str.count(r"\|") == 0)]
        h = d.headline.fillna("")
        d = d[h.str.contains(r"\bEPS\b") & h.str.contains(EPS_RE) & ~h.str.contains(NOT_RE)]
        out.append(d[["symbols", "ts"]])
    N = pd.concat(out, ignore_index=True).rename(columns={"symbols": "symbol"})
    N["symbol"] = N.symbol.str.replace(".", "-", regex=False)
    N["t"] = pd.to_datetime(N.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return N[N.symbol.isin(syms)][["symbol", "t"]].sort_values("t").reset_index(drop=True)


def earnings_flags(P, cols, idx):
    """Wide boolean frames on entry days idx (night t = close t -> open t+1):
    EW  calendar report date is t or the next trading day t+1 (known in advance, timing not needed)
    EN  report confirmed inside the night: an EPS-result headline between 16:00 of t and 09:30 of t+1, or the
        calendar's own time field (after-hours on t / pre-market on t+1)
    EG  gap rule (study 74, robustness only: uses the reaction itself): EW and the night's |return| >= 1% and >= 2x
        the other candidate night (t-1 for a date-t row, t+1 for a date-t+1 row)."""
    days = P["c"].index
    E = store.read("earnings")
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    E["symbol"] = E.symbol.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)
    E = E[E.symbol.isin(set(cols)) & (E.date >= "2023-12-01") & (E.date <= "2026-10-31")].drop_duplicates(
        ["symbol", "date"])
    ci = {t: i for i, t in enumerate(cols)}
    di = {d: i for i, d in enumerate(idx)}
    shape = (len(idx), len(cols))
    EW, EN, EG = np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool)
    o, c = P["o"][cols].reindex(days).values, P["c"][cols].reindex(days).values
    pos = {d: i for i, d in enumerate(days)}
    night = lambda i, j: o[i + 1, j] / c[i, j] - 1 if 0 <= i < len(days) - 1 else np.nan   # noqa: E731
    for sym, d, tm in zip(E.symbol, E.date, E.time):
        j = ci[sym]
        k = days.searchsorted(d)                       # first trading day >= d
        if k >= len(days):
            continue
        # date-t row (report after the close of t = d, if d trades), date-t+1 row (report before the open of d)
        cands = []
        if days[k] == d:
            cands.append((days[k], "AMC", k - 1))
        if k >= 1:
            cands.append((days[k - 1], "BMO", k))
        for t, typ, other in cands:
            if t not in di:
                continue
            i = di[t]
            EW[i, j] = True
            if (tm == "time-after-hours" and typ == "AMC") or (tm == "time-pre-market" and typ == "BMO"):
                EN[i, j] = True
            g, go = night(pos[t], j), night(other, j)
            if np.isfinite(g) and np.isfinite(go) and abs(g) >= 0.01 and abs(g) >= 2 * abs(go):
                EG[i, j] = True
    N = eps_headlines(set(cols))
    for sym, g in N.groupby("symbol"):
        j = ci[sym]
        for t_ in g.t:
            # night whose window [16:00 t, 09:30 t+1) contains the headline
            k = days.searchsorted(t_.normalize() + pd.Timedelta(0), side="left")
            for kk in (k - 1, k):
                if 0 <= kk < len(days) - 1:
                    a0 = days[kk] + pd.Timedelta(hours=16)
                    b0 = days[kk + 1] + pd.Timedelta(hours=9, minutes=30)
                    if a0 <= t_ < b0 and days[kk] in di:
                        EN[di[days[kk]], j] = True
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)   # noqa: E731
    return f(EW), f(EN), f(EG), N


# ------------------------------------------------------------------ shared setup
def setup():
    K = S94.build_picks()
    P, cols, days = K["P"], K["cols"], K["days"]
    S = K["S"].loc["2023-12-01":"2026-09-30"]
    X = all_crosses()
    Q = ratios(K, X)
    ro = Q["ro"].reindex(index=days, columns=cols)
    rc = Q["rc"].reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    G = {}
    G["Rp"] = (o.shift(-1) / c - 1).reindex_like(S)
    G["Rx"] = ((1 + G["Rp"]) * ro.fillna(1.0).shift(-1).reindex_like(S) / rc.fillna(1.0).reindex_like(S) - 1)
    G["miss"] = (ro.shift(-1).isna() | rc.isna()).reindex_like(S)
    G["C"] = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    G["bias_o"] = (1e4 * (1 / ro.shift(-1) - 1)).reindex_like(S)       # panel open t+1 vs open cross, bp
    G["bias_c"] = (1e4 * (1 / rc - 1)).reindex_like(S)                 # panel close t vs close cross, bp
    G["c_next"] = (c.shift(-1) / c - 1).reindex_like(S)                # close t -> close t+1 (panel)
    G["dv"] = P["dv"][cols].reindex_like(S)
    W = K["W"].reindex(index=S.index).fillna(0.0)
    Wc = K["Wc"].reindex(index=S.index).fillna(0.0)
    EW, EN, EG, N = earnings_flags(P, cols, S.index)
    nxt = pd.Series(days[1:], index=days[:-1])
    return dict(K=K, P=P, cols=cols, days=days, S=S, X=X, Q=Q, G=G, W=W, Wc=Wc, EW=EW, EN=EN, EG=EG, N=N, nxt=nxt)


def pick_frame(Wx, D, extra=None):
    """One row per held pick (entry day t in START..END)."""
    s = Wx.loc[START:END].stack()
    s = s[s > 0]
    df = pd.DataFrame({"t": s.index.get_level_values(0), "ticker": s.index.get_level_values(1), "w": s.values})
    pnl = dict(D["G"])
    pnl.update(EW=D["EW"], EN=D["EN"], EG=D["EG"])
    pnl.update(extra or {})
    for k, v in pnl.items():
        df[k] = v.stack(future_stack=True).reindex(s.index).values
    df["net_p"] = 1e4 * df.Rp - 2 * df.C
    df["net_x"] = 1e4 * df.Rx - 2 * df.C
    df["t1"] = df.t.map(D["nxt"])
    return df


def in_per(t, a0, b0):
    return (t >= pd.Timestamp(a0)) & (t <= pd.Timestamp(b0) + pd.offsets.MonthEnd(0))
