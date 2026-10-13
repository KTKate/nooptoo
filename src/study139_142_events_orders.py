"""Studies 139 (earnings nights at the official crosses) and 142 (live-order risks at the open).

Question 139: the blend often holds stocks that report earnings that night. Priced at the official crosses rather
than the daily panel, are those picks better or worse than the rest, how large is the panel-minus-cross open bias on
them, and should the live book exclude them or hold them at half weight? (Study 80: an earnings-night model added
about 0.1 Sharpe at 25% of capital on report nights, on panel prices.)
Question 142: what can go wrong with the market-on-open sell order? (a) How often a pick has no listing-exchange
opening cross by 09:45, why, what the stock did before its first trade, and what an Alpaca MOO order would have done;
(b) how often a pick is halted around the open or opens more than 20% from the close, and the P&L share of those
nights; (c) how large our order is relative to the opening cross for $10k / $25k / $100k accounts at 10% per name.

Data: blend as study 69 lines 22-46 (top 10 of (2 rank(ensemble) + rank(p_jump - p_drop)) / 3) and the live version
with at most 3 names per industry (paper_overnight.cap_by_industry rule, data/store/sectors.parquet), both built by
study 94's build_picks. Official crosses: union of the caches data/local/auctions{94,99,108,110,113,118,121,128}.parquet
and study 88's opens (listing-venue filter as study 94), plus 251 crosses fetched here for names that enter only under
rule (a) (data/local/auctions139.parquet, study 94's rule: first 'O' / '6' print of the listing exchange, second pass
the largest print of any listing exchange). Prices on the cross scale with study 94's method (night return x open
cross / panel open on t+1 / (close cross / panel close on t); legs without a usable cross at the panel price).
Costs exec_cost_bps(P, "auction") + 2.5 bp per side. Earnings: store.read("earnings") (Nasdaq calendar; 99% of rows
have time 'time-not-supplied'); Benzinga EPS-result headlines from data/local/news_scored (study 74's pattern).
Trades for study 142: Alpaca SIP trades of the sell-day morning (09:29-09:50, up to 4 pages of 10,000; for picks
without a listing cross also the last pre-market trade and the first trades 09:50-16:00), cached in
data/local/study142_trades.parquet, about 60 requests a minute.

Design 139. Report timing rule (night t = closing cross of t -> opening cross of t+1):
  confirmed earnings night = an EPS-result headline for the symbol between 16:00 of t and 09:30 of t+1, or the
  calendar's own time field says after-hours on t / pre-market on t+1. Headlines are timestamped when published, so
  this recovers a timing that companies announce in advance; it does not select on the reaction.
  report date t or t+1, not confirmed = calendar row dated t or t+1 without such a headline (mostly reports before
  the open of t or after the close of t+1, i.e. not inside the held night, plus a few with no headline).
  robustness: study 74's gap rule (night |return| >= 1% and >= 2x the other candidate night); it uses the reaction,
  so it is shown only as a check.
Per group and period: count, mean net bp per pick at panel and at cross prices, open bias (panel open / open cross -
1, bp), close bias, hit rate, 5% quantile, worst picks, share of the book's net P&L. Rules on the live and the
uncapped series, with the flag = confirmed nights (EN) and = calendar window (EW, knowable without timing):
  a  exclude flagged picks, refill from the next-ranked names (industry cap reapplied)
  b  half weight on flagged picks, rest cash;  b2 half weight, freed weight spread over the other picks
The variant with the best cross-price Sharpe on 2024-01..2025-06 is checked on 2025-07..2026-09 (paired t of the
daily difference vs the base). Data check: raw cross-to-cross night returns vs the ratio method, picks that differ
by > 1% listed; group statistics repeated without them.
Design 142. (a) Status of each live pick's sell-day open from the caches: listing cross by 09:30:03 / 09:30:20 /
09:32 / 09:45, cross only at another listing exchange ('largest' rule), or none by 09:45. Rows without a listing
cross are refetched and classified (no trades, late open, reopening print, cross under another exchange code, official
open without a cross print). (b) Halt proxies from the refetched trades: a reopening print (condition '5'), a pause of
>= 5 minutes after the first regular trade (stocks with >= 30 trades in the window), or a first regular trade after
09:35. Sample: every pick with a late or missing cross, |panel night return| >= 5%, plus 400 random other picks
(seed 142) to estimate the rate among the rest. Opens > 20% from the close: panel or cross night return. (c) Order
= 10% of the account; share = order / (open cross price x size of the listing-venue 'O' print, which carries the
whole auction volume); closing cross for reference.
Periods 2024-01..2025-06, 2025-07..2026-09, 2024-01..2026-09 by entry date.

Data errors found: base series reproduce study 94 exactly (live Sharpe at crosses 2.65 / 2.23 / 2.45). 19 of 6,850
live picks have raw cross-to-cross returns that disagree with the ratio method by > 1%: 4 reverse splits on the sell
day, ex-dividend days (TRMD, CVX, TTE, WU, UVV, VTS), RIME (4 nights at ~3 cents, panel returns -20% to -33% vs
crosses -8% to -15%), TNXP, MEI, FLG, and PEW 2025-07-15 (panel +24%, crosses -7%); none is an earnings night;
without them the other picks gain up to 6 bp (2024-01..2025-06: +39 instead of +34 bp), earnings nights unchanged. 12 sell-day opens without a listing cross (2024-11-19
and -20) and 12 'largest' rows had the cross printed under SIP exchange code 'T' (feed labelling, not a missing
auction). Opening-cross sizes for the 'largest' rule are left out of (c) (another exchange's print can be small).
Results (live series, cross prices, 2024-01..2025-06 / 2025-07..2026-09): confirmed earnings nights are 11% / 8% of
picks and earned +111 / +15 bp net per pick (rest +34 / +38); the open bias on them was 2 / 33 bp, the latter from 67
picks whose listing cross came after 09:30:20 (mostly NYSE openings delayed by the DMM; +95 bp bias, -37 bp at the
cross vs +54 bp on the panel). Every rule lost on the first period (exclude -12 bp/day, t -2.0; half weight -6 bp,
t -2.3); the first-period choice is to keep the picks. 142: 88% of live picks have a listing cross by 09:30:03,
7.5% a later listing cross (up to 09:45; 40% of those are earnings nights), 2.9% a cross only at another exchange,
0.2% none by 09:45 (all explained above, 2 real cases without auction volume). Halts: 14 picks (0.2%) paused >= 5
minutes or reopened by a '5' print by 09:50, 12 of them after the opening cross; 0 of 400 random other picks.
Opens > 20% from the close: 1.1% of picks, +9.7% of the net P&L (down gaps -37%, up gaps +47%). Order vs opening
cross: $10k account 1.1% of picks over 5% of the cross, $25k 5.4%, $100k 24% (13% over 10%).

Output: results/study139_earnings_nights.csv (tables groups, worst, data_check, groups_without_data_flags, rules,
choice, timing_check); results/study142_open_orders.csv (a_open_status, a_nocross_reasons, b_halts_gaps,
c_cross_size).

    python src/study139_142_events_orders.py fetch   # refill crosses + sell-day trades (resumable)
    python src/study139_142_events_orders.py         # both studies
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


# ------------------------------------------------------------------ study 139
GROUPS = [("earnings night (headline or calendar time)", lambda d: d.EN.astype(bool)),
          ("report date t or t+1, not confirmed for the night", lambda d: d.EW.astype(bool) & ~d.EN.astype(bool)),
          ("no report on t or t+1", lambda d: ~d.EW.astype(bool) & ~d.EN.astype(bool)),
          ("all picks", lambda d: pd.Series(True, index=d.index)),
          ("robustness: earnings night by gap rule (uses the reaction)", lambda d: d.EG.astype(bool))]


def group_rows(series, df):
    rows, worst = [], []
    for lab, a0, b0 in PER:
        x = df[in_per(df.t, a0, b0)]
        tot = (x.w * x.net_x.fillna(0)).sum()
        tot_p = (x.w * x.net_p.fillna(0)).sum()
        rest = x[~x.EN.astype(bool)]
        for g, f in GROUPS:
            y = x[f(x)]
            v = y.net_x.dropna()
            r = rest.net_x.dropna()
            t_diff = ((v.mean() - r.mean()) / np.sqrt(v.var() / len(v) + r.var() / len(r))
                      if g.startswith("earnings night (") and len(v) > 2 else np.nan)
            rows.append(dict(table="groups", series=series, period=lab, group=g, n=len(y), share_of_picks=len(y) / len(x),
                             no_return=int(y.Rp.isna().sum()), cross_missing=float(y.miss.mean()),
                             net_bp_panel=y.net_p.mean(), net_bp_cross=v.mean(), median_bp_cross=v.median(),
                             open_bias_bp=y.bias_o.mean(), open_bias_median_bp=y.bias_o.median(),
                             close_bias_bp=y.bias_c.mean(), hit_cross=(v > 0).mean(), sd_bp_cross=v.std(),
                             p05_bp_cross=v.quantile(0.05), worst_bp_cross=v.min(),
                             pnl_share_cross=(y.w * y.net_x.fillna(0)).sum() / tot,
                             pnl_share_panel=(y.w * y.net_p.fillna(0)).sum() / tot_p, t_vs_rest_cross=t_diff))
            if g.startswith(("earnings night (", "no report")):
                for _, z in y.nsmallest(5, "net_x").iterrows():
                    worst.append(dict(table="worst", series=series, period=lab, group=g, ticker=z.ticker,
                                      entry=z.t.date(), net_bp_cross=z.net_x, net_bp_panel=z.net_p,
                                      open_bias_bp=z.bias_o))
    return rows + worst


def rule_rows(series, D, sel):
    """Rules (a) exclude flagged picks and refill from the next-ranked names; (b) half weight on flagged picks
    (rest cash, or the freed weight spread over the other picks)."""
    S, G = D["S"], D["G"]
    base = D["W"] if sel is topk else D["Wc"]
    rows, nets = [], {}
    for fname in ["EN", "EW"]:
        F = D[fname].reindex_like(S).fillna(False)
        Wex = sel(S.where(~F))
        Wh = base.where(~F, base * 0.5)
        Whr = Wh.div(Wh.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        miss_new = ((Wex > 0) & G["miss"]).loc[START:END].sum().sum() / max((Wex > 0).loc[START:END].sum().sum(), 1)
        for vname, Wv in [("base", base), ("a: exclude, refill next-ranked", Wex), ("b: half weight, rest cash", Wh),
                          ("b2: half weight, freed weight to the others", Whr)]:
            for pr, R in [("panel", G["Rp"]), ("cross", G["Rx"])]:
                net = bt.run(Wv, R, G["C"]).net.loc[START:END]
                nets[(fname, vname, pr)] = net
                bnet = nets[(fname, "base", pr)]
                for lab, a0, b0 in PER:
                    x = net.loc[a0:b0]
                    st = ann_stats(x)
                    dd = (x - bnet.loc[a0:b0])
                    rows.append(dict(table="rules", series=series, flag=fname, variant=vname, pricing=pr, period=lab,
                                     n=st["n"], sharpe=st["sharpe"], net_bp=1e4 * x.mean(), ann=st["ann_ret"],
                                     maxdd=st["maxdd"], diff_bp_vs_base=1e4 * dd.mean(),
                                     t_diff=dd.mean() / dd.std() * np.sqrt(len(dd)) if dd.std() > 0 else np.nan,
                                     flagged_share=float(((base > 0) & F).loc[a0:b0].sum().sum() /
                                                         max((base > 0).loc[a0:b0].sum().sum(), 1)),
                                     cross_missing_legs=miss_new if vname.startswith("a") else np.nan))
    # choice on the first period by cross Sharpe, checked on the holdout
    r = pd.DataFrame(rows)
    r = r[(r.pricing == "cross")]
    out = []
    for fname, g in r.groupby("flag"):
        p1 = g[g.period == PER[0][0]].set_index("variant").sharpe
        best = p1.idxmax()
        h = g[(g.period == PER[1][0])].set_index("variant")
        out.append(dict(table="choice", series=series, flag=fname, variant=best, period=PER[1][0],
                        sharpe_first=p1[best], sharpe_base_first=p1["base"], sharpe=h.sharpe[best],
                        sharpe_base_holdout=h.sharpe["base"], diff_bp_vs_base=h.diff_bp_vs_base[best],
                        t_diff=h.t_diff[best]))
    return rows + out


def study139(D):
    rows = []
    for series, sel in [("live: top 10, max 3 per industry", capped), ("blend top 10 (study 69 base)", topk)]:
        Wx = D["Wc"] if sel is capped else D["W"]
        df = pick_frame(Wx, D)
        rows += group_rows(series, df)
        rows += rule_rows(series, D, sel)
    # timing agreement (picks of the live series in EW)
    df = pick_frame(D["Wc"], D)
    x = df[df.EW.astype(bool)]
    ct = pd.crosstab(x.EN.astype(bool), x.EG.astype(bool))
    for a in ct.index:
        for b in ct.columns:
            rows.append(dict(table="timing_check", series="live: top 10, max 3 per industry", group=f"EN={a}|EG={b}",
                             n=int(ct.loc[a, b])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ study 142
def open_status(df, X):
    """Attach the sell day's (t+1) opening-cross row: rule, px, sz, seconds after 09:30:00, and a status."""
    xo = X[X.kind == "open"][["ticker", "date", "px", "sz", "ts", "x", "rule", "first_px"]].rename(
        columns={"date": "t1", "px": "ox_px", "sz": "ox_sz", "ts": "ox_ts", "x": "ox_x", "rule": "ox_rule",
                 "first_px": "ox_first"})
    d = df.merge(xo, on=["ticker", "t1"], how="left", indicator=True)
    ts = pd.to_datetime(d.ox_ts, utc=True, format="ISO8601").dt.tz_convert("America/New_York").dt.tz_localize(None)
    d["ox_sec"] = (ts - (d.t1 + pd.Timedelta(hours=9, minutes=30))).dt.total_seconds()
    st = np.select([d._merge == "left_only", d.ox_px.isna(), d.ox_rule == "largest",
                    d.ox_sec <= 3, d.ox_sec <= 20, d.ox_sec <= 120],
                   ["not in the caches", "no opening cross by 09:45", "cross only at another exchange (largest print)",
                    "listing cross 09:30:00-03", "listing cross 09:30:03-20", "listing cross 09:30:20-09:32"],
                   "listing cross 09:32-09:45")
    d["status"] = st
    return d.drop(columns="_merge")


EXT = {"T", "U", "Z"}          # extended-hours / out-of-sequence forms


def _trades(sym, day, a, b, max_pages=4, sort="asc", limit=10000):
    p = dict(symbols=sym, start=S94._utc(day, a), end=S94._utc(day, b), limit=limit, feed="sip", sort=sort)
    out, trunc = [], False
    for k in range(max_pages):
        j = A.get("trades", p)
        out += (j.get("trades") or {}).get(sym, [])
        if not j.get("next_page_token"):
            break
        if k == max_pages - 1:
            trunc = True
        p["page_token"] = j["next_page_token"]
    return out, trunc


def _summ(trs, day, ven, t0="09:30:00", t1="09:50:00"):
    """Summary of a trade list in [t0, t1): first regular trade, opening ('O') and reopening ('5') prints, the
    longest pause between trades after the first regular trade (including the pause to the window end)."""
    o = {}
    if not trs:
        return dict(n=0)
    ts = pd.to_datetime([z["t"] for z in trs], utc=True, format="ISO8601").tz_convert("America/New_York").tz_localize(None)
    a, b = pd.Timestamp(f"{day} {t0}"), pd.Timestamp(f"{day} {t1}")
    reg = [(t, z) for t, z in zip(ts, trs) if a <= t < b and not set(z.get("c", [])) & EXT]
    o["n"] = len(reg)
    if reg:
        o["first_ts"], o["first_px"] = reg[0][0], reg[0][1]["p"]
        tt = [t for t, _ in reg] + [b]
        gaps = np.diff(np.array(tt, dtype="datetime64[ns]")).astype("timedelta64[ms]").astype(float) / 1e3
        k = int(np.argmax(gaps))
        o["max_gap_s"], o["max_gap_at"] = gaps[k], tt[k]
        o["max_gap_inner_s"] = gaps[:-1].max() if len(gaps) > 1 else np.nan
    for cond, nm in [("O", "O"), ("5", "R"), ("Q", "OQ")]:
        hit = [(t, z) for t, z in zip(ts, trs) if cond in z.get("c", []) and a - pd.Timedelta(minutes=1) <= t < b]
        hv = [(t, z) for t, z in hit if z.get("x") == ven] if ven else hit
        h = hv or hit
        if h:
            o[f"{nm}_ts"], o[f"{nm}_px"], o[f"{nm}_x"], o[f"{nm}_sz"] = h[0][0], h[0][1]["p"], h[0][1].get("x"), h[0][1]["s"]
            o[f"{nm}_n"] = len(hit)
    return o


def fetch142():
    """Trades of the sell-day morning for a sample of picks (see docstring); resumable."""
    D = setup()
    df = open_status(pick_frame(D["Wc"], D), D["X"])
    df = df[df.t1.notna()]
    rng = np.random.default_rng(142)
    late = df.status.isin(["listing cross 09:30:20-09:32", "listing cross 09:32-09:45"])
    nox = df.status.isin(["no opening cross by 09:45", "not in the caches", "cross only at another exchange (largest print)"])
    big = df.Rp.abs() >= 0.05
    rest = df[~(late | nox | big)]
    rnd = rest.iloc[np.sort(rng.choice(len(rest), 400, replace=False))]
    s = pd.concat([df[late | nox | big].assign(sample="flagged"), rnd.assign(sample="random")])
    s = s[["ticker", "t1", "sample", "status"]].drop_duplicates(["ticker", "t1"])
    have = pd.read_parquet(TR142) if os.path.exists(TR142) else pd.DataFrame(columns=["ticker", "date"])
    have_k = set(zip(have.ticker, pd.to_datetime(have.date)))
    todo = [r for r in s.itertuples() if (r.ticker, r.t1) not in have_k]
    print("sample", len(s), "to fetch", len(todo), flush=True)
    ven = S94.listing_venue()
    rows, t0 = [], time.time()
    for i, r in enumerate(todo):
        sym, day = r.ticker.replace("-", "."), r.t1.strftime("%Y-%m-%d")
        v = ven.get(sym)
        try:
            trs, tr = _trades(sym, day, "09:29:00", "09:50:00")
            o = _summ(trs, day, v)
            o["trunc"] = tr
            if r.status != "listing cross 09:30:00-03" and r.status.startswith(("no ", "not ", "cross only")):
                pre, _ = _trades(sym, day, "04:00:00", "09:30:00", max_pages=1, sort="desc", limit=5)
                pre = [z for z in pre]
                if pre:
                    o["pre_last_ts"] = pd.Timestamp(pre[0]["t"]).tz_convert("America/New_York").tz_localize(None)
                    o["pre_last_px"] = pre[0]["p"]
                if o.get("n", 0) == 0 or "O_ts" not in o:
                    day_trs, _ = _trades(sym, day, "09:50:00", "16:00:00", max_pages=1, limit=2000)
                    od = _summ(day_trs, day, v, "09:50:00", "16:00:00")
                    o.update({f"late_{k}": val for k, val in od.items() if k in ("n", "first_ts", "first_px", "O_ts",
                                                                                   "O_px", "O_x", "R_ts", "R_px")})
        except RuntimeError as e:
            print("err", sym, day, e, flush=True)
            continue
        rows.append(dict(ticker=r.ticker, date=r.t1, sample=r.sample, listing=v, **o))
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            for k in out.columns:
                if out[k].dtype == object and k not in ("ticker", "sample", "listing"):
                    out[k] = out[k].astype("string") if not k.endswith(("_ts", "_at")) else pd.to_datetime(out[k])
            out.to_parquet(TR142, index=False)
            print(i + 1, "/", len(todo), f"{time.time() - t0:.0f}s", flush=True)


def fetch_refill():
    """Crosses for names that enter the books only under rule (a) (exclude and refill), study 94's rule."""
    D = setup()
    nxt = D["nxt"]
    need = []
    for sel in (capped, topk):
        for fname in ["EN", "EW"]:
            F = D[fname].reindex_like(D["S"]).fillna(False)
            Wx = sel(D["S"].where(~F)).loc[START:END]
            s = Wx.stack()
            s = s[s > 0].reset_index()
            s.columns = ["t", "ticker", "w"]
            need += [pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                     pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]
    need = pd.concat(need).dropna().drop_duplicates()
    need["date"] = pd.to_datetime(need.date)
    have = D["X"][["ticker", "date", "kind"]]
    m = need.merge(have, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    todo = todo.assign(ticker=todo.ticker.str.replace("-", ".", regex=False))
    print("refill pairs to fetch", len(todo), flush=True)
    ven = S94.listing_venue()
    rows = []
    for (d, k), g in todo.groupby(["date", "kind"]):
        day = d.strftime("%Y-%m-%d")
        try:
            r = S94._day_kind(day, k, sorted(g.ticker), ven)
            for x in r:
                x["rule"] = "venue"
            miss = [x["ticker"] for x in r if np.isnan(x["px"])]
            if miss:
                lg = {x["ticker"]: x for x in S94._day_kind_largest(day, k, miss)}
                for x in r:
                    if x["ticker"] in lg:
                        x["rule"] = "largest"
                        if not np.isnan(lg[x["ticker"]]["px"]):
                            x.update({q: lg[x["ticker"]][q] for q in ["px", "sz", "ts", "x"]})
            rows += r
        except RuntimeError as e:
            print("err", day, k, e, flush=True)
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_parquet(AUC139) if os.path.exists(AUC139) else pd.DataFrame()
        out = pd.concat([old, new], ignore_index=True)
        for k in ["ts", "x", "rule", "ticker"]:
            out[k] = out[k].astype("string")
        out.to_parquet(AUC139, index=False)
        print("saved", len(out), "found", int(new.px.notna().sum()), flush=True)


def data_check_rows(D, df):
    """Raw cross-to-cross night return vs the ratio method (study 94): picks that disagree by more than 1%."""
    XO, XC = D["Q"]["XO"], D["Q"]["XC"]
    a = XC.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([df.t, df.ticker])).values
    b = XO.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([df.t1, df.ticker])).values
    rr = b / a - 1
    dif = np.abs((1 + rr) / (1 + df.Rx.values) - 1)
    sp = pd.read_csv(f"{DATA}/local/events/splits_yf.csv", parse_dates=["date"])
    spk = set(zip(sp.ticker, sp.date))
    bad = np.where(dif > 0.01)[0]
    rows = []
    for i in bad:
        z = df.iloc[i]
        rows.append(dict(table="data_check", ticker=z.ticker, entry=z.t.date(), raw_cross_bp=1e4 * rr[i],
                         ratio_cross_bp=1e4 * z.Rx, panel_bp=1e4 * z.Rp, split_on_sell_day=(z.ticker, z.t1) in spk,
                         group="earnings night" if z.EN else "other"))
    return rows, df.index[bad]


def study139(D):
    rows = []
    for series, sel in [("live: top 10, max 3 per industry", capped), ("blend top 10 (study 69 base)", topk)]:
        Wx = D["Wc"] if sel is capped else D["W"]
        df = pick_frame(Wx, D)
        rows += group_rows(series, df)
        if sel is capped:
            dc, bad = data_check_rows(D, df)
            rows += dc
            rows += [dict(r, table="groups_without_data_flags") for r in group_rows(series, df.drop(index=bad))
                     if r["table"] == "groups"]
        rows += rule_rows(series, D, sel)
    df = pick_frame(D["Wc"], D)
    x = df[df.EW.astype(bool)]
    ct = pd.crosstab(x.EN.astype(bool), x.EG.astype(bool))
    for a in ct.index:
        for b in ct.columns:
            rows.append(dict(table="timing_check", series="live: top 10, max 3 per industry",
                             group=f"confirmed={a}|gap rule={b}", n=int(ct.loc[a, b])))
    return pd.DataFrame(rows)


def classify_nocross(z):
    """Reason a pick has no listing-venue opening cross by 09:45 (refetched trades, 09:29-09:50 and later)."""
    if z.get("n", 0) == 0 and z.get("late_n", 0) in (0, np.nan) and pd.isna(z.get("late_first_ts")):
        return "no trades that day"
    if pd.notna(z.get("R_ts")) and (pd.isna(z.get("O_ts")) or z["R_ts"] <= z["O_ts"]):
        return "opened by a reopening print (halt cross)"
    if pd.notna(z.get("O_ts")):
        if z.get("O_x") == z.get("listing"):
            return "listing cross found on refetch"
        if z.get("O_x") in ("T",):
            return "cross printed under exchange code T (feed label)"
        return "cross at another listing exchange (listing transfer or asset-list venue wrong)"
    if z.get("n", 0) == 0:
        return "late open after 09:50"
    if pd.notna(z.get("OQ_ts")):
        return "official open reported, no opening-cross print (no auction volume)"
    return "trades but no opening or official-open print"


def study142(D):
    rows = []
    df = open_status(pick_frame(D["Wc"], D), D["X"])
    series = "live: top 10, max 3 per industry"
    XC = D["Q"]["XC"]
    df["xc_t"] = XC.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([df.t, df.ticker])).values
    prc = D["Q"]["PRC"]
    df["prc_t"] = prc.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([df.t, df.ticker])).values
    tot = {lab: (x.w * x.net_x.fillna(0)).sum() for lab, a0, b0 in PER for x in [df[in_per(df.t, a0, b0)]]}
    # (a) status of the opening cross
    for lab, a0, b0 in PER:
        x = df[in_per(df.t, a0, b0)]
        for st, y in x.groupby("status"):
            rows.append(dict(table="a_open_status", series=series, period=lab, status=st, n=len(y), share=len(y) / len(x),
                             earnings_night_share=y.EN.mean(), net_bp_cross=y.net_x.mean(), net_bp_panel=y.net_p.mean(),
                             open_bias_bp=y.bias_o.mean(), median_sec_after_0930=y.ox_sec.median(),
                             pnl_share_cross=(y.w * y.net_x.fillna(0)).sum() / tot[lab]))
    T = pd.read_parquet(TR142) if os.path.exists(TR142) else None
    if T is not None:
        T["date"] = pd.to_datetime(T.date)
        for k in [c for c in T.columns if c.endswith(("_ts", "_at"))]:
            T[k] = pd.to_datetime(T[k])
        m = df.merge(T.rename(columns={"date": "t1"}), on=["ticker", "t1"], how="left", suffixes=("", "_tr"))
        m["fetched"] = m["sample"].notna()
        for k in ["R_ts", "O_ts", "O_x", "OQ_ts", "first_ts", "first_px", "late_n", "late_first_ts", "late_first_px",
                  "max_gap_s", "max_gap_inner_s", "n", "trunc"]:
            if k not in m:
                m[k] = pd.NaT if k.endswith("_ts") else np.nan
        # refetched pairs without a listing cross by 09:45
        nox = m[m.status.isin(["no opening cross by 09:45", "not in the caches",
                               "cross only at another exchange (largest print)"]) & m.fetched].copy()
        nox["reason"] = [classify_nocross(z) for z in nox.to_dict("records")]
        start = nox.t1 + pd.Timedelta(hours=9, minutes=30)
        ft = nox.first_ts.fillna(nox.late_first_ts) if "late_first_ts" in nox else nox.first_ts
        fp = nox.first_px.fillna(nox.late_first_px) if "late_first_px" in nox else nox.first_px
        ref_c = nox.xc_t.fillna(nox.prc_t)
        nox["close_to_first_bp"] = 1e4 * (fp / ref_c - 1)
        nox["delay_s"] = (ft - start).dt.total_seconds()
        nox["close_to_close_bp"] = 1e4 * nox.c_next
        for (st, rs), y in nox.groupby(["status", "reason"]):
            rows.append(dict(table="a_nocross_reasons", series=series, period="2024-26", status=st, reason=rs,
                             n=len(y), years=",".join(sorted(set(y.t1.dt.year.astype(str)))),
                             first_trade_delay_s_median=y.delay_s.median(),
                             close_to_first_trade_bp=y.close_to_first_bp.mean(),
                             close_to_panel_open_bp=1e4 * y.Rp.mean(), close_to_next_close_bp=y.close_to_close_bp.mean(),
                             examples=";".join(f"{a} {b.date()}" for a, b in zip(y.ticker.head(4), y.t1.head(4)))))
        # (b) halts and big gaps
        m["gap20"] = (m.Rp.abs() > 0.2) | (m.Rx.abs() > 0.2)
        m["gap20_dn"] = m.gap20 & (m.Rx < 0)
        m["gap20_up"] = m.gap20 & (m.Rx > 0)
        liquid = m.n.fillna(0) >= 30
        inner = m.max_gap_inner_s.fillna(0)
        trunc = m.trunc.astype("string").fillna("True").str.lower().isin(["true", "1", "1.0"])
        tail = np.where(trunc, 0, m.max_gap_s.fillna(0))
        m["h_reopen"] = m.R_ts.notna()
        m["h_pause"] = liquid & (np.maximum(inner, tail) >= 295)
        m["h_late"] = m.fetched & (m.first_ts.isna() | (m.first_ts > m.t1 + pd.Timedelta(hours=9, minutes=35)))
        m["halt_any"] = m.h_reopen | m.h_pause | m.h_late
        # before the opening cross (affects the MOO): reopening print before or instead of the 'O' print, first
        # trade after 09:35, or the listing cross itself after 09:35
        m["h_pre"] = ((m.R_ts.notna() & (m.O_ts.isna() | (m.R_ts <= m.O_ts))) | m.h_late |
                      (m.fetched & (m.ox_sec > 300)))
        m["h_post"] = m.halt_any & ~m.h_pre
        flagged = m["sample"] == "flagged"
        rnd = m["sample"] == "random"
        for lab, a0, b0 in PER:
            x = m[in_per(m.t, a0, b0)]
            fl, rn = x[x["sample"] == "flagged"], x[x["sample"] == "random"]
            n_rest = int((~x.fetched | (x["sample"] == "random")).sum())
            for nm, col in [("reopening print ('5') 09:29-09:50", "h_reopen"),
                            ("trading pause >= 5 min after the first trade (>= 30 trades)", "h_pause"),
                            ("first regular trade after 09:35", "h_late"),
                            ("any of the three (halt proxy)", "halt_any"),
                            ("  halt or delay before the opening cross (cross after 09:35 or opened by a reopening "
                             "print)", "h_pre"),
                            ("  halt only after the opening cross (MOO already filled)", "h_post"),
                            ("opens > 20% from the close (panel or cross)", "gap20"),
                            ("  of which down", "gap20_dn"), ("  of which up", "gap20_up")]:
                if col.startswith("gap20"):
                    y = x[x[col]]
                    est = len(y)
                else:
                    y = fl[fl[col]]
                    rate_rest = rn[col].mean() if len(rn) else np.nan
                    est = len(y) + rate_rest * (n_rest - len(rn)) + rn[col].sum()
                rows.append(dict(table="b_halts_gaps", series=series, period=lab, event=nm, n=len(y),
                                 share=len(y) / len(x), est_total_incl_unsampled=est, est_share=est / len(x),
                                 random_rest_rate=(rn[col].mean() if not col.startswith("gap20") and len(rn) else np.nan),
                                 random_rest_n=len(rn), net_bp_cross=y.net_x.mean(), net_bp_panel=y.net_p.mean(),
                                 pnl_share_cross=(y.w * y.net_x.fillna(0)).sum() / tot[lab],
                                 pnl_share_panel=(y.w * y.net_p.fillna(0)).sum() / (x.w * x.net_p.fillna(0)).sum(),
                                 earnings_night_share=y.EN.mean(), open_bias_bp=y.bias_o.mean(),
                                 examples=";".join(f"{a} {b.date()}" for a, b in zip(y.ticker.head(5), y.t1.head(5)))))
        del flagged, rnd
    # (c) opening-cross size relative to the order (sell order at t+1's open; buy order at t's close for reference)
    xc = D["X"][D["X"].kind == "close"][["ticker", "date", "px", "sz", "rule"]].rename(
        columns={"date": "t", "px": "cc_px", "sz": "cc_sz", "rule": "cc_rule"})
    df = df.merge(xc, on=["ticker", "t"], how="left")
    df["open_cross_usd"] = (df.ox_px * df.ox_sz).where(df.ox_rule == "venue")
    df["close_cross_usd"] = (df.cc_px * df.cc_sz).where(df.cc_rule == "venue")
    for lab, a0, b0 in PER:
        x = df[in_per(df.t, a0, b0)]
        for kind, col in [("opening cross (sell)", "open_cross_usd"), ("closing cross (buy, reference)", "close_cross_usd")]:
            v = x[col].dropna()
            for acct in [10_000, 25_000, 100_000]:
                sh = 0.1 * acct / v
                rows.append(dict(table="c_cross_size", series=series, period=lab, cross=kind, account=acct,
                                 order_usd=0.1 * acct, n=len(v), no_listing_cross=1 - len(v) / len(x),
                                 cross_usd_median=v.median(), cross_usd_p10=v.quantile(0.1),
                                 order_share_median=sh.median(), order_share_p90=sh.quantile(0.9),
                                 share_over_5pct=(sh > 0.05).mean(), share_over_10pct=(sh > 0.10).mean(),
                                 share_over_25pct=(sh > 0.25).mean(),
                                 earn_night_over_5pct=(sh[x.loc[v.index].EN.astype(bool)] > 0.05).mean()))
    return pd.DataFrame(rows)


def show(df):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 600)
    pd.set_option("display.max_colwidth", 60)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(3).to_string(index=False))


def main():
    D = setup()
    r139 = study139(D)
    r139.to_csv(OUT139, index=False)
    show(r139)
    r142 = study142(D)
    r142.to_csv(OUT142, index=False)
    show(r142)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch_refill()
    fetch142()
    sys.exit()

if __name__ == "__main__":
    main()
