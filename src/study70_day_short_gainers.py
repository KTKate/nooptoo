"""Study 70: day-session shorts of the largest overnight gainers (beyond our own picks).

Question: study 53/68 found the overnight blend's picks rise at night and fall in the next day session, and study 62
found premarket gap-ups > 10% fall from the opening auction to the close. Does shorting the market-wide largest
overnight gainers at the opening auction (cover at the closing auction) earn money after costs and borrow, and does it
add to the overnight blend?

Data / signal (decision at about 09:25, before the market-on-open cut-off):
  pm_m5pre  premarket gain = last 5-minute SIP bar close before 09:25 / previous day's 15:55 close - 1
            (data/local/m5pre via study 25's cache data/local/m5pre/_study25_features.pkl, read-only; m5snap
            universe, 2024-01 .. 2026-09). Known in time (needs real-time SIP or IEX premarket prints).
  pm_s62    premarket gain = last trade before 09:25 / previous official close - 1, from data/local/m5s62 (study 62's
            5-minute bars, 2020-01 ..). Those bars were fetched only for stock-days with an OFFICIAL opening gap of
            at least 2% in absolute value, so the candidate set is selected with the open (mild look-ahead: a stock up
            6% at 09:25 that opened only +1.5% is missing). Scale check: (1 + premarket gain) / (1 + official gap)
            within [0.7, 1.43].
  gap_off   official overnight gap = opening auction price / previous close - 1 (2020-01 ..). UPPER BOUND: the open
            is not known when a market-on-open order must be sent; it is the price the order would trade at.
Universe each day t (known before the open): traded price of t-1 > $5 (Yahoo raw close times later splits,
s6162_common.traded_price), 20-day median dollar volume through t-1 > $20M, open and close of t present; ticker-days
with a split dated t or t-1 (splits_yf.csv) are dropped; names that fell >= 10% on t-1 are dropped (short-sale
restriction in force on t, no market short at the open).
Trade: short at the opening auction (P["o"]), cover at the closing auction (P["c"]), adjusted panel, same day.
Costs: auction cost per side = core.exec_cost_bps on traded prices + 2.5 bp, both sides. Borrow: 0 for names on
Alpaca's current easy-to-borrow list (data/local/alpaca_assets_active.parquet, today's list: flatters the past),
30%/yr (11.9 bp) charged once per short-day for the others (an intraday short normally pays no borrow fee, but a
hard-to-borrow locate does cost; this is a rough stand-in). No intraday stops (daily data only).
Sizing: equal weight among the day's selected names, capped at 10% of capital per name (top 5 -> 50% gross, cash rest).
Variants: source (3) x selection (top 5, top 10 by gain; buckets 2-5%, 5-10%, >10%) x catalyst (all; overnight news
or earnings; none) x borrow (all names with the fee; easy to borrow only) = 90 portfolios.
  catalyst: >= 1 Benzinga article (news_features.NF.load_news) for the symbol with timestamp in (16:00 of t-1,
  09:25 of t], or an earnings date (store 'earnings') on t-1 or t (report times are mostly not supplied, so an
  after-close report on t is also counted).
Comparison: the overnight blend (study 68 construction from results/study33_pred.parquet + study23_pred.parquet,
top 10, close t -> open t+1, 2024-01 ..), dated by the morning it ends; correlation with the day short of the same
day, and the blend + quarter-size day short (0.25 x the variant's daily P&L).
Periods: 2020-23, 2024-26, 2024H1-25H1 (2024-01..2025-06), 2025H2-26 (2025-07..2026-09-25).
Survivorship: the panel only holds tickers alive in 2026; gainers that later collapsed and delisted are missing, which
biases the short side DOWN slightly (one-day holds: small).
Output: results/study70_day_short_gainers.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store
import bt
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
from s6162_common import with_traded_price, auction_cost_bps

END = pd.Timestamp("2026-09-25")          # news archive and model predictions end here
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-30"),
       ("2024H1-25H1", "2024-01-01", "2025-06-30"), ("2025H2-26", "2025-07-01", "2026-09-30")]
FEE_HTB = 0.30 / 252                      # per short-day, decimal

P = load_panel()
PT = with_traded_price(P)
cols = stock_cols(P)
days = P["c"].index
days = days[(days >= "2020-01-01") & (days <= END)]
o, c = P["o"][cols], P["c"][cols]
TP = PT["rawc"][cols]
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
ret1 = c / c.shift(1) - 1
univ = (TP.shift(1) > 5) & (adv > 20e6) & o.notna() & c.notna() & c.shift(1).notna() & ~(ret1.shift(1) <= -0.10)
spl = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
spl = spl[spl.ticker.isin(cols)]
splitday = pd.DataFrame(False, index=P["c"].index, columns=cols)
di = pd.Series(np.arange(len(P["c"].index)), index=P["c"].index)
for t, d in zip(spl.ticker, spl.date):
    i = P["c"].index.searchsorted(d)
    if i < len(P["c"].index):
        splitday.iloc[i, cols.index(t)] = True
        if i + 1 < len(P["c"].index):
            splitday.iloc[i + 1, cols.index(t)] = True
univ &= ~splitday
gap_off = o / c.shift(1) - 1
R = c / o - 1
COST = auction_cost_bps(PT)[cols]       # bp per side
a = pd.read_parquet(os.path.join(DATA, "local", "alpaca_assets_active.parquet"))
ETB = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)].str.replace(".", "-", regex=False))


def stack(df, name):
    s = df.loc[days].stack()
    s.index.names = ["date", "ticker"]
    return s.rename(name)


def base_frame(sig):
    """sig: Series indexed (date, ticker) with the overnight gain. Joins returns, costs, flags; keeps universe rows."""
    x = sig.rename("g").to_frame()
    x = x[x.g > 0]
    u = stack(univ, "u")
    x = x.join(u, how="inner")
    x = x[x.u].drop(columns="u")
    x = x.join(stack(R, "R"), how="inner").join(stack(COST, "cost"), how="left").join(stack(gap_off, "gap_off"))
    x = x[x.R.notna()]
    x["cost"] = x.cost.fillna(100.0)
    return x.sort_index()


def overnight_catalyst(idx):
    import news_features as NF
    d = NF.load_news()[["ts", "sym"]]
    d = d[d.sym.isin(cols)]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).values
    alld = P["c"].index
    cut = pd.DatetimeIndex(alld) + pd.Timedelta(hours=9, minutes=25)
    start = pd.DatetimeIndex(np.r_[[pd.Timestamp("1990-01-01")], (pd.DatetimeIndex(alld[:-1]) + pd.Timedelta(hours=16)).values])
    i = cut.searchsorted(ts, side="left")              # first day whose 09:25 cutoff >= ts
    ok = i < len(alld)
    i = np.where(ok, i, 0)
    ok &= ts > start[i].values
    nn = pd.DataFrame({"date": alld[i[ok]], "ticker": d.sym.values[ok]}).groupby(["date", "ticker"]).size()
    E = store.read("earnings")[["symbol", "date"]]
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    E = E[E.date.isin(alld) & E.symbol.isin(cols)]
    k = di.reindex(E.date).values
    pairs = set(zip(k, E.symbol)) | set(zip(k + 1, E.symbol))
    dd = di.reindex(idx.get_level_values(0)).values
    earn = np.array([(a_, t) in pairs for a_, t in zip(dd, idx.get_level_values(1))])
    news = nn.reindex(idx).fillna(0).values > 0
    return pd.Series(np.where(earn, "earn", np.where(news, "news", "none")), index=idx)


def src_gap_off():
    return base_frame(stack(gap_off, "g"))


def src_pm_s62():
    parts = []
    d0 = os.path.join(DATA, "local", "m5s62")
    for fn in sorted(f for f in os.listdir(d0) if f.endswith(".parquet")):
        d = pd.read_parquet(os.path.join(d0, fn), columns=["ts", "ticker", "c"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d = d[hm < 925]                                   # bars starting before 09:25 (trades up to 09:25)
        d["date"] = d.ts.dt.normalize()
        parts.append(d.groupby(["date", "ticker"]).c.last())
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].reset_index()
    s["ticker"] = s.ticker.str.replace(".", "-", regex=False)
    s = s.set_index(["date", "ticker"]).c
    F = P["c"] / P["rawc"]
    pc = (P["rawc"].shift(1) * F.shift(1) / F)[cols]       # previous close in split-only units (Alpaca bar scale)
    pcs = stack(pc, "pc").reindex(s.index)
    g = s / pcs - 1
    go = stack(gap_off, "go").reindex(s.index)
    ratio = (1 + g) / (1 + go)
    okk = (ratio > 0.7) & (ratio < 1 / 0.7)
    print("pm_s62 rows", len(g), "scale mismatch dropped", int((~okk & g.notna()).sum()), flush=True)
    return base_frame(g[okk])


def src_pm_m5pre():
    x = pd.read_pickle(os.path.join(DATA, "local", "m5pre", "_study25_features.pkl"))[["date", "ticker", "gap"]]
    return base_frame(x.set_index(["date", "ticker"]).gap)


def portfolio(x, sel, cat, borrow, d0, d1):
    y = x
    if cat == "catalyst":
        y = y[y.cat != "none"]
    elif cat == "none":
        y = y[y.cat == "none"]
    if borrow == "etb":
        y = y[y.etb]
    if sel.startswith("top"):
        k = int(sel[3:])
        y = y.sort_values("g", ascending=False).groupby(level=0).head(k)
    else:
        lo, hi = {"b2-5": (0.02, 0.05), "b5-10": (0.05, 0.10), "b>10": (0.10, np.inf)}[sel]
        y = y[(y.g >= lo) & (y.g < hi)]
    n = y.groupby(level=0).size()
    w = np.minimum(0.10, 1.0 / n.reindex(y.index.get_level_values(0)).values)
    fee = np.where(y.etb, 0.0, FEE_HTB)
    gross = -w * y.R.values
    cost = w * (2 * y.cost.values / 1e4 + fee)
    ev = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "w": w}, index=y.index).sort_index()
    dd = days[(days >= d0) & (days <= d1)]
    g_ = ev.groupby(level=0)
    daily = pd.DataFrame({"gross": g_.gross.sum(), "cost": g_.cost.sum(), "net": g_.net.sum(),
                          "n": g_.size(), "wsum": g_.w.sum()}).reindex(dd).fillna(0.0)
    return daily, ev


def blend_night():
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    alld = P["c"].index
    ens = ens.loc[ens.index < alld[-2]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    W = bt.select_topk(S, S.notna(), 10)
    night = (o.shift(-1) / c - 1).reindex_like(W)
    cost = (exec_cost_bps(P, "auction")[cols].shift(-1) + 2.5).reindex_like(W)
    r = bt.run(W, night, cost).net
    nxt = pd.Series(alld[1:], index=alld[:-1])
    r.index = pd.DatetimeIndex(nxt.reindex(r.index).values)     # dated by the morning the night ends
    picks = W.copy()
    picks.index = pd.DatetimeIndex(nxt.reindex(W.index).values)
    return r.dropna(), (picks > 0)


def main():
    srcs = {"gap_off": src_gap_off(), "pm_s62": src_pm_s62(), "pm_m5pre": src_pm_m5pre()}
    allidx = pd.concat([v[[]] for v in srcs.values()]).index.unique()
    catall = overnight_catalyst(allidx)
    for k, x in srcs.items():
        x["cat"] = catall.reindex(x.index).values
        x["etb"] = np.array([t in ETB for t in x.index.get_level_values(1)])
        print(k, "rows", len(x), x.index.get_level_values(0).min().date(), x.index.get_level_values(0).max().date(),
              "cat shares", x.cat.value_counts(normalize=True).round(3).to_dict(), "etb", round(x.etb.mean(), 3), flush=True)
    bn, bpicks = blend_night()
    span = {"gap_off": ("2020-01-01", END), "pm_s62": ("2020-01-01", END), "pm_m5pre": ("2024-01-03", END)}
    rows, keep = [], {}
    for sname, x in srcs.items():
        for sel in ["top5", "top10", "b2-5", "b5-10", "b>10"]:
            for cat in ["all", "catalyst", "none"]:
                for borrow in ["all", "etb"]:
                    daily, ev = portfolio(x, sel, cat, borrow, *span[sname])
                    keep[(sname, sel, cat, borrow)] = (daily, ev)
                    for per, a_, b_ in PER:
                        y = daily.loc[a_:b_]
                        if len(y) < 20:
                            continue
                        st = ann_stats(y.net)
                        e = ev.loc[a_:b_]
                        row = dict(source=sname, sel=sel, catalyst=cat, borrow=borrow, period=per, days=len(y),
                                   days_active=int((y.n > 0).sum()), names=y.n.mean(), gross_w=y.wsum.mean(),
                                   gross_bp=1e4 * y.gross.mean(), cost_bp=1e4 * y.cost.mean(), net_bp=1e4 * y.net.mean(),
                                   sharpe=st["sharpe"], tstat=st["tstat"], maxdd=st["maxdd"], ann_ret=st["ann_ret"],
                                   ev_net_bp=1e4 * (e.net / e.w).mean() if len(e) else np.nan,
                                   ev_net_trim_bp=1e4 * (e.net / e.w).clip((e.net / e.w).quantile(.01), (e.net / e.w).quantile(.99)).mean() if len(e) > 50 else np.nan)
                        z = pd.concat([y.net.rename("s"), bn.rename("b")], axis=1, sort=True).dropna()
                        if len(z) > 50:
                            row["corr_blend"] = z.s.corr(z.b)
                            comb = z.b + 0.25 * z.s
                            sc = ann_stats(comb)
                            row.update(comb_net_bp=1e4 * comb.mean(), comb_sharpe=sc["sharpe"], comb_maxdd=sc["maxdd"])
                        rows.append(row)
    for per, a_, b_ in PER:
        y = bn.loc[a_:b_]
        if len(y) > 50:
            st = ann_stats(y)
            rows.append(dict(source="blend_night_only", sel="top10", catalyst="", borrow="", period=per, days=len(y),
                             net_bp=1e4 * y.mean(), sharpe=st["sharpe"], tstat=st["tstat"], maxdd=st["maxdd"],
                             ann_ret=st["ann_ret"], comb_net_bp=1e4 * y.mean(), comb_sharpe=st["sharpe"], comb_maxdd=st["maxdd"]))
    spy = (P["c"]["SPY"] / P["o"]["SPY"] - 1)
    for per, a_, b_ in PER:
        st = ann_stats(spy.loc[a_:b_].loc[:END])
        rows.append(dict(source="SPY_day_session_long", period=per, net_bp=1e4 * spy.loc[a_:b_].loc[:END].mean(),
                         sharpe=st["sharpe"], maxdd=st["maxdd"]))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study70_day_short_gainers.csv"), index=False)

    # ---------------- checks
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print("\n== overlap: share of top-10 gainers (pm_m5pre, all) that are blend picks the night before:")
    _, ev = keep[("pm_m5pre", "top10", "all", "all")]
    bp = bpicks.stack()
    bp = bp[bp]
    print(round(ev.index.isin(bp.index).mean(), 4))
    for key in [("gap_off", "top10", "all", "all"), ("pm_m5pre", "top10", "all", "all"), ("pm_s62", "top10", "all", "all")]:
        _, ev = keep[key]
        x = srcs[key[0]].reindex(ev.index)
        z = ev.assign(R=x.R, g=x.g, go=x.gap_off).sort_values("net")
        print("\n== largest single-name losses / gains (per unit weight)", key)
        print(z.head(6)[["g", "go", "R", "net"]].to_string())
        print(z.tail(4)[["g", "go", "R", "net"]].to_string())
    # same events, both sources (2024+): does the premarket gain rank like the official gap?
    a1 = srcs["pm_m5pre"][["g", "gap_off"]].loc["2024-01-01":]
    print("\ncorr premarket gain vs official gap (m5pre rows, gain>0):", round(a1.g.corr(a1.gap_off), 3))
    show = res[(res.catalyst.isin(["all", "catalyst", "none"])) & res.period.isin(["2020-23", "2024-26"])]
    v = show.pivot_table(index=["source", "sel", "catalyst", "borrow"], columns="period",
                         values=["net_bp", "sharpe", "maxdd", "corr_blend", "comb_sharpe"])
    print(v.round(2).to_string())
    print(res[res.source.isin(["blend_night_only", "SPY_day_session_long"])][["source", "period", "net_bp", "sharpe", "maxdd"]].round(3))


if __name__ == "__main__":
    main()
