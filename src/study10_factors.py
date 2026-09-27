"""Study 10: news, sentiment, event, regulation, short-volume and sector factors for the overnight hold.

All factors are known at 15:45 ET of day t (news_features.py). Target: close(t) -> open(t+1), the trade the
surviving strategy makes (market-on-close buy, market-on-open sell). Universes:
  L+M   price > $5 (traded), 20d median dollar volume > $5M (the ML ranker's universe)
  S     price > $2 (traded), $1-5M
Tests
  A  single factors: each day rank stocks by the factor; mean overnight return of the top and bottom decile
     (and of stocks with the event vs without), val 2024-01..2025-06, holdout 2025-07..2026-09, and 2020-23.
  B  news-conditioned reversal (Chan 2003): intraday losers (open -> 15:45, bottom decile) with news in the
     window vs without; same for winners.
  C  macro calendar: overnight return of the ML strategy and of SPY on nights before FOMC / CPI / jobs days
     (data/store/macro_events.csv).
  D  sector regulation news: sector-level count of government/regulation headlines vs next overnight sector
     return.
Returns are gross (bp); a factor that is not clearly positive gross cannot pay the ~10 bp auction round trip.
Output: results/study10_factors.csv
"""
import os
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, traded_close, ann_stats, RES, DATA
from study8_exec_retest import price_1545
import news_features as NF
import bt

P = load_panel()
cols = stock_cols(P)
F = NF.load()
days = P["c"].loc["2020-01-02":].index[:-1]
rs = lambda x: x.reindex(index=days, columns=cols)
Rn = rs(P["o"][cols].shift(-1) / P["c"][cols] - 1)                       # overnight, total return
adv = rs(P["dv"][cols].rolling(20, min_periods=10).median().shift(1))
pxT = rs(traded_close(P)[cols].shift(1))
pxY = rs(P["rawc"][cols].shift(1))
px = pxT.fillna(pxY) if True else pxT                                     # traded price from 2023-12, Yahoo before
p1545 = rs(price_1545("none"))
o = rs(P["o"][cols] / (P["c"][cols] / P["rawc"][cols]))                   # raw-scale open
intra = (p1545 / o - 1).where(p1545.notna(), rs(P["c"][cols] / P["o"][cols] - 1))   # 15:45 where available
U = {"LM": (px > 5) & (adv > 5e6), "S": (px > 2) & (adv > 1e6) & (adv <= 5e6)}
PER = [("2020-23", "2020-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]
rows = []


def rec(test, name, uni, x, **kw):
    """x: Series indexed by day, mean overnight return of a daily portfolio (decimal)."""
    for per, a, b in PER:
        y = x.loc[a:b].dropna()
        if len(y) < 20:
            continue
        rows.append(dict(test=test, factor=name, universe=uni, period=per, n_days=len(y), bps=1e4 * y.mean(),
                         t=y.mean() / y.std() * np.sqrt(len(y)) if y.std() > 0 else np.nan, **kw))


def daily_mean(mask):
    return Rn.where(mask).mean(axis=1)


newsok = F["n_news"].reindex(index=days, columns=cols).notna()
for uni, E in U.items():
    E = E & Rn.notna() & newsok
    base = daily_mean(E)
    rec("A", "universe_mean", uni, base)
    # continuous factors: decile spread (demeaned by the universe mean each day)
    for k in ["n_news", "n_news5", "sent", "sent5", "short_ratio", "short_z"]:
        if k not in F:
            continue
        f = F[k].reindex(index=days, columns=cols).where(E)
        pct = f.rank(axis=1, pct=True)
        top, bot = daily_mean(E & (pct > 0.9)), daily_mean(E & (pct <= 0.1))
        rec("A", k + "_top_decile", uni, top - base)
        rec("A", k + "_bottom_decile", uni, bot - base)
        rec("A", k + "_top_minus_bottom", uni, top - bot)
    # event flags: with event vs universe
    for k in ["ev_analyst", "ev_up", "ev_down", "ev_fda", "ev_offering", "ev_mna", "ev_gov", "ev_earn"]:
        m = F[k].reindex(index=days, columns=cols) > 0
        rec("A", k, uni, daily_mean(E & m) - base, avg_names=float((E & m).sum(1).loc["2024":].mean()))
    # sentiment within news stocks
    s = F["sent"].reindex(index=days, columns=cols)
    for lo, hi, nm in [(-1.1, -0.5, "sent_neg"), (0.5, 1.1, "sent_pos")]:
        rec("A", nm, uni, daily_mean(E & (s > lo) & (s <= hi)) - base)
    # B: news-conditioned reversal
    ir = intra.where(E).rank(axis=1, pct=True)
    news = F["n_news"].reindex(index=days, columns=cols) > 0
    for side, m in [("loser", ir <= 0.1), ("winner", ir > 0.9)]:
        rec("B", f"intraday_{side}_with_news", uni, daily_mean(E & m & news) - base,
            avg_names=float((E & m & news).sum(1).loc["2024":].mean()))
        rec("B", f"intraday_{side}_no_news", uni, daily_mean(E & m & ~news) - base,
            avg_names=float((E & m & ~news).sum(1).loc["2024":].mean()))
        for lo, hi, nm in [(-1.1, -0.3, "neg"), (0.3, 1.1, "pos")]:
            ms = (s > lo) & (s <= hi)
            rec("B", f"intraday_{side}_news_{nm}", uni, daily_mean(E & m & news & ms) - base,
                avg_names=float((E & m & news & ms).sum(1).loc["2024":].mean()))
    print(uni, "done", flush=True)

# C: macro calendar
fn = os.path.join(DATA, "store", "macro_events.csv")
if os.path.exists(fn):
    ev = pd.read_csv(fn, parse_dates=["date"])
    spyN = (P["o"]["SPY"].shift(-1) / P["c"]["SPY"] - 1).reindex(days)
    ser = pd.read_parquet(f"{RES}/final_series.parquet")["ml_overnight_k10"]
    for e, g in ev.groupby("event"):
        evd = pd.DatetimeIndex(g.date)
        pos = days.searchsorted(evd) - 1                      # the night before the event day (08:30 releases)
        if "FOMC" in e.upper():
            pos = days.searchsorted(evd)                      # FOMC at 14:00: the night after the decision
        nights = days[pos[(pos >= 0) & (pos < len(days))]]
        m = pd.Series(days.isin(nights), index=days)
        for nm, x in [("spy_night", spyN), ("ml_k10_net", ser.reindex(days))]:
            rec("C", f"{e}_{nm}_event_nights", "mkt", x.where(m))
            rec("C", f"{e}_{nm}_other_nights", "mkt", x.where(~m))

# D: sector regulation news -> sector overnight return (equal weight, L+M universe)
sec = F["_sector"]
E = U["LM"] & Rn.notna() & newsok
gov = F["ev_gov"].reindex(index=days, columns=cols)
out = []
for sname in sec.dropna().unique():
    c = [t for t in cols if sec.get(t) == sname]
    if len(c) < 30:
        continue
    cnt = gov[c].where(E[c]).sum(1)
    r = Rn[c].where(E[c]).mean(1)
    z = (cnt - cnt.rolling(60, min_periods=20).mean()) / cnt.rolling(60, min_periods=20).std()
    out.append(pd.DataFrame({"z": z, "r": r - Rn.where(E).mean(1)}))
X = pd.concat(out)
rec("D", "sector_gov_news_high", "LM", X[X.z > 2].groupby(level=0).r.mean())
rec("D", "sector_gov_news_low", "LM", X[X.z < 0].groupby(level=0).r.mean())

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study10_factors.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(df.pivot_table(index=["test", "universe", "factor"], columns="period", values=["bps", "t"]).round(2).to_string())
