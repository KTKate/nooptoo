"""Studies 96 and 98: index-event closes and the closing-auction volume share, 2024-01..2026-09.

Question.
96  (a) Is the overnight blend (study 33: top 10 by 2 x ensemble rank + jump-minus-drop rank at 15:45, bought in the
    closing auction, sold in the next opening auction) better or worse on nights that start at an index-event close:
    quarterly options expiration (third Friday of Mar/Jun/Sep/Dec, the previous trading day when that Friday is a
    holiday, e.g. 2026-06-18), which is also the S&P 500 quarterly rebalance day; the Russell reconstitution day (last
    Friday of June: 2024-06-28, 2025-06-27, 2026-06-26; the first semiannual December run falls after the sample);
    off-cycle S&P 500 add/delete days (close before an effective date that is not a quarterly rebalance);
    monthly options expiration in other months; month end; quarter end?
    (b) Standalone: on those days, do stocks with unusually large closing crosses (cross size / own 20-day average
    cross size) reverse overnight (close -> next open)? Signed by the auction move (closing price vs the last trade
    before 16:00), net of auction costs; top / bottom decile, event days vs other days.
98  Does the closing-auction share of daily volume (cross size / daily volume), known at 15:45 only up to yesterday,
    predict the overnight return of liquid stocks, and does it add to the blend as a tilt or a filter?

Data.
Closing-cross size: volume of the SIP 5-minute bar starting 16:00 (data/local/m5ah, study 35; raw shares, every
liquid stock-day 2024-01..2026-09). That bar is the cross plus after-hours trades to 16:05; checked against the
official closing print ('6' from the listing venue, Alpaca SIP trades) for a random sample of stock-days fetched
here into data/local/close_cross96.parquet ('validate' step). Daily volume and traded close: Alpaca raw daily bars
(data/local/d1raw.parquet, to 2026-09-25). Last trade before the cross: close of the 15:55 5-minute bar
(data/local/m5snap). Overnight return: panel open(t+1) / close(t) - 1 (the panel open is the first trade, a few bp
above the official cross, study 88; in paired comparisons this mostly cancels). S&P 500 changes:
data/local/events/sp500_events.csv. Blend scores: results/study23_pred.parquet, results/study33_pred.parquet.

Design.
Costs: core.exec_cost_bps(P, "auction") + 2.5 bp per side. Periods: 2024-01..2025-06 (P1, selection) and
2025-07..2026-09 (P2, holdout). Liquid universe: traded close > $5 and 20-day median dollar volume (to t-1) > $5M.
96a: mean blend net return on event nights vs all other nights, Welch t, per period; also skip-the-event rules.
96b: per event day, deciles of the cross ratio; mean overnight return of top / bottom decile (gross, net of two
    auction legs), and for the top decile the reversal portfolio long the names pushed down in the auction and short
    the names pushed up (descriptive: cross size and price are only known after 16:00, so a market-on-close order
    cannot condition on them; reported as an upper bound). Executable check: the same long-short entered after the
    cross at the last price of the 16:00-16:05 bar (gross; after-hours spreads are wider than auction costs).
98: features share_lag1 = share(t-1), share_avg20 = mean share(t-20..t-1), abnormal = share_lag1 - share_avg20,
    and share_avg20 net of its ADV-decile median (the share rises with size). Daily rank IC with the overnight
    return, decile means and D10 - D1 per period. Blend: tilt S + w x rank(feature) with sign and w chosen on P1;
    filter: drop the picks in the feature's worst decile (chosen on P1); paired t of the daily net difference vs
    the baseline.

Output: results/study96_close_events.csv, results/study98_close_share.csv

    python src/study96_98_close_events.py validate   # fetch the official closing print for ~250 stock-days
    python src/study96_98_close_events.py            # both studies
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES
import bt

A.RL = A.RateLimiter(60)
LOC = A.LOCAL
VAL_FN = os.path.join(LOC, "close_cross96.parquet")
PRIMARY = {"NASDAQ": "Q", "NYSE": "N", "AMEX": "A", "ARCA": "P", "BATS": "Z"}
PER = [("P1 2024-01..2025-06", "2024-01", "2025-06"), ("P2 2025-07..2026-09", "2025-07", "2026-09"),
       ("all 2024-01..2026-09", "2024-01", "2026-09")]


# ------------------------------------------------------------------ data
def cross_bars():
    """16:00 5-minute bar (cross + after-hours to 16:05) per stock-day: volume and last price."""
    parts = []
    d = os.path.join(LOC, "m5ah")
    for f in sorted(os.listdir(d)):
        x = pd.read_parquet(os.path.join(d, f))
        x = x[(x.ts.dt.hour == 16) & (x.ts.dt.minute == 0)]
        parts.append(x[["day", "ticker", "v", "c"]])
    x = pd.concat(parts, ignore_index=True)
    x["day"] = pd.to_datetime(x["day"])
    return x


def listing_venue():
    a = pd.concat([pd.read_parquet(os.path.join(LOC, f"alpaca_assets_{k}.parquet")) for k in ["active", "inactive"]])
    a = a.drop_duplicates("symbol")
    return dict(zip(a.symbol, a.exchange.map(PRIMARY)))


def _close_cross(t, day, venue):
    """Official closing print ('6' from the listing venue) for ticker t on day: price and size."""
    for wa, wb in [("15:59:59", "16:00:30"), ("16:00:30", "16:15:00")]:
        a = pd.Timestamp(f"{day} {wa}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        b = pd.Timestamp(f"{day} {wb}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        p = dict(symbols=t, start=a, end=b, limit=10000, feed="sip")
        while True:
            j = A.get("trades", p)
            for x in (j.get("trades") or {}).get(t, []):
                if "6" in x.get("c", []) and (venue is None or x.get("x") == venue):
                    return dict(ticker=t, day=pd.Timestamp(day), cross_px=x["p"], cross_sz=x["s"], cross_x=x.get("x"))
            if not j.get("next_page_token"):
                break
            p["page_token"] = j["next_page_token"]
    return dict(ticker=t, day=pd.Timestamp(day), cross_px=np.nan, cross_sz=np.nan, cross_x=None)


def validate(n=250):
    xb = cross_bars()
    smp = xb.sample(n, random_state=96)
    ven = listing_venue()
    if os.path.exists(VAL_FN):
        have = pd.read_parquet(VAL_FN)
        smp = smp.merge(have[["ticker", "day"]], how="left", indicator=True)
        smp = smp[smp._merge == "left_only"]
    items = [(r.ticker, r.day.strftime("%Y-%m-%d"), ven.get(r.ticker)) for r in smp.itertuples()]
    with ThreadPoolExecutor(3) as ex:
        out = list(ex.map(lambda z: _close_cross(*z), items))
    new = pd.DataFrame(out)
    if os.path.exists(VAL_FN):
        new = pd.concat([pd.read_parquet(VAL_FN), new], ignore_index=True)
    new.to_parquet(VAL_FN, index=False)
    print("fetched", len(out), "found", int(pd.DataFrame(out).cross_sz.notna().sum()))


def check_validation(xb, craw):
    v = pd.read_parquet(VAL_FN).merge(xb, on=["ticker", "day"], how="left")
    v["c_raw"] = [craw.at[d, t] if (d in craw.index and t in craw.columns) else np.nan for d, t in zip(v.day, v.ticker)]
    v = v.dropna(subset=["cross_sz", "v"])
    ratio = v.cross_sz / v.v
    pxd = (v.c / v.cross_px - 1).abs() * 1e4
    cd = (v.c_raw / v.cross_px - 1).abs() * 1e4
    rc = np.corrcoef(np.log(v.cross_sz), np.log(v.v))[0, 1]
    print(f"validation n={len(v)}: cross size / 16:00 bar volume median {ratio.median():.3f}, "
          f"p10 {ratio.quantile(.1):.3f}, p90 {ratio.quantile(.9):.3f}; log corr {rc:.3f}; "
          f"|bar close - cross px| median {pxd.median():.1f} bp; |daily close - cross| median {cd.median():.1f} bp, "
          f">5 bp {np.mean(cd > 5):.2f}")
    return dict(n=len(v), ratio_med=ratio.median(), ratio_p10=ratio.quantile(.1), ratio_p90=ratio.quantile(.9),
                logcorr=rc, close_vs_cross_bp_med=cd.median())


def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def paired_t(d):
    d = pd.Series(d).dropna()
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan


# ------------------------------------------------------------------ event calendar
def event_days(days):
    days = pd.DatetimeIndex(days)
    s = pd.Series(days, index=days)

    def on_or_before(d):
        i = days.searchsorted(d, side="right") - 1
        return days[i] if i >= 0 else None

    ev = {}
    q_opex, m_opex = set(), set()
    for y in range(2024, 2027):
        for m in range(1, 13):
            first = pd.Timestamp(y, m, 1)
            fri3 = pd.date_range(first, periods=31, freq="D")
            fri3 = fri3[(fri3.weekday == 4) & (fri3.month == m)][2]
            d = on_or_before(fri3)
            if d is None or d < days[0] or fri3 > days[-1]:
                continue
            (q_opex if m in (3, 6, 9, 12) else m_opex).add(d)
    ev["quarterly opex / S&P rebalance"] = q_opex
    ev["monthly opex (other months)"] = m_opex
    ev["Russell reconstitution"] = {pd.Timestamp("2024-06-28"), pd.Timestamp("2025-06-27"), pd.Timestamp("2026-06-26")}
    me = [pd.Timestamp(x) for x in s.groupby([s.index.year, s.index.month]).max()]
    ev["month end"] = set(me)
    ev["quarter end"] = {d for d in ev["month end"] if d.month in (3, 6, 9, 12)}
    ev["month end (not quarter end)"] = ev["month end"] - ev["quarter end"]
    sp = pd.read_csv(os.path.join(LOC, "events", "sp500_events.csv"), parse_dates=["eff"])
    sp_days = set()
    for e in sp.eff.dropna().unique():
        i = days.searchsorted(pd.Timestamp(e)) - 1      # index trades at the close before the effective date
        if 0 <= i < len(days):
            sp_days.add(days[i])
    ev["S&P add/delete close (off-cycle)"] = sp_days - q_opex
    ev["any index event (opex, Russell, S&P change)"] = q_opex | ev["Russell reconstitution"] | sp_days
    return {k: pd.DatetimeIndex(sorted(v)) for k, v in ev.items()}


# ------------------------------------------------------------------ main
def main():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    pd.set_option("display.width", 220)
    # ---- blend baseline (study 69 lines 22-46)
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    W = bt.select_topk(S, S.notna(), 10)
    base = bt.run(W, R, C).net
    base = base.loc["2024-01":"2026-09"]
    for p, a, b in PER:
        print(p, "baseline Sharpe", round(ann_stats(base.loc[a:b])["sharpe"], 2), "net bp", round(1e4 * base.loc[a:b].mean(), 1))

    # ---- closing cross panel
    xb = cross_bars()
    xb = xb[xb.ticker.isin(cols)]
    XV = xb.pivot(index="day", columns="ticker", values="v").reindex(index=days, columns=cols).astype("float32")
    XC = xb.pivot(index="day", columns="ticker", values="c").reindex(index=days, columns=cols).astype("float32")
    d1 = pd.read_parquet(os.path.join(LOC, "d1raw.parquet"))
    d1 = d1[d1.ticker.isin(cols)]
    VR = d1.pivot(index="date", columns="ticker", values="v_raw").reindex(index=days, columns=cols).astype("float32")
    tc = traded_close(P)[cols].astype("float32")
    del d1, xb
    rows = []
    vres = check_validation(cross_bars(), tc) if os.path.exists(VAL_FN) else {}
    if vres:
        rows.append(dict(study="96/98", table="validation_16:00_bar_vs_official_cross", **vres))

    adv = P["dv"][cols].rolling(20).median().shift(1)
    liq = (tc > 5) & (adv > 5e6)
    # data checks: cross volume > daily volume, zero daily volume
    share = (XV / VR.where(VR > 0))
    bad = share > 1.0
    print("stock-days with 16:00 bar:", int(XV.notna().sum().sum()), "| share > 1 (bad, set NaN):", int(bad.sum().sum()),
          "| liquid stock-days with bar:", int((liq & XV.notna()).sum().sum()), "of", int(liq.loc["2024":"2026-09"].sum().sum()))
    share = share.mask(bad)
    q = share.where(liq).stack()
    print("share of daily volume in the 16:00 bar (liquid): median", round(q.median(), 3), "p10", round(q.quantile(.1), 3),
          "p90", round(q.quantile(.9), 3))
    # last trade before the cross: close of the 15:55 5-minute bar
    parts = []
    for f in sorted(os.listdir(os.path.join(LOC, "m5snap"))):
        x = pd.read_parquet(os.path.join(LOC, "m5snap", f), columns=["ts", "ticker", "c"])
        x = x[(x.ts.dt.hour == 15) & (x.ts.dt.minute == 55)]
        x["day"] = x.ts.dt.normalize()
        parts.append(x[["day", "ticker", "c"]])
    x = pd.concat(parts, ignore_index=True)
    pre = x.pivot_table(index="day", columns="ticker", values="c", aggfunc="last").reindex(index=days, columns=cols)
    del x, parts
    # auction move = traded close / last trade before 16:00 - 1 (m5snap is split-adjusted as of its fetch, the
    # traded close is raw: a split after the fetch would show as a huge move, so |move| >= 10% is dropped)
    amove = tc / pre - 1
    amove = amove.where(amove.abs() < 0.10)

    # ================================================================ study 96a
    ev = event_days(days[(days >= "2024-01-01") & (days <= "2026-09-30")])
    for k, v in ev.items():
        print(f"{k:45s} {len(v):3d}", ", ".join(d.strftime("%Y-%m-%d") for d in v[:12]), "..." if len(v) > 12 else "")
    # controls: most quarterly-opex nights are Friday (weekend) nights, and month-end nights follow the market;
    # compare with other nights of the same weekday, and use the blend's excess over SPY's overnight return
    spy = (P["o"]["SPY"].shift(-1) / P["c"]["SPY"] - 1).reindex(base.index)
    exc = base - spy
    wd = pd.Series(base.index.weekday, index=base.index)
    for k, v in ev.items():
        for p, a, b in PER:
            bp, xp, sp_, wp = base.loc[a:b], exc.loc[a:b], spy.loc[a:b], wd.loc[a:b]
            ie = bp.index.isin(v)
            e, o = bp[ie], bp[~ie]
            same_wd = (~ie) & wp.isin(set(wp[ie])).values
            skip = bp.where(~ie, 0.0)        # skip rule: stay flat on event nights
            rows.append(dict(study=96, table="blend_event_nights", event=k, period=p, n_event=len(e), n_other=len(o),
                             event_net_bp=1e4 * e.mean(), other_net_bp=1e4 * o.mean(), diff_bp=1e4 * (e.mean() - o.mean()),
                             welch_t=welch(e, o), event_hit=(e > 0).mean() if len(e) else np.nan,
                             diff_vs_same_weekday_bp=1e4 * (e.mean() - bp[same_wd].mean()), welch_t_same_weekday=welch(e, bp[same_wd]),
                             spy_event_bp=1e4 * sp_[ie].mean(), spy_other_bp=1e4 * sp_[~ie].mean(),
                             excess_vs_spy_diff_bp=1e4 * (xp[ie].mean() - xp[~ie].mean()), welch_t_excess=welch(xp[ie], xp[~ie]),
                             sharpe_base=ann_stats(bp)["sharpe"], sharpe_skip=ann_stats(skip)["sharpe"]))
    t96a = pd.DataFrame([r for r in rows if r.get("table") == "blend_event_nights"])
    print(t96a.pivot_table(index="event", columns="period", values=["n_event", "diff_bp", "welch_t", "welch_t_same_weekday", "spy_event_bp", "welch_t_excess", "sharpe_skip"], sort=False).round(1).to_string())

    # ================================================================ study 96b
    xavg = XV.rolling(20, min_periods=15).mean().shift(1)
    ratio = (XV / xavg).where(liq & (xavg > 0))
    Rn = R.reindex(index=days, columns=cols)
    Cs = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=days, columns=cols)
    # sanity: overnight returns beyond +-50% are data errors here
    Rn = Rn.where(Rn.abs() < 0.5)
    # post-close entry: last price of the 16:00-16:05 bar (after the cross is printed and published) -> next open
    Rpost = ((1 + Rn) * tc / XC - 1).where(lambda z: z.abs() < 0.5)
    groups = {k: ev[k] for k in ["quarterly opex / S&P rebalance", "Russell reconstitution", "S&P add/delete close (off-cycle)",
                                 "month end", "quarter end", "monthly opex (other months)"]}
    allev = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in groups.values()])))
    test_days = days[(days >= "2024-02-01") & (days <= "2026-09-29")]
    groups["other days"] = test_days[~test_days.isin(allev)]
    daily = []
    for d in test_days:
        rr = ratio.loc[d].dropna()
        if len(rr) < 200:
            continue
        r = Rn.loc[d, rr.index]
        c = Cs.loc[d, rr.index] / 1e4
        m = amove.loc[d, rr.index]
        dec = pd.qcut(rr.rank(method="first"), 10, labels=False)
        top, botm = dec == 9, dec == 0
        g = dict(day=d, med_ratio=rr.median(), top_ratio=rr[top].median(),
                 all_gross=r.mean(), top_gross=r[top].mean(), bot_gross=r[botm].mean(),
                 top_net=(r[top] - 2 * c[top]).mean(), bot_net=(r[botm] - 2 * c[botm]).mean())
        # reversal within the top decile: long pushed down at the cross, short pushed up
        mt = m[top].dropna()
        if len(mt) >= 20:
            dn, up = mt.index[mt < mt.quantile(0.3)], mt.index[mt > mt.quantile(0.7)]
            g["top_rev_ls_gross"] = r[dn].mean() - r[up].mean()
            g["top_rev_long_net"] = (r[dn] - 2 * c[dn]).mean()
            g["top_rev_short_net"] = (-r[up] - 2 * c[up]).mean()
            g["top_move_ic"] = mt.rank().corr(r[mt.index].rank())
            rp = Rpost.loc[d]
            g["top_rev_ls_post_gross"] = rp[dn].mean() - rp[up].mean()
        ma = m.dropna()
        g["all_move_ic"] = ma.rank().corr(r[ma.index].rank())
        daily.append(g)
    D = pd.DataFrame(daily).set_index("day")
    D["top_minus_all"] = D.top_gross - D.all_gross
    D["bot_minus_all"] = D.bot_gross - D.all_gross
    print("median ratio by group:")
    for k, v in groups.items():
        dd = D[D.index.isin(v)]
        oth = D[D.index.isin(groups["other days"])]
        for p, a, b in PER:
            x, y = dd.loc[a:b], oth.loc[a:b]
            r_ = dict(study=96, table="cross_ratio_deciles", event=k, period=p, n_days=len(x),
                      median_ratio=x.med_ratio.mean(), top_decile_ratio=x.top_ratio.mean())
            for col in ["all_gross", "top_gross", "bot_gross", "top_net", "bot_net", "top_minus_all", "bot_minus_all",
                        "top_rev_ls_gross", "top_rev_long_net", "top_rev_short_net", "top_rev_ls_post_gross", "top_move_ic", "all_move_ic"]:
                s_ = x[col].dropna()
                scale = 1 if col.endswith("ic") else 1e4
                r_[col] = scale * s_.mean()
                r_[col + "_t"] = paired_t(s_)
            if k != "other days":
                r_["top_minus_all_vs_other_t"] = welch(x.top_minus_all, y.top_minus_all)
                r_["top_rev_ls_vs_other_t"] = welch(x.top_rev_ls_gross, y.top_rev_ls_gross)
            rows.append(r_)
    t96b = pd.DataFrame([r for r in rows if r.get("table") == "cross_ratio_deciles"])
    print(t96b[["event", "period", "n_days", "median_ratio", "top_decile_ratio", "top_minus_all", "top_minus_all_t",
                "top_net", "bot_net", "top_rev_ls_gross", "top_rev_ls_gross_t", "top_rev_long_net", "top_rev_short_net",
                "top_rev_ls_post_gross", "top_rev_ls_post_gross_t", "top_move_ic", "all_move_ic", "all_move_ic_t"]].round(2).to_string())
    out96 = pd.DataFrame([r for r in rows if r.get("study") in (96, "96/98")])
    out96.to_csv(f"{RES}/study96_close_events.csv", index=False)

    # ================================================================ study 98
    rows98 = []
    if vres:
        rows98.append(dict(table="validation_16:00_bar_vs_official_cross", **vres))
    sh = share.where(VR > 0)
    f_lag1 = sh.shift(1)
    f_avg20 = sh.rolling(20, min_periods=15).mean().shift(1)
    feats = {"share_lag1": f_lag1, "share_avg20": f_avg20, "abnormal_lag1_minus_avg20": f_lag1 - f_avg20}
    # ADV-neutral version: avg20 minus median of its within-day ADV decile
    advr = adv.where(liq).rank(axis=1, pct=True)
    advd = np.floor(advr * 10).clip(upper=9)
    fa = f_avg20.where(liq)
    st = pd.DataFrame({"f": fa.stack(), "g": advd.stack()}).dropna()
    st["med"] = st.groupby([st.index.get_level_values(0), "g"]).f.transform("median")
    feats["share_avg20_adv_neutral"] = (st.f - st.med).unstack().reindex(index=days, columns=cols)
    del st
    print("corr of share_avg20 rank with ADV rank (mean daily):",
          round(fa.rank(axis=1).corrwith(advr.rank(axis=1), axis=1).mean(), 3))
    tdays = days[(days >= "2024-01-01") & (days <= "2026-09-29")]
    Ru = Rn.where(liq)
    for name, F in feats.items():
        Fu = F.where(liq).reindex(tdays)
        Rt = Ru.reindex(tdays)
        Fu = Fu.where(Rt.notna())
        ic = Fu.rank(axis=1).corrwith(Rt.where(Fu.notna()).rank(axis=1), axis=1)
        dec = Fu.rank(axis=1, pct=True)
        dm = {}
        for k in range(10):
            msk = (dec > k / 10) & (dec <= (k + 1) / 10)
            dm[k] = Rt.where(msk).mean(axis=1)
        dm = pd.DataFrame(dm)
        cn = Cs.reindex(tdays).where(liq.reindex(tdays))
        for p, a, b in PER:
            icp = ic.loc[a:b].dropna()
            spread = (dm[9] - dm[0]).loc[a:b]
            r_ = dict(table="feature_ic", feature=name, period=p, n_days=len(icp), n_names=Fu.loc[a:b].notna().sum(axis=1).mean(),
                      ic_mean=icp.mean(), ic_t=paired_t(icp), d10_minus_d1_bp=1e4 * spread.mean(), d10_d1_t=paired_t(spread))
            for k in range(10):
                r_[f"d{k + 1}_bp"] = 1e4 * dm[k].loc[a:b].mean()
            r_["univ_auction_cost_rt_bp"] = 2 * cn.loc[a:b].stack().mean()
            rows98.append(r_)
    t98 = pd.DataFrame(rows98[1:] if vres else rows98)
    print(t98[["feature", "period", "n_names", "ic_mean", "ic_t", "d1_bp", "d5_bp", "d10_bp", "d10_minus_d1_bp", "d10_d1_t"]].round(3).to_string())

    # ---- blend: tilt and filter (feature ranks within the blend's eligible names; missing feature -> neutral)
    def rec(name, net, choose=""):
        net = net.loc["2024-01":"2026-09"]
        for p, a, b in PER:
            n_, b_ = net.loc[a:b], base.loc[a:b]
            dd = (n_ - b_).dropna()
            st_ = ann_stats(n_)
            rows98.append(dict(table="blend", variant=name, choice=choose, period=p, sharpe=st_["sharpe"], net_bp=1e4 * n_.mean(),
                               base_sharpe=ann_stats(b_)["sharpe"], diff_bp=1e4 * dd.mean(), paired_t=paired_t(dd),
                               overlap=np.nan))
    rec("base: blend top 10", base)
    P1 = slice("2024-01", "2025-06")
    for name in ["share_lag1", "share_avg20", "abnormal_lag1_minus_avg20", "share_avg20_adv_neutral"]:
        F = feats[name].reindex_like(S).where(S.notna())
        Fr = F.rank(axis=1, pct=True).fillna(0.5).where(S.notna())
        # pick-level relation in the baseline picks: which side of the feature is bad?
        res = {}
        for sgn in (1, -1):
            for w in (0.25, 0.5, 1.0):
                St = S + w * sgn * (Fr - 0.5)
                net = bt.run(bt.select_topk(St, St.notna(), 10), R, C).net
                res[(sgn, w)] = net
        best = max(res, key=lambda k: ann_stats(res[k].loc[P1])["sharpe"])
        for k, net in res.items():
            rec(f"98 tilt {name} sign {k[0]:+d} w {k[1]}", net, "chosen on P1" if k == best else "")
        fres = {}
        for side in ("low", "high"):
            for q in (0.1, 0.2):
                m = (Fr <= q) if side == "low" else (Fr > 1 - q)
                Sf = S.where(~(m & F.notna()))
                fres[(side, q)] = bt.run(bt.select_topk(Sf, Sf.notna(), 10), R, C).net
        bestf = max(fres, key=lambda k: ann_stats(fres[k].loc[P1])["sharpe"])
        for k, net in fres.items():
            rec(f"98 filter {name}: drop {k[0]} {int(k[1] * 100)}%", net, "chosen on P1" if k == bestf else "")
    # picks' feature coverage
    cov = feats["share_lag1"].reindex_like(S)[W > 0].notna().sum().sum() / (W > 0).sum().sum()
    print("share of blend picks with share_lag1:", round(cov, 3))
    out = pd.DataFrame(rows98)
    out.to_csv(f"{RES}/study98_close_share.csv", index=False)
    tb = out[out.table == "blend"]
    print(tb.pivot_table(index=["variant", "choice"], columns="period", values=["sharpe", "diff_bp", "paired_t"], sort=False).round(2).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
    else:
        main()
