"""Studies 197, 198 and 200: the full day cycle (night leg + day long + day short) at official cross prices.

197  Cycle weights. Question: study 195 found that the study 71 cycle (night + 0.5 day long + 0.25 day short) beat the
     night leg alone in 2026Q3 (+14.9 vs -3.1 bp a night, t 1.45) and over 2024H2-26 (Sharpe 2.44 vs 2.17, t 1.7).
     Which day-leg sizes are best, chosen on 2024H2-25H1 only, and what do they need in gross exposure and margin?
198  Day-short size by regime. Question: does a rule fixed in advance, w_S = 0.25 normally and 0.5 when the night leg
     is weak or VIX is high, beat a fixed 0.25 day short?
200  How the legs move together. Question: are the night, day-short and day-long returns correlated, and is the day
     short a dependable offset (higher return when the night leg is weak)?

Data
  Everything as study 195 (src/study194_195_alternatives.py, whose prep / day_long_picks are reused): core.load_panel
  (ends 2026-10-19), the live capped blend (study135_137 build() score, top 10, at most 3 per industry), official
  crosses from every data/local/auctions*.parquet (study 166 cached_crosses, including auctions194 fetched by study
  195; nothing new is fetched here, so no Alpaca requests), study 94 ratio method (panel return x cross / panel price;
  legs without a usable cross at the panel price), easy-to-borrow from today's Alpaca asset list, study 60 p_all
  predictions (results/study60_pred.parquet, 2024-07..2026-09-25), VIX = panel column ^VIX (close).
Legs, all aligned to the entry night t (night t = close cross of t to open cross of t+1; the day legs run in the day
  session of t+1, from the open cross to the close cross):
  night      capped blend top 10, equal weight, cost core.exec_cost_bps(P, 'auction') + 2.5 bp per side.
  day short  short the easy-to-borrow names among night t's capped picks, 1/10 of capital each (so standalone gross
             = names / 10), cost auction + 2.5 bp per side of day t+1 (study 68 / 195). No borrow fee.
  day long   study 60 p_all top 10 of day t+1 (price > $5, ADV > $5M), equal weight, study 25 cost column x 2.
  Book = night + w_L day long + w_S day short (returns on equity; a missing day long counts as 0, as study 195).
Design 197: grid w_L in {0, 0.25, 0.5} x w_S in {0, 0.25, 0.5}. Per period (2024H2-25H1 choose, 2025H2-26H1,
  2026Q3, 2024H2-26): nights, Sharpe, mean bp, max drawdown, worst night, compounded, mean difference vs the night
  leg alone and its paired t, and the day-session gross exposure (w_L + w_S x names shorted / 10, mean and max) and
  net exposure. The combo with the best 2024H2-25H1 Sharpe is the pick; later periods are out of sample.
Design 198: w_L = 0.5 (study 71) and also w_L = 0; w_S = 0.25, raised to 0.5 when any of (a) the mean of the 20
  night-leg returns through night t-1 is below 0, (b) night t-1 lost more than 1%, (c) VIX close of day t is above
  its median over the 60 trading days through t. All of these are known at the close of day t, before the day-short
  orders for the open of t+1. A sensitivity version uses night t itself in (a) and (b) (that night ends at the same
  opening auction the short enters, so it is not known when on-open orders must be in; upper bound only). Per
  period vs fixed 0.25: Sharpe, mean, max drawdown, worst night, share of days at 0.5, paired t; and the day-short
  leg on raised vs normal days for each condition.
Design 200: correlations (Pearson and Spearman) of the three standalone legs, over all nights with all three legs,
  per period, on day sessions after night t lost more than 1%, and when the 20-night night-leg mean through night t
  is below 0. Mean day-short (and day-long) return by tercile of the same cycle's night-leg return (night t, then
  the day session of t+1), per period, with the tercile cut points from the whole sample, plus the share of weak-
  night days on which the day short made money.

Output: results/study197_cycle_weights.csv (table = grid / chosen), results/study198_dayshort_regime.csv
  (table = book / leg_by_condition), results/study200_leg_corr.csv (table = corr / terciles).

    python src/study197_200_cycle.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from core import exec_cost_bps, ann_stats, RES   # noqa: E402
import bt   # noqa: E402
import study166_167_blend_parts as S166   # noqa: E402
import study194_195_alternatives as S195   # noqa: E402

OUT197 = os.path.join(RES, "study197_cycle_weights.csv")
OUT198 = os.path.join(RES, "study198_dayshort_regime.csv")
OUT200 = os.path.join(RES, "study200_leg_corr.csv")
PER = S195.PER195          # 2024H2-25H1 / 2025H2-26H1 / 2026Q3 / 2024H2-26
GRID = [0, 0.25, 0.5]


# ------------------------------------------------------------------ legs (study 195 code, capped picks only)
def build_legs():
    import study99_cross_retrain as S99
    K = S195.prep(with_oct=False)
    P, cols, days, S = K["P"], K["cols"], K["days"], K["S"]
    X = S166.cached_crosses().dropna(subset=["px"])[["ticker", "date", "kind", "px"]]
    X["ticker"] = X.ticker.str.replace(".", "-", regex=False)
    ro, rc, err = S99.cross_ratios(P, cols, days, X)
    del X
    print("cross data errors (legs at panel price):", err, flush=True)
    ro, rc = ro.reindex(index=days, columns=cols), rc.reindex(index=days, columns=cols)
    fo_, fc_ = ro.fillna(1.0), rc.fillna(1.0)
    o, c = P["o"][cols], P["c"][cols]
    ecost = exec_cost_bps(P, "auction")[cols]
    # night
    Rp = (o.shift(-1) / c - 1).reindex(index=S.index)
    Rx = (1 + Rp) * fo_.shift(-1).reindex(index=S.index) / fc_.reindex(index=S.index) - 1
    C = (ecost + 2.5).reindex(index=S.index)
    W = K["W"]["blend"]
    has = W.sum(axis=1) > 0
    night = bt.run(W, Rx, C).net.where(has).dropna()
    # day short of the easy-to-borrow capped picks
    etb = np.array([t in K["etb"] for t in cols])
    etbm = pd.DataFrame(np.broadcast_to(etb, S.shape), index=S.index, columns=cols)
    Rd_p = (c.shift(-1) / o.shift(-1) - 1).reindex(index=S.index)
    Rd_x = (1 + Rd_p) * fc_.shift(-1).reindex(index=S.index) / fo_.shift(-1).reindex(index=S.index) - 1
    cost_d = (ecost.shift(-1) + 2.5).reindex(index=S.index)
    ws = ((W > 0) & etbm).astype(float).div(10)
    short = bt.run(ws, Rd_x, cost_d, side=-1).net.where(has).dropna()
    nshort = ws.gt(0).sum(axis=1).where(has).dropna()
    # day long
    prev = pd.Series(days[:-1], index=days[1:])
    dl = S195.day_long_picks()
    dk = pd.MultiIndex.from_arrays([dl.date, dl.ticker])
    fac = fc_.stack(future_stack=True).reindex(dk).values / fo_.stack(future_stack=True).reindex(dk).values
    dl["R_x"] = (1 + dl.R.values) * np.nan_to_num(fac, nan=1.0) - 1
    gq = dl.groupby("date")
    dln = gq.R_x.mean() - 2 * gq.cost.mean() / 1e4
    dln.index = dln.index.map(prev)
    dln = dln[dln.index.notna()]
    vix = P["c"]["^VIX"].copy()
    del K, P, ro, rc, fo_, fc_, o, c, Rp, Rx, Rd_p, Rd_x
    idx = night.loc["2024-07":"2026-09"].index
    L = pd.DataFrame({"night": night.reindex(idx), "day_short": short.reindex(idx), "day_long": dln.reindex(idx),
                      "n_short": nshort.reindex(idx)})
    L["vix"] = vix.reindex(idx)
    # regime inputs: full-history night series so the 20-night window is full from 2024-07
    tr20 = night.rolling(20, min_periods=20).mean()
    L["tr20_thru_t"] = tr20.reindex(idx)
    L["tr20_thru_tm1"] = tr20.shift(1).reindex(idx)
    L["night_tm1"] = night.shift(1).reindex(idx)
    vmed = vix.rolling(60, min_periods=60).median()
    L["vix_med60"] = vmed.reindex(idx)
    return L


# ------------------------------------------------------------------ helpers
def stat(x):
    x = x.dropna()
    s = ann_stats(x)
    return dict(nights=len(x), sharpe=s["sharpe"], mean_bp=1e4 * x.mean(), maxdd=s["maxdd"],
                worst_night_bp=1e4 * x.min(), compounded=float((1 + x).prod() - 1), hit=float((x > 0).mean()))


def book(L, wl, ws):
    return L.night + wl * L.day_long.fillna(0) + ws * L.day_short.fillna(0)


# ------------------------------------------------------------------ 197
def study197(L):
    rows = []
    for wl in GRID:
        for ws in GRID:
            b = book(L, wl, ws)
            gross = wl * L.day_long.notna() + ws * L.n_short / 10
            net = wl * L.day_long.notna() - ws * L.n_short / 10
            for lab, a, z in PER:
                x, n = b.loc[a:z], L.night.loc[a:z]
                rows.append(dict(table="grid", w_long=wl, w_short=ws, period=lab, **stat(x),
                                 diff_vs_night_bp=1e4 * (x - n).mean(),
                                 t_vs_night=S166.paired_t(x, n) if (wl or ws) else np.nan,
                                 night_sharpe=ann_stats(n)["sharpe"], night_mean_bp=1e4 * n.mean(),
                                 day_gross_mean=gross.loc[a:z].mean(), day_gross_max=gross.loc[a:z].max(),
                                 day_net_mean=net.loc[a:z].mean(),
                                 peak_gross_at_close_orders=1 + gross.loc[a:z].max()))
    D = pd.DataFrame(rows)
    g = D[D.period == PER[0][0]].sort_values("sharpe", ascending=False)
    best = g.iloc[0]
    for _, r in D[(D.w_long == best.w_long) & (D.w_short == best.w_short)].iterrows():
        rows.append({**r.to_dict(), "table": "chosen"})
    return pd.DataFrame(rows), (best.w_long, best.w_short)


# ------------------------------------------------------------------ 198
def regime_flags(L, same_night=False):
    tr = L.tr20_thru_t if same_night else L.tr20_thru_tm1
    nt = L.night if same_night else L.night_tm1
    a = tr < 0
    b = nt < -0.01
    c = L.vix > L.vix_med60
    return pd.DataFrame({"a_trail20_neg": a, "b_prev_night_lt_-1pct": b, "c_vix_above_med60": c, "any": a | b | c})


def study198(L):
    rows = []
    for vers, same in [("rule (known at close of t)", False), ("sensitivity: uses night t (not knowable in time)", True)]:
        F = regime_flags(L, same)
        wsv = np.where(F["any"], 0.5, 0.25)
        for wl in [0.5, 0.0]:
            fixed = book(L, wl, 0.25)
            reg = L.night + wl * L.day_long.fillna(0) + wsv * L.day_short.fillna(0)
            fixed5 = book(L, wl, 0.5)
            for lab, a, z in PER:
                n = L.night.loc[a:z]
                for name, x in [("fixed 0.25", fixed), ("regime 0.25/0.5", reg), ("fixed 0.5", fixed5),
                                ("night alone", L.night)]:
                    if vers.startswith("sens") and name != "regime 0.25/0.5":
                        continue
                    xx = x.loc[a:z]
                    rows.append(dict(table="book", version=vers, w_long=wl, book=name, period=lab, **stat(xx),
                                     share_days_at_0p5=float(F["any"].loc[a:z].mean()) if name.startswith("regime") else np.nan,
                                     diff_vs_fixed025_bp=1e4 * (xx - fixed.loc[a:z]).mean(),
                                     t_vs_fixed025=S166.paired_t(xx, fixed.loc[a:z]) if name != "fixed 0.25" else np.nan,
                                     diff_vs_night_bp=1e4 * (xx - n).mean(),
                                     t_vs_night=S166.paired_t(xx, n) if name != "night alone" else np.nan))
        # day-short leg on flagged vs other days
        for cond in F.columns:
            for lab, a, z in PER:
                f = F[cond].loc[a:z]
                s = L.day_short.loc[a:z]
                for on in [True, False]:
                    v = s[f == on].dropna()
                    rows.append(dict(table="leg_by_condition", version=vers, book=cond, period=lab,
                                     flag=on, nights=len(v), day_short_bp=1e4 * v.mean(),
                                     day_short_hit=float((v > 0).mean()) if len(v) else np.nan,
                                     night_bp=1e4 * L.night.loc[a:z][f == on].mean(),
                                     day_short_t=v.mean() / v.std() * np.sqrt(len(v)) if len(v) > 4 else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 200
def study200(L):
    rows = []
    J = L[["night", "day_short", "day_long"]]
    subsets = [("all nights", lambda d: d.index == d.index)]
    for lab, a, z in PER:
        subsets.append((f"period {lab}", lambda d, a=a, z=z: (d.index >= pd.Timestamp(a)) &
                        (d.index <= pd.Timestamp(z) + pd.offsets.MonthEnd(0))))
    subsets += [("night t < -1% (day after a losing night)", lambda d: (d.night < -0.01).values),
                ("trailing 20-night mean through t < 0", lambda d: (L.tr20_thru_t.reindex(d.index) < 0).values),
                ("night t >= -1%", lambda d: (d.night >= -0.01).values)]
    for lab, f in subsets:
        for legs_used in [["night", "day_short", "day_long"], ["night", "day_short"]]:
            d = J[legs_used].dropna()
            d = d[f(d)]
            if len(d) < 5:
                continue
            cp, cs = d.corr(), d.corr(method="spearman")
            r = dict(table="corr", subset=lab, legs=" ".join(legs_used), nights=len(d),
                     night_bp=1e4 * d.night.mean(), day_short_bp=1e4 * d.day_short.mean(),
                     corr_night_short=cp.loc["night", "day_short"], spearman_night_short=cs.loc["night", "day_short"],
                     day_short_hit=float((d.day_short > 0).mean()))
            if "day_long" in legs_used:
                r.update(day_long_bp=1e4 * d.day_long.mean(), corr_night_long=cp.loc["night", "day_long"],
                         corr_short_long=cp.loc["day_short", "day_long"],
                         spearman_night_long=cs.loc["night", "day_long"],
                         spearman_short_long=cs.loc["day_short", "day_long"])
            rows.append(r)
    d = J.dropna(subset=["night", "day_short"])
    q = d.night.quantile([1 / 3, 2 / 3]).values
    terc = pd.cut(d.night, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
    for lab, a, z in [("all", "2024-07", "2026-09")] + PER:
        dd, tt = d.loc[a:z], terc.loc[a:z]
        for k in ["low", "mid", "high"]:
            v = dd[tt == k]
            rows.append(dict(table="terciles", subset=lab, tercile=k, cut_low_bp=1e4 * q[0], cut_high_bp=1e4 * q[1],
                             nights=len(v), night_bp=1e4 * v.night.mean(), day_short_bp=1e4 * v.day_short.mean(),
                             day_short_hit=float((v.day_short > 0).mean()) if len(v) else np.nan,
                             day_short_t=v.day_short.mean() / v.day_short.std() * np.sqrt(len(v)) if len(v) > 4 else np.nan,
                             day_long_bp=1e4 * v.day_long.mean(),
                             book71_bp=1e4 * (v.night + 0.5 * v.day_long.fillna(0) + 0.25 * v.day_short).mean()))
        lo, hi = dd.day_short[tt == "low"], dd.day_short[tt == "high"]
        rows.append(dict(table="terciles", subset=lab, tercile="low minus high", nights=len(lo) + len(hi),
                         day_short_bp=1e4 * (lo.mean() - hi.mean()),
                         day_short_t=(lo.mean() - hi.mean()) / np.sqrt(lo.var() / len(lo) + hi.var() / len(hi))))
    return pd.DataFrame(rows)


def main():
    L = build_legs()
    chk = {lab: (1e4 * L.night.loc[a:z].mean(), 1e4 * L.day_short.loc[a:z].mean(), 1e4 * L.day_long.loc[a:z].mean())
           for lab, a, z in PER}
    print("check vs study 195 (night / day short / day long bp):", {k: tuple(round(v, 1) for v in t) for k, t in chk.items()})
    D197, best = study197(L)
    D197.to_csv(OUT197, index=False)
    D198 = study198(L)
    D198.to_csv(OUT198, index=False)
    D200 = study200(L)
    D200.to_csv(OUT200, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    g = D197[D197.table == "grid"]
    for v in ["sharpe", "mean_bp", "maxdd", "worst_night_bp", "t_vs_night"]:
        print("\n197", v)
        print(g.pivot_table(index=["w_long", "w_short"], columns="period", values=v)[[p[0] for p in PER]].round(3).to_string())
    print("\n197 exposure")
    print(g[g.period == "2024H2-26"][["w_long", "w_short", "day_gross_mean", "day_gross_max", "day_net_mean",
                                      "peak_gross_at_close_orders"]].round(3).to_string(index=False))
    print("chosen on 2024H2-25H1:", best)
    b = D198[D198.table == "book"]
    print("\n198")
    print(b[["version", "w_long", "book", "period", "nights", "sharpe", "mean_bp", "maxdd", "worst_night_bp",
             "share_days_at_0p5", "diff_vs_fixed025_bp", "t_vs_fixed025", "t_vs_night"]].round(3).to_string(index=False))
    lc = D198[D198.table == "leg_by_condition"]
    print(lc[["version", "book", "period", "flag", "nights", "day_short_bp", "day_short_hit", "night_bp",
              "day_short_t"]].round(2).to_string(index=False))
    print("\n200")
    print(D200[D200.table == "corr"].drop(columns=["table"]).dropna(axis=1, how="all").round(3).to_string(index=False))
    print(D200[D200.table == "terciles"].drop(columns=["table"]).dropna(axis=1, how="all").round(2).to_string(index=False))
    pr = L.loc["2024-07":"2026-09"]
    print("nights", len(pr), "day long missing", int(pr.day_long.isna().sum()), "day short missing",
          int(pr.day_short.isna().sum()), "mean names shorted", round(pr.n_short.mean(), 2),
          "vix missing", int(pr.vix.isna().sum()), "vix med missing", int(pr.vix_med60.isna().sum()))


if __name__ == "__main__":
    main()
    import resource
    print("peak memory GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
