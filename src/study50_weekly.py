"""Study 50: weekly and monthly calendar windows for ETFs and the liquid stock universe, and for the overnight blend.

Instruments: SPY, QQQ, IWM, DIA, 11 sector ETFs (XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC) and an equal-weight
basket of the liquid stock universe of src/horizon_lib.py (traded price > $5, 20-day median dollar volume > $20M,
membership fixed as of the previous close; survivors only, so its level is biased up - compare rules with the
basket's own buy-and-hold, not with SPY, when judging timing).

Each trading day t is split into two segments, the night ON(t) = close t-1 -> open t and the session ID(t) = open t
-> close t (Yahoo dividend-adjusted o/c). A rule is the set of segments in which it is long (cash otherwise, 0%).
Every switch in or out costs one side: core.exec_cost_bps(P, "auction") + 2.5 bp (opening or closing auction).
Holds are at most 5 trading days. Daily return = compounded segments of the day minus costs paid that day.

Rules (long only)
  week_mon_open_fri_close   first session of the week (open) .. last session of the week (close)
  weekend_fri_close_mon_open  last close of the week -> first open of the next week (one night)
  week_mon_close_fri_close  close of the first session -> close of the last session of the week
  dow_<Mon..Fri>            one close-to-close day ending on that weekday
  tom4                      close of the 2nd-to-last session of the month -> close of the 3rd session (4 days)
  tom3                      last close of the month -> close of the 3rd session of the next month (3 days)
  preholiday_day            close-to-close day ending on the session before a weekday market holiday
  preholiday_session        open -> close of that session
  preholiday_to_post        close before the pre-holiday session -> close of the first session after the holiday
  opex_week                 close before the monthly options-expiration week -> close of opex day (3rd Friday,
                            Thursday when Friday is a holiday)
  opex_mon_after            opex close -> close of the next session
  week_after_opex           opex close -> close of the last session of the next week
  ex_<window>               buy-and-hold except during that window (for the windows above that are not weekday
                            fragments), i.e. "avoid the bad window"
  bh                        buy and hold (one entry)
Periods 2020-23 and 2024-26 (to 2026-09). A rule "beats SPY" when its net Sharpe exceeds SPY buy-and-hold's in both
periods; annual return is shown too (a part-time rule usually earns less in total).

Blend: the study 69 baseline (top 10 overnight picks, auction cost + 2.5 bp/side, 2024-01 .. 2026-09) by calendar
bucket of the entry night (turn of month, opex week, Monday after opex, week after opex, pre-holiday, first/last
session of the week), mean net bp and t, halves 2024-25H1 / 2025H2-26, plus skip-rules' Sharpe.

    python src/study50_weekly.py     # writes results/study50_weekly.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import horizon_lib as H
import bt
from core import exec_cost_bps, ann_stats, stock_cols, RES

warnings.filterwarnings("ignore")
P = H.P
days = P["c"].index
days = days[days >= "2019-12-01"]
END = "2026-09-30"
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", END)]
ETF = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
CA = exec_cost_bps(P, "auction") + 2.5

# ------------------------------------------------------------------ calendar
D = pd.DataFrame(index=days)
D["pos"] = np.arange(len(days))
iso = days.isocalendar()
wk = iso.year.astype(str).values + "-" + iso.week.astype(str).values
D["wk"] = wk
D["first_wk"] = D.wk != D.wk.shift(1)
D["last_wk"] = D.wk != D.wk.shift(-1)
ym = days.to_period("M")
D["ym"] = ym
D["mday"] = D.groupby("ym").cumcount() + 1
D["mday_rev"] = D.groupby("ym").cumcount(ascending=False) + 1      # 1 = last session of the month
nxt = pd.Series(days[1:].append(pd.DatetimeIndex([days[-1] + pd.offsets.BDay(1)])), index=days)
bd_gap = np.busday_count(days.values.astype("datetime64[D]"), nxt.values.astype("datetime64[D]"))
D["preholiday"] = bd_gap > 1                                        # next session more than one weekday away
# monthly opex: 3rd Friday; Thursday if that Friday is not a session
opex = []
for m in pd.period_range(days[0], days[-1], freq="M"):
    f = pd.date_range(m.start_time, m.end_time, freq="W-FRI")[2]
    while f not in days and f > m.start_time:
        f -= pd.Timedelta(days=1)
    if f in days:
        opex.append(f)
opex = pd.DatetimeIndex(opex)
D["opex"] = D.index.isin(opex)
D["opex_wk"] = D.wk.isin(set(D.loc[D.opex, "wk"]))
after = D.pos[D.opex].values + 1
D["mon_after_opex"] = D.pos.isin(after)
D["wk_after_opex"] = D.wk.isin(set(D.loc[D.mon_after_opex, "wk"])) & ~D.opex_wk
T = len(days)


def seg_mask(name):
    """Boolean arrays (ON, ID) of length T: long during the night before / the session of day t."""
    on, idm = np.zeros(T, bool), np.zeros(T, bool)
    f, l = D.first_wk.values, D.last_wk.values
    if name == "bh":
        on[:], idm[:] = True, True
    elif name == "week_mon_open_fri_close":
        inweek = np.ones(T, bool)
        idm[:] = inweek
        on[:] = ~f
    elif name == "weekend_fri_close_mon_open":
        on[:] = f
    elif name == "week_mon_close_fri_close":
        on[:] = ~f
        idm[:] = ~f
    elif name.startswith("dow_"):
        k = ["Mon", "Tue", "Wed", "Thu", "Fri"].index(name[4:])
        m = days.weekday == k
        on[:], idm[:] = m, m
    elif name == "tom4":
        m = (D.mday_rev.values == 1) | (D.mday.values <= 3)
        on[:], idm[:] = m, m
    elif name == "tom3":
        m = D.mday.values <= 3
        on[:], idm[:] = m, m
    elif name == "preholiday_day":
        m = D.preholiday.values
        on[:], idm[:] = m, m
    elif name == "preholiday_session":
        idm[:] = D.preholiday.values
    elif name == "preholiday_to_post":
        ph = D.preholiday.values
        post = np.r_[False, ph[:-1]]
        on[:], idm[:] = ph | post, ph | post
    elif name == "opex_week":
        m = D.opex_wk.values
        on[:], idm[:] = m, m
    elif name == "opex_mon_after":
        m = D.mon_after_opex.values
        on[:], idm[:] = m, m
    elif name == "week_after_opex":
        m = D.wk_after_opex.values
        on[:], idm[:] = m, m
    elif name.startswith("ex_"):
        a, b = seg_mask(name[3:])
        on, idm = ~a, ~b
    else:
        raise KeyError(name)
    return on, idm


WINDOWS = ["week_mon_open_fri_close", "weekend_fri_close_mon_open", "week_mon_close_fri_close",
           "dow_Mon", "dow_Tue", "dow_Wed", "dow_Thu", "dow_Fri", "tom4", "tom3", "preholiday_day",
           "preholiday_session", "preholiday_to_post", "opex_week", "opex_mon_after", "week_after_opex"]
RULES = ["bh"] + WINDOWS + ["ex_" + w for w in ["tom4", "preholiday_day", "opex_week", "opex_mon_after",
                                                 "week_after_opex", "dow_Mon", "weekend_fri_close_mon_open"]]


def rule_returns(r_on, r_id, cost, name):
    """Daily net returns of a rule from segment returns (arrays length T) and per-side cost (decimal)."""
    on, idm = seg_mask(name)
    seq = np.empty(2 * T, bool)
    seq[0::2], seq[1::2] = on, idm
    sw = np.r_[seq[0], seq[1:] != seq[:-1]].astype(float)          # switches at the start of each segment
    if name == "bh":
        sw[:] = 0
        sw[0] = 1
    # cost of a switch at the start of ON(t) is paid at close t-1: use cost known for day t-1 (same day's close
    # auction); at the start of ID(t) the opening auction of day t
    c_on = np.r_[cost[0], cost[:-1]]
    paid = sw[0::2] * c_on + sw[1::2] * cost
    g = (1 + np.where(on, r_on, 0)) * (1 + np.where(idm, r_id, 0)) - 1
    return pd.Series(np.nan_to_num(g) - np.nan_to_num(paid), index=days), on | idm


def segs_etf(t):
    o, c = P["o"][t].reindex(days).values.astype(float), P["c"][t].reindex(days).values.astype(float)
    r_on = o / np.r_[np.nan, c[:-1]] - 1
    r_id = c / o - 1
    return r_on, r_id, CA[t].reindex(days).fillna(5.0).values.astype(float) / 1e4


def segs_basket():
    U = H.universe().shift(1).reindex(days).fillna(False)
    o, c = P["o"][H.cols].reindex(days), P["c"][H.cols].reindex(days)
    on = (o / c.shift(1) - 1).where(U)
    idr = (c / o - 1).where(U)
    on = on.where(on.abs() < 1)
    idr = idr.where(idr.abs() < 1)
    cost = CA[H.cols].reindex(days).where(U).mean(1) / 1e4
    print("basket names per day", U.sum(1).loc["2020":].describe().round(0).to_dict(), flush=True)
    return on.mean(1).values, idr.mean(1).values, cost.values


rows = []


def evaluate(inst, r_on, r_id, cost):
    for name in RULES:
        r, live = rule_returns(r_on, r_id, cost, name)
        live = pd.Series(live, index=days)
        for per, a, b in PER:
            x = r.loc[a:b]
            st = ann_stats(x)
            lv = live.loc[a:b]
            rows.append(dict(part="rule", inst=inst, rule=name, period=per, n_days=st["n"],
                             in_market=float(lv.mean()), sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                             ann_vol=st["ann_vol"], maxdd=st["maxdd"],
                             net_bp_per_live_day=1e4 * x[lv].mean() if lv.any() else np.nan,
                             t_live_day=(x[lv].mean() / x[lv].std() * np.sqrt(lv.sum())) if lv.sum() > 2 else np.nan))


for t in ETF:
    evaluate(t, *segs_etf(t))
    print(t, flush=True)
evaluate("stock_basket", *segs_basket())

# ------------------------------------------------------------------ blend
cols = stock_cols(P)
alld = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < alld[-1]]
pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
W = bt.select_topk(S, S.notna(), 10)
base = bt.run(W, R, C).net.loc["2024-01":"2026-09"].dropna()
Dn = D.reindex(base.index)
nidx = base.index
buckets = {
    "weekday": pd.Series(nidx.day_name(), index=nidx),
    "turn_of_month_entry": pd.Series(np.where(Dn.mday_rev == 2, "2nd-last day", np.where(Dn.mday_rev == 1, "last day",
                                     np.where(Dn.mday <= 3, "day 1-3", "other"))), index=nidx),
    "opex": pd.Series(np.where(Dn.opex, "opex day", np.where(Dn.opex_wk, "opex week (not opex day)",
                      np.where(Dn.mon_after_opex, "day after opex", np.where(Dn.wk_after_opex, "week after opex (rest)",
                                                                             "other")))), index=nidx),
    "preholiday": pd.Series(np.where(Dn.preholiday, "pre-holiday night", "other"), index=nidx),
    "week_position": pd.Series(np.where(Dn.first_wk, "first session of week", np.where(Dn.last_wk, "last session of week",
                               "middle")), index=nidx),
}
BPER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
for g, s in buckets.items():
    for per, a, b in BPER:
        x, gg = base.loc[a:b], s.loc[a:b]
        for lev, v in x.groupby(gg):
            rows.append(dict(part="blend_bucket", inst="blend", rule=f"{g}={lev}", period=per, n_days=len(v),
                             net_bp_per_live_day=1e4 * v.mean(),
                             t_live_day=v.mean() / v.std() * np.sqrt(len(v)) if len(v) > 2 else np.nan))
skips = {"base": pd.Series(False, index=nidx), "skip_monday_entries": pd.Series(nidx.weekday == 0, index=nidx),
         "skip_first_session_of_week": Dn.first_wk.astype(bool),
         "skip_opex_week": Dn.opex_wk.astype(bool), "skip_week_after_opex": (Dn.wk_after_opex | Dn.mon_after_opex).astype(bool),
         "skip_tom4": ((Dn.mday_rev <= 2) | (Dn.mday <= 3)).astype(bool)}
for k, m in skips.items():
    x = base.where(~m, 0.0)
    for per, a, b in BPER:
        st = ann_stats(x.loc[a:b])
        rows.append(dict(part="blend_skip", inst="blend", rule=k, period=per, n_days=st["n"],
                         in_market=float(1 - m.loc[a:b].mean()), sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                         maxdd=st["maxdd"]))

df = pd.DataFrame(rows)
spy = df[(df.part == "rule") & (df.inst == "SPY") & (df.rule == "bh")].set_index("period").sharpe
df["spy_bh_sharpe"] = df.period.map(spy)
r = df[df.part == "rule"]
wide = r.pivot_table(index=["inst", "rule"], columns="period", values="sharpe")
beat = wide[(wide["2020-23"] > spy["2020-23"]) & (wide["2024-26"] > spy["2024-26"])]
df["beats_spy_both"] = df.set_index(["inst", "rule"]).index.isin(beat.index) & (df.part == "rule")
df.to_csv(os.path.join(RES, "study50_weekly.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 600)
print("SPY buy-and-hold Sharpe", spy.round(2).to_dict())
print("\n== rules with Sharpe above SPY buy-and-hold in both periods")
print(beat.round(2).to_string())
for inst in ["SPY", "QQQ", "IWM", "stock_basket"]:
    t = r[r.inst == inst].pivot_table(index="rule", columns="period",
                                      values=["sharpe", "ann_ret", "in_market", "net_bp_per_live_day", "t_live_day"],
                                      sort=False)
    print("\n==", inst)
    print(t.round(2).to_string())
# average over the 15 ETFs: net bp per live day and share of ETFs with positive t
e = r[r.inst.isin(ETF)]
agg = e.groupby(["rule", "period"]).agg(mean_bp=("net_bp_per_live_day", "mean"), mean_sharpe=("sharpe", "mean"),
                                        share_pos=("net_bp_per_live_day", lambda v: (v > 0).mean())).unstack()
print("\n== 15 ETFs average")
print(agg.round(2).to_string())
print("\n== blend buckets")
print(df[df.part == "blend_bucket"].pivot_table(index="rule", columns="period", values=["net_bp_per_live_day", "t_live_day", "n_days"]).round(1).to_string())
print("\n== blend skip rules")
print(df[df.part == "blend_skip"].pivot_table(index="rule", columns="period", values=["sharpe", "ann_ret"]).round(2).to_string())
