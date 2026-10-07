"""Study 83: the study-60 day-session long model rebuilt with the inputs the free Alpaca plan delivers at 09:25 ET.

Free plan: SIP data is delayed 15 minutes, so at 09:25 (opening-auction orders by about 09:28) only SIP 5-minute bars
ending by 09:10 exist; IEX trades are real time but IEX premarket volume is thin.
Versions of study 25's premarket inputs (same ticker-day rows as data/local/m5pre/_study25_features.pkl):
  sip0925   reference: the original cache (m5pre bars ending by 09:25)
  sip0910   (a) m5pre bars ending by 09:10 (bar starts <= 09:05); pm_late / pm_900 window moved 15 minutes earlier
            (last price at bar start <= 08:45); pm_mins measured to 09:25 as before
  sip0910_iex  (b) (a) plus the latest IEX trade 09:10:00-09:25:00 (src/study83_fetch_iex.py, data/local/iex83):
            gap, gap_rel, gap_z, mkt_gap use the IEX price when there was an IEX trade, the 09:10 SIP price otherwise;
            extra inputs gap_sip (09:10 SIP gap), iex_move (IEX last / SIP 09:10 last - 1), iex_n, iex_relvol, iex_mins
The price inputs keep study 25's per ticker-month rescaling (derived from the original cache). Everything else (close,
analyst, fundamental inputs, walk-forward by quarter 2024Q3..2026Q3 with 5-day embargo, LightGBM settings, top-10/20
book at the auctions, cost) is study 60's 'all' set. The study-71 day cycle (night blend + a * day long + b * day
short) is recomputed with each version's day-long leg.
LightGBM runs in this one process (no fork after OpenMP).
Output: results/study83_day_long_freeplan.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import analyst_features as AF
from core import load_panel, stock_cols, ann_stats, RES, DATA

CACHE = os.path.join(DATA, "local", "study83")
os.makedirs(CACHE, exist_ok=True)
PRICE = ["pm_first", "pm_last", "pm_hi", "pm_lo", "pm_900"]


# ------------------------------------------------------------------ premarket aggregates (study 25's pm_features)
def pm_raw(cut_hm, late_hm):
    """Study 25's per ticker-day aggregates from m5pre bars with bar start <= cut_hm; 'late' price at start <= late_hm."""
    fn = os.path.join(CACHE, f"pm_{cut_hm}_{late_hm}.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    out = []
    for f in sorted(f for f in os.listdir(os.path.join(DATA, "local", "m5pre")) if f.endswith(".parquet")):
        d = pd.read_parquet(os.path.join(DATA, "local", "m5pre", f))
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d = d[hm <= cut_hm].copy()
        d["date"] = d.ts.dt.normalize()
        d["dvb"] = d.v * d.vwap.astype("float64")
        d["late"] = np.where(d.ts.dt.hour * 100 + d.ts.dt.minute <= late_hm, d.c, np.nan)
        g = d.groupby(["ticker", "date"], sort=False)
        out.append(g.agg(pm_first=("o", "first"), pm_last=("c", "last"), pm_hi=("h", "max"), pm_lo=("l", "min"),
                         pm_dv=("dvb", "sum"), pm_n=("n", "sum"), pm_bars=("c", "size"), pm_900=("late", "last"),
                         pm_tlast=("ts", "last")).reset_index())
    x = pd.concat(out, ignore_index=True)
    x.to_pickle(fn)
    return x


def derive(x):
    """Study 25's derived premarket inputs."""
    x["gap"] = x.pm_last / x.prevc - 1
    x["pm_trend"] = x.pm_last / x.pm_first - 1
    x["pm_range"] = (x.pm_hi - x.pm_lo) / x.prevc
    x["pm_late"] = x.pm_last / x.pm_900 - 1
    x["pm_mins"] = (x.date + pd.Timedelta(hours=9, minutes=25) - x.pm_tlast).dt.total_seconds() / 60
    x["relvol"] = x.pm_dv / x.adv20
    regap(x)
    return x


def regap(x):
    x["mkt_gap"] = x.groupby("date").gap.transform("median")
    x["gap_rel"] = x.gap - x.mkt_gap
    x["gap_z"] = x.gap / x.vol20


def versions():
    x0 = pd.read_pickle(f"{DATA}/local/m5pre/_study25_features.pkl")
    x0["date"] = pd.to_datetime(x0.date)
    x0 = x0.reset_index(drop=True)
    pmcols = ["pm_first", "pm_last", "pm_hi", "pm_lo", "pm_dv", "pm_n", "pm_bars", "pm_900", "pm_tlast"]
    r25 = pm_raw(920, 900)
    k = x0[["ticker", "date", "pm_last"]].merge(r25[["ticker", "date", "pm_last"]], on=["ticker", "date"], how="left",
                                                suffixes=("", "_raw"))
    k["mon"] = k.date.dt.strftime("%Y-%m")
    rr = (k.pm_last_raw / k.pm_last).groupby([k.ticker, k.mon]).transform("median").fillna(1.0)
    rr = rr.where((rr - 1).abs() > 1e-3, 1.0).values           # study 25's rescale factor per ticker-month
    chk = x0.copy().drop(columns=["pm_tlast"], errors="ignore")
    print("rows", len(x0), "rescaled rows", int((rr != 1).sum()), flush=True)
    out = {"sip0925": x0}
    r10 = pm_raw(905, 845)
    base = x0.drop(columns=[c for c in pmcols if c in x0.columns])
    a = base.merge(r10, on=["ticker", "date"], how="left")
    assert len(a) == len(x0) and (a.ticker.values == x0.ticker.values).all()
    for c in PRICE:
        a[c] = a[c] / rr
    a = derive(a).drop(columns=["pm_tlast"])
    # study 25's data-consistency rule (premarket gap vs Yahoo open gap > 30%/43% apart = price-scale mismatch); those
    # rows were dropped from the cache; here the premarket price inputs are blanked instead so the rows stay the same
    ratio = (1 + a.gap) / (1 + a.ygap)
    bad = (ratio < 0.7) | (ratio > 1 / 0.7)
    print("sip0910 scale-mismatch rows blanked", int(bad.sum()), flush=True)
    a.loc[bad, ["pm_first", "pm_last", "pm_hi", "pm_lo", "pm_900", "gap", "pm_trend", "pm_range", "pm_late"]] = np.nan
    regap(a)
    out["sip0910"] = a[x0.columns]
    # sanity: rebuilding the 09:25 version the same way must reproduce the cache
    c25 = base.merge(r25, on=["ticker", "date"], how="left")
    for c in PRICE:
        c25[c] = c25[c] / rr
    c25 = derive(c25)
    for c in ["gap", "pm_late", "pm_mins", "relvol", "gap_z", "pm_range"]:
        print("rebuild check", c, "max abs diff", float(np.nanmax(np.abs(c25[c].values - chk[c].values))), flush=True)
    print("sip0910 coverage: rows with any bar by 09:10", round(a.pm_last.notna().mean(), 4),
          "| gap rank corr with 09:25 gap", round(a.gap.corr(x0.gap, method="spearman"), 4),
          "| median abs diff bp", round(float((a.gap - x0.gap).abs().median() * 1e4), 1), flush=True)
    # (b) IEX trades 09:10-09:25
    fs = sorted(f for f in os.listdir(os.path.join(DATA, "local", "iex83")) if f.endswith(".parquet"))
    nmon = x0.date.dt.strftime("%Y-%m").nunique()
    assert len(fs) == nmon or os.environ.get("ONLY"), f"IEX fetch incomplete: {len(fs)} of {nmon} months"
    iex = pd.concat([pd.read_parquet(os.path.join(DATA, "local", "iex83", f)) for f in fs], ignore_index=True)
    iex["date"] = pd.to_datetime(iex.date)
    b = a.merge(iex, on=["ticker", "date"], how="left")
    assert len(b) == len(x0)
    # trades are not split-adjusted (bars are, as of their fetch date): per ticker-month median ratio to the 09:10
    # SIP price (or to the previous close when there is no SIP bar) as in study 25's rescaling
    b["mon"] = b.date.dt.strftime("%Y-%m")
    rx = (b.iex_last / b.pm_last.fillna(b.prevc)).groupby([b.ticker, b.mon]).transform("median")
    fx = ((rx < 0.8) | (rx > 1.25)) & b.iex_last.notna()
    print("IEX ticker-days rescaled", int(fx.sum()), flush=True)
    b.loc[fx, "iex_last"] = b.iex_last[fx] / rx[fx]
    q = b.iex_last / b.pm_last.fillna(b.prevc)
    bad = (q < 0.7) | (q > 1 / 0.7)                                        # remaining mismatch -> drop the IEX print
    print("iex prints dropped for scale", int(bad.sum()), flush=True)
    b.loc[bad, ["iex_last", "iex_n", "iex_v", "iex_pv"]] = np.nan
    have = b.iex_last.notna()
    print("IEX coverage: rows with an IEX trade 09:10-09:25", round(have.mean(), 4), "| months fetched", len(fs), flush=True)
    b["gap_sip"] = b.gap
    b["iex_move"] = b.iex_last / b.pm_last - 1
    b["iex_n"] = b.iex_n.fillna(0)
    b["iex_relvol"] = (b.iex_pv.fillna(0) / b.adv20).astype("float64").replace([np.inf, -np.inf], np.nan)
    b["iex_mins"] = (b.date + pd.Timedelta(hours=9, minutes=25) - b.iex_t).dt.total_seconds() / 60
    now = b.iex_last.where(have, b.pm_last)
    b["gap"] = now / b.prevc - 1
    regap(b)
    out["sip0910_iex"] = b[list(x0.columns) + ["gap_sip", "iex_move", "iex_n", "iex_relvol", "iex_mins"]]
    cov = pd.DataFrame({"date": b.date, "have": have, "pm910": a.pm_last.notna()})
    return out, cov


# ------------------------------------------------------------------ study 60 frame and walk-forward
P = load_panel()
cols = stock_cols(P)
days = P["c"].index
prevd = pd.Series(days[:-1], index=days[1:])
V, cov = versions()
x0 = V["sip0925"]
keep = (x0.R.notna() & x0.ticker.isin(cols) & (x0.R.abs() < 1)).values
for k in V:
    V[k] = V[k][keep].reset_index(drop=True)
cov = cov[keep].reset_index(drop=True)
X = V["sip0925"][["ticker", "date"]].copy()
X["prev"] = X.date.map(prevd)
M = pd.read_parquet(f"{DATA}/ml_frame.parquet")
M = M[M.index.get_level_values(0) >= "2023-12-01"]
close_f = [c for c in M.columns if not c.startswith("y_")]
M = M[close_f].add_prefix("c_").reset_index()
M.columns = ["prev", "ticker"] + list(M.columns[2:])
X = X.merge(M, on=["prev", "ticker"], how="left")
assert (X.ticker.values == V["sip0925"].ticker.values).all()
close_f = ["c_" + c for c in close_f]
del M
new = []


def put(name, panel, key):
    s = panel.reindex(columns=cols).stack(future_stack=True)
    X[name] = s.reindex(pd.MultiIndex.from_arrays([X[key], X.ticker])).values.astype("float32")
    new.append(name)


A = AF.load()
for k in ["pt_n", "pt_up", "pt_dn", "pt_net20", "pt_gap", "pt_firms", "pt_gap_chg20", "ev_guid_up20", "ev_guid_dn20",
          "ev_buyback20", "ev_exec20"]:
    put("an_" + k, A[k], "prev")
del A
FP = pickle.load(open(f"{DATA}/local/fundamentals_panels.pkl", "rb"))
stale = FP["days_since_filing"] > 200
for k in ["market_cap", "ev_sales", "pe_ttm", "fcf_yield", "sbc_to_revenue", "op_margin", "revenue_growth_yoy"]:
    put("fu_" + k, FP[k].where(~stale).rank(axis=1, pct=True), "date")
del FP, stale
pr = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)
DROP = ("ticker", "date", "R", "cost", "o930", "ygap", "prevc", "pm_first", "pm_last", "pm_hi", "pm_lo", "px")
preds = {}
for name, v in V.items():
    pre = [c for c in v.columns if c not in DROP]
    x = pd.concat([v, X.drop(columns=["ticker", "date"])], axis=1)
    x["y"] = x.groupby("date").R.rank(pct=True) - 0.5
    f = pre + close_f + new
    fn = os.path.join(CACHE, f"pred_{name}.parquet")
    if os.environ.get("ONLY") and name not in os.environ["ONLY"].split(","):
        continue
    if os.path.exists(fn):
        preds[name] = pd.read_parquet(fn).p
        continue
    p = pd.Series(np.nan, index=x.index)
    for q in pd.period_range("2024Q3", "2026Q3", freq="Q"):
        cut = days[max(0, days.searchsorted(q.start_time) - 6)]
        tr = x.date < cut
        te = (x.date >= q.start_time) & (x.date <= q.end_time)
        if te.sum() == 0:
            continue
        m = lgb.train(pr, lgb.Dataset(x.loc[tr, f], x.y[tr]), num_boost_round=300)
        p[te] = m.predict(x.loc[te, f])
        print(name, q, len(f), "inputs", flush=True)
    pd.DataFrame({"p": p}).to_parquet(fn)
    preds[name] = p
    del x
if os.environ.get("ONLY"):
    sys.exit(0)                 # ONLY=<versions>: train and cache those predictions only
base = V["sip0925"][["date", "ticker", "R", "cost", "px", "adv20"]].copy()
for k, p in preds.items():
    base["p_" + k] = p.values
ref = pd.read_parquet(f"{RES}/study60_pred.parquet")
chk = base[["date", "ticker", "p_sip0925"]].merge(ref[["date", "ticker", "p_all"]], on=["date", "ticker"])
print("reference vs study60 p_all: rows", len(chk), "max abs diff", float((chk.p_sip0925 - chk.p_all).abs().max()), flush=True)

# ------------------------------------------------------------------ books
liq = (base.px > 5) & (base.adv20 > 5e6)
PER = [("2024H2-25H1", "2024-07", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024H2-26", "2024-07", "2026-09")]
rows, dl, picks = [], {}, {}
for k in preds:
    for n in [10, 20]:
        sel = base[liq & base["p_" + k].notna()].sort_values("p_" + k, ascending=False).groupby("date").head(n)
        g = sel.groupby("date")
        net, gross = g.R.mean() - 2 * g.cost.mean() / 1e4, g.R.mean()
        if n == 10:
            dl[k], picks[k] = net, set(zip(sel.date, sel.ticker))
        for p, a, b in PER:
            st = ann_stats(net.loc[a:b])
            rows.append(dict(section="day_long_book", variant=k, k=n, period=p, sharpe=st["sharpe"],
                             gross_bp=1e4 * gross.loc[a:b].mean(), net_bp=1e4 * net.loc[a:b].mean(),
                             maxdd=st["maxdd"], days=len(net.loc[a:b])))
oos = base[base.p_sip0925.notna()]
for k in preds:
    if k == "sip0925":
        continue
    ic = oos.groupby("date").apply(lambda d: d["p_" + k].corr(d.p_sip0925, method="spearman")).mean()
    ov = len(picks[k] & picks["sip0925"]) / len(picks["sip0925"])
    print(f"{k}: mean daily rank corr with reference {ic:.3f}, top-10 overlap {ov:.2%}", flush=True)
    rows.append(dict(section="vs_reference", variant=k, k=10, period="2024H2-26", rank_corr=ic, top10_overlap=ov))
for k in preds:
    ic = oos.groupby("date").apply(lambda d: d["p_" + k].corr(d.R, method="spearman"))
    for p, a, b in PER:
        rows.append(dict(section="ic", variant=k, period=p, ic=ic.loc[a:b].mean()))
cv = cov.assign(per=np.where(cov.date < "2025-07-01", "2024-25H1", "2025H2-26"))
for p, g in cv.groupby("per"):
    rows.append(dict(section="coverage", variant="sip0910_iex", period=p, iex_trade_share=g.have.mean(),
                     sip0910_bar_share=g.pm910.mean()))
    print("coverage", p, "IEX trade 09:10-09:25", round(g.have.mean(), 4), "SIP bar by 09:10", round(g.pm910.mean(), 4))
top = base[liq & base.p_sip0910_iex.notna()].sort_values("p_sip0910_iex", ascending=False).groupby("date").head(10)
th = top.merge(cov.assign(ticker=V["sip0925"].ticker)[["date", "ticker", "have"]].drop_duplicates(["date", "ticker"]),
               on=["date", "ticker"], how="left")
print("IEX trade share among sip0910_iex top-10 picks", round(th.have.mean(), 4), flush=True)
rows.append(dict(section="coverage", variant="sip0910_iex_top10", period="2024H2-26", iex_trade_share=th.have.mean()))

# ------------------------------------------------------------------ study 71 day cycle with each version's day long
del V, X, base
cwd = os.getcwd()
os.chdir(SRC)
exec(open("study68_day_short_picks.py").read().split("variants = {")[0])
os.chdir(cwd)
night_leg = bt.run(W, night, cost).net
m = (W > 0) & etbm
short_leg = bt.run(m.astype(float).div(10), day, cost + fee, side=-1).net
idx = night_leg.loc["2024-07":"2026-09"].dropna().index
for k, d in list(dl.items()) + [("none", None)]:
    if d is not None:
        d = d.copy()
        d.index = d.index.map(prevd)
    grid = [(0.0, 0.0), (0.0, 0.25)] if d is None else [(0.25, 0.25), (0.5, 0.0), (0.5, 0.25)]
    for a, b in grid:
        tot = night_leg.reindex(idx) + b * short_leg.reindex(idx).fillna(0)
        if d is not None:
            tot = tot + a * d.reindex(idx).fillna(0)
        for p, a0, b0 in PER:
            st = ann_stats(tot.loc[a0:b0])
            rows.append(dict(section="cycle71", variant=k, day_long=a, day_short=b, period=p, sharpe=st["sharpe"],
                             ann=st["ann_ret"], maxdd=st["maxdd"], net_bp=1e4 * tot.loc[a0:b0].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study83_day_long_freeplan.csv", index=False)
pd.set_option("display.width", 220)
print(df[df.section == "day_long_book"].pivot_table(index=["variant", "k"], columns="period",
                                                     values=["net_bp", "sharpe"]).round(2).to_string())
print(df[df.section == "ic"].pivot_table(index="variant", columns="period", values="ic").round(4).to_string())
print(df[df.section == "cycle71"].pivot_table(index=["variant", "day_long", "day_short"], columns="period",
                                               values=["sharpe", "net_bp", "maxdd"]).round(2).to_string())
