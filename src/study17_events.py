"""Study 17: corporate-event and calendar effects with free data.

Event types
  1. S&P 500 additions and deletions (Wikipedia 'Historical components of the S&P 500', changes table;
     announcement date from the cited S&P press release, src/fetch_events_sp500.py). Anchors: first trading day
     after the announcement (announcements come after the close) and the effective date (index funds trade in the
     closing auction of the day before).
  2. Russell reconstitution: no reliable free membership list, so the calendar effect on IWM (and IWM - SPY) around
     the reconstitution day (last Friday of June) is tested instead.
  3. Ex-dividend days from the store's dividend factor q (in this store q = f_t / f_{t-1} > 1 on ex-dates;
     dividend yield = 1 - 1/q of the previous close). Prices are total-return adjusted, so the overnight return
     into the ex-date is already net of the dividend.
  4. Stock splits: yfinance Ticker.splits (src/fetch_events_splits.py) for stocks that ever had 20d median dollar
     volume > $1M since 2020, plus steps in Yahoo split-adjusted / Alpaca raw close ratio (2024+, covers delisted).
  5. Monthly options expiration (third Friday) and quad witching (Mar/Jun/Sep/Dec).

Profiles: mean market-adjusted (minus SPY) overnight (close t-1 -> open t) and intraday (open t -> close t) returns for
days -5..+10 around day 0. Trades: auction entries/exits only, daily portfolio of up to 10 names at 1/10 capital
each (cash otherwise), cost per side exec_cost_bps(P, "auction") + 2.5 bp. ETF calendar rules use 100% in the ETF.
Output: results/study17_profiles.csv, results/study17_trades.csv
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, RES, DATA
import store

EV = os.path.join(DATA, "local", "events")
PERIODS = [("2020-23", "2020-01-01", "2023-12-31"), ("val", "2024-01-01", "2025-06-30"),
           ("oos", "2025-07-01", "2026-09-30")]
K = range(-5, 11)

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
N = len(days)
allc = list(P["c"].columns)
ci = {t: i for i, t in enumerate(allc)}
O, C = P["o"].values.astype("float64"), P["c"].values.astype("float64")
night = np.full_like(C, np.nan)
night[1:] = O[1:] / C[:-1] - 1
intra = C / O - 1
cc = np.full_like(C, np.nan)
cc[1:] = C[1:] / C[:-1] - 1
js = ci["SPY"]
spy_n, spy_i, spy_cc = night[:, js], intra[:, js], pd.Series(cc[:, js], index=days)
cost = (exec_cost_bps(P, "auction") + 2.5).reindex(columns=allc)
cost[["SPY", "IWM"]] = 1.3 + 2.5                               # ETFs: fees + 1 bp + 2.5 bp
Cv = cost.values.astype("float64")
adv = P["dv"].rolling(20, min_periods=10).median().shift(1).values
pxT = traded_close(P).shift(1).reindex(columns=allc)
px = pxT.fillna(P["rawc"].shift(1)).values   # traded price where known (2023-12+), else Yahoo raw
del pxT


def dayidx(dates, on_or_after=True):
    """index of first trading day >= date (or > date when on_or_after is False)"""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    return days.searchsorted(d, side="left" if on_or_after else "right")


def profile(ev, name, groups=None):
    """ev: DataFrame with j (column index) and d0 (day-0 index). Mean market-adjusted overnight and intraday (bps)."""
    out = []
    groups = groups or {"all": np.ones(len(ev), bool)}
    for k in K:
        i = ev.d0.values + k
        ok = (i >= 1) & (i < N)
        ii = np.clip(i, 0, N - 1)
        en = np.where(ok, night[ii, ev.j.values] - spy_n[ii], np.nan)
        ei = np.where(ok, intra[ii, ev.j.values] - spy_i[ii], np.nan)
        for g, m in groups.items():
            out.append(dict(event=name, group=g, k=k, n=int(np.isfinite(en[m]).sum()),
                            night_bps=1e4 * np.nanmean(en[m]), intra_bps=1e4 * np.nanmean(ei[m]),
                            night_t=np.nanmean(en[m]) / (np.nanstd(en[m]) + 1e-12) * np.sqrt(np.isfinite(en[m]).sum()),
                            intra_t=np.nanmean(ei[m]) / (np.nanstd(ei[m]) + 1e-12) * np.sqrt(np.isfinite(ei[m]).sum())))
    return out


def book(trades, maxpos=10, w=0.1):
    """trades: list of (a, b, j, kind, prio): buy in the closing auction of day a, sell in the opening (kind 'open')
    or closing ('close') auction of day b > a. Slots: at most maxpos positions; entries on a day are taken in order of
    prio (higher first) while slots are free; a slot is freed for a new entry at the close of the exit day.
    Returns daily net return Series and the list of accepted trades."""
    tr = sorted([t for t in trades if 0 <= t[0] < t[1] < N and np.isfinite(C[t[0], t[2]])],
                key=lambda t: (t[0], -t[4]))
    pnl = np.zeros(N)
    busy = np.zeros(N, int)
    acc = []
    for a, b, j, kind, pr in tr:
        if busy[a] >= maxpos:
            continue
        # skip trades with missing exit price
        if not np.isfinite(O[b, j] if kind == "open" else C[b, j]):
            continue
        busy[a:b] += 1
        acc.append((a, b, j, kind))
        last = b if kind == "close" else b - 1
        prev = C[a, j]
        for t in range(a + 1, last + 1):
            x = C[t, j]
            if np.isfinite(x):
                pnl[t] += w * (x / prev - 1)
                prev = x
        if kind == "open":
            pnl[b] += w * (O[b, j] / prev - 1)
        pnl[a + 1] -= w * Cv[a, j] / 1e4 if np.isfinite(Cv[a, j]) else w * 0.01
        pnl[b] -= w * Cv[b, j] / 1e4 if np.isfinite(Cv[b, j]) else w * 0.01
    return pd.Series(pnl, index=days), acc


TRADES = []


def report(name, s, acc, extra=""):
    for per, a, b in PERIODS:
        x = s.loc[a:b]
        st = ann_stats(x)
        m = spy_cc.reindex(x.index).fillna(0)
        beta = np.cov(x, m)[0, 1] / m.var() if x.std() > 0 else np.nan
        alpha = (x - beta * m).mean() * 252 if x.std() > 0 else np.nan
        ne = sum(1 for t in acc if a <= str(days[t[0]].date()) <= b)
        TRADES.append(dict(rule=name, period=per, n_trades=ne, sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                           alpha=alpha, beta=beta, maxdd=st["maxdd"], invested=float((x != 0).mean()), note=extra))
    r = [t for t in TRADES if t["rule"] == name]
    print(f"{name:34s} " + "  ".join(f"{t['period']}: n={t['n_trades']:4d} SR={t['sharpe']:5.2f} a={t['alpha']:6.3f}"
                                     for t in r), flush=True)


def elig(a, j, pmin=2.0, amin=1e6):
    return np.isfinite(px[a, j]) and px[a, j] > pmin and np.isfinite(adv[a, j]) and adv[a, j] > amin


PROF = []

# ============================================================== 1. S&P 500 changes
sp = pd.read_csv(os.path.join(EV, "sp500_events.csv"), parse_dates=["eff", "ann"])
sp = sp[(sp.eff >= "2020-01-01") & sp.ticker.isin(cols)].copy()
sp["j"] = sp.ticker.map(ci)
sp["e0"] = dayidx(sp.eff)                            # effective day (index trade at close of e0 - 1)
sp["a0"] = np.where(sp.ann.notna(), dayidx(sp.ann.fillna(sp.eff), on_or_after=False), -1)  # first day after ann.
sp = sp[sp.e0 < N]
# data must exist around the effective date (ticker reuse / acquired names drop out)
sp = sp[[np.isfinite(C[max(e - 1, 0), j]) for e, j in zip(sp.e0, sp.j)]]
print("S&P events with data:", sp.groupby("side").size().to_dict(), flush=True)
for side in ["add", "del"]:
    x = sp[sp.side == side]
    grp = {"all": np.ones(len(x), bool), "mcap": x.mcap_change.values, "other": ~x.mcap_change.values}
    PROF += profile(x.assign(d0=x.e0), f"sp500_{side}_eff", grp)
    y = x[x.a0 >= 0]
    grp = {"all": np.ones(len(y), bool), "mcap": y.mcap_change.values, "other": ~y.mcap_change.values}
    PROF += profile(y.assign(d0=y.a0), f"sp500_{side}_ann", grp)
    # cumulative market-adjusted close-to-close from first day after announcement close to effective-1 close
    run = []
    for a, e, j in zip(y.a0, y.e0, y.j):
        if e - 1 > a:
            run.append(np.nansum(cc[a + 1:e, j] - spy_cc.values[a + 1:e]))
    print(f"S&P {side}: mean adj. return close(ann day0) -> close(eff-1): {1e4*np.mean(run):.0f} bps, n={len(run)}")

sp_rules = {}
A = sp[(sp.side == "add") & (sp.a0 >= 0)]
D = sp[sp.side == "del"]
Dm = D[D.mcap_change]
sp_rules["sp_add_ann0_to_eff-1"] = [(a, e - 1, j, "close", 0) for a, e, j in zip(A.a0, A.e0, A.j) if e - 1 > a]
sp_rules["sp_add_ann0_night"] = [(a, a + 1, j, "open", 0) for a, j in zip(A.a0, A.j)]
sp_rules["sp_add_eff-1_night"] = [(e - 1, e, j, "open", 0) for e, j in zip(A.e0, A.j)]
sp_rules["sp_del_mcap_eff-1_night"] = [(e - 1, e, j, "open", 0) for e, j in zip(Dm.e0, Dm.j)]
sp_rules["sp_del_mcap_eff-1_hold5"] = [(e - 1, e + 4, j, "close", 0) for e, j in zip(Dm.e0, Dm.j)]
sp_rules["sp_del_mcap_eff-1_hold10"] = [(e - 1, e + 9, j, "close", 0) for e, j in zip(Dm.e0, Dm.j)]
Dma = Dm[Dm.a0 >= 0]
sp_rules["sp_del_mcap_ann0_to_eff+4"] = [(a, e + 4, j, "close", 0) for a, e, j in zip(Dma.a0, Dma.e0, Dma.j)]
for k, v in sp_rules.items():
    v = [t for t in v if elig(t[0], t[2])]
    s, acc = book(v)
    report(k, s, acc)

# ============================================================== 2. Russell reconstitution (calendar, IWM)
recon = []
for y in range(2020, 2027):
    d = pd.Timestamp(f"{y}-06-30")
    while d.dayofweek != 4:
        d -= pd.Timedelta(days=1)
    recon.append(d)
rc = pd.DataFrame({"d0": dayidx(recon), "j": ci["IWM"]})
PROF += profile(rc, "russell_IWM_minus_SPY")
# raw IWM (not market-adjusted) for reference
for k in K:
    i = rc.d0.values + k
    PROF.append(dict(event="russell_IWM_raw", group="all", k=k, n=len(i), night_bps=1e4 * np.nanmean(night[i, ci["IWM"]]),
                     intra_bps=1e4 * np.nanmean(intra[i, ci["IWM"]])))
jI = ci["IWM"]
ru = {"russell_IWM_close-5_to_close0": [(d - 5, d, jI, "close", 0) for d in rc.d0],
      "russell_IWM_close0_to_close5": [(d, d + 5, jI, "close", 0) for d in rc.d0],
      "russell_IWM_close0_to_close10": [(d, d + 10, jI, "close", 0) for d in rc.d0]}
for k, v in ru.items():
    s, acc = book(v, maxpos=1, w=1.0)
    report(k, s, acc, "100% IWM, one event per year")

# ============================================================== 3. ex-dividend
dv = store.read("daily", start="2019-12")[["date", "ticker", "q"]]
dv = dv[(dv.q - 1).abs() > 1e-4]
dv["y"] = 1 - 1 / dv.q
dv = dv[dv.ticker.isin(cols) & (dv.y > 0) & (dv.y < 0.15)]
dv["d0"] = days.searchsorted(pd.to_datetime(dv.date))
dv["j"] = dv.ticker.map(ci)
dv = dv[(dv.d0 >= 6) & (dv.d0 < N)]
dv = dv[[elig(d - 1, j) for d, j in zip(dv.d0, dv.j)]]
hy = dv[dv.y > 0.01]
print("ex-div events (liquid):", len(dv), "with yield > 1%:", len(hy), flush=True)
PROF += profile(hy, "exdiv_y>1%", {"all": np.ones(len(hy), bool), "y>2%": hy.y.values > 0.02,
                                   "adv>20M": np.array([adv[d - 1, j] > 2e7 for d, j in zip(hy.d0, hy.j)])})
PROF += profile(dv[dv.y <= 0.01], "exdiv_y<=1%")
# drop vs dividend: adjusted overnight into ex-date (total return) in bps of price; > 0 means price fell less
xn = night[hy.d0.values, hy.j.values]
print(f"ex-div y>1%: mean yield {1e4*hy.y.mean():.0f} bps, mean adj. overnight into ex-date {1e4*np.nanmean(xn):.1f} bps,"
      f" SPY-adj {1e4*np.nanmean(xn - spy_n[hy.d0.values]):.1f} bps", flush=True)
exr = {}
exr["exdiv_y>1%_night"] = [(d - 1, d, j, "open", y) for d, j, y in zip(hy.d0, hy.j, hy.y)]
exr["exdiv_y>1%_close-1_to_close0"] = [(d - 1, d, j, "close", y) for d, j, y in zip(hy.d0, hy.j, hy.y)]
exr["exdiv_y>1%_close-6_to_close-1"] = [(d - 6, d - 1, j, "close", y) for d, j, y in zip(hy.d0, hy.j, hy.y)]
hy2 = hy[[adv[d - 1, j] > 2e7 for d, j in zip(hy.d0, hy.j)]]
exr["exdiv_y>1%_adv>20M_night"] = [(d - 1, d, j, "open", y) for d, j, y in zip(hy2.d0, hy2.j, hy2.y)]
exr["exdiv_y>1%_close0_to_close5"] = [(d, d + 5, j, "close", y) for d, j, y in zip(hy.d0, hy.j, hy.y)]
for k, v in exr.items():
    s, acc = book(v)
    report(k, s, acc)

# ============================================================== 4. splits
sl = []
fy = os.path.join(EV, "splits_yf.csv")
if os.path.exists(fy):
    y = pd.read_csv(fy, parse_dates=["date"])
    y = y[(y.ratio >= 1.5) | (y.ratio <= 0.67)]      # drop spin-off style adjustments (e.g. 1.28)
    y["src"] = "yf"
    sl.append(y)
# Alpaca raw vs Yahoo split-adjusted ratio steps (2024+)
raw = traded_close(P)
ratio = (P["rawc"] / raw).values
step = ratio[1:] / ratio[:-1]
ii, jj = np.where(np.isfinite(step) & ((step > 1.4) | (step < 0.7)))
st = pd.DataFrame({"ticker": [allc[j] for j in jj], "date": days[ii + 1], "ratio": step[ii, jj], "src": "alpaca"})
st = st[st.ticker.isin(cols)]
sl.append(st)
S = pd.concat(sl).sort_values(["ticker", "date"])
S["kind"] = np.where(S.ratio > 1, "forward", "reverse")
# merge duplicates (same ticker, kind, within 5 days): keep the earliest date
S["gap"] = S.groupby(["ticker", "kind"]).date.diff().dt.days
S = S[~(S.gap <= 5)].copy()
S = S[S.ticker.isin(cols) & (S.date >= "2020-01-01")]
S["d0"] = dayidx(S.date)
S["j"] = S.ticker.map(ci)
S = S[(S.d0 >= 6) & (S.d0 < N)]
S = S[[elig(d - 6, j, pmin=1.0, amin=5e5) for d, j in zip(S.d0, S.j)]]
# data sanity: drop events whose adjusted overnight return on day 0 looks like an unadjusted split (Yahoo applied the
# split on a different day): |return| > 40% and |log return| > half of |log ratio|
n0 = night[S.d0.values, S.j.values]
bad = (np.abs(n0) > 0.4) & (np.abs(np.log1p(n0)) > 0.5 * np.abs(np.log(S.ratio.values)))
print("split events dropped as unadjusted-split data errors:", int(bad.sum()))
S = S[~bad]
S.to_csv(os.path.join(EV, "splits_events_used.csv"), index=False)
print("split events:", S.groupby(["kind", "src"]).size().to_dict(), flush=True)
for kd in ["forward", "reverse"]:
    x = S[S.kind == kd]
    PROF += profile(x, f"split_{kd}", {"all": np.ones(len(x), bool), "yf": (x.src == "yf").values,
                                        "alpaca_only": (x.src == "alpaca").values})
    run = [np.nansum(cc[d:d + 10, j] - spy_cc.values[d:d + 10]) for d, j in zip(x.d0, x.j)]
    pre = [np.nansum(cc[d - 5:d, j] - spy_cc.values[d - 5:d]) for d, j in zip(x.d0, x.j)]
    print(f"split {kd}: n={len(x)} adj. cc return days -5..-1 {1e4*np.nanmean(pre):.0f} bps, days 0..+9 "
          f"{1e4*np.nanmean(run):.0f} bps (median {1e4*np.nanmedian(run):.0f})", flush=True)
F = S[S.kind == "forward"]
R = S[S.kind == "reverse"]
spl = {"split_fwd_close-6_to_close-1": [(d - 6, d - 1, j, "close", 0) for d, j in zip(F.d0, F.j)],
       "split_fwd_close-1_to_close+9": [(d - 1, d + 9, j, "close", 0) for d, j in zip(F.d0, F.j)],
       "split_fwd_close-1_night": [(d - 1, d, j, "open", 0) for d, j in zip(F.d0, F.j)],
       "split_fwd_close-1_night_cleanratio_adv>20M": [(d - 1, d, j, "open", 0) for d, j, r in zip(F.d0, F.j, F.ratio)
                                                      if abs(2 * r - round(2 * r)) < 0.02 and adv[d - 1, j] > 2e7],
       "split_rev_close0_to_close+9(long)": [(d, d + 9, j, "close", 0) for d, j in zip(R.d0, R.j)]}
for k, v in spl.items():
    v = [t for t in v if elig(t[0], t[2])]
    s, acc = book(v)
    report(k, s, acc)

# ============================================================== 5. options expiration
opex = []
for m in pd.period_range("2020-01", "2026-09", freq="M"):
    d = pd.Timestamp(m.start_time)
    fr = d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)
    i = days.searchsorted(fr, side="right") - 1      # third Friday, or the trading day before if a holiday
    opex.append((i, m.month in (3, 6, 9, 12)))
ox = pd.DataFrame(opex, columns=["d0", "quad"])
ox = ox[ox.d0 < N]
for g, sub in [("all", ox), ("quad", ox[ox.quad]), ("monthly_only", ox[~ox.quad])]:
    for et, j in [("SPY", js), ("IWM", ci["IWM"])]:
        for k in K:
            i = sub.d0.values + k
            i = i[i < N]
            PROF.append(dict(event=f"opex_{et}_raw", group=g, k=k, n=len(i), night_bps=1e4 * np.nanmean(night[i, j]),
                             intra_bps=1e4 * np.nanmean(intra[i, j])))
# comparison: opex day vs all other days (SPY raw), and the ML strategy
fs = pd.read_parquet(os.path.join(RES, "final_series.parquet"))["ml_overnight_k10"]   # row t: close t -> open t+1
isop = np.zeros(N, bool)
isop[ox.d0.values] = True
isq = np.zeros(N, bool)
isq[ox[ox.quad].d0.values] = True
rows = []
for lab, mask in [("night_into_opex", isop), ("night_into_quad", isq), ("opex_day_intraday", isop),
                  ("night_after_opex", np.roll(isop, 1)), ("night_into_thu_before", np.roll(isop, -1))]:
    for et, j in [("SPY", js), ("IWM", ci["IWM"])]:
        v = intra[:, j] if "intraday" in lab else night[:, j]
        sel = mask & (days >= "2020-01-01")
        oth = ~mask & (days >= "2020-01-01")
        rows.append(dict(test=lab, series=et, n=int(sel.sum()), mean_bps=1e4 * np.nanmean(v[sel]),
                         other_bps=1e4 * np.nanmean(v[oth]),
                         t_diff=(np.nanmean(v[sel]) - np.nanmean(v[oth])) / np.sqrt(np.nanvar(v[sel]) / sel.sum() + np.nanvar(v[oth]) / oth.sum())))
    if "intraday" not in lab:
        # ML row dated t is the night t -> t+1; the night into day d is the row of day d-1
        nm = pd.Series(np.roll(mask, -1), index=days).reindex(fs.index).fillna(False).values
        rows.append(dict(test=lab, series="ml_overnight_k10", n=int(nm.sum()), mean_bps=1e4 * fs[nm].mean(),
                         other_bps=1e4 * fs[~nm].mean(),
                         t_diff=(fs[nm].mean() - fs[~nm].mean()) / np.sqrt(fs[nm].var() / nm.sum() + fs[~nm].var() / (~nm).sum())))
OPX = pd.DataFrame(rows)
print(OPX.round(2).to_string(index=False), flush=True)
OPX.to_csv(os.path.join(RES, "study17_opex_tests.csv"), index=False)
opr = {"opex_SPY_night_into_opex": [(d - 1, d, js, "open", 0) for d in ox.d0],
       "opex_SPY_night_into_quad": [(d - 1, d, js, "open", 0) for d in ox[ox.quad].d0],
       "opex_SPY_night_after_opex": [(d, d + 1, js, "open", 0) for d in ox.d0],
       "opex_IWM_night_into_opex": [(d - 1, d, ci["IWM"], "open", 0) for d in ox.d0]}
for k, v in opr.items():
    s, acc = book(v, maxpos=1, w=1.0)
    report(k, s, acc, "100% ETF")
# ML strategy skipping the night into opex (cash that night)
skip = pd.Series(np.roll(isop, -1), index=days).reindex(fs.index).fillna(False)
for lab, s in [("ml_overnight_k10_base", fs), ("ml_overnight_k10_skip_night_into_opex", fs.where(~skip, 0.0))]:
    s = s.reindex(days).fillna(0.0)
    report(lab, s, [], "from results/final_series.parquet (2024+ only)")

pd.DataFrame(PROF).to_csv(os.path.join(RES, "study17_profiles.csv"), index=False)
T = pd.DataFrame(TRADES)
T.to_csv(os.path.join(RES, "study17_trades.csv"), index=False)
print("variants tried:", T.rule.nunique())
