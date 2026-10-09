"""Study 95: overnight drift in index, sector and leveraged ETFs (buy in the closing auction, sell in the next
opening auction).

Question: (a) is the close -> next open return of SPY, QQQ, IWM, DIA, TQQQ, SQQQ, SOXL, SOXS, UPRO, TNA, SMH, XLK,
XLF, XLE positive net of costs, per period, and how does it compare with holding the same ETF during the day (open ->
close)? (b) do filters known at 15:45 (the ETF's day return to 15:45, its 5-day return, VIX level and change, weekday,
nights before a holiday, month end) pick better nights? Rules are chosen on 2020-23 and 2024-01..2025-06 and checked
on 2025-07..2026-09. (c) does an ETF overnight leg added to the study-33 blend (25% or 50% extra notional, which
needs margin) raise the combined Sharpe?
Data: daily panel (core.load_panel; all these ETFs and ^VIX are in it, Yahoo-adjusted so dividends are included),
15:45 prices from the Alpaca 1-minute bars (data/local/m1, the last bar 15 minutes before the session's last bar, as
in study 27), Alpaca SIP NBBO quote samples at 15:44-15:45 and 09:31-09:32 (4 days per year per ETF, cached in
data/local/study95_quotes.parquet), official opening/closing crosses for a sample of days (alpaca_data._official,
cached in data/local/study95_official.parquet) to check the panel prices (study 88).
Design: cost per side = median quoted half-spread of that ETF and year (15:45 quote for the buy, 09:31 quote for
the sell) + 1 bp. Sensitivity: core.exec_cost_bps 'auction' + 2.5 bp per side (the blend's cost model).
Periods: 2020-23, 2024-01..2025-06, 2025-07..2026-09. Filters: sign and dev-tercile of the 15:45 day return, sign of
the 5-day return, VIX (previous close) above/below 20 and up/down on the previous day, one weekday only or skip one
weekday, before a holiday (or not), month end (or not). A rule trades only on nights where it is true (cash
otherwise). Selection per ETF: positive net mean in both 2020-23 and 2024-25H1, then the highest of
min(Sharpe 2020-23, Sharpe 2024-25H1); its 2025-07..2026-09 result is the test.
(c) combined daily return = blend net + w x ETF overnight net (w = 0.25, 0.5; also -0.25 / -0.5 as a hedge),
2024-01..2026-09, for SPY, QQQ and the selected filtered legs.
Output: results/study95_etf_overnight.csv (part, etf, variant, period, n, mean_bp, sharpe, ...)
"""
import os
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA
import alpaca_data as A
import bt

ETF = ["SPY", "QQQ", "IWM", "DIA", "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "TNA", "SMH", "XLK", "XLF", "XLE"]
PER = [("2020-23", "2020-01", "2023-12"), ("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09")]
LOCAL = f"{DATA}/local"
QFN, OFN = f"{LOCAL}/study95_quotes.parquet", f"{LOCAL}/study95_official.parquet"

P = load_panel()
days = P["c"].index
C = P["c"][ETF].astype("float64")
O = P["o"][ETF].astype("float64")
RAWC = P["rawc"][ETF].astype("float64")
VIX = P["c"]["^VIX"].astype("float64")
NIGHT = O.shift(-1) / C - 1                 # close t -> open t+1 (dividend-adjusted)
DAY = C / O - 1                              # open t -> close t
CC = C / C.shift(1) - 1
# data errors: Yahoo misses some reverse splits of the inverse ETFs (SOXS 2026-05-26: 1,132 -> 67), so moves larger
# than 50% (the largest real one is 35%, 2020-03-13) are set to NaN
for X_ in (NIGHT, DAY, CC):
    X_.mask(np.log1p(X_).abs() > np.log(1.5), inplace=True)
nxt = pd.Series(days[1:].append(pd.DatetimeIndex([pd.NaT])), index=days)
GAP = (nxt - pd.Series(days, index=days)).dt.days


# ---------------------------------------------------------------- 15:45 snapshot (study 27 method)
def snapshot():
    out = []
    for m in A.months("2019-06", str(days[-1])[:7]):
        d = A.read("m1", start=m, end=m, tickers=ETF, columns=["ts", "ticker", "c"])
        if d.empty:
            continue
        tod = d.ts.dt.hour * 60 + d.ts.dt.minute
        d = d[(tod >= 570) & (tod < 960)].copy()
        d["tod"] = tod[d.index]
        d["date"] = d.ts.dt.normalize()
        last = d.groupby(["date", "ticker"]).tail(1).set_index(["date", "ticker"])
        d = d.join(last["tod"].rename("end"), on=["date", "ticker"])
        s = d[d.tod <= d.end - 14].groupby(["date", "ticker"]).tail(1).set_index(["date", "ticker"])["c"]
        out.append(pd.DataFrame({"snap": s, "last": last["c"]}).dropna())
    x = pd.concat(out)
    ratio = (x.snap / x["last"]).unstack("ticker")
    ratio.index = pd.DatetimeIndex(ratio.index.astype("datetime64[ns]"))
    return ratio.reindex(index=days, columns=ETF)


# ---------------------------------------------------------------- quoted spreads
def quote_sample():
    if os.path.exists(QFN):
        return pd.read_parquet(QFN)
    rng = np.random.default_rng(95)
    rows = []
    for yr in range(2020, 2027):
        dd = days[(days.year == yr) & (days <= "2026-09-30")]
        pick = dd[np.sort(rng.choice(len(dd), 4, replace=False))]
        for d in pick:
            for hm in ["15:44", "09:31"]:
                ts = pd.Timestamp(f"{d:%Y-%m-%d} {hm}").tz_localize("America/New_York").tz_convert("UTC")
                a, b = ts.strftime("%Y-%m-%dT%H:%M:%SZ"), (ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                for t in ETF:
                    try:
                        j = A.get("quotes", dict(symbols=t, start=a, end=b, limit=1000, feed="sip"))
                    except RuntimeError:
                        continue
                    q = pd.DataFrame((j.get("quotes") or {}).get(t, []))
                    if q.empty:
                        continue
                    q = q[(q.bp > 0) & (q.ap > q.bp)]
                    hs = ((q.ap - q.bp) / (q.ap + q.bp) * 1e4).median()
                    rows.append(dict(ticker=t, date=d, hm=hm, half_spread_bps=hs, n=len(q)))
        print("quotes", yr, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(QFN, index=False)
    return df


def official_check(n_days=30, tickers=("SPY", "QQQ", "IWM", "TQQQ", "SOXL", "SMH", "XLK", "UPRO", "TNA", "DIA",
                                          "XLE")):
    """Official closing cross on day t and opening cross on t+1 vs the panel's raw close and open."""
    df = pd.read_parquet(OFN) if os.path.exists(OFN) else pd.DataFrame(columns=["ticker"])
    todo = [t for t in tickers if t not in set(df.ticker)]
    if todo:
        rng = np.random.default_rng(88)
        dd = days[(days >= "2023-01-01") & (days <= "2026-09-20")]
        pick = dd[np.sort(rng.choice(len(dd) - 1, n_days, replace=False))]
        rows = []
        for d in pick:
            d1 = nxt[d]
            for t in todo:
                try:
                    cl = A._official(t, f"{d:%Y-%m-%d}", "close")[0]
                    op = A._official(t, f"{d1:%Y-%m-%d}", "open")[0]
                except RuntimeError:
                    cl = op = np.nan
                rows.append(dict(ticker=t, date=d, close_off=cl, open_next_off=op))
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if len(df) else pd.DataFrame(rows)
        df.to_parquet(OFN, index=False)
    f = C / RAWC
    out = []
    for r in df.itertuples():
        i = days.get_loc(r.date)
        rc = RAWC.at[r.date, r.ticker]
        ro = O.iat[i + 1, ETF.index(r.ticker)] / f.iat[i + 1, ETF.index(r.ticker)]
        out.append(dict(ticker=r.ticker, date=r.date, close_diff_bp=1e4 * np.log(rc / r.close_off),
                        open_diff_bp=1e4 * np.log(ro / r.open_next_off),
                        night_diff_bp=1e4 * (np.log(ro / rc) - np.log(r.open_next_off / r.close_off))))
    return pd.DataFrame(out)


def stats(x, part, etf, variant, period, **kw):
    x = x.dropna()
    st = ann_stats(x)
    return dict(part=part, etf=etf, variant=variant, period=period, n=len(x), mean_bp=1e4 * x.mean(),
                sharpe=st["sharpe"], t=st["tstat"], ann=st["ann_ret"], maxdd=st["maxdd"], **kw)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    rows = []
    # ---- data checks
    chk = official_check()
    print("panel vs official crosses (bp), median |diff| / 90th pct |diff|:")
    print(chk.groupby("ticker")[["close_diff_bp", "open_diff_bp", "night_diff_bp"]]
          .agg(lambda s: f"{s.abs().median():.1f} / {s.abs().quantile(0.9):.1f}").to_string())
    for t, g in chk.groupby("ticker"):
        rows.append(dict(part="check official crosses", etf=t, variant="panel minus official, mean bp",
                         period="sample 2023-26", n=g.night_diff_bp.notna().sum(), mean_bp=g.night_diff_bp.mean(),
                         med_abs_close_bp=g.close_diff_bp.abs().median(), med_abs_open_bp=g.open_diff_bp.abs().median()))
    bias = chk.groupby("ticker").night_diff_bp.mean()      # panel overnight return minus official, bp
    big = NIGHT.loc["2020":"2026-09"].abs().stack().sort_values().tail(6)
    print("largest |overnight| moves:\n", big)

    # ---- costs
    qs = quote_sample()
    qs["year"] = pd.to_datetime(qs.date).dt.year
    hs = qs.groupby(["ticker", "hm", "year"]).half_spread_bps.median().unstack("hm")
    print("median quoted half-spread (bp) by year, 15:44 / 09:31:")
    print(hs.unstack("year").round(2).to_string())
    yrs = pd.Series(days.year, index=days)
    cost_close = pd.DataFrame({t: yrs.map(hs["15:44"].xs(t)) for t in ETF}) + 1.0
    cost_open = pd.DataFrame({t: yrs.map(hs["09:31"].xs(t)) for t in ETF}) + 1.0
    # sell side happens on t+1 (next session's open)
    cost_rt = (cost_close + cost_open.shift(-1).fillna(cost_open)) / 1e4
    ec = exec_cost_bps({k: P[k][ETF] for k in ["c", "h", "l", "rawc", "dv"]}, "auction") + 2.5
    cost_alt = 2 * ec.astype("float64") / 1e4
    for t in ETF:
        for p, a, b in PER:
            rows.append(dict(part="cost", etf=t, variant="round trip bp (quoted half-spread + 1 bp per side)",
                             period=p, mean_bp=1e4 * cost_rt[t].loc[a:b].mean(),
                             alt_bp=1e4 * cost_alt[t].loc[a:b].mean()))
    net = NIGHT - cost_rt
    net_alt = NIGHT - cost_alt
    day_net = DAY - (cost_open + cost_close) / 1e4        # open auction buy, close auction sell, same day

    # ---- (a) unconditional
    for t in ETF:
        for p, a, b in PER:
            rows.append(stats(NIGHT[t].loc[a:b], "a unconditional", t, "overnight gross", p))
            rows.append(stats(net[t].loc[a:b], "a unconditional", t, "overnight net", p))
            rows.append(stats(net_alt[t].loc[a:b], "a unconditional", t, "overnight net, blend cost model", p))
            rows.append(stats(DAY[t].loc[a:b], "a unconditional", t, "day (open to close) gross", p))
            rows.append(stats(day_net[t].loc[a:b], "a unconditional", t, "day net", p))
            if t in bias:
                rows.append(stats((net[t] - bias[t] / 1e4).loc[a:b], "a unconditional", t,
                                  "overnight net, less panel open bias vs official cross", p))
            rows.append(stats(CC[t].loc[a:b], "a unconditional", t, "buy and hold (close to close)", p))

    # ---- (b) 15:45 filters
    ratio = snapshot()
    print("15:45 snapshot coverage 2020-26:", ratio.loc["2020":].notna().mean().round(3).min())
    S = (C * ratio).fillna(C.shift(1))          # no bar: fall back to yesterday's close
    d1 = S / C.shift(1) - 1
    r5 = S / C.shift(5) - 1
    vl = VIX.shift(1)
    vc = VIX.shift(1) / VIX.shift(2) - 1
    wd = pd.Series(days.weekday, index=days)
    preh = (GAP > 3) | ((GAP == 2) & (wd != 4)) | ((GAP == 3) & (wd != 4))   # next session is not the next weekday
    mm = pd.Series(days.month, index=days)
    me = mm != mm.shift(-1)
    dev = slice("2020-01", "2023-12")
    rules = {}
    for t in ETF:
        lo, hi = d1[t].loc[dev].quantile([1 / 3, 2 / 3])
        R_ = {"day ret < 0": d1[t] < 0, "day ret > 0": d1[t] > 0, "day ret bottom tercile": d1[t] < lo,
              "day ret top tercile": d1[t] > hi, "5d ret < 0": r5[t] < 0, "5d ret > 0": r5[t] > 0,
              "VIX > 20": vl > 20, "VIX <= 20": vl <= 20, "VIX up yesterday": vc > 0, "VIX down yesterday": vc <= 0,
              "pre-holiday": preh, "not pre-holiday": ~preh, "month end": me, "not month end": ~me}
        for k, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
            R_[f"only {nm}"] = wd == k
            R_[f"skip {nm}"] = wd != k
        rules[t] = R_
    sel = {}
    for t in ETF:
        cand = []
        for rn, m in rules[t].items():
            x = net[t].where(m, 0.0)
            res = {}
            for p, a, b in PER:
                xs = x.loc[a:b]
                act = m.loc[a:b] & net[t].loc[a:b].notna()
                st = ann_stats(xs)
                res[p] = (st["sharpe"], 1e4 * net[t].loc[a:b][act].mean(), int(act.sum()))
                rows.append(dict(part="b filter", etf=t, variant=rn, period=p, n=int(act.sum()),
                                 mean_bp=1e4 * net[t].loc[a:b][act].mean(), sharpe=st["sharpe"], ann=st["ann_ret"],
                                 maxdd=st["maxdd"]))
            if res["2020-23"][1] > 0 and res["2024-25H1"][1] > 0 and min(res["2020-23"][2], res["2024-25H1"][2]) >= 20:
                cand.append((min(res["2020-23"][0], res["2024-25H1"][0]), rn, res))
        if cand:
            cand.sort(key=lambda z: -z[0])
            sc, rn, res = cand[0]
            sel[t] = rn
            rows.append(dict(part="b selected rule", etf=t, variant=rn, period="2025H2-26", n=res["2025H2-26"][2],
                             mean_bp=res["2025H2-26"][1], sharpe=res["2025H2-26"][0], sel_score=sc,
                             sharpe_2020_23=res["2020-23"][0], sharpe_2024_25H1=res["2024-25H1"][0],
                             uncond_sharpe_hold=ann_stats(net[t].loc["2025-07":"2026-09"])["sharpe"]))
    print("selected rules:", sel)

    # ---- (c) blend + ETF leg (SKIP_C=1 skips it, for a quick look at parts a and b)
    if os.environ.get("SKIP_C") == "1":
        pd.DataFrame(rows).to_csv(f"{RES}/study95_etf_overnight.csv", index=False)
        raise SystemExit
    cols = stock_cols(P)
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    p33 = pd.read_parquet(f"{RES}/study33_pred.parquet")
    pj = p33.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = p33.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del p33
    ok = ens.notna() & pj.notna()
    SB = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr, ok
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(SB)
    CB = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(SB)
    base = bt.run(bt.select_topk(SB, SB.notna(), 10), R, CB).net.loc[:"2026-09"]
    del R, CB, SB
    legs = {f"{t} unconditional": net[t] for t in ["SPY", "QQQ", "IWM", "TQQQ", "UPRO"]}
    for t, rn in sel.items():
        legs[f"{t} {rn}"] = net[t].where(rules[t][rn], 0.0)
    pc = [p for p in PER[1:]] + [("2024-26", "2024-01", "2026-09")]
    for p, a, b in pc:
        rows.append(stats(base.loc[a:b], "c blend + ETF leg", "-", "blend alone", p))
    for ln, leg in legs.items():
        for w in [0.25, 0.5, -0.25, -0.5]:
            comb = base + w * leg.reindex(base.index).fillna(0.0)
            for p, a, b in pc:
                rows.append(stats(comb.loc[a:b], "c blend + ETF leg", ln, f"w={w}", p,
                                  corr=base.loc[a:b].corr(leg.reindex(base.index).loc[a:b])))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study95_etf_overnight.csv", index=False)
    a_ = df[df.part == "a unconditional"].pivot_table(index=["etf", "variant"], columns="period",
                                                       values=["mean_bp", "sharpe"], sort=False)
    print(a_.round(2).to_string())
    print(df[df.part == "b selected rule"].round(2).to_string())
    c_ = df[df.part == "c blend + ETF leg"].pivot_table(index=["etf", "variant"], columns="period",
                                                         values="sharpe", sort=False)
    print(c_.round(2).to_string())
