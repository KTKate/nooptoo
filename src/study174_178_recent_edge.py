"""Studies 174, 176 and 178: is the live paper loss of October 2026 a model failure, a market-wide change, or luck?

Background: the live virtual book (logs/paper/virtual_days.csv, scored at official auction prints) lost about 10.5%
over 12 nights 2026-09-29..10-15, against a backtest mean of +38 bp a night at cross prices; study 137 found 2026Q3
was the only negative backtest quarter; study 172 found the picks concentrated in tech hardware.

174  Recent edge. Question: has the capped blend's edge been weak since about July 2026, and how unusual is the last
     60 backtest nights plus the live nights against the full 2024-26 sample?
176  Tech hardware overnight. Question: did the overnight (close-to-next-open) drift of tech hardware stocks, and of
     SMH and SPY, turn negative recently, compared with the rest of the liquid universe?
178  Base overnight premium. Question: the blend partly earns the known overnight premium of volatile, high-beta
     stocks. Was that premium unusual in 2026Q3 and October 2026 (market-wide change), and how much of the live
     loss does it explain?

Data: core.load_panel (split- and dividend-adjusted Yahoo daily o/c, rawc, dv; ends 2026-10-15, so the last panel
night is 2026-10-14 -> 10-15). Backtest net: study135_137_expectations.build() (capped blend, at most 3 names per
industry, nightly net at official crosses, costs auction + 2.5 bp per side; predictions end 2026-09-25, so the last
backtest night is 09-25 -> 09-28). Live nights: logs/paper/virtual_days.csv net column (read only; 2026-09-29..10-15,
entry day; no entry on 09-30) and logs/paper/virtual.csv (per name). Night 2026-10-15 -> 10-16 for the universe:
Alpaca SIP daily bars (alpaca_data.bars, adjustment=split, 60 requests a minute), cached in
data/local/study176_bars.parquet. Tech hardware: study 90/92 definition (TECH_HW industries, plus 'Electrical
Products' and 'Industrial Machinery/Components' inside the Technology sector; data/store/sectors.parquet snapshot).
Liquid universe: stocks (no ETFs) with raw close > $5 and 20-day median dollar volume > $5M on the entry day t.

Design 174: monthly table of the backtest net (nights, mean bp, Sharpe) and the rolling 20- and 60-night mean and
Sharpe at each month's last night, on the backtest series and on the combined series (backtest, then the live nights
appended). Windows: last 60 backtest nights, the same plus the 12 live nights (72), 2026-07-01..09-25 and the same plus
live; each window's mean and compounded net is placed as a percentile among all rolling windows of the same length of
the backtest series 2024-01..2026-09 (overlapping windows, so a description of the sample, not a test). Welch t of
2026-07..09 (and with live) against 2024-01..2026-06. Live check: the live picks' close-to-next-open return from the
panel against the live gross at official prints, per night.
Design 176: per entry night t, equal-weighted mean overnight return o(t+1)/c(t)-1 (single returns clipped at +-50%
for data errors) of liquid tech hardware and of the rest of the liquid universe, and of the top-beta half (60-day
beta of daily close returns to SPY, as study 90 roll_beta, split at each night's median within the group); nightly
means averaged by month of t, 2024-01..2026-10 (October 2026 includes the Alpaca night 10-15). SMH and SPY overnight
the same way (ETFs from the panel; 10-15 night from Alpaca). Panel opens are the first trade, not the opening cross;
for random liquid stock-days the panel open lies on average +1.0 bp above the cross (study 94; close ~0 bp), so the
panel-vs-cross bias is small next to the effects here.
Design 178: each night, top decile of the liquid universe by 20-day volatility of daily close returns, and by 60-day
beta; their EW overnight return minus the liquid universe EW mean ('premium'), and the same sort's next day session
o(t+1)->c(t+1) minus the universe mean, for contrast. Monthly means 2020-01..2026-10 (earlier years have some
survivorship bias in the panel). 2026Q3, October 2026 and 2026-07-01..10-15 are placed as percentiles among all
rolling windows of the same number of nights 2020-2026 and 2024-2026. Attribution: OLS of the backtest capped nightly
net (2024-01..2026-09) on the universe EW overnight mean, the top-vol premium and tech hardware minus rest overnight;
the fitted values for the live nights give the net expected from the market moves alone, and the live residual
(live net minus fitted) is the part the factors do not explain, with a t from the backtest residual sd. Live picks:
each pick's official-print return minus its group's EW overnight mean that night (tech hardware or rest of liquid).

Data checks: build() reproduces study 94/137 (capped mean 38.2 bp, Sharpe 2.45). Alpaca close 2026-10-15 equals the
panel raw close (median difference 0 bp); 2,593 of 2,594 liquid names have the 10-15 night. Live picks at panel
prices vs official prints: per-night gross within about 15 bp, no systematic gap (live scoring is not the issue).
Peak memory 1.75 GB.

Result 174: monthly backtest net 2026-05..09: +113, +25, -23, +52, -44 bp (Sept to 09-25). Rolling 60-night mean at
09-25: -4.5 bp (Sharpe -0.28), 5.9th percentile of all 626 60-night windows 2024-26 (5th pct -5.4 bp). 2026-07-01..
09-25: -3.1 bp over 61 nights vs +42.2 bp in 2024-01..2026-06 (Welch t -1.35). Live 12 nights: -109 bp a night,
compounded -12.5% (net column), 1.6th percentile of 12-night windows. Last 60 backtest + 12 live nights (72): -21.9
bp a night, compounded -16.5%, below every one of the 614 72-night windows in 2024-26 (worst -16.8 bp, -12.8%;
5th pct +1.9 bp); 2026-07..live (73 nights) -20.5 bp, t -2.05 vs 2024-01..2026-06. So the
edge has been weak since July 2026 (one positive month, August, in three) and the live nights extend it. Earlier
weak stretches exist (2024-08..10 rolling 60 at -3 bp; 2026-02..04 at +8 bp) and recovered. The panel now runs to
2026-10-15, so an October rescore with the frozen 2026Q4 models is possible (not done).
Result 176 (bp a night, EW, overnight): tech hardware 2024-01..2026-06 +14.1, 2026Q3 +22.9 (normal, 77th pct), Oct
2026 (11 nights to 10-15) -74.9 (3rd pct of 11-night windows), top-beta half -102.7 (2nd pct); rest of liquid +5.4,
+8.7, +3.9 (38th pct), top-beta half -11.8. Tech hardware minus rest: +8.7, +14.1, -78.8 (lowest 11-night window, t
-2.1). SMH +21.8, +24.1, -77.2 (3rd pct); SPY +5.8, +7.4, +1.4 (27th pct). Tech hardware's overnight drift turned
sharply negative in October 2026 only; it was strong through 2026Q3; the broad market's overnight return was normal.
Result 178 (top decile minus liquid universe, bp a night, overnight): vol20 premium 2020-26H1 +5.8, 2024-26H1 +6.2,
2026Q3 +2.4 (45th pct vs 2020), Oct 2026 -49.0 (0.9th pct, t -2.0); beta60 premium +5.0 / +9.6, 2026Q3 +16.9 (85th
pct), Oct -74.5 (0.5th pct). Day session (contrast): vol20 -10.6 normally, Oct -71.9 (7th pct); beta60 -3.8, Oct
-81.6 (3rd pct), so high-beta names fell in both sessions in October. Universe overnight Oct -0.2 bp (34th pct).
Attribution (685 backtest nights, R2 0.49): net = 24.8 bp (t 3.6) + 0.76 x universe night (t 6.1) + 1.52 x top-vol
premium (t 10.0) + 0.03 x tech hw minus rest (t 0.3). Live nights: fitted -36.6 bp a night (factor part -61.3 bp),
actual -108.6, residual -72.0 bp (t -1.40, 12 nights); backtest 2026-07..09 residual -34.9 bp (t -1.53, in-sample
fit); both together -41.0 bp over 73 nights (t -1.97). Live picks vs own group that night: tech hardware picks (45 of
119, 79% of the gross loss) -206 bp vs group -120 bp (excess -86); other picks -34 bp vs +18 bp (excess -52); picks on
one night are correlated, so pick-level t (-3.1 / -2.2) overstates significance.
Reading: about half of the live shortfall against the backtest mean (38 -> -109 bp, a 147 bp gap) is the market: the
high-beta / high-vol overnight premium and tech hardware overnight had one of their worst stretches since 2020 in
October 2026 (top-decile premium at the 1st percentile), while the broad market's overnight return was normal. The
rest (about -72 bp a night after factors) is not explained by these factors; on its own it is within 1.4 sd for 12
nights, but together with 2026Q3, when the premium was normal and the blend still earned about 0, the unexplained
shortfall is about -41 bp a night over 73 nights (t about -2). Market-wide change plus a weak model stretch since July,
with luck not ruled out for either part.
Caveats: sectors.parquet is a current snapshot; the factor regression is fit on the same 2024-26 nights it judges
(2026Q3 residual is in-sample) and its tech hardware slope is near 0 because picks were rarely tech-heavy before
2025H2, so the live tech concentration enters through the premium factor and the residual; factor returns use panel
opens (first trade, ~1 bp above the cross for liquid names) and Alpaca bars for the 10-15 night, the live net uses
official prints; percentile windows overlap; 12 live nights is a small sample.

Output: results/study174_recent_edge.csv (table = monthly / windows / live_vs_panel / panel_extent),
results/study176_techhw_overnight.csv (table = monthly / recent), results/study178_premium.csv (table = monthly /
windows / attribution / live_nights / live_picks).

    python src/study174_178_recent_edge.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")
from core import stock_cols, ann_stats, RES, DATA   # noqa: E402
import study135_137_expectations as S135   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS = os.path.join(DATA, "local", "study176_bars.parquet")
EXTRA_NIGHT = pd.Timestamp("2026-10-15")      # entry day of the night not in the panel (exit 10-16)
EXTRA_EXIT = pd.Timestamp("2026-10-16")


def sh(x):
    x = pd.Series(x).dropna()
    return ann_stats(x)["sharpe"] if len(x) >= 5 else np.nan


def pct_rank(dist, v):
    dist = np.asarray(dist)
    dist = dist[~np.isnan(dist)]
    return (dist < v).mean() + 0.5 * (dist == v).mean()


def live():
    v = pd.read_csv(os.path.join(ROOT, "logs", "paper", "virtual_days.csv"), parse_dates=["day"]).set_index("day")
    return v


# ------------------------------------------------------------------ 174
def study174(K):
    net = K["nets"]["capped"]
    v = live()
    lv = v.net
    comb = pd.concat([net, lv]).sort_index()
    rows = []
    for name, s in [("backtest", net), ("backtest + live", comb)]:
        r20m, r60m = s.rolling(20).mean(), s.rolling(60).mean()
        r20s = s.rolling(20).mean() / s.rolling(20).std() * np.sqrt(252)
        r60s = s.rolling(60).mean() / s.rolling(60).std() * np.sqrt(252)
        for m, x in s.groupby(s.index.to_period("M")):
            if name == "backtest + live" and m < pd.Period("2026-05", "M"):
                continue
            d = x.index[-1]
            rows.append(dict(table="monthly", series=name, period=str(m), n=len(x), net_bp=1e4 * x.mean(),
                             sharpe=sh(x), hit=(x > 0).mean(), cum_net=(1 + x).prod() - 1,
                             roll20_bp=1e4 * r20m[d], roll20_sharpe=r20s[d], roll60_bp=1e4 * r60m[d],
                             roll60_sharpe=r60s[d], asof=str(d.date())))
    # windows vs the full-sample distribution
    bt_ = net.loc["2024-01":"2026-09"]
    win = {
        "last 60 backtest nights": bt_.iloc[-60:],
        "last 60 backtest + live": pd.concat([bt_.iloc[-60:], lv]),
        "last 20 backtest + live": pd.concat([bt_.iloc[-20:], lv]),
        "2026-07-01..09-25 backtest": bt_.loc["2026-07-01":],
        "2026-07-01..09-25 backtest + live": pd.concat([bt_.loc["2026-07-01":], lv]),
        "live only": lv,
    }
    for lab, x in win.items():
        n = len(x)
        rm = bt_.rolling(n).mean().dropna()
        rc = np.exp(np.log1p(bt_).rolling(n).sum().dropna()) - 1
        cum = (1 + x).prod() - 1
        rows.append(dict(table="windows", series=lab, period=f"{x.index[0].date()}..{x.index[-1].date()}", n=n,
                         net_bp=1e4 * x.mean(), sharpe=sh(x), hit=(x > 0).mean(), cum_net=cum,
                         pct_mean=pct_rank(rm.values, x.mean()), pct_cum=pct_rank(rc.values, cum),
                         dist_mean_min_bp=1e4 * rm.min(), dist_mean_p5_bp=1e4 * rm.quantile(0.05), dist_mean_p50_bp=1e4 * rm.median(),
                         dist_cum_min=rc.min(), dist_cum_p5=rc.quantile(0.05), n_windows=len(rm)))
    a, b = bt_.loc[:"2026-06"], bt_.loc["2026-07":]
    for lab, x in [("2026-07..09 vs 2024-01..2026-06", b), ("2026-07..09 + live vs 2024-01..2026-06",
                                                           pd.concat([b, lv]))]:
        rows.append(dict(table="windows", series=lab, n=len(x), net_bp=1e4 * x.mean(),
                         base_bp=1e4 * a.mean(), t_welch=S135.welch(x, a), sharpe=sh(x), base_sharpe=sh(a)))
    # live check: panel close-to-next-open for the live picks vs official prints
    P = K["P"]
    vv = pd.read_csv(os.path.join(ROOT, "logs", "paper", "virtual.csv"), parse_dates=["day"])
    days = P["c"].index
    o, c = P["o"], P["c"]
    for d, g in vv.groupby("day"):
        j = days.get_indexer([d])[0]
        rp = np.nan
        if j >= 0 and j + 1 < len(days):
            t = [s.replace(".", "-") for s in g.symbol if s.replace(".", "-") in c.columns]
            rp = float((o.iloc[j + 1][t] / c.iloc[j][t] - 1).mean())
        rows.append(dict(table="live_vs_panel", period=str(d.date()), n=len(g), gross_official_bp=1e4 * g.ret.mean(),
                         gross_panel_bp=1e4 * rp, net_bp=1e4 * v.net.get(d, np.nan),
                         techhw_names=int(g.symbol.str.replace(".", "-").isin(K["techhw_set"]).sum())))
    rows.append(dict(table="panel_extent", period=f"{days[0].date()}..{days[-1].date()}",
                     n=int(c.iloc[-1].notna().sum()),
                     note="daily panel extends to 2026-10-15 (last panel night 10-14 -> 10-15); backtest predictions "
                          "end 2026-09-25. An October rescore with the frozen 2026Q4 models (data/models/*2026Q4*) "
                          "on panel features is possible for entry days 2026-09-28..10-14 (not done; no retraining)."))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ data for 176 / 178
def extra_night(P, tick):
    """Alpaca daily bars 2026-10-15 and 10-16 for the given panel tickers (cached)."""
    if os.path.exists(BARS):
        return pd.read_parquet(BARS)
    import alpaca_data as A
    sym = sorted({t.replace("-", ".") for t in tick})
    parts = []
    for i in range(0, len(sym), 200):
        try:
            d = A.bars(sym[i:i + 200], "1Day", "2026-10-15T00:00:00Z", "2026-10-17T00:00:00Z")
        except RuntimeError as e:
            print("alpaca error", e, flush=True)
            continue
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["ticker"] = d.ticker.str.replace(".", "-", regex=False)
    d["date"] = d.ts.dt.normalize()
    d = d[["ticker", "date", "o", "c", "v"]]
    d.to_parquet(BARS, index=False)
    return d


def roll_beta(x, m, n=60, mp=40):
    """as study 90"""
    xm = x.rolling(n, min_periods=mp).mean()
    mm = m.rolling(n, min_periods=mp).mean()
    cov = x.mul(m, axis=0).rolling(n, min_periods=mp).mean() - xm.mul(mm, axis=0)
    return cov.div(m.rolling(n, min_periods=mp).var(ddof=0), axis=0)


def build_frames(K):
    P = K["P"]
    cols = K["cols"]
    days = P["c"].index
    c, o, rawc = P["c"][cols], P["o"][cols], P["rawc"][cols]
    r1 = (c / c.shift(1) - 1).astype("float32")
    spy = (P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1)
    vol20 = r1.rolling(20, min_periods=15).std().replace(0, np.nan).astype("float32")
    beta60 = roll_beta(r1, spy).astype("float32")
    adv = P["dv"][cols].rolling(20, min_periods=15).median()
    liq = (rawc > 5) & (adv > 5e6)
    del adv
    night = (o.shift(-1) / c - 1).clip(-0.5, 0.5)
    day = (c.shift(-1) / o.shift(-1) - 1).clip(-0.5, 0.5)
    etf_n = {k: P["o"][k].shift(-1) / P["c"][k] - 1 for k in ["SPY", "SMH"]}
    # the night 10-15 -> 10-16 from Alpaca
    if days[-1] == EXTRA_NIGHT:
        tick = list(liq.columns[liq.iloc[-1].values]) + ["SPY", "SMH"]
        B = extra_night(P, tick)
        bc = B[B.date == EXTRA_NIGHT].set_index("ticker").c
        bo = B[B.date == EXTRA_EXIT].set_index("ticker").o
        bc1 = B[B.date == EXTRA_EXIT].set_index("ticker").c
        rn = (bo / bc - 1).clip(-0.5, 0.5)
        rd = (bc1 / bo - 1).clip(-0.5, 0.5)
        night.iloc[-1] = rn.reindex(cols).values
        day.iloc[-1] = rd.reindex(cols).values
        for k in etf_n:
            etf_n[k].iloc[-1] = rn.get(k, np.nan)
        print("Alpaca night 10-15: names with a return", int(night.iloc[-1].notna().sum()),
              "of", int(liq.iloc[-1].sum()), "liquid", flush=True)
        # consistency of the Alpaca close with the panel close on 10-15 (raw scale)
        chk = (bc.reindex(cols) / rawc.iloc[-1] - 1).abs()
        print("Alpaca vs panel close 10-15: median abs diff bp", round(1e4 * float(chk.median()), 2),
              "share > 1%", round(float((chk > 0.01).mean()), 4), flush=True)
    liq = liq & night.notna()
    return dict(liq=liq, night=night, day=day, vol20=vol20.where(liq), beta60=beta60.where(liq), etf=etf_n)


def gmean(R, M):
    return R.where(M).mean(axis=1)


def top_half(score, M):
    med = score.where(M).median(axis=1)
    return M & score.ge(med, axis=0)


# ------------------------------------------------------------------ 176
def study176(K, F):
    techhw = pd.Series(K["cols"]).isin(K["techhw_set"]).values
    th = pd.DataFrame(np.broadcast_to(techhw, F["liq"].shape), index=F["liq"].index, columns=F["liq"].columns)
    L = F["liq"]
    g = {"tech hardware": L & th, "rest of liquid": L & ~th}
    series = {}
    for k, M in g.items():
        series[f"{k} EW"] = gmean(F["night"], M)
        series[f"{k} top-beta half"] = gmean(F["night"], top_half(F["beta60"], M))
        series[f"{k} names"] = M.sum(axis=1).astype(float)
    series["tech hw minus rest EW"] = series["tech hardware EW"] - series["rest of liquid EW"]
    series["SMH"] = F["etf"]["SMH"]
    series["SPY"] = F["etf"]["SPY"]
    D = pd.DataFrame(series).loc["2024-01":].dropna(subset=["SPY"])
    rows = []
    for m, x in D.groupby(D.index.to_period("M")):
        r = dict(table="monthly", period=str(m), nights=len(x))
        for k in D.columns:
            r[k + (" bp" if "names" not in k else "")] = (1e4 if "names" not in k else 1) * x[k].mean()
        rows.append(r)
    # recent windows vs 2024-01..2026-06 monthly distribution
    base = D.loc[:"2026-06"]
    for lab, a, b in [("2024-01..2026-06", "2024-01", "2026-06"), ("2026Q3", "2026-07", "2026-09"),
                      ("2026-10 (to 10-15)", "2026-10", "2026-10"), ("2026-07..10", "2026-07", "2026-10"),
                      ("live nights 09-29..10-15", "2026-09-29", "2026-10-15")]:
        x = D.loc[a:b]
        if lab.startswith("live"):
            x = x[x.index.isin(live().index)]
        r = dict(table="recent", period=lab, nights=len(x))
        for k in ["tech hardware EW", "tech hardware top-beta half", "rest of liquid EW", "rest of liquid top-beta half",
                  "tech hw minus rest EW", "SMH", "SPY"]:
            r[k + " bp"] = 1e4 * x[k].mean()
            if not lab.startswith("2024"):
                r[k + " t_vs_base"] = S135.welch(x[k], base[k])
                rm = base[k].rolling(len(x)).mean()
                r[k + " pct"] = pct_rank(rm.values, x[k].mean())
        rows.append(r)
    return pd.DataFrame(rows), D


# ------------------------------------------------------------------ 178
def study178(K, F, D176):
    L = F["liq"]
    um_n, um_d = gmean(F["night"], L), gmean(F["day"], L)
    out = {"universe night": um_n, "universe day": um_d, "universe names": L.sum(axis=1).astype(float)}
    for key, sc in [("vol20", F["vol20"]), ("beta60", F["beta60"])]:
        q9 = sc.quantile(0.9, axis=1)
        M = L & sc.ge(q9, axis=0)
        out[f"top {key} night premium"] = gmean(F["night"], M) - um_n
        out[f"top {key} day premium"] = gmean(F["day"], M) - um_d
        out[f"top {key} night raw"] = gmean(F["night"], M)
    D = pd.DataFrame(out).loc["2020-01":]
    D = D[D["universe names"] > 100]
    rows = []
    vals = [k for k in D.columns if "names" not in k]
    for m, x in D.groupby(D.index.to_period("M")):
        r = dict(table="monthly", period=str(m), nights=len(x), universe_names=x["universe names"].mean())
        for k in vals:
            r[k + " bp"] = 1e4 * x[k].mean()
        rows.append(r)
    lvd = live().index
    for lab, a, b in [("2020-01..2026-06", "2020-01", "2026-06"), ("2024-01..2026-06", "2024-01", "2026-06"),
                      ("2026Q3", "2026-07", "2026-09"), ("2026-10 (to 10-15)", "2026-10", "2026-10"),
                      ("2026-07..10", "2026-07", "2026-10"), ("live nights", "2026-09-29", "2026-10-15")]:
        x = D.loc[a:b]
        if lab == "live nights":
            x = x[x.index.isin(lvd)]
        for k in vals:
            r = dict(table="windows", period=lab, series=k, nights=len(x), mean_bp=1e4 * x[k].mean(),
                     sd_bp=1e4 * x[k].std())
            if not lab.startswith("20") or lab.startswith("2026"):
                for blab, ba in [("2020", "2020-01"), ("2024", "2024-01")]:
                    base = D.loc[ba:"2026-06", k]
                    r[f"pct_vs_{blab}"] = pct_rank(base.rolling(len(x)).mean().values, x[k].mean())
                    r[f"t_vs_{blab}"] = S135.welch(x[k], base)
            rows.append(r)
    # attribution of the backtest net to market moves
    net = K["nets"]["capped"]
    X = pd.DataFrame({"universe night": D["universe night"], "top vol20 night premium": D["top vol20 night premium"],
                      "tech hw minus rest night": D176["tech hw minus rest EW"]})
    Z = X.reindex(net.index).dropna()
    y = net.reindex(Z.index)
    A = np.column_stack([np.ones(len(Z)), Z.values])
    bcoef = np.linalg.lstsq(A, y.values, rcond=None)[0]
    e = y.values - A @ bcoef
    s = e.std(ddof=A.shape[1])
    cov = np.linalg.inv(A.T @ A) * s ** 2
    r2 = 1 - e.var() / y.values.var()
    for i, nm in enumerate(["const"] + list(Z.columns)):
        rows.append(dict(table="attribution", series=nm, coef=bcoef[i] * (1e4 if i == 0 else 1),
                         t=bcoef[i] / np.sqrt(cov[i, i]), nights=len(Z), r2=r2, resid_sd_bp=1e4 * s,
                         note="const in bp; slopes per unit of factor return"))
    # one-factor version (universe night only) for comparison
    A1 = np.column_stack([np.ones(len(Z)), Z["universe night"].values])
    b1 = np.linalg.lstsq(A1, y.values, rcond=None)[0]
    e1 = y.values - A1 @ b1
    rows.append(dict(table="attribution", series="one factor: universe night", coef=b1[1], const_bp=1e4 * b1[0],
                     r2=1 - e1.var() / y.values.var(), nights=len(Z)))
    lv = live().net
    XL = X.reindex(lv.index)
    fit = bcoef[0] + XL.values @ bcoef[1:]
    fit_mkt = XL.values @ bcoef[1:]
    res = lv.values - fit
    for d, a_, f_, fm, rs in zip(lv.index, lv.values, fit, fit_mkt, res):
        rows.append(dict(table="live_nights", period=str(d.date()), live_net_bp=1e4 * a_, fitted_bp=1e4 * f_,
                         factor_part_bp=1e4 * fm, resid_bp=1e4 * rs,
                         **{k + " bp": 1e4 * XL.loc[d, k] for k in X.columns}))
    ok = ~np.isnan(res)
    rows.append(dict(table="live_nights", period="mean over live nights", nights=int(ok.sum()),
                     live_net_bp=1e4 * lv.values[ok].mean(), fitted_bp=1e4 * fit[ok].mean(),
                     factor_part_bp=1e4 * fit_mkt[ok].mean(), resid_bp=1e4 * res[ok].mean(),
                     t=res[ok].mean() / (s / np.sqrt(ok.sum())),
                     cum_live=(1 + lv.values[ok]).prod() - 1, cum_fitted=(1 + fit[ok]).prod() - 1,
                     **{k + " bp": 1e4 * XL[k].values[ok].mean() for k in X.columns}))
    # same check for 2026Q3 in the backtest
    q3 = (Z.index >= "2026-07-01")
    rows.append(dict(table="live_nights", period="backtest 2026-07..09 (in-sample fit)", nights=int(q3.sum()),
                     live_net_bp=1e4 * y.values[q3].mean(), fitted_bp=1e4 * (A @ bcoef)[q3].mean(),
                     factor_part_bp=1e4 * (Z.values @ bcoef[1:])[q3].mean(), resid_bp=1e4 * e[q3].mean(),
                     t=e[q3].mean() / (s / np.sqrt(q3.sum()))))
    allr = np.concatenate([e[q3], res[ok]])
    rows.append(dict(table="live_nights", period="2026-07..09 backtest + live nights", nights=len(allr),
                     resid_bp=1e4 * allr.mean(), t=allr.mean() / (s / np.sqrt(len(allr)))))
    # live picks against their own group's overnight mean that night (tech hardware or rest of liquid)
    vv = pd.read_csv(os.path.join(ROOT, "logs", "paper", "virtual.csv"), parse_dates=["day"])
    vv["techhw"] = vv.symbol.str.replace(".", "-", regex=False).isin(K["techhw_set"])
    gm = np.where(vv.techhw, D176["tech hardware EW"].reindex(vv.day).values,
                  D176["rest of liquid EW"].reindex(vv.day).values)
    vv["excess"] = vv.ret - gm
    for lab, x in [("tech hardware picks", vv[vv.techhw]), ("other picks", vv[~vv.techhw]), ("all picks", vv)]:
        rows.append(dict(table="live_picks", series=lab, nights=x.day.nunique(), n=len(x),
                         mean_ret_bp=1e4 * x.ret.mean(), group_mean_bp=1e4 * (x.ret - x.excess).mean(),
                         excess_bp=1e4 * x.excess.mean(),
                         t=x.excess.mean() / (x.excess.std() / np.sqrt(len(x))),
                         share_of_live_gross=x.ret.sum() / vv.ret.sum()))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    K = S135.build()
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    ind, sct = sec.industry, sec.sector
    th = ind.isin(S135.TECH_HW) | (ind.isin(S135.TECH_HW_TECHSECTOR) & (sct == "Technology"))
    K["techhw_set"] = set(th[th].index)
    df = study174(K)
    df.to_csv(os.path.join(RES, "study174_recent_edge.csv"), index=False)
    print("\n== study174")
    for t in df.table.unique():
        print(df[df.table == t].dropna(axis=1, how="all").round(3).to_string(index=False))
    F = build_frames(K)
    d176, D176 = study176(K, F)
    d176.to_csv(os.path.join(RES, "study176_techhw_overnight.csv"), index=False)
    print("\n== study176")
    print(d176[d176.table == "monthly"].dropna(axis=1, how="all").round(1).to_string(index=False))
    print(d176[d176.table == "recent"].dropna(axis=1, how="all").round(2).T.to_string())
    d178 = study178(K, F, D176)
    d178.to_csv(os.path.join(RES, "study178_premium.csv"), index=False)
    print("\n== study178")
    m = d178[d178.table == "monthly"].dropna(axis=1, how="all")
    print(m[m.period >= "2024-01"].round(1).to_string(index=False))
    for t in ["windows", "attribution", "live_nights", "live_picks"]:
        print(d178[d178.table == t].dropna(axis=1, how="all").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
    print([ln for ln in open("/proc/self/status") if ln.startswith("VmHWM")], flush=True)
