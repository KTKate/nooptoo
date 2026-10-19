"""Studies 187 and 189: after a bad overnight stretch like October 2026, did the groups and the strategy recover?

Background: the live virtual book lost about 12.5% over 12 paper nights to 2026-10-15; the backtest has been weak
since July 2026 (study 174: last 60 backtest + 12 live nights -21.9 bp a night, compounded -16.5%); studies 176/178
found a market-wide overnight reversal of high-beta / tech hardware stocks in October 2026.

187  Group momentum overnight. Question: after a losing stretch of the tech hardware group, the top-beta decile and
     the top-volatility decile (each relative to the liquid universe), what happened over the next 1, 5 and 20
     nights, compared with all nights? And after a bad 10-night stretch of the blend itself, the next 20 and 60?
189  Earlier drawdowns. Question: what were the 5 worst drawdowns of the ML ranker backtest and of the blend at
     crosses, how long did they take to recover, what was the market doing, and how does the current one compare?

Data: core.load_panel (split- and dividend-adjusted Yahoo daily o/c, rawc, dv; ends 2026-10-15) through
study174_178_recent_edge.build_frames, which adds the night 2026-10-15 -> 10-16 from Alpaca SIP daily bars (cache
data/local/study176_bars.parquet from study 176; no new Alpaca requests). Liquid universe: stocks with raw close > $5
and 20-day median dollar volume > $5M on the entry day t. Tech hardware: study 90/92 definition (TECH_HW industries,
plus 'Electrical Products' and 'Industrial Machinery/Components' inside the Technology sector;
data/store/sectors.parquet, a current snapshot applied to all dates). Top-beta / top-vol decile: top 10% of the
liquid universe by 60-day beta of daily close returns to SPY / by 20-day volatility (study 178). Blend:
study135_137_expectations.build() nets 'capped' (at most 3 names per industry, official crosses, auction costs +
2.5 bp per side; 2024-01-02 .. 2026-09-25). Live nights: logs/paper/virtual_days.csv net (read only; 2026-09-29 ..
10-15). ML ranker: results/final_series.parquet 'ml_overnight_k10' (15:45 features, top 10, panel opens, auction
costs + 2.5 bp; 2024-01-02 .. 2026-09-24; it does NOT reach back to 2020, the file starts in 2024). For a longer ML
history: the close-feature LightGBM walk-forward predictions results/study3_pred_night.parquet (quarterly retrain,
out-of-sample from 2022Q1; no earlier predictions exist), top 10 by prediction, panel night return, core.exec_cost_bps
'auction' + 2.5 bp per side, 2022-01 .. 2026-09 ('ml_close_k10'). Both ML series use panel opens (first trade), which
lie a few bp above the opening cross, so their levels are slightly optimistic; drawdown shapes are less affected.

Design 187: nightly relative return x(t) = group EW overnight o(t+1)/c(t)-1 (single returns clipped at +-50%) minus
the liquid universe EW mean, 2020-01 .. 2026-10-15 (nights with > 100 liquid names). Events at night t: (a) the k-th
consecutive night with x < 0 (k = 3, 5, 8; each streak counts once per k, on the night it reaches k), (b) 10-night
sum x(t-9..t) below its 5th (and 1st) percentile over the whole sample (in-sample cut; every qualifying night counts, so
events cluster). Forward outcome: sum of x over nights t+1..t+h (h = 1, 5, 20), reported as bp a night (sum / h).
Comparison: mean over all nights with a full forward window. Difference test: OLS of the forward h-sum on a constant
and the event dummy, Newey-West standard errors with h + 10 lags (forward windows overlap by h - 1 nights, and
clustered events by up to about 10). Episodes: events separated by more than 20 nights count as separate episodes,
reported next to the event count as the effective sample size; t is left empty below 5 episodes. Blend (capped, 2024-26) and ml_close_k10 (2022-26):
same 10-night 5th-percentile event on the net series, forward 20 and 60 nights, NW lags h + 10. Current state per
series at its last night (streak, 10-night sum and its percentile).
Design 189: compounded equity per series; a drawdown episode runs from a running peak to the first night the equity
regains it (or to the end, 'not recovered'). The 5 deepest episodes of: ml_overnight_k10 (final_series),
ml_close_k10 (2022-26), the blend backtest, and the blend backtest with the live nights appended. Per episode: depth,
peak night (for an episode starting on the first night, the day before the series starts), trough night, nights peak -> trough, nights trough -> recovery, and market context over the nights from
peak to trough: SPY overnight mean bp, top-beta decile overnight mean bp (raw and minus universe), tech hardware
minus universe. Current drawdown: the open episode at the end of each series, plus the window from 2026-07-01 to the
last night (blend + live).

Data checks: build() reproduces study 94/137 (capped mean 38.2 bp, Sharpe 2.45); Alpaca night 10-15 has 2,593 of
2,594 liquid names, Alpaca close = panel close (median 0 bp). Group panel 2020-01-02..2026-10-15, 1,706 nights.
ml_close_k10: mean 21.3 bp 2022-26, 31.5 bp 2024-26, daily correlation 0.83 with final_series ml_overnight_k10.
Peak memory 1.76 GB, run time about 40 s.

Result 187 (relative to the liquid universe, bp a night; all-nights means: tech hw +3.0, top beta +5.0, top vol +5.3):
state at 10-15: all three groups on 5 losing nights in a row; 10-night sums tech hw -935 bp, top beta -973, top vol
-612, the lowest (tech hw) or near-lowest (0.2nd, 0.4th pct) of any 10-night window since 2020, far below the p5 cut
(-223 / -366 / -286), so past events are milder than now. After 10-night sum < p5 (15-20 episodes): next 20 nights
tech hw +7.0 (diff +3.9, t 1.1), top beta +9.0 (+3.8, t 0.6), top vol +5.2 (-0.4, t -0.1); next 1 and 5 nights
similar, all |t| < 1.1. After < p1 (4-5 episodes): next 20 nights +16 / +27 / +13 (top vol t 2.1, others too few
episodes for a t). After k losing nights: no consistent sign; the only |t| > 2 is top vol over the next 20 nights
after 3 or 5 losing nights (-1.1 / -3.0 bp vs +5.6 all, t -3.0 / -2.6, 22 / 13 episodes), i.e. mild continuation,
out of about 40 tests. 8-night streaks happened once or twice (2022-09, 2025-04), too few to read. Reading: after
past bad stretches these groups went back to roughly their normal small positive overnight premium; there is weak,
not significant, evidence of a bounce, and no evidence of a long continuation except a small one for top vol.
Blend capped (2024-26, 8 episodes): after 10-night sum < p5, next 20 nights +30.9 bp a night (all +39.9, t -0.8),
next 60 nights +51.8 (all +41.3, t 1.4), 90% / 96% of forward windows positive. With live nights appended the current
10-night sum is -1,516 bp (0.7th pct, below the p5 -953). ml_close_k10 (2022-26, 14 episodes): next 20 +31.2 (all
+22.2, t 1.2), next 60 +24.0 (t 0.3). So earlier bad 10-night stretches of the strategy were followed by roughly
normal returns, not by continued losses.
Result 189 (depth, peak -> trough, nights to trough, nights trough -> recovery):
blend backtest: -22.8% 2026-02-12 -> 03-19 (24, 32); -17.0% 2025-02-25 -> 03-07 (8, 44); -16.9% 2024-07-30 -> 10-28
(63, 25); -16.8% 2026-06-18 -> 07-28 (26, 21); -16.4% 2025-10-31 -> 11-17 (11, 45). All recovered within 47-88
nights from the peak. Market context: the 2025-02, 2025-11 and 2026-02 drawdowns came with weak SPY nights (-27 to
-37 bp) and a weak top-beta premium (-34 to -61 bp); 2026-06..07 with top beta -38 and tech hw -45 at a flat SPY;
2024-07..10 with normal markets (a model-only stretch). Current (backtest + live): -22.4% from 2026-08-26 to 10-15,
33 nights, not recovered, 2nd deepest after -22.8%; SPY nights +2.7 bp, top beta premium -18.5, tech hw -20.8.
2026-07-01..10-15 (73 nights): -15.8%, 0.3rd pct of 73-night windows (worst other -16.1%, itself overlapping now).
ml_overnight_k10 (final_series, 2024-26): -23.8% 2026-01-28 -> 09-11 (156 nights, not recovered, -20.5% at 09-24);
-18.5% 2025-01..04 (61, 28); -14.1% 2024-07..09 (24, 66); -9.6%; -7.8%. So the 15:45-feature ranker has been below
its January 2026 peak for 165 nights, in normal markets (SPY +4 bp, top beta +6). ml_close_k10 (2022-26): -18.1%
2022-04 -> 10 (132, 97: 229 nights peak to recovery, the 2022 bear market, SPY -13 bp, top beta -19); -17.4% 2025-01
-> 04 (50, 27); -17.3% 2026-01 -> 05 (75, 20); -14.8% 2022-01; -14.3% 2023-07 -> 10 (62, 28); it was near a peak
in September 2026. Reading: the current blend drawdown is about as deep as the worst earlier one, but longer in
nights to trough than most blend drawdowns (33 vs 8-63) and not yet at a known trough; every earlier blend and
ML drawdown recovered, in 26-229 nights.
Caveats: event thresholds use the whole sample (in-sample); events cluster, so the effective sample is the episode
count (4-31), small; many tests; past bad stretches were much milder than October 2026, so they are weak analogs;
2020-21 panel has some survivorship bias; sectors.parquet is a current snapshot; group returns and ML series use
panel opens (a few bp above the cross); the blend sample is 2024-26 only (3 years, 5 drawdowns of 16-23%), and
'recovered' in a backtest says nothing about whether the edge itself is still there (study 174/178: about -41 bp a
night unexplained since July).

Output: results/study187_group_momentum.csv (table = events / blend_events / current), results/study189_drawdowns.csv
(table = drawdowns / current / windows).

    python src/study187_189_recovery.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")
from core import RES, DATA, exec_cost_bps   # noqa: E402
import bt   # noqa: E402
import study135_137_expectations as S135   # noqa: E402
import study174_178_recent_edge as S174   # noqa: E402

HS = (1, 5, 20)
KS = (3, 5, 8)


def nw_t(y, d, lags):
    """OLS y = a + b d; returns b, Newey-West t of b."""
    ok = ~(np.isnan(y) | np.isnan(d))
    y, d = y[ok], d[ok]
    X = np.column_stack([np.ones(len(y)), d])
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    g = X * e[:, None]
    S = g.T @ g
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = g[L:].T @ g[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b[1], b[1] / np.sqrt(V[1, 1])


def fwd_sum(x, h):
    """sum of x over t+1..t+h (NaN when incomplete)."""
    return x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


def streak(x):
    neg = (x < 0).astype(int)
    grp = (neg == 0).cumsum()
    return neg.groupby(grp).cumsum()


def episodes(idx_pos, gap=20):
    if len(idx_pos) == 0:
        return 0
    return int(1 + (np.diff(idx_pos) > gap).sum())


def event_rows(name, x, hs, events, table):
    rows = []
    pos = np.arange(len(x))
    for h in hs:
        f = fwd_sum(x, h)
        base = f.dropna()
        for ev_lab, ev in events.items():
            e = ev & f.notna()
            if e.sum() == 0:
                rows.append(dict(table=table, series=name, event=ev_lab, h=h, n_events=0))
                continue
            b, t = nw_t(f.values, e.values.astype(float), h + 10)
            fe = f[e]
            rows.append(dict(table=table, series=name, event=ev_lab, h=h, n_events=int(e.sum()),
                             n_episodes=episodes(pos[e.values]),
                             fwd_bp_per_night=1e4 * fe.mean() / h, all_nights_bp_per_night=1e4 * base.mean() / h,
                             diff_bp_per_night=1e4 * b / h, t_nw=t if episodes(pos[e.values]) >= 5 else np.nan, fwd_sum_bp=1e4 * fe.mean(),
                             share_fwd_positive=(fe > 0).mean(), all_share_positive=(base > 0).mean(),
                             first=str(x.index[e.values][0].date()), last=str(x.index[e.values][-1].date()),
                             sample=f"{x.index[0].date()}..{x.index[-1].date()}"))
    return rows


def current_row(name, x, table="current"):
    s10 = x.rolling(10).sum()
    v = s10.iloc[-1]
    return dict(table=table, series=name, last_night=str(x.index[-1].date()), losing_streak=int(streak(x).iloc[-1]),
                sum10_bp=1e4 * v, sum10_p5_bp=1e4 * s10.quantile(0.05), sum10_pct=float((s10.dropna() < v).mean()),
                sum10_below_p5=bool(v < s10.quantile(0.05)),
                oct2026_bp_per_night=1e4 * x[x.index >= "2026-10-01"].mean(),
                mean_bp_per_night=1e4 * x.mean())


# ------------------------------------------------------------------ 187
def study187(G, nets):
    rows, cur = [], []
    for name, x in G.items():
        x = x.dropna()
        st = streak(x)
        s10 = x.rolling(10).sum()
        ev = {f"{k} losing nights": st == k for k in KS}
        ev["10-night sum < p5"] = s10 < s10.quantile(0.05)
        ev["10-night sum < p1"] = s10 < s10.quantile(0.01)
        rows += event_rows(name, x, HS, ev, "events")
        cur.append(current_row(name, x))
    for name, x in nets.items():
        x = x.dropna()
        s10 = x.rolling(10).sum()
        ev = {"10-night sum < p5": s10 < s10.quantile(0.05), "10-night sum < p10": s10 < s10.quantile(0.10)}
        rows += event_rows(name, x, (20, 60), ev, "blend_events")
        cur.append(current_row(name, x))
    return pd.DataFrame(rows + cur)


# ------------------------------------------------------------------ 189
def drawdowns(x):
    eq = (1 + x).cumprod()
    eq0 = pd.concat([pd.Series([1.0], index=[x.index[0] - pd.Timedelta(days=1)]), eq])
    peak = eq0.cummax()
    out, i, n = [], 0, len(eq0)
    v = eq0.values
    while i < n:
        if v[i] >= peak.values[i] and i + 1 < n and v[i + 1] < v[i]:
            p = i
            j = i + 1
            while j < n and v[j] < v[p]:
                j += 1
            seg = v[p:j]
            tr = p + int(np.argmin(seg))
            out.append(dict(peak_i=p, trough_i=tr, rec_i=j if j < n else None, depth=v[tr] / v[p] - 1))
            i = j
        else:
            i += 1
    idx = eq0.index
    res = []
    for d in out:
        res.append(dict(depth=d["depth"], peak=str(idx[d["peak_i"]].date()), trough=str(idx[d["trough_i"]].date()),
                        nights_to_trough=d["trough_i"] - d["peak_i"],
                        recovered=str(idx[d["rec_i"]].date()) if d["rec_i"] is not None else "not recovered",
                        nights_trough_to_recovery=(d["rec_i"] - d["trough_i"]) if d["rec_i"] is not None else np.nan,
                        nights_peak_to_recovery=(d["rec_i"] - d["peak_i"]) if d["rec_i"] is not None else np.nan,
                        open_at_end=d["rec_i"] is None, nights_since_peak=(n - 1 - d["peak_i"]),
                        _a=idx[d["peak_i"]], _b=idx[d["trough_i"]], _ai=d["peak_i"], _bi=d["trough_i"]))
    return pd.DataFrame(res), eq0


def context(M, a, b):
    """market nights strictly after the peak night a up to the trough night b (entry days)."""
    w = M.loc[(M.index > a) & (M.index <= b)]
    return {k + " bp": 1e4 * w[k].mean() for k in M.columns} | {"ctx_nights": len(w)}


def study189(series, M):
    rows = []
    for name, x in series.items():
        x = x.dropna()
        D, eq0 = drawdowns(x)
        top = D.sort_values("depth").head(5)
        for rk, (_, r) in enumerate(top.iterrows(), 1):
            ctx = context(M, r._a, r._b)
            rows.append(dict(table="drawdowns", series=name, rank=rk,
                             sample=f"{x.index[0].date()}..{x.index[-1].date()}",
                             **{k: v for k, v in r.items() if not k.startswith("_")}, **ctx))
        last = D.iloc[-1] if len(D) and D.iloc[-1].open_at_end else None
        if last is not None:
            rank = int((D.depth < last.depth).sum() + 1)
            rows.append(dict(table="current", series=name, rank=rank, n_episodes=len(D),
                             sample=f"{x.index[0].date()}..{x.index[-1].date()}",
                             **{k: v for k, v in last.items() if not k.startswith("_")},
                             depth_now=float(eq0.iloc[-1] / eq0.iloc[last._ai] - 1), **context(M, last._a, last._b)))
        else:
            rows.append(dict(table="current", series=name, depth_now=float(eq0.iloc[-1] / eq0.cummax().iloc[-1] - 1),
                             note="at a peak at the end of the series"))
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
    F = S174.build_frames(K)
    L, N = F["liq"], F["night"]
    um = S174.gmean(N, L)
    thm = pd.DataFrame(np.broadcast_to(pd.Series(K["cols"]).isin(K["techhw_set"]).values, L.shape),
                       index=L.index, columns=L.columns)
    out = {"universe night": um, "SPY night": F["etf"]["SPY"], "names": L.sum(axis=1).astype(float)}
    out["tech hardware - universe"] = S174.gmean(N, L & thm) - um
    for key in ["beta60", "vol20"]:
        sc = F[key]
        M_ = L & sc.ge(sc.quantile(0.9, axis=1), axis=0)
        out[f"top {key} decile - universe"] = S174.gmean(N, M_) - um
        out[f"top {key} decile raw"] = S174.gmean(N, M_)
    D = pd.DataFrame(out).loc["2020-01":]
    D = D[D.names > 100]
    print("group panel", D.index[0].date(), D.index[-1].date(), len(D), flush=True)
    del F, N, L, thm
    G = {k: D[k] for k in ["tech hardware - universe", "top beta60 decile - universe", "top vol20 decile - universe"]}

    # ML series
    P, cols = K["P"], K["cols"]
    fs = pd.read_parquet(os.path.join(RES, "final_series.parquet"))
    print("final_series", fs.index[0].date(), fs.index[-1].date(), list(fs.columns), flush=True)
    pred = pd.read_parquet(os.path.join(RES, "study3_pred_night.parquet"))["pred"].unstack().reindex(columns=cols)
    pred = pred.loc[pred.index < P["c"].index[-1]].loc[:"2026-09-25"]
    W = bt.select_topk(pred, pred.notna(), 10)
    del pred
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex(W.index)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(W.index)
    ml_close = bt.run(W, R, C).net.where(W.sum(axis=1) > 0).dropna()
    del W, R, C
    print("ml_close_k10", ml_close.index[0].date(), ml_close.index[-1].date(), "mean bp",
          round(1e4 * ml_close.mean(), 1), "2024-26 mean bp", round(1e4 * ml_close.loc["2024":].mean(), 1),
          "corr with final_series ml", round(ml_close.reindex(fs.index).corr(fs.ml_overnight_k10), 3), flush=True)
    blend = K["nets"]["capped"]
    live = S174.live().net
    blend_live = pd.concat([blend, live]).sort_index()

    d187 = study187(G, {"blend capped (backtest 2024-26)": blend, "blend capped (backtest + live)": blend_live,
                        "ml_close_k10 (2022-26)": ml_close})
    d187.to_csv(os.path.join(RES, "study187_group_momentum.csv"), index=False)
    print("\n== study187")
    for t in d187.table.unique():
        print(d187[d187.table == t].dropna(axis=1, how="all").round(3).to_string(index=False))

    M = D[["SPY night", "universe night", "top beta60 decile raw", "top beta60 decile - universe",
           "tech hardware - universe"]]
    d189 = study189({"ml_overnight_k10 (final_series)": fs.ml_overnight_k10.astype(float),
                     "ml_close_k10 (2022-26)": ml_close,
                     "blend capped (backtest)": blend, "blend capped (backtest + live)": blend_live}, M)
    # windows: 2026-07-01 .. end, and the study 174 window
    for name, x in [("blend capped (backtest + live)", blend_live), ("blend capped (backtest)", blend),
                    ("ml_overnight_k10 (final_series)", fs.ml_overnight_k10.astype(float)),
                    ("ml_close_k10 (2022-26)", ml_close)]:
        for lab, w in [("2026-07-01..end", x.loc["2026-07-01":]),
                       ("last 72 nights", x.iloc[-72:])]:
            rows = dict(table="windows", series=name, window=lab, nights=len(w),
                        first=str(w.index[0].date()), last_night=str(w.index[-1].date()),
                        mean_bp=1e4 * w.mean(), cum=(1 + w).prod() - 1)
            # same-length windows of the whole series: share with a lower compounded return
            rc = np.exp(np.log1p(x).rolling(len(w)).sum().dropna()) - 1
            rows["pct_cum_vs_all_windows"] = float((rc < rows["cum"]).mean())
            rows["worst_cum_any_window"] = float(rc.min())
            d189 = pd.concat([d189, pd.DataFrame([rows])], ignore_index=True)
    d189.to_csv(os.path.join(RES, "study189_drawdowns.csv"), index=False)
    print("\n== study189")
    for t in d189.table.unique():
        print(d189[d189.table == t].dropna(axis=1, how="all").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
    print([ln for ln in open("/proc/self/status") if ln.startswith("VmHWM")], flush=True)
