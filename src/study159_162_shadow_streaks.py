"""Studies 159 and 162: shadow candidates at official crosses, and losing streaks of the live blend.

159  Shadow candidates at official crosses. Question: the two overnight candidates tracked in shadow (src/shadow.py:
     smallcap_nonews and adr_loser) were backtested at panel prices (Yahoo open/close for small caps, Alpaca daily bars
     for ADRs). What are their Sharpe, mean net and drawdown when the close buy and the open sell are priced at the
     listing exchange's official closing and opening crosses? How far is the panel open from the opening cross for
     these picks (small caps and ADRs may open with large gaps between the first trade and the cross), how many legs
     have no cross, and how does the live shadow record so far compare?
162  Losing streaks. Question: the paper virtual book lost on 5 of its last 6 nights (10/6, 10/7, 10/9, 10/12, 10/13)
     and about -5% over its nights since 9/29. Is a streak like that unusual for the live (3-per-industry capped)
     blend at cross prices, and do streaks predict more losses over the next 5, 10, 20 nights?

Data: core.load_panel (Yahoo, split- and dividend-adjusted o/c, rawc). Small-cap rule as src/final_series.py
s_intraday_loser (traded price > $2, 20-day median dollar volume $1-5M, 15:45 price from study8_exec_retest.price_1545,
signal = loss from the open to 15:45, top 10) restricted to stocks without a news article in the window from 15:45 of
the previous trading day to 15:45 of day t (news_features n_news, the window used by study 10 and by shadow.py; the
news archive ends 2026-09-25, so this rule's backtest ends with the night of 2026-09-25). Cost core.exec_cost_bps(P,
"auction") + 2.5 bp per side as final_series. ADR rule: study 16 (data/local/intl/daily.parquet, Alpaca daily bars,
split-adjusted, plus store.adj_factor; universe data/local/intl/universe.csv without ETFs; cost 1.3 bp + 10% of
half_spread_model + 2.5 bp per side), two versions: (a) study 16 as defined: 20-day median dollar volume > $5M,
previous close > $3, signal = loss from the open to the close of day t (uses the 16:00 close, so it is not tradable as
written), top 5; (b) live-like as shadow.py: price > $2, signal = loss from the open to 15:45 (15:30-15:45 Alpaca SIP
15-minute bar fetched here into data/local/intl/m15_1545_159.parquet), top 5. Alpaca raw daily bars of the ADRs and of
the small-cap picks (data/local/intl/d1raw159.parquet, data/local/d1raw159_small.parquet) put crosses on the panel's
scale for ADRs and show what the live scorer (shadow.py score, Alpaca daily bar close and next open) measures. Official
crosses: all cached data/local/auctions*.parquet (studies 88-149) plus data/local/auctions159.parquet fetched here with
study 94's rule (listing-venue 'O' / '6' print in widening windows, second pass: the largest 'O' / '6' print of any
listing exchange). Live: logs/paper/shadow_*.json (picks), shadow_scores.csv (scores), virtual_days.csv (read only).
Study 162 series: study135_137_expectations.build() (capped blend at crosses, 2024-01..2026-09, legs without a usable
cross at the panel price), the series study 136 used.

Design 159: night return at crosses as study 94: panel night return x (open cross / panel open on t+1) / (close cross
/ panel close on t); small caps on the Yahoo traded scale (study 99 cross_ratios: scale check against the traded close,
ratio outside 0.8..1.25 is a data error), ADRs on the Alpaca raw daily bar scale (data error if raw and adjusted bars
disagree on the day's open/close ratio by more than 0.5%, or the ratio is outside 0.8..1.25). A leg without a usable
cross keeps the panel price (share reported). Bias per pick: panel / cross - 1 in bp (positive = panel above the cross)
for the sell-day open and the buy-day close, plus first trade vs opening cross and Alpaca daily bar vs cross. Periods
2024-01..2025-06, 2025-07..2026-09 and both. Live record: mean net per night at the live scorer's prices
(shadow_scores.csv) and the same picks rescored at crosses (pairs fetched here), compared with the backtest's mean at
crosses (t of the difference, and the percentile of the live mean among all rolling windows of the same length in the
backtest).
Design 162: losing night = net < 0. Trigger on night t when nights t-5..t hold 3+, 4+ or 5+ losses, or their compounded
net is below -5%, or 5+ losses and below -5% (the live case). After each trigger: nights t+1..t+h (h = 5, 10, 20) mean
net bp, annualized Sharpe of those nights (pooled, overlapping), hit rate, mean compounded h-night return, versus the
same over all nights. Triggers overlap, so significance comes from a circular-shift test: the trigger dates are
shifted by a random offset (30..n-30 nights, 2,000 draws, seed 162) relative to the return series, which keeps the
clustering of triggers and of returns but breaks the link; p = share of shifts with a forward mean at least as low
as the observed one. Persistence: autocorrelation of nightly net at lags 1-5 and of non-overlapping 5-night sums at
lag 1 (with 1.96/sqrt(n) bands). Bad stretches: share of rolling 10- and 11-night windows with compounded net at or
below the live figure and at or below -5%, and of rolling 6-night windows with 5+ losses. Bootstrap as study 136
(stationary blocks, mean length 10, 10,000 paths, seed 136, 2024-26 sample, full and half mean): percentile of the
live compounded net over its nights, and the chance of at least one 6-night window with 5+ losses within 63 and
252 nights.

Output: results/study159_shadow_cross.csv (table = rescore / bias / fallback / live / data_errors),
results/study162_streaks.csv (table = after_streak / persistence / stretches / bootstrap / live).

    python src/study159_162_shadow_streaks.py fetch   # 15:45 ADR bars, raw daily bars, crosses (resumable)
    python src/study159_162_shadow_streaks.py 159
    python src/study159_162_shadow_streaks.py 162
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402
import bt   # noqa: E402
import study94_cross_rescore as S94   # noqa: E402  (sets A.RL at import; reset below)

A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(DATA, "local")
INTL = os.path.join(LOC, "intl")
LOG = os.path.join(ROOT, "logs", "paper")
AUC159 = os.path.join(LOC, "auctions159.parquet")
M1545 = os.path.join(INTL, "m15_1545_159.parquet")
RAW_ADR = os.path.join(INTL, "d1raw159.parquet")
RAW_SMALL = os.path.join(LOC, "d1raw159_small.parquet")
OUT159 = os.path.join(RES, "study159_shadow_cross.csv")
OUT162 = os.path.join(RES, "study162_streaks.csv")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
NEWS_END = "2026-09-25"


def _utc(day, hms):
    return pd.Timestamp(f"{day} {hms}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ picks
def smallcap_picks():
    """smallcap_nonews backtest: returns dict with P, cols, days, W (entry day t x ticker), Rn panel, cost."""
    from study8_exec_retest import price_1545
    import news_features as NF
    P = load_panel()
    cols = stock_cols(P)
    alld = P["c"].index
    days = P["c"].loc["2024-01-02":NEWS_END].index
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).reindex(days)
    px = traded_close(P)[cols].shift(1).reindex(days)
    p1545 = price_1545("none").reindex(index=days, columns=cols)
    yo = (P["o"][cols] / (P["c"][cols] / P["rawc"][cols])).reindex(days)
    tierS = (px > 2) & (adv > 1e6) & (adv <= 5e6) & p1545.notna()
    sig = -(p1545 / yo - 1)
    news = NF.load()["n_news"].reindex(index=days, columns=cols)
    nonews = ~(news > 0) & news.notna()
    W = bt.select_topk(sig, tierS & sig.notna() & nonews, 10)
    Wall = bt.select_topk(sig, tierS & sig.notna(), 10)        # final_series s_intraday_loser (reference)
    Rn = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex(days)
    cost = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(days)
    return dict(P=P, cols=cols, alld=alld, days=days, W=W, Wall=Wall, Rn=Rn, cost=cost)


def adr_panels():
    import study16_international as S16
    P = S16.panels()
    D = S16.derived(P)
    U = pd.read_csv(os.path.join(INTL, "universe.csv"))
    tick = [t for t in U[U.kind != "etf"].ticker if t in P["c"].columns]
    return P, D, tick


def adr_picks(P, D, tick, m1545=None):
    """Study 16 rule (close signal) and the live-like 15:45 rule; W index = entry day t."""
    E = D["elig"][tick]
    S = -D["oc"][tick]
    EE = E & S.notna() & D["co_next"][tick].notna()
    out = {"adr_loser_s16_close": bt.select_topk(S, EE, 5).loc["2024-01":"2026-09"]}
    if m1545 is not None:
        osplit = (P["o"] / P["c"] * P["rawc"])[tick]
        p = m1545.reindex(index=P["c"].index, columns=tick)
        S2 = -(p / osplit - 1)
        E2 = (D["adv"][tick] > 5e6) & (P["rawc"][tick].shift(1) > 2) & S2.notna() & D["co_next"][tick].notna()
        out["adr_loser_live_1545"] = bt.select_topk(S2, E2, 5).loc["2024-01":"2026-09"]
    return out


def load_m1545():
    if not os.path.exists(M1545):
        return None
    m = pd.read_parquet(M1545)
    m["date"] = m.ts.dt.normalize()
    return m.pivot_table(index="date", columns="ticker", values="c").astype("float64")


def legs(W, days):
    """(ticker, date, kind) of every pick: close on t, open on t+1."""
    nxt = pd.Series(days[1:], index=days[:-1])
    s = W.stack()
    s = s[s > 0].reset_index()
    s.columns = ["t", "ticker", "w"]
    return pd.concat([pd.DataFrame({"ticker": s.ticker, "date": s.t, "kind": "close"}),
                      pd.DataFrame({"ticker": s.ticker, "date": s.t.map(nxt), "kind": "open"})]).dropna()


def live_picks():
    """Shadow picks per recorded day (not replays): DataFrame day, strategy, ticker."""
    rows = []
    for f in sorted(glob.glob(os.path.join(LOG, "shadow_2*.json"))):
        if "replay" in f:
            continue
        r = json.load(open(f))
        for strat in ["smallcap_nonews", "adr_loser"]:
            for t in r.get(strat, {}) or {}:
                rows.append(dict(day=pd.Timestamp(r["day"]), strategy=strat, ticker=t))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ crosses
def cached_crosses():
    """All cached crosses in the study 94 format (ticker Alpaca style). Returns found (px notna) and attempted keys."""
    parts = []
    for f in sorted(glob.glob(os.path.join(LOC, "auctions*.parquet"))):
        b = os.path.basename(f)
        if b in ("auctions.parquet",):
            continue
        d = pd.read_parquet(f)
        if b == "auctions88.parquet":
            ven = S94.listing_venue()
            lv = d.ticker.map(ven)
            d = d[d.open_px.isna() | lv.isna() | (d.open_x == lv)]
            d = pd.DataFrame({"ticker": d.ticker, "date": d.date, "kind": "open", "first_px": d.first_px,
                              "px": d.open_px, "rule": "venue"})
        if "rule" not in d:
            d["rule"] = "venue"
        d = d[["ticker", "date", "kind", "first_px", "px", "rule"]].copy()
        d["src"] = b
        parts.append(d)
    x = pd.concat(parts, ignore_index=True)
    x["ticker"] = x.ticker.astype(str).str.replace("-", ".", regex=False)
    x["date"] = pd.to_datetime(x.date)
    x["rule"] = x.rule.astype(str).replace({"<NA>": "venue", "nan": "venue", "None": "venue"})
    # prefer rows with a price, then this study's file
    x["has"] = x.px.notna()
    x = x.sort_values(["has", "src"], ascending=[False, True]).drop_duplicates(["ticker", "date", "kind"])
    return x.drop(columns="has").reset_index(drop=True)


def fetch_1545(days, tick):
    have = pd.read_parquet(M1545) if os.path.exists(M1545) else pd.DataFrame(columns=["ts", "ticker", "c"])
    done = set(pd.to_datetime(have.ts).dt.normalize().unique()) if len(have) else set()
    # days fetched with no bars at all are recorded in a sidecar so a rerun skips them
    side = M1545 + ".done.json"
    done |= set(pd.to_datetime(json.load(open(side)))) if os.path.exists(side) else set()
    todo = [d for d in days if d not in done]
    print("15:45 ADR bars: days to fetch", len(todo), flush=True)
    rows, empt = [], sorted(str(d.date()) for d in done)
    for i, d in enumerate(todo):
        ds = str(d.date())
        b = A.bars(tick, "15Min", _utc(ds, "15:30:00"), _utc(ds, "15:31:00"))
        if len(b):
            rows.append(b[b.ts.dt.strftime("%H:%M") == "15:30"])
        empt.append(ds)
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            if rows:
                have = pd.concat([have] + rows, ignore_index=True)
                have.to_parquet(M1545, index=False)
                rows = []
            json.dump(empt, open(side, "w"))
            print("  15:45 bars", i + 1, "/", len(todo), flush=True)


def fetch_raw(tickers, fn, start="2023-12-01", end="2026-10-01"):
    have = pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(columns=["ts", "ticker"])
    todo = sorted(set(tickers) - set(have.ticker))
    print("raw daily bars:", os.path.basename(fn), "tickers to fetch", len(todo), flush=True)
    for i in range(0, len(todo), 100):
        b = A.bars(todo[i:i + 100], "1Day", start + "T00:00:00Z", end + "T00:00:00Z", adjustment="raw")
        if len(b):
            b["date"] = b.ts.dt.normalize()
            have = pd.concat([have, b[["date", "ticker", "o", "c", "v"]]], ignore_index=True)
        have = have.drop(columns=[k for k in ["ts"] if k in have])
        # placeholder rows so tickers without bars are not re-requested
        miss = set(todo[i:i + 100]) - set(have.ticker)
        if miss:
            have = pd.concat([have, pd.DataFrame({"ticker": sorted(miss)})], ignore_index=True)
        have.to_parquet(fn, index=False)
        print("  raw", min(i + 100, len(todo)), "/", len(todo), flush=True)


def fetch_crosses(need):
    need = need.copy()
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need["date"] = pd.to_datetime(need.date)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    if os.path.exists(AUC159):
        have = pd.read_parquet(AUC159)
        have["date"] = pd.to_datetime(have.date)
    else:
        have = pd.DataFrame(columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)
    S94.AUC_FN = AUC159
    rows, t0 = [], time.time()
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        try:
            return S94._day_kind(job[0], job[1], job[2], ven)
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(2) as ex:
            for r in ex.map(one, jobs[i:i + 24]):
                rows += r
        S94._save(pd.concat([have, pd.DataFrame(rows)], ignore_index=True))
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(rows),
              f"found {np.mean([not np.isnan(r['px']) for r in rows]) if rows else 0:.3f}", f"{time.time() - t0:.0f}s",
              flush=True)
    if os.path.exists(AUC159):
        S94.refetch_missing()


def fetch():
    K = smallcap_picks()
    need = [legs(K["W"], K["alld"])]
    small_tick = sorted(set(K["W"].columns[(K["W"] > 0).any()]))
    del K
    P, D, tick = adr_panels()
    days16 = P["c"].index
    elig = D["adv"][tick].loc["2023-12-20":] > 5e6
    adr_live = sorted(elig.columns[elig.any()])
    fetch_1545(days16[(days16 >= "2024-01-02") & (days16 <= "2026-09-30")], adr_live)
    Wd = adr_picks(P, D, tick, load_m1545())
    for W in Wd.values():
        need.append(legs(W, days16))
    adr_t = sorted(set().union(*[set(W.columns[(W > 0).any()]) for W in Wd.values()]))
    lp = live_picks()
    cal = pd.DatetimeIndex(A.trading_days("2026-09-20", "2026-10-31"))
    if len(cal) and lp.day.max() >= cal[-1]:
        cal = cal.append(pd.DatetimeIndex([cal[-1] + pd.offsets.BDay(1)]))
    nxt = pd.Series(cal[1:], index=cal[:-1])
    need.append(pd.concat([pd.DataFrame({"ticker": lp.ticker, "date": lp.day, "kind": "close"}),
                           pd.DataFrame({"ticker": lp.ticker, "date": lp.day.map(nxt), "kind": "open"})]).dropna())
    fetch_raw(adr_t + sorted(set(lp.ticker[lp.strategy == "adr_loser"])), RAW_ADR)
    fetch_raw([t.replace("-", ".") for t in small_tick] + sorted(set(lp.ticker[lp.strategy == "smallcap_nonews"])),
              RAW_SMALL)
    fetch_crosses(pd.concat(need, ignore_index=True))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch()
    sys.exit()


# ------------------------------------------------------------------ 162
def _st(x):
    x = pd.Series(x).dropna()
    s = ann_stats(x)
    return dict(n=len(x), net_bp=1e4 * x.mean() if len(x) else np.nan, sharpe=s["sharpe"],
                hit=(x > 0).mean() if len(x) else np.nan)


def virtual_book():
    v = pd.read_csv(os.path.join(LOG, "virtual_days.csv"), parse_dates=["day"])
    return v.set_index("day").net


def forward(r, trig, h):
    """Forward nights t+1..t+h after trigger nights (positions in r). Returns pooled nights, h-night compounded sums."""
    n = len(r)
    pos = np.flatnonzero(trig)
    pos = pos[pos + h < n]
    if not len(pos):
        return np.array([]), np.array([]), pos
    idx = pos[:, None] + np.arange(1, h + 1)[None, :]
    fw = r[idx]
    return fw.ravel(), np.prod(1 + fw, axis=1) - 1, pos


def study162():
    import study135_137_expectations as S135
    K = S135.build()
    net = K["nets"]["capped"].loc["2024-01":"2026-09"]
    del K
    rows = []
    r = net.values
    n = len(r)
    loss = (r < 0).astype(int)
    L6 = pd.Series(loss).rolling(6).sum().values
    C6 = (pd.Series(1 + r).rolling(6).apply(np.prod, raw=True) - 1).values
    trig = {"3+ losses in 6": L6 >= 3, "4+ losses in 6": L6 >= 4, "5+ losses in 6": L6 >= 5,
            "6-night net < -5%": C6 < -0.05, "5+ losses in 6 and 6-night net < -5% (live case)": (L6 >= 5) & (C6 < -0.05)}
    per_mask = {lab: (net.index >= a) & (net.index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0)) for lab, a, b in PER}
    rng = np.random.default_rng(162)
    shifts = rng.integers(30, n - 30, 2000)
    for lab, a, b in PER:
        pm = per_mask[lab]
        for h in [5, 10, 20]:
            # all nights: every start in the period
            allf, allc, _ = forward(r, pm & (np.arange(n) >= 5), h)
            rows.append(dict(table="after_streak", period=lab, trigger="all nights (baseline)", horizon=h,
                             triggers=int((pm & (np.arange(n) >= 5)).sum()), episodes=np.nan, **_st(allf),
                             cum_h_mean=allc.mean(), cum_h_median=np.median(allc), share_h_loss=(allc < 0).mean()))
            for tn, tg in trig.items():
                tgp = tg & pm
                f, cmp_, pos = forward(r, tgp, h)
                if not len(pos):
                    rows.append(dict(table="after_streak", period=lab, trigger=tn, horizon=h, triggers=0))
                    continue
                ep = int(1 + (np.diff(pos) > 6).sum())
                d = dict(table="after_streak", period=lab, trigger=tn, horizon=h, triggers=len(pos), episodes=ep,
                         **_st(f), cum_h_mean=cmp_.mean(), cum_h_median=np.median(cmp_),
                         share_h_loss=(cmp_ < 0).mean(), diff_vs_all_bp=1e4 * (f.mean() - allf.mean()))
                if lab == "2024-26":       # circular-shift test on the whole sample
                    obs = f.mean()
                    sims = []
                    for s in shifts:
                        ff, _, _ = forward(r, np.roll(tg, s), h)
                        sims.append(ff.mean() if len(ff) else np.nan)
                    sims = np.array(sims)
                    d.update(p_shift_lower=np.nanmean(sims <= obs), shift_mean_bp=1e4 * np.nanmean(sims),
                             shift_p5_bp=1e4 * np.nanquantile(sims, 0.05), shift_p95_bp=1e4 * np.nanquantile(sims, 0.95))
                rows.append(d)
        # persistence
        x = net[pm]
        for lag in range(1, 6):
            rows.append(dict(table="persistence", period=lab, series="nightly net", lag=lag, n=len(x),
                             autocorr=x.autocorr(lag), band95=1.96 / np.sqrt(len(x))))
        s5 = x.groupby(np.arange(len(x)) // 5).sum()
        s5 = s5[x.groupby(np.arange(len(x)) // 5).size() == 5]
        rows.append(dict(table="persistence", period=lab, series="non-overlapping 5-night sums", lag=1, n=len(s5),
                         autocorr=s5.autocorr(1), band95=1.96 / np.sqrt(len(s5))))
        s10 = x.groupby(np.arange(len(x)) // 10).sum()
        s10 = s10[x.groupby(np.arange(len(x)) // 10).size() == 10]
        rows.append(dict(table="persistence", period=lab, series="non-overlapping 10-night sums", lag=1, n=len(s10),
                         autocorr=s10.autocorr(1), band95=1.96 / np.sqrt(len(s10))))
        # correlation of past 6-night compounded net with the next 5/10/20 nights (overlapping)
        for h in [5, 10, 20]:
            fwd = (pd.Series(1 + r).rolling(h).apply(np.prod, raw=True) - 1).shift(-h).values
            m = pm & ~np.isnan(C6) & ~np.isnan(fwd)
            rows.append(dict(table="persistence", period=lab, series=f"past 6-night net vs next {h}-night net (overlapping)",
                             lag=h, n=int(m.sum()), autocorr=np.corrcoef(C6[m], fwd[m])[0, 1]))
    # bad stretches
    vb = virtual_book()
    live_cum = (1 + vb).prod() - 1
    live_eq = pd.read_csv(os.path.join(LOG, "virtual_days.csv")).equity.iloc[-1] / 1e4 - 1   # the log's own equity
    vloss = (vb < 0).astype(int)
    live_L6 = int(vloss.iloc[-6:].sum())
    live_C6 = (1 + vb.iloc[-6:]).prod() - 1
    for lab, a, b in PER:
        x = net[per_mask[lab]]
        for w in [6, 10, 11, 20]:
            cw = (1 + x).rolling(w).apply(np.prod, raw=True).dropna() - 1
            lw = (x < 0).astype(int).rolling(w).sum().dropna()
            d = dict(table="stretches", period=lab, window=w, windows=len(cw), median_cum=cw.median(),
                     p5_cum=cw.quantile(0.05), p1_cum=cw.quantile(0.01), worst_cum=cw.min(),
                     share_le_minus5pct=(cw <= -0.05).mean(), share_le_live=(cw <= live_cum).mean(), share_le_live_equity=(cw <= live_eq).mean(),
                     share_5plus_losses=(lw >= 5).mean() if w == 6 else np.nan,
                     share_6_losses=(lw >= 6).mean() if w == 6 else np.nan)
            # distinct episodes (non-overlapping) at or below -5%
            hit = (cw <= -0.05).values
            starts = int(hit[0]) + int(((hit[1:]) & ~hit[:-1]).sum()) if len(hit) else 0
            d["episodes_le_minus5pct"] = starts
            d["episodes_per_year"] = starts / (len(x) / 252)
            rows.append(d)
    # live record vs the bootstrap (study 136 method)
    r_all = net.values
    mu = r_all.mean()
    rng = np.random.default_rng(S135.SEED)
    base = S135.boot_paths(r_all, S135.NPATH, 252, S135.BLOCK, rng)
    nv = len(vb)
    for case, shift in [("backtest", 0.0), ("half mean (haircut)", mu / 2)]:
        xb = base - shift
        for nn, (meas, lc) in [(a_, b_) for a_ in sorted({nv, 11}) for b_ in
                               [("compounded net column", live_cum), ("equity column", live_eq)]]:
            fin = np.prod(1 + xb[:, :nn], axis=1) - 1
            rows.append(dict(table="bootstrap", case=case, nights=nn, measure=meas, live_cum=lc, median=np.median(fin),
                             p5=np.quantile(fin, 0.05), p10=np.quantile(fin, 0.1),
                             share_le_minus5pct=(fin <= -0.05).mean(),
                             percentile=(fin < lc).mean() + 0.5 * (fin == lc).mean(),
                             note="11 nights = live compounded net with the missed night of 2026-09-30 as zero"
                             if nn == 11 and nv != 11 else ""))
        ls = pd.DataFrame((xb < 0).astype(np.int8)).T.rolling(6).sum().T.values >= 5
        for hz in [21, 63, 252]:
            rows.append(dict(table="bootstrap", case=case, nights=hz, statistic="chance of 5+ losses in some 6-night window",
                             value=ls[:, 5:hz].any(axis=1).mean()))
        lp = np.log1p(xb)
        cs = np.cumsum(lp, axis=1)
        c11 = cs[:, 10:] - np.concatenate([np.zeros((xb.shape[0], 1)), cs[:, :-11]], axis=1)
        for hz in [63, 252]:
            rows.append(dict(table="bootstrap", case=case, nights=hz,
                             statistic="chance of some 11-night stretch at or below -5%",
                             value=(c11[:, :hz - 10] <= np.log(0.95)).any(axis=1).mean()))
        del xb, lp, cs, c11, ls
    rows.append(dict(table="live", period=f"{vb.index[0].date()}..{vb.index[-1].date()}", n=nv, net_bp=1e4 * vb.mean(),
                     hit=(vb > 0).mean(), live_cum=live_cum, live_equity_cum=live_eq, live_losses_last6=live_L6, live_cum_last6=live_C6,
                     backtest_net_bp=1e4 * mu, backtest_sd_bp=1e4 * r_all.std(ddof=1),
                     t_live_vs_backtest=(vb.mean() - mu) / (r_all.std(ddof=1) / np.sqrt(nv))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT162, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(4).to_string(index=False))


# ------------------------------------------------------------------ 159
def _pmask(idx, a, b):
    return (idx >= a) & (idx <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))


def rescore_rows(name, out, pricing):
    rows = []
    for lab, a, b in PER:
        x = out.loc[a:b]
        x = x[x.n > 0]
        st = ann_stats(x.net)
        rows.append(dict(table="rescore", strategy=name, pricing=pricing, period=lab, n=st["n"], sharpe=st["sharpe"],
                         net_bp=1e4 * x.net.mean(), gross_bp=1e4 * x.gross.mean(), cost_bp=1e4 * x.cost.mean(),
                         maxdd=st["maxdd"], hit=st["hit"], total=st["total"], names=x.n.mean()))
    return rows


def bias_rows(name, leg, sel, ratio, ref="panel"):
    """sel: DataFrame date, ticker (price date); ratio: wide cross / reference price. bp = reference / cross - 1."""
    rows = []
    v = ratio.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([sel.date, sel.ticker])).values
    d = pd.DataFrame({"date": sel.date.values, "bp": 1e4 * (1 / v - 1)})
    for lab, a, b in PER:
        x = d[_pmask(d.date, a, b)]
        y = x.bp.dropna()
        rows.append(dict(table="bias", strategy=name, leg=leg, reference=ref, period=lab, n=len(x),
                         found=y.size / max(len(x), 1), mean_bp=y.mean(), median_bp=y.median(),
                         p10_bp=y.quantile(0.1), p90_bp=y.quantile(0.9), mean_abs_bp=y.abs().mean(),
                         share_pos=(y > 0.5).mean(), share_zero=(y.abs() <= 0.5).mean(),
                         t=y.mean() / (y.std() / np.sqrt(len(y))) if len(y) > 2 else np.nan))
    return rows


def fallback_rows(name, W, miss):
    rows = []
    h = W > 0
    for lab, a, b in PER:
        hh, mm = h.loc[a:b], (miss.reindex_like(W).fillna(True) & h).loc[a:b]
        rows.append(dict(table="fallback", strategy=name, period=lab, legs=int(hh.sum().sum()),
                         share_panel=mm.sum().sum() / max(hh.sum().sum(), 1)))
    return rows


def picks_frame(W, days, shift=0):
    nxt = pd.Series(days[1:], index=days[:-1])
    s = W.stack()
    s = s[s > 0].reset_index()
    s.columns = ["t", "ticker", "w"]
    d = s.t
    for _ in range(shift):
        d = d.map(nxt)
    return pd.DataFrame({"date": d, "ticker": s.ticker}).dropna()


def wide(x, kind, col, index, columns):
    g = x[x.kind == kind].drop_duplicates(["date", "ticker"])
    return g.pivot(index="date", columns="ticker", values=col).reindex(index=index, columns=columns)


def raw_wide(fn, index, columns, alpaca_style=True):
    r = pd.read_parquet(fn).dropna(subset=["date"])
    r["date"] = pd.to_datetime(r.date)
    if not alpaca_style:
        r["ticker"] = r.ticker.str.replace(".", "-", regex=False)
    r = r.drop_duplicates(["date", "ticker"])
    return {k: r.pivot(index="date", columns="ticker", values=k).reindex(index=index, columns=columns) for k in ["o", "c"]}


def study159():
    import study99_cross_retrain as S99
    rows = []
    X = cached_crosses()
    X = X[X.px.notna()].copy()
    Xp = X.assign(ticker=X.ticker.str.replace(".", "-", regex=False))
    # ---------------- small caps
    K = smallcap_picks()
    P, cols, alld, days = K["P"], K["cols"], K["alld"], K["days"]
    ro, rc, err = S99.cross_ratios(P, cols, alld, Xp[["ticker", "date", "kind", "px"]])
    print("small caps: cross data errors", err, flush=True)
    for k, v in err.items():
        rows.append(dict(table="data_errors", strategy="small caps (all cached crosses)", leg=k, n=v["found"],
                         scale_mismatch=v["scale"], ratio_out_of_range=v["ratio"]))
    fo_, fc_ = ro.fillna(1.0), rc.fillna(1.0)
    Rp = K["Rn"]
    Rx = (1 + Rp) * fo_.shift(-1).reindex(days) / fc_.reindex(days) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(days)
    nets = {}
    for name, W in [("smallcap_nonews (top 10, no news)", K["W"]), ("ref: s_intraday_loser (top 10, news or not)", K["Wall"])]:
        for pr, R in [("panel", Rp), ("cross", Rx)]:
            o = bt.run(W, R, K["cost"])
            nets[(name, pr)] = o.net.where(o.n > 0)
            rows += rescore_rows(name, o, pr)
        rows += fallback_rows(name, W, miss)
    W = K["W"]
    so, sc = picks_frame(W, alld, 1), picks_frame(W, alld, 0)
    nm = "smallcap_nonews (top 10, no news)"
    rows += bias_rows(nm, "sell-day open (t+1)", so, ro, "panel (Yahoo open)")
    rows += bias_rows(nm, "buy-day close (t)", sc, rc, "panel (Yahoo close)")
    ix = alld[alld >= "2023-12-01"]
    Xs = Xp[Xp.ticker.isin(cols)]
    XO, XC, FO = (wide(Xs, "open", "px", ix, cols), wide(Xs, "close", "px", ix, cols),
                  wide(Xs, "open", "first_px", ix, cols))
    rows += bias_rows(nm, "sell-day open (t+1)", so, XO / FO, "first trade")
    if os.path.exists(RAW_SMALL):
        R_ = raw_wide(RAW_SMALL, ix, cols, alpaca_style=False)
        rows += bias_rows(nm, "sell-day open (t+1)", so, XO / R_["o"], "Alpaca daily bar open (live scorer)")
        rows += bias_rows(nm, "buy-day close (t)", sc, XC / R_["c"], "Alpaca daily bar close (live scorer)")
        del R_
    bt_small = nets[(nm, "cross")]
    del K, P, Rp, Rx, ro, rc, fo_, fc_, XO, XC, FO, Xs
    # ---------------- ADRs
    P16, D, tick = adr_panels()
    d16 = P16["c"].index
    Wd = adr_picks(P16, D, tick, load_m1545())
    alpt = tick
    Xa = X[X.ticker.isin(alpt)]
    ix = d16[d16 >= "2023-12-01"]
    XO, XC, FO = (wide(Xa, "open", "px", ix, alpt), wide(Xa, "close", "px", ix, alpt), wide(Xa, "open", "first_px", ix, alpt))
    R_ = raw_wide(RAW_ADR, ix, alpt)
    osplit = (P16["o"] / P16["c"] * P16["rawc"])[alpt].reindex(ix)
    csplit = P16["rawc"][alpt].reindex(ix)
    scale_ok = ((R_["o"] / osplit) / (R_["c"] / csplit) - 1).abs() < 0.005
    ra, rca = {}, {}
    for k, x, pr in [("ro", XO, R_["o"]), ("rc", XC, R_["c"])]:
        r = x / pr
        bad_s = x.notna() & ~scale_ok
        bad_r = x.notna() & scale_ok & ((r < 0.8) | (r > 1.25))
        rows.append(dict(table="data_errors", strategy="ADRs (all cached crosses)", leg=k, n=int(x.notna().sum().sum()),
                         scale_mismatch=int(bad_s.sum().sum()), ratio_out_of_range=int(bad_r.sum().sum())))
        ra[k] = r.where(scale_ok & ~bad_r).reindex(d16)
    ro, rc = ra["ro"], ra["rc"]
    Rp = D["co_next"][alpt]
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1) / rc.fillna(1.0) - 1
    miss = ro.shift(-1).isna() | rc.isna()
    for name, W in Wd.items():
        for pr, R in [("panel", Rp), ("cross", Rx)]:
            o = bt.run(W, R.reindex(W.index), D["cost"][alpt].reindex(W.index))
            nets[(name, pr)] = o.net.where(o.n > 0)
            rows += rescore_rows(name, o, pr)
        rows += fallback_rows(name, W, miss.reindex(W.index))
        so, sc = picks_frame(W, d16, 1), picks_frame(W, d16, 0)
        rows += bias_rows(name, "sell-day open (t+1)", so, ro, "panel (Alpaca daily bar open)")
        rows += bias_rows(name, "buy-day close (t)", sc, rc, "panel (Alpaca daily bar close)")
        rows += bias_rows(name, "sell-day open (t+1)", so, (XO / FO).reindex(d16), "first trade")
    # ---------------- live record
    sh = pd.read_csv(os.path.join(LOG, "shadow_scores.csv"), parse_dates=["day"]).set_index("day")
    tr = pd.read_csv(os.path.join(LOG, "shadow_trades.csv"), parse_dates=["day"])
    tr = tr[tr.strategy.isin(["smallcap_nonews", "adr_loser"])].copy()
    cal = pd.DatetimeIndex(A.trading_days("2026-09-20", "2026-10-31"))
    nxt = pd.Series(cal[1:], index=cal[:-1])
    xk = X.set_index(["ticker", "date", "kind"]).px
    tr["xc"] = [xk.get((t, d, "close"), np.nan) for t, d in zip(tr.ticker, tr.day)]
    tr["xo"] = [xk.get((t, nxt.get(d), "open"), np.nan) for t, d in zip(tr.ticker, tr.day)]
    tr["ret_x"] = np.where(tr.xc.notna() & tr.xo.notna(), tr.xo / tr.xc - 1, tr.ret)
    tr["net_x"] = tr.ret_x - tr.cost
    tr["fallback"] = tr.xc.isna() | tr.xo.isna()
    tr = tr[~tr.flag_split]
    for strat, btname in [("smallcap_nonews", "smallcap_nonews (top 10, no news)"),
                          ("adr_loser", "adr_loser_live_1545"), ("adr_loser", "adr_loser_s16_close")]:
        if (btname, "cross") not in nets:
            continue
        g = tr[tr.strategy == strat]
        lx = g.groupby("day").net_x.mean()
        ls = sh[strat].dropna()
        b = nets[(btname, "cross")].loc["2024-01":"2026-09"].dropna()
        nv = len(lx)
        roll = b.rolling(nv).mean().dropna()
        for pricing, lv in [("live scorer (Alpaca daily bar close -> next open)", ls), ("official crosses", lx)]:
            rows.append(dict(table="live", strategy=strat, compared_with=btname, pricing=pricing,
                             period=f"{lv.index.min().date()}..{lv.index.max().date()}", n=len(lv),
                             net_bp=1e4 * lv.mean(), hit=(lv > 0).mean(), total=(1 + lv).prod() - 1,
                             sd_bp=1e4 * lv.std(), backtest_cross_net_bp=1e4 * b.mean(), backtest_sd_bp=1e4 * b.std(),
                             t_vs_backtest=(lv.mean() - b.mean()) / (b.std() / np.sqrt(len(lv))),
                             pct_of_backtest_rolling_mean=(roll < lv.mean()).mean(),
                             legs=len(g), share_fallback=g.fallback.mean()))
        # live per-leg bias: scorer prices vs crosses
        m = g.dropna(subset=["xo"])
        mc = g.dropna(subset=["xc"])
        if btname != "adr_loser_s16_close":
            rows.append(dict(table="live", strategy=strat, pricing="bias: scorer next open / opening cross - 1",
                             n=len(m), mean_bp=1e4 * (m.next_open / m.xo - 1).mean(),
                             median_bp=1e4 * (m.next_open / m.xo - 1).median()))
            rows.append(dict(table="live", strategy=strat, pricing="bias: scorer close / closing cross - 1",
                             n=len(mc), mean_bp=1e4 * (mc.close / mc.xc - 1).mean(),
                             median_bp=1e4 * (mc.close / mc.xc - 1).median()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT159, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    for t, x in df.groupby("table", sort=False):
        print("\n==", t)
        print(x.dropna(axis=1, how="all").drop(columns="table").round(4).to_string(index=False))
    r = df[df.table == "rescore"].pivot_table(index="strategy", columns=["period", "pricing"], values=["sharpe", "net_bp"],
                                              sort=False)
    print(r.round(2).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("159", "all"):
        study159()
    if what in ("162", "all"):
        study162()
