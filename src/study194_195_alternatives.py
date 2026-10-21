"""Studies 194 and 195: alternatives to the overnight blend in its weak stretch since July 2026.

194  Blend by beta. Question: studies 176/178 found that high-beta and tech hardware stocks had a record overnight
     reversal in October 2026, and that the blend earns part of its return from the high-beta overnight premium.
     If the blend's candidates are restricted to lower-beta names (or the picks are kept beta-neutral), does that
     avoid the 2026 losses, and what does it cost in the good periods?
195  Other legs recently. Question: did the day-session long (studies 60/67/71) and the day short of the blend
     picks (study 68) hold up since July 2026, and would the combined cycle (study 71: night + half-size day long +
     quarter-size day short) have done better than the blend alone in 2026Q3? October: the live day-short record.

Data
  core.load_panel (split- and dividend-adjusted o/c, rawc, dv; now ends 2026-10-19, so every October night including
  10-16 -> 10-19 is priced by the study 94 ratio method). Blend score as
  study135_137_expectations.build(): S = (2 rank(ensemble) + rank(p_jump - p_drop)) / 3 (results/study23_pred.parquet,
  study33_pred.parquet; entry days 2024-01-02..2026-09-25), live version capped at 3 names per industry
  (paper_overnight.cap_by_industry rule, data/store/sectors.parquet; names without an industry are their own group).
  October 2026: study 190's frame-B blend scores from the frozen 2026Q4 models (study190_191_live_leads.all_scores:
  data/local/study179_scores.parquet + study190_scores.parquet, entry days 2026-09-28..10-16), Alpaca-tradable names,
  as study 190. Beta: 60-day beta of daily close-to-close returns to SPY (study 90 / 176 roll_beta, 40 days minimum),
  taken through day t-1 so it is known at the 15:45 decision on day t. Tech hardware: study 90/92 definition
  (study166_167_blend_parts.industry_map). Official crosses: every data/local/auctions*.parquet (study 166
  cached_crosses) plus data/local/auctions194.parquet fetched here with study 94's rule (listing-venue 'O' / '6' print
  in widening windows, second pass the largest 'O' / '6' print of any listing exchange; Alpaca SIP trades, 60 requests
  a minute). Day-session long: results/study60_pred.parquet p_all (predictions 2024-07..2026-09-25), price > $5, ADV >
  $5M, top 10, study 25 cost column (as study 94). Easy to borrow: today's Alpaca asset list (as studies 68/94). Live
  day short: logs/paper/dayshort_<day>.json (orders), logs/paper/dayshort_names.csv (src/week_dayshort.py output:
  official open and close per name, auction cost) for 10-05..10-16; the days 10-19 and 10-20 are priced here the
  same way (alpaca_data._official open and close, week_dayshort.auction_cost), cached in
  data/local/study195_dayshort_live.parquet. Live night book: logs/paper/virtual_days.csv (read only).

Design 194: each entry day t, universe = names with a blend score and a beta. Variants, all top 10 equal weight with
  at most 3 per industry, taken down the blend ranking: blend (live, no beta rule); low beta (beta below the
  universe median that day); mid beta (beta between the universe's 20th and 80th percentiles); beta-neutral (walk
  down the ranking and accept a name unless it pushes the sum of (beta - universe mean beta) over the picks so far
  beyond +-1.0 in absolute value without moving it closer to 0; so the 10 picks' mean beta ends within +-0.1 of the
  universe mean). Night return at crosses as study 94 (panel night return x open cross / panel open on t+1 / (close
  cross / panel close on t); legs without a usable cross at the panel price), cost core.exec_cost_bps(P, 'auction')
  + 2.5 bp per side. October (entry days 09-28..10-16) priced with study 190's night_returns (study 94 ratio method;
  it switches to raw cross to cross only for a night past the panel's end, none here). Per variant and period: nights, Sharpe, mean net bp, max drawdown, compounded, hit rate,
  mean pick beta, universe mean beta, tech hardware picks a night, overlap with the blend's picks, mean nightly
  difference vs the blend and its paired t. Periods 2024-25H1 (2024-01..2025-06), 2025H2-26 (2025-07..2026-09),
  2024-26, 2024-01..2026-06 (before the weak stretch), 2026Q3, October 2026 (15 nights 09-28..10-16, and the 13 of
  those with a live entry). Monthly mean net bp 2025-07..2026-10.
Design 195: all legs aligned to the entry night t of the cycle (night t, then the day session of t+1). Night leg:
  capped blend (live) and uncapped blend (study 94's study 71 night leg). Day short: easy-to-borrow picks of night
  t, short from t+1's opening cross to t+1's closing cross, 1/10 of capital each (standalone; the combined books use
  a quarter of it, as the live job's 2.5% of equity per name), cost auction + 2.5 bp per side of day t+1; on the
  capped picks (what the live job shorts) and on the uncapped picks (study 94 / 68). Day long: top 10 by p_all of
  day t+1, bought at the opening cross, sold at the closing cross, cost column x 2 (standalone; half size in the
  combined books). Day returns at crosses: panel day return x close-cross ratio / open-cross ratio (study 94).
  Combined: 67 = night + 0.5 day long, 68 = night + 0.25 day short, 71 = night + 0.5 day long + 0.25 day short.
  Monthly (by month of t) 2024-07..2026-09: nights, mean bp and Sharpe per leg and book; periods 2024H2-25H1,
  2025H2-26H1, 2026Q3, 2024H2-26 with Sharpe, mean, max drawdown and the paired t of each combined book vs the
  night leg alone. October: backtest-style day short of study 190's frame-B capped blend picks (entries 09-28..10-16,
  day sessions 09-29..10-19, raw cross to cross on the day, panel open/close where no cross; cost auction + 2.5 bp per
  side of the day), and the live record (auction P&L / account equity at the short; x4
  = standalone equivalent), with the live night book for the combined October picture. No day-long predictions
  exist after 2026-09-25 and the live job does not trade a day long, so the day long has no October figure.

Data checks: the blend reproduces build() / study 94 at crosses (Sharpe 2.65 / 2.23 / 2.45, mean 38.2 bp) and study
  190's October blend (-47.2 bp over 15 nights, -57.6 over the 13 live nights). 47,754 cross legs needed, 40,075
  cached, 7,679 fetched into auctions194 (98.9% found, 70 of 81 more in the second pass). Backtest legs at the panel
  price 0.6-0.8% of picks; no October leg without a cross. SLN (a low-beta pick on 10-16) rose from 8.68 to 19.58
  overnight on $1.4B volume (a real event, no split), worth about +1,250 bp to the low-beta book that night; the
  October rows without 10-16 show the result without it. Peak memory 2.4 GB.

Result 194 (net bp a night, Sharpe; 2024-25H1 / 2025H2-26 / 2024-26): blend 39.3 / 36.8 / 38.2, Sharpe 2.65 / 2.23 /
  2.45, mean pick beta 2.2 against a universe mean of 1.08, 1.3 tech hardware picks a night. Low beta 5.5 / 2.3 / 4.0
  (Sharpe 0.36 2024-26, max drawdown -37%; t vs blend -3.9); mid beta 6.3 / 25.2 / 14.9 (Sharpe 1.27, t -2.9);
  beta-neutral 12.5 / 23.2 / 17.3 (Sharpe 1.42, max drawdown -30%, t -2.6). Before the weak stretch (2024-01..
  2026-06) the blend made 42.2 bp a night; the beta rules give up 22-36 bp of it. 2026Q3 (61 nights): blend -3.1 bp,
  low beta -21.7, mid beta -10.5, beta-neutral -11.6 (all worse; t -0.3 to -0.5). October (15 nights 09-28..10-16):
  blend -47.2 (compounded -7.3%), low beta +64.5 (+9.3%; without 10-16: -23.5 vs blend -86.5), mid beta +4.2,
  beta-neutral +0.4 (-2.2 vs -86.5 without 10-16; t vs blend +0.9, +1.9 without 10-16). Blend pick beta in October 3.1
  (universe 0.95), 3.8 tech hardware picks a night; beta-neutral 1.0 and 1.1.
Reading 194: the high-beta tilt is where the blend's edge comes from. Taking it out cuts the 2024-26 return by more
  than half (beta-neutral 17 vs 38 bp) and did not help in 2026Q3, when the blend was already flat. It would have
  avoided most of the October loss (about 85 bp a night better on the 14 nights to 10-15), which is the high-beta
  reversal of study 178, but 14-15 nights give t of 1-2 only.
Result 195 (bp a night by entry night, 2024H2-25H1 / 2025H2-26H1 / 2026Q3 / 2024H2-26): night (capped blend) 32.1 /
  46.6 / -3.1 / 34.7; day short of the capped easy-to-borrow picks (standalone) 4.3 / 4.7 / +40.4 / 8.4 (Sharpe 2.15
  in 2026Q3; uncapped picks 7.2 / 5.3 / 38.2 / 9.7); day long top 10 (standalone) 9.3 / 7.1 / +16.2 / 9.0. Both day
  legs had their best quarter in 2026Q3 and were positive in each of July..September except the day long in August
  (-16.9). Study 71 book (night + 0.5 day long + 0.25 day short, capped): 38.0 / 51.3 / +14.9 / 41.5, Sharpe 2.33 /
  2.91 / 0.86 / 2.44 vs night alone 2.07 / 2.80 / -0.20 / 2.17; 2026Q3 difference +18.1 bp a night (t 1.45), max
  drawdown -12.8% vs -14.7%. 2024H2-26 difference +6.7 bp (t 1.68). October: backtest-style day short of study 190's
  blend picks +122 bp a day (standalone, 15 days 09-29..10-19, hit 73%); live record (12 days 10-05..10-20) +9.3 bp a
  day on equity (quarter size; about +37 bp standalone), live nights followed by a day short -86.1, together -76.8
  (t +0.9 vs night alone). On the 11 shared days the live day short earned 56 bp standalone vs 159 backtest-style
  (correlation 0.69): the live job skips names it cannot size (e.g. 10-14 four names, 10-16 two) and its fills are
  scored at the crosses on the names actually shorted.
Reading 195: both day legs held up since July; the day short gained most when the blend lost (the high-beta picks
  also fell in the day session, study 178). The full cycle would have turned 2026Q3 from -3 to +15 bp a night, a
  gain of the right sign but not significant. It needs a margin account for the short and opening-auction (on-open)
  orders for the day long, which the live account does not have yet.
Caveats: today's easy-to-borrow list and sectors.parquet are applied to all dates; the beta-neutral rule is one simple
  greedy choice; October uses study 179/190 scores (backtest-style frames, not the live 15:45 inputs) and 15 nights;
  the day long has no predictions after 2026-09-25; live day short P&L is scored at official crosses with the model
  cost, not the paper fills; borrow fees are not charged.

Output: results/study194_beta_blend.csv (table = summary / monthly / october_nightly / data),
  results/study195_other_legs.csv (table = monthly / period / october_backtest / october_live / october_summary /
  data).

    python src/study194_195_alternatives.py fetch    # official crosses missing from the caches -> auctions194.parquet
    python src/study194_195_alternatives.py          # both studies
"""
import json
import os
import sys
import time

os.environ.setdefault("ALPACA_PER_MIN", "60")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402
import study94_cross_rescore as S94   # noqa: E402  (sets A.RL at import; reset in rearm)
import study166_167_blend_parts as S166   # noqa: E402
import study190_191_live_leads as S190   # noqa: E402
from study174_178_recent_edge import roll_beta   # noqa: E402

LOC = os.path.join(DATA, "local")
LOG = os.path.join(A.ROOT, "logs", "paper")
AUC194 = os.path.join(LOC, "auctions194.parquet")
LIVE195 = os.path.join(LOC, "study195_dayshort_live.parquet")
OUT194 = os.path.join(RES, "study194_beta_blend.csv")
OUT195 = os.path.join(RES, "study195_other_legs.csv")
VARIANTS = ["blend", "low_beta", "mid_beta", "beta_neutral"]
PER194 = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09"),
          ("2024-01..2026-06", "2024-01", "2026-06"), ("2026Q3", "2026-07", "2026-09")]
PER195 = [("2024H2-25H1", "2024-07", "2025-06"), ("2025H2-26H1", "2025-07", "2026-06"), ("2026Q3", "2026-07", "2026-09"),
          ("2024H2-26", "2024-07", "2026-09")]
NEUTRAL_BAND = 1.0       # max |sum(beta - universe mean)| over the picks; / 10 picks = +-0.1 on the mean


def rearm():
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))


# ------------------------------------------------------------------ selection
def pick_day(v, b, g, mode, ustat, cap=3, n=10):
    """v: score array (NaN = no candidate), b: beta array, g: group array, ustat: universe beta stats dict."""
    idx = np.where(~np.isnan(v))[0]
    order = idx[np.argsort(-v[idx], kind="stable")]
    if mode == "low_beta":
        order = order[b[order] < ustat["median"]]
    elif mode == "mid_beta":
        order = order[(b[order] >= ustat["p20"]) & (b[order] <= ustat["p80"])]
    elif mode == "beta_neutral":
        order = order[~np.isnan(b[order])]
    cnt, out, dev = {}, [], 0.0
    for j in order:
        if cnt.get(g[j], 0) >= cap:
            continue
        if mode == "beta_neutral":
            nd = dev + b[j] - ustat["mean"]
            if abs(nd) > NEUTRAL_BAND and abs(nd) >= abs(dev):
                continue
            dev = nd
        cnt[g[j]] = cnt.get(g[j], 0) + 1
        out.append(j)
        if len(out) == n:
            break
    return out


def ustats(b):
    b = b[~np.isnan(b)]
    if len(b) < 50:
        return None
    return dict(median=np.median(b), p20=np.quantile(b, 0.2), p80=np.quantile(b, 0.8), mean=b.mean())


def blend_scores(P, cols):
    """As study135_137_expectations.build()."""
    days = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    return S.loc["2024-01":"2026-09"]


def betas(P, cols):
    c = P["c"][cols]
    r1 = (c / c.shift(1) - 1).astype("float32")
    spy = P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1
    return roll_beta(r1, spy).shift(1).astype("float32")       # through t-1: known at 15:45 on day t


def backtest_picks(S, B, grp):
    """variant -> weight frame (index S.index), plus universe mean beta per day."""
    g = grp.reindex(S.columns).values
    Bv = B.reindex(index=S.index, columns=S.columns).values.astype("float64")
    Sv = S.values
    W = {k: np.zeros(S.shape) for k in VARIANTS}
    um = np.full(len(S), np.nan)
    for i in range(len(S)):
        v, b = Sv[i], Bv[i]
        us = ustats(b[~np.isnan(v)])
        if us is None:
            continue
        um[i] = us["mean"]
        for k in VARIANTS:
            out = pick_day(v, b, g, k if k != "blend" else "none", us)
            if out:
                W[k][i, out] = 1.0 / len(out)
    W = {k: pd.DataFrame(x, index=S.index, columns=S.columns) for k, x in W.items()}
    return W, pd.Series(um, index=S.index)


def october_picks(P, cols, B, grp):
    """variant -> {day: [tickers]} from study 190's frame-B blend scores (tradable names), plus universe mean beta."""
    from paper_overnight import tradable
    sc = S190.all_scores()
    sc = sc[sc.kind == "B"]
    ok = set(tradable(sorted(set(sc.index.get_level_values(1)))))
    out, um = {k: {} for k in VARIANTS}, {}
    for day, x in sc.groupby(level=0):
        v = x.droplevel(0).blend
        v = v[v.index.isin(cols)]
        b = B.loc[day].reindex(v.index).values.astype("float64") if day in B.index else np.full(len(v), np.nan)
        us = ustats(b)
        um[day] = us["mean"]
        cand = v.where(v.index.isin(ok)).values.astype("float64")
        g = grp.reindex(v.index).values
        for k in VARIANTS:
            idx = pick_day(cand, b, g, k if k != "blend" else "none", us)
            out[k][day] = list(v.index[idx])
    return out, pd.Series(um)


def prep(with_oct=True):
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    grp, th = S166.industry_map(cols)
    S = blend_scores(P, cols)
    B = betas(P, cols)
    W, um = backtest_picks(S, B, grp)
    K = dict(P=P, cols=cols, days=days, grp=grp, th=th, S=S, B=B, W=W, um=um)
    if with_oct:
        K["oct"], K["oct_um"] = october_picks(P, cols, B, grp)
    a = pd.read_parquet(f"{DATA}/local/alpaca_assets_active.parquet")
    K["etb"] = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)].str.replace(".", "-", regex=False))
    return K


def uncapped(S):
    return bt.select_topk(S, S.notna(), 10)


def day_long_picks():
    x = pd.read_parquet(f"{RES}/study60_pred.parquet")
    liq = (x.px > 5) & (x.adv20 > 5e6)
    dl = x[liq & x.p_all.notna()].sort_values("p_all", ascending=False).groupby("date").head(10)
    return dl[["date", "ticker", "R", "cost"]].reset_index(drop=True)


def live_dayshort_orders():
    out = {}
    for f in sorted(os.listdir(LOG)):
        if f.startswith("dayshort_") and f.endswith(".json"):
            r = json.load(open(os.path.join(LOG, f)))
            if r.get("submitted") and r.get("orders"):
                out[r["day"]] = r
    return out


# ------------------------------------------------------------------ fetch
def fetch():
    from concurrent.futures import ThreadPoolExecutor
    K = prep()
    days = K["days"]
    nxt = S190.next_days(days)
    need = []
    # 194 backtest: close of t, open of t+1 for every variant (blend is cached)
    for k in VARIANTS[1:]:
        need.append(S166.legs(K["W"][k], days))
    # 194 October: close of t, open of the next trading day
    rows = []
    for k in VARIANTS:
        for d, t in K["oct"][k].items():
            rows += [dict(ticker=x, date=d, kind="close") for x in t] + [dict(ticker=x, date=nxt[d], kind="open") for x in t]
    # 195 day legs: easy-to-borrow capped and uncapped picks of t, open and close of t+1; day long open and close
    for Wx in [K["W"]["blend"], uncapped(K["S"])]:
        s = Wx.stack()
        s = s[s > 0].reset_index()
        s.columns = ["t", "ticker", "w"]
        s = s[s.ticker.isin(K["etb"])]
        d1 = s.t.map(nxt)
        need += [pd.DataFrame({"ticker": s.ticker, "date": d1, "kind": k}) for k in ["open", "close"]]
    dl = day_long_picks()
    need += [pd.DataFrame({"ticker": dl.ticker, "date": dl.date, "kind": k}) for k in ["open", "close"]]
    # 195 October backtest-style day short: open and close of t+1 for the blend's October picks
    for d, t in K["oct"]["blend"].items():
        rows += [dict(ticker=x, date=nxt[d], kind=k) for x in t if x in K["etb"] for k in ["open", "close"]]
    need.append(pd.DataFrame(rows))
    need = pd.concat(need, ignore_index=True).dropna()
    need["date"] = pd.to_datetime(need.date)
    del K
    need["ticker"] = need.ticker.astype(str).str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    have = pd.read_parquet(AUC194) if os.path.exists(AUC194) else pd.DataFrame(
        columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    have["date"] = pd.to_datetime(have.date)
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)
    S94.AUC_FN = AUC194
    rearm()
    rows, t0 = [], time.time()

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
    if os.path.exists(AUC194):
        S94.refetch_missing()
    fetch_live_dayshort()


def fetch_live_dayshort():
    """Official open / close and auction cost for live day-short names on days not in dayshort_names.csv."""
    import week_dayshort as WD
    have = pd.read_csv(os.path.join(LOG, "dayshort_names.csv")) if os.path.exists(os.path.join(LOG, "dayshort_names.csv")) else pd.DataFrame(columns=["day"])
    cache = pd.read_parquet(LIVE195) if os.path.exists(LIVE195) else pd.DataFrame(columns=["day", "symbol"])
    today = str(pd.Timestamp.now(tz="America/New_York").date())
    rows = []
    rearm()
    for day, r in live_dayshort_orders().items():
        if day in set(have.day) or day in set(cache.day) or day >= today:
            continue
        for od in r["orders"]:
            t, q = od["symbol"], int(od["qty"])
            o0 = A._official(t, day, "open")[0]
            c0 = A._official(t, day, "close")[0]
            cost = WD.auction_cost(t, day, o0) if o0 == o0 else np.nan
            rows.append(dict(day=day, symbol=t, qty=q, open=o0, close=c0,
                             auction_pnl=q * (o0 - c0) - cost * q * (o0 + c0), cost=cost))
        print("live day short priced", day, len(r["orders"]), flush=True)
    if rows:
        pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).to_parquet(LIVE195, index=False)


# ------------------------------------------------------------------ stats helpers
def stat(x):
    x = x.dropna()
    s = ann_stats(x)
    return dict(nights=len(x), sharpe=s["sharpe"], net_bp=1e4 * x.mean() if len(x) else np.nan, maxdd=s["maxdd"],
                compounded=float((1 + x).prod() - 1) if len(x) else np.nan, hit=float((x > 0).mean()) if len(x) else np.nan)


def ptt(a, b):
    return S166.paired_t(a, b)


# ------------------------------------------------------------------ 194
def study194(K, Rx, C):
    P, cols, days, th = K["P"], K["cols"], K["days"], K["th"]
    B = K["B"].reindex(index=K["S"].index)
    rows, nets, info = [], {}, {}
    thv = th.reindex(cols).fillna(False).astype(float)
    for k, W in K["W"].items():
        nets[k] = bt.run(W, Rx, C).net.where(W.sum(axis=1) > 0).dropna()
        h = (W > 0)
        info[k] = pd.DataFrame({"pick_beta": B.where(h).mean(axis=1), "techhw": h.mul(thv, axis=1).sum(axis=1),
                                "overlap": (h & (K["W"]["blend"] > 0)).sum(axis=1), "names": h.sum(axis=1)})
    for k in VARIANTS:
        for lab, a, b in PER194:
            x = nets[k].loc[a:b]
            inf = info[k].loc[x.index]
            base = nets["blend"].reindex(x.index)
            rows.append(dict(table="summary", variant=k, period=lab, **stat(x), pick_beta=inf.pick_beta.mean(),
                             universe_beta=K["um"].reindex(x.index).mean(), techhw_per_night=inf.techhw.mean(),
                             overlap_blend=inf.overlap.mean(), names=inf.names.mean(),
                             diff_vs_blend_bp=1e4 * (x - base).mean(), t_vs_blend=ptt(x, base) if k != "blend" else np.nan))
    for k in VARIANTS:
        x = nets[k].loc["2025-07":]
        for m, v in x.groupby(x.index.to_period("M")):
            rows.append(dict(table="monthly", variant=k, period=str(m), nights=len(v), net_bp=1e4 * v.mean(),
                             compounded=float((1 + v).prod() - 1), pick_beta=info[k].pick_beta.reindex(v.index).mean(),
                             techhw_per_night=info[k].techhw.reindex(v.index).mean()))
    # October from study 190's scores, priced as study 190
    idx = days[(days >= S190.OCT[0]) & (days <= S190.OCT[1])]
    Rxo, misso, Co = S190.night_returns(P, cols, idx)
    live = set(pd.read_csv(os.path.join(LOG, "virtual_days.csv")).day)
    vd = pd.read_csv(os.path.join(LOG, "virtual_days.csv"), parse_dates=["day"]).set_index("day").net
    bo = K["B"]
    on = {}
    for k in VARIANTS:
        for d in idx:
            t = [x for x in K["oct"][k].get(d, []) if x in Rxo.columns]
            r = Rxo.loc[d, t]
            if d == days[-1]:
                r = r.dropna()
                t = list(r.index)
            else:
                r = r.fillna(0.0)
            net = float(r.mean() - 2 * Co.loc[d, t].mean() / 1e4) if t else np.nan
            on[(k, d)] = net
            pk = K["oct"][k].get(d, [])
            rows.append(dict(table="october_nightly", variant=k, period=str(d.date()), live_night=str(d.date()) in live,
                             net_bp=1e4 * net, names=len(t), pick_beta=float(bo.loc[d].reindex(pk).mean()),
                             universe_beta=K["oct_um"].get(d), techhw_per_night=int(th.reindex(pk).fillna(False).sum()),
                             overlap_blend=len(set(pk) & set(K["oct"]["blend"].get(d, []))),
                             live_book_bp=1e4 * vd.get(d, np.nan), picks=" ".join(pk)))
    on = pd.Series(on)
    for k in VARIANTS:
        x = on[k]
        x.index = pd.to_datetime(x.index)
        base = on["blend"]
        base.index = pd.to_datetime(base.index)
        nt = pd.DataFrame([r for r in rows if r["table"] == "october_nightly" and r["variant"] == k]).set_index("period")
        nt.index = pd.to_datetime(nt.index)
        for lab, sel in [("Oct 2026: 15 nights 09-28..10-16", np.ones(len(x), bool)),
                         ("Oct 2026: 13 live nights", nt.live_night.reindex(x.index).values.astype(bool)),
                         ("Oct 2026: 14 nights 09-28..10-15 (without 10-16)", x.index < "2026-10-16")]:
            xx = x[sel]
            rows.append(dict(table="summary", variant=k, period=lab, **stat(xx), pick_beta=nt.pick_beta[sel].mean(),
                             universe_beta=nt.universe_beta[sel].mean(), techhw_per_night=nt.techhw_per_night[sel].mean(),
                             overlap_blend=nt.overlap_blend[sel].mean(), names=nt.names[sel].mean(),
                             diff_vs_blend_bp=1e4 * (xx - base[sel]).mean(),
                             t_vs_blend=ptt(xx, base[sel]) if k != "blend" else np.nan))
    rows.append(dict(table="data", variant="october legs at panel price or dropped",
                     nights=int(sum(misso.loc[d, [x for x in K["oct"][k].get(d, []) if x in misso.columns]].sum()
                                    for k in VARIANTS for d in idx if d in misso.index))))
    return pd.DataFrame(rows), nets


# ------------------------------------------------------------------ 195
def study195(K, Rx, ro, rc, nets194):
    P, cols, days = K["P"], K["cols"], K["days"]
    S = K["S"]
    nxt = pd.Series(days[1:], index=days[:-1])
    prev = pd.Series(days[:-1], index=days[1:])
    fo_, fc_ = ro.fillna(1.0), rc.fillna(1.0)
    o, c = P["o"][cols], P["c"][cols]
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=S.index)
    Wc, Wu = K["W"]["blend"], uncapped(S)
    etbm = pd.DataFrame(np.broadcast_to(np.array([t in K["etb"] for t in cols]), S.shape), index=S.index, columns=cols)
    Rd_p = (c.shift(-1) / o.shift(-1) - 1).reindex(index=S.index)
    Rd_x = (1 + Rd_p) * fc_.shift(-1).reindex(index=S.index) / fo_.shift(-1).reindex(index=S.index) - 1
    cost_d = (exec_cost_bps(P, "auction")[cols].shift(-1) + 2.5).reindex(index=S.index)
    L = {}
    L["night capped (live blend)"] = nets194["blend"]
    L["night uncapped (study 94/71)"] = bt.run(Wu, Rx, C).net.where(Wu.sum(axis=1) > 0).dropna()
    for lab, Wx in [("capped", Wc), ("uncapped", Wu)]:
        ws = ((Wx > 0) & etbm).astype(float).div(10)
        L[f"day short {lab} etb (standalone, 1/10 each)"] = bt.run(ws, Rd_x, cost_d, side=-1).net.where(
            Wx.sum(axis=1) > 0).dropna()
    dl = day_long_picks()
    dk = pd.MultiIndex.from_arrays([dl.date, dl.ticker])
    fac = fc_.stack(future_stack=True).reindex(dk).values / fo_.stack(future_stack=True).reindex(dk).values
    dl["R_x"] = (1 + dl.R.values) * np.nan_to_num(fac, nan=1.0) - 1
    gq = dl.groupby("date")
    dln = gq.R_x.mean() - 2 * gq.cost.mean() / 1e4
    dln.index = dln.index.map(prev)                              # align to the preceding night t
    dln = dln[dln.index.notna()]
    L["day long top 10 (standalone)"] = dln
    idx = L["night capped (live blend)"].loc["2024-07":"2026-09"].index
    dla = dln.reindex(idx).fillna(0)
    for lab in ["capped", "uncapped"]:
        n_ = L["night capped (live blend)" if lab == "capped" else "night uncapped (study 94/71)"].reindex(idx)
        sh = L[f"day short {lab} etb (standalone, 1/10 each)"].reindex(idx).fillna(0)
        L[f"67 {lab}: night + 0.5 day long"] = n_ + 0.5 * dla
        L[f"68 {lab}: night + 0.25 day short"] = n_ + 0.25 * sh
        L[f"71 {lab}: night + 0.5 day long + 0.25 day short"] = n_ + 0.5 * dla + 0.25 * sh
    rows = []
    for k, x in L.items():
        x = x.loc["2024-07":"2026-09"].dropna()
        for m, v in x.groupby(x.index.to_period("M")):
            rows.append(dict(table="monthly", leg=k, period=str(m), nights=len(v), net_bp=1e4 * v.mean(),
                             sharpe=ann_stats(v)["sharpe"], compounded=float((1 + v).prod() - 1)))
        base = L["night capped (live blend)" if "uncapped" not in k else "night uncapped (study 94/71)"]
        for lab, a, b in PER195:
            xx = x.loc[a:b]
            r = dict(table="period", leg=k, period=lab, **stat(xx))
            if k[:2] in ("67", "68", "71"):
                bb = base.reindex(xx.index)
                r.update(night_alone_sharpe=ann_stats(bb)["sharpe"], night_alone_bp=1e4 * bb.mean(),
                         night_alone_maxdd=ann_stats(bb)["maxdd"], diff_vs_night_bp=1e4 * (xx - bb).mean(),
                         t_vs_night=ptt(xx, bb))
            if k.startswith("day "):
                nn = L["night capped (live blend)"].reindex(xx.index)
                r["corr_with_night"] = float(xx.corr(nn))
            rows.append(r)
    # ---- October, backtest style: day short of study 190's capped blend picks (entries 09-28..10-16)
    X = S166.cached_crosses().dropna(subset=["px"])
    X["ticker"] = X.ticker.str.replace(".", "-", regex=False)
    nx = S190.next_days(days)
    ecost = exec_cost_bps(P, "auction")[cols]
    oc = []
    for d, pk in K["oct"]["blend"].items():
        d1 = nx[d]
        e = [t for t in pk if t in K["etb"]]
        xo = X[(X.date == d1) & (X.kind == "open")].set_index("ticker").px.reindex(e)
        xc = X[(X.date == d1) & (X.kind == "close")].set_index("ticker").px.reindex(e)
        r = xc / xo - 1
        if d1 in P["c"].index:                               # panel fallback where a cross is missing
            r = r.fillna((P["c"].loc[d1].reindex(e) / P["o"].loc[d1].reindex(e) - 1))
        sane = (xo / P["rawc"][cols].loc[d].reindex(e) - 1).abs() < 0.5
        r = r.where(sane | xo.isna())
        cd = ecost.loc[min(d1, days[-1])].reindex(e) + 2.5
        ok = r.notna()
        net = float((-(r[ok]) - 2 * cd[ok] / 1e4).sum() / 10) if ok.any() else 0.0
        oc.append(dict(table="october_backtest", period=str(d1.date()), entry_day=str(d.date()), names=int(ok.sum()),
                       etb_of_picks=f"{len(e)}/{len(pk)}", net_bp=1e4 * net, legs_no_cross=int(len(e) - (xo.notna() & xc.notna()).sum()),
                       symbols=" ".join(e)))
    rows += oc
    # ---- October live record
    nm = pd.read_csv(os.path.join(LOG, "dayshort_names.csv"))
    nm["notional"] = nm.qty * nm.open
    if os.path.exists(LIVE195):
        lv = pd.read_parquet(LIVE195)
        lv["notional"] = lv.qty * lv.open
        nm = pd.concat([nm, lv], ignore_index=True)
    orders = live_dayshort_orders()
    vd = pd.read_csv(os.path.join(LOG, "virtual_days.csv")).set_index("day")
    ocb = pd.DataFrame(oc).set_index("period")
    lrows = []
    for day, g in nm.groupby("day"):
        g = g[g.open.notna() & g.close.notna()]
        r = orders.get(day, {})
        eq = r.get("equity", np.nan)
        ent = str(r.get("from_entry", "")).replace("entry_", "").replace(".json", "")
        pnl = g.auction_pnl.sum()
        lrows.append(dict(table="october_live", period=day, entry_day=ent, names=len(g), notional=g.notional.sum(),
                          equity=eq, auction_pnl=pnl, ret_on_equity_bp=1e4 * pnl / eq,
                          standalone_equiv_bp=4e4 * pnl / eq, short_gross_bp=1e4 * float((g.qty * (g.open - g.close)).sum() / g.notional.sum()),
                          night_live_book_bp=1e4 * vd.net.get(ent, np.nan),
                          backtest_style_standalone_bp=ocb.net_bp.get(day, np.nan)))
    rows += lrows
    lr = pd.DataFrame(lrows)
    lr = lr[lr.night_live_book_bp.notna()]
    night = lr.night_live_book_bp / 1e4
    comb = night + lr.ret_on_equity_bp / 1e4
    allnight = vd.net.loc["2026-09-29":]
    rows += [dict(table="october_summary", leg="live night book (all live nights 09-29..)", nights=len(allnight),
                  net_bp=1e4 * allnight.mean(), compounded=float((1 + allnight).prod() - 1)),
             dict(table="october_summary", leg="live night book (nights followed by a live day short)", nights=len(night),
                  net_bp=1e4 * night.mean(), compounded=float((1 + night).prod() - 1)),
             dict(table="october_summary", leg="live day short (return on equity, quarter size)", nights=len(lr),
                  net_bp=lr.ret_on_equity_bp.mean(), compounded=float((1 + lr.ret_on_equity_bp / 1e4).prod() - 1),
                  hit=float((lr.ret_on_equity_bp > 0).mean())),
             dict(table="october_summary", leg="live day short (standalone equivalent, x4)", nights=len(lr),
                  net_bp=lr.standalone_equiv_bp.mean()),
             dict(table="october_summary", leg="live night + live day short", nights=len(lr), net_bp=1e4 * comb.mean(),
                  compounded=float((1 + comb).prod() - 1), t_vs_night=ptt(comb, night))]
    ob = pd.DataFrame(oc)
    rows.append(dict(table="october_summary", leg="backtest-style day short, capped picks (standalone), days 09-29..10-19",
                     nights=len(ob), net_bp=ob.net_bp.mean(), hit=float((ob.net_bp > 0).mean()),
                     compounded=float((1 + ob.net_bp / 1e4).prod() - 1)))
    both = lr.dropna(subset=["backtest_style_standalone_bp"])
    rows.append(dict(table="october_summary", leg="live vs backtest-style day short, same days (standalone bp)",
                     nights=len(both), net_bp=both.standalone_equiv_bp.mean(),
                     night_alone_bp=both.backtest_style_standalone_bp.mean(),
                     corr_with_night=float(both.standalone_equiv_bp.corr(both.backtest_style_standalone_bp))))
    return pd.DataFrame(rows)


def main():
    import study99_cross_retrain as S99
    K = prep()
    P, cols, days, S = K["P"], K["cols"], K["days"], K["S"]
    X = S166.cached_crosses().dropna(subset=["px"])[["ticker", "date", "kind", "px"]]
    X["ticker"] = X.ticker.str.replace(".", "-", regex=False)
    ro, rc, err = S99.cross_ratios(P, cols, days, X)
    del X
    print("cross data errors (legs at panel price):", err, flush=True)
    ro, rc = ro.reindex(index=days, columns=cols), rc.reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rp = (o.shift(-1) / c - 1).reindex(index=S.index)
    Rx = (1 + Rp) * ro.fillna(1.0).shift(-1).reindex(index=S.index) / rc.fillna(1.0).reindex(index=S.index) - 1
    miss = (ro.shift(-1).isna() | rc.isna()).reindex(index=S.index)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=S.index)
    D194, nets = study194(K, Rx, C)
    fb = [dict(table="data", variant=f"backtest legs at panel price: {k}",
               nights=int((miss & (W > 0)).sum().sum()), names=float((miss & (W > 0)).sum().sum() / (W > 0).sum().sum()))
          for k, W in K["W"].items()]
    D194 = pd.concat([D194, pd.DataFrame(fb)], ignore_index=True)
    D194.to_csv(OUT194, index=False)
    D195 = study195(K, Rx, ro, rc, nets)
    D195.to_csv(OUT195, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    s = D194[D194.table == "summary"]
    print(s[["variant", "period", "nights", "sharpe", "net_bp", "maxdd", "compounded", "pick_beta", "universe_beta",
             "techhw_per_night", "overlap_blend", "diff_vs_blend_bp", "t_vs_blend"]].round(3).to_string(index=False))
    print(D194[D194.table == "monthly"].pivot_table(index="period", columns="variant", values="net_bp").round(1).to_string())
    print(D194[D194.table == "october_nightly"].pivot_table(index="period", columns="variant", values="net_bp").round(0).to_string())
    print(D194[D194.table == "data"].dropna(axis=1, how="all").to_string(index=False))
    p = D195[D195.table == "period"].dropna(axis=1, how="all")
    print(p.drop(columns="table").round(3).to_string(index=False))
    mo = D195[D195.table == "monthly"].pivot_table(index="period", columns="leg", values="net_bp")
    print(mo[[c_ for c_ in mo.columns if "capped" in c_ and "uncapped" not in c_ or c_.startswith("day long")]].round(1).to_string())
    for t in ["october_backtest", "october_live", "october_summary"]:
        print("\n==", t)
        print(D195[D195.table == t].dropna(axis=1, how="all").drop(columns="table").round(2).to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"fetch": fetch, "report": main}[what]()
    import resource
    print("peak memory GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
