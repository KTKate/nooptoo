"""Studies 173 and 175: correlation-aware selection and pause rules for the live overnight blend.

Context: the paper virtual book lost about 10.5% over 12 nights to 2026-10-15 with picks concentrated in tech
hardware (average pairwise 60-day overnight correlation 0.38, 0.57-0.64 on the worst nights). Study 172 found no
halving or tech-hardware cap rule that helps in both periods; study 92's hard correlation cap (skip a candidate whose
correlation with any chosen pick is above 0.6 / 0.7) lost.

173  Correlation-aware selection. Question: does choosing the 10 picks with their correlation in mind (instead of
     the plain top 10 with at most 3 per industry) improve the blend, or at least reduce its bad nights, without
     losing mean return? Rules fixed before looking at results:
     (a) greedy penalty: walk down the blend ranking; at each step take the candidate with the highest
         score_i - lambda * sd_t * mean correlation with the names already chosen, lambda in {0.1, 0.25, 0.5},
         sd_t = cross-sectional sd of the blend score that day (first pick = top score);
     (b) correlation budget: walk down the ranking and add a name only if the book's average pairwise correlation
         including it stays <= 0.25 / 0.30 (greedy approximation of "max sum of scores subject to average pairwise
         correlation <= c");
     (c) inverse-correlation weights on the live picks: weight proportional to 1 / (1 + mean correlation with the
         other picks), normalized to 100%.
     All variants keep the live industry cap (at most 3 per industry) and 10 names. The variant with the best
     2024-01..2025-06 Sharpe is the chosen one; its 2025-07..2026-09 numbers are the test.
175  Pause rules. Question: would a strategy-level stop (go to cash for a while after losses) have helped the base
     nightly series? Rules fixed before looking at results (the shadow series = the base series, always computed;
     the traded series = shadow while trading, 0 while paused; decisions for night t use nights up to t-1):
     (a) drawdown stop: pause when the traded book is 8% / 12% below its running peak; resume after 5 nights, after
         10 nights, or when the shadow equity makes a new 20-night high (higher than each of the 20 nights before),
         each resume rule reported separately; at resume the traded peak is reset to the current traded equity;
     (b) trend stop: pause when the trailing 20-night mean of the shadow series is below 0 / below -20 bp; resume
         when it is back above 0;
     (c) streak stop: pause 1 / 3 nights after 3 losing shadow nights in a row (a continuing streak extends the
         pause).
     Rules run continuously over 2024-01..2026-09 and then on the 12 live nights (logs/paper/virtual_days.csv net,
     appended after the backtest so the rule state carries over), and also on the live nights alone (fresh state).

Data: blend scores as study 69 lines 22-46 (results/study23_pred.parquet ensemble, results/study33_pred.parquet
p_jump - p_drop; S = (2 rank(ensemble) + rank(p_jump - p_drop)) / 3). Live capped blend: top 10 with at most 3 per
industry (paper_overnight.cap_by_industry rule, data/store/sectors.parquet, names without an industry their own
group), equal weight, as study135_137_expectations.build(). Night return at official crosses as study 94 /
study166_167_blend_parts.returns(): every cached data/local/auctions*.parquet plus data/local/auctions173.parquet
fetched here for the variants' new legs (study 94 rule: listing-venue 'O' / '6' print in widening windows, second
pass the largest 'O' / '6' print of any listing exchange); legs without a usable cross at the panel price. Costs
core.exec_cost_bps(P, "auction") + 2.5 bp per side. Correlation: Pearson correlation of overnight returns (open t /
close t-1) over the 60 trading days ending the day before the entry day (at least 40 common observations; pairs
with fewer are ignored in means and in the budget). Tech hardware as studies 90/92/172. Live nights:
logs/paper/virtual_days.csv (net) and logs/paper/virtual.csv (per-name returns at official prints), read only.

Data checks: the base reproduces study 135 (capped Sharpe 2.65 / 2.23 / 2.45, mean 38.2 bp) with all caches; the
series ends with the night of 2026-09-25, so 2026Q3 is 61 nights. 1,693 new legs fetched into auctions173 (97% found
in the first pass, 36 of 49 in the second); legs at the panel price 0.3-1.1% for every variant. Book correlation of
the base (window ending the day before): 0.29 / 0.31, 0.38 in 2026Q3; live nights 0.41. The live virtual book is
-12.5% compounded on its net column over 12 nights (-10.5% on its equity column; study 136 noted the difference).

Result 173 (2024-25H1 / 2025H2-26 / 2026Q3; base Sharpe 2.65 / 2.23 / -0.20, 39.3 / 36.8 / -3.1 bp, maxdd -17% /
-23%, nights <= -2% 11.3% / 14.1%): every selection rule lowered the mean. Penalty lambda 0.1: 2.44 / 2.19 / -0.69,
-4.1 / -2.7 / -7.0 bp (t -1.3 / -0.8 / -1.1); lambda 0.25: 1.82 / 1.68, -12.7 / -12.0 bp (t -2.5 / -2.2); lambda 0.5:
1.76 / 2.24 / 0.09, -14.2 / -5.8 / +4.2 bp (t -2.2 / -0.8 / 0.3), book correlation 0.14 / 0.13, maxdd -16% / -16%,
nights <= -2% 8.8% / 11.2%, tech hardware 0.74 / 1.30 names (base 0.84 / 1.77). Budget 0.25: 1.82 / 1.87, -14.8 /
-9.8 bp; budget 0.30: 1.98 / 2.18, -11.4 / -4.6 bp. Inverse-correlation weights: 2.63 / 2.17 / -0.33, -0.1 / -1.2 /
-2.0 bp (t -0.1 / -1.2 / -0.8). Chosen on 2024-25H1 (best Sharpe among the six, still below the base): inverse-
correlation weights, which on 2025H2-26 gave Sharpe 2.17 vs 2.23 and -1.2 bp a night. On the live nights the same
weights would have changed -108.6 bp a night to -107.9 bp (-12.4% vs -12.5% compounded). Worst nights barely move
(the -13.4% night of 2024-25H1 is kept by most variants). No correlation-aware rule is worth adopting.
Result 175 (Sharpe / total / maxdd / share paused, 2024-25H1 and 2025H2-26; base 2.65 / +290% / -17% and 2.23 /
+183% / -23%): drawdown 8% resume 5 nights 2.55 / 2.10, -4.0 / -3.5 bp (maxdd -21% / -17%, paused 9% / 13%);
drawdown 12% resume 5 nights 2.55 / 2.30, -3.1 / +0.3 bp (maxdd -20% / -17%, paused 5% / 8%); 10-night resumes -5 to
-8 bp; shadow 20-night-high resumes -10 to -12 bp (paused 25-39%, t -2.1 / -2.5 on 2024-26). Trailing mean below 0:
1.89 / 2.37, -14.3 / -2.5 bp (t -2.6 / -0.4, maxdd -29% / -16%); below -20 bp: 1.98 / 2.40, -12.1 / -1.6 bp. Three
losses pause 1: 2.62 / 2.12, -0.9 / -2.6 bp; pause 3: 2.52 / 2.13, -3.1 / -2.8 bp (2026Q3 +6.2 bp). Every rule lowers
the 2024-26 mean (by 1.5 to 11 bp a night); some cut the 2025H2-26 drawdown (-17% instead of -23%) but none improves
both periods' Sharpe. Live 12 nights (compounded net -12.5%): the 8% drawdown rules and both 3-loss rules would have
skipped 10-14 and 10-15 (-6.2%), the 12% drawdown rules 10-15 only (-10.9%); with the rule state carried over from
the backtest, the shadow-20-high and trailing-mean rules would have been in cash all 12 nights (0%), because 2026Q3
ended in a losing stretch; started fresh on 2026-09-29 the trailing-mean rules cannot trigger in 12 nights.

Caveats: rules and thresholds were fixed before the run, but the live nights are the reason the study was asked, so
the live comparison is not a test; 12 live nights is far too few to judge; the pause rules' backtest is one path of
685 nights, so their differences are mostly within noise; sectors.parquet is today's classification for all dates.

Output: results/study173_corr_select.csv (table = variants / choice / live_inverse_corr; one row per variant x
period: n, sharpe, net_bp, maxdd, worst_bp, share_le_2pct, book_corr, techhw, names, overlap_base, diff_bp,
t_vs_base, legs_at_panel), results/study175_pause.csv (table = backtest / live; per rule x period: sharpe, total,
maxdd, share_paused, net_bp, diff_bp, t_vs_base; live rows with the rule's compounded return over the 12 nights
next to the actual).

    python src/study173_175_selection_pause.py fetch    # crosses for the new picks (resumable, ~60 Alpaca requests a minute)
    python src/study173_175_selection_pause.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402
import study94_cross_rescore as S94   # noqa: E402  (sets A.RL at import; reset below)
import study166_167_blend_parts as S166   # noqa: E402

A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(DATA, "local")
PAPER = os.path.join(ROOT, "logs", "paper")
AUC173 = os.path.join(LOC, "auctions173.parquet")
OUT173 = os.path.join(RES, "study173_corr_select.csv")
OUT175 = os.path.join(RES, "study175_pause.csv")
PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
PERQ = PER + [("2026Q3", "2026-07", "2026-09")]
LAMBDAS = [0.1, 0.25, 0.5]
BUDGETS = [0.25, 0.30]
WIN, MINP, DEPTH = 60, 40, 300
CACHE = os.environ.get("STUDY173_CACHE")   # optional pickle of picks (development only)


# ------------------------------------------------------------------ scores, correlations, picks
def blend(cols, last_day):
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < last_day]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    return S.loc["2024-01":"2026-09"].astype("float64")


def corr_matrix(onv, i, js):
    """Pairwise correlation of overnight returns over rows i-WIN..i-1 for columns js (NaN where < MINP common)."""
    X = onv[i - WIN:i][:, js].astype("float64")
    return np.array(pd.DataFrame(X).corr(min_periods=MINP).values, dtype="float64")


def avg_pair(Cm):
    if Cm.shape[0] < 2:
        return np.nan
    iu = np.triu_indices(Cm.shape[0], 1)
    return np.nanmean(Cm[iu]) if np.isfinite(Cm[iu]).any() else np.nan


def make_picks(P, cols, days, S, grp):
    """Weight frames for the base and every 173 variant, plus the book correlation per night."""
    if CACHE and os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    on = (P["o"][cols] / P["c"][cols].shift(1) - 1).values.astype("float32")
    g = grp.reindex(cols).values
    pos = days.get_indexer(S.index)
    names = ["base"] + [f"a_penalty_lambda{l:g}" for l in LAMBDAS] + [f"b_budget_{c:.2f}" for c in BUDGETS] + \
        ["c_inverse_corr_weights"]
    W = {k: np.zeros(S.shape) for k in names}
    bc = {k: np.full(len(S), np.nan) for k in names}
    Sv = S.values
    t0 = time.time()
    for r in range(len(S)):
        s = Sv[r]
        idx = np.where(np.isfinite(s))[0]
        order = idx[np.argsort(-s[idx], kind="stable")][:DEPTH]
        sd = np.nanstd(s)
        Cm = corr_matrix(on, pos[r], order)          # candidates in ranking order
        sc = s[order]
        gg = g[order]

        def capok(cnt, q):
            return cnt.get(gg[q], 0) < 3

        # base
        cnt, ch = {}, []
        for q in range(len(order)):
            if capok(cnt, q):
                cnt[gg[q]] = cnt.get(gg[q], 0) + 1
                ch.append(q)
                if len(ch) == 10:
                    break
        base = ch
        # (a) penalty
        sel = {"base": base}
        for lam in LAMBDAS:
            cnt, ch = {}, []
            avail = np.ones(len(order), bool)
            while len(ch) < 10:
                ok = avail & np.array([capok(cnt, q) for q in range(len(order))])
                if not ok.any():
                    break
                if ch:
                    pm = np.nanmean(np.where(np.isfinite(Cm[:, ch]), Cm[:, ch], np.nan), axis=1) \
                        if np.isfinite(Cm[:, ch]).any() else np.zeros(len(order))
                    pm = np.where(np.isfinite(pm), pm, 0.0)
                else:
                    pm = np.zeros(len(order))
                adj = np.where(ok, sc - lam * sd * pm, -np.inf)
                q = int(np.argmax(adj))           # first max = higher ranked on ties
                ch.append(q)
                avail[q] = False
                cnt[gg[q]] = cnt.get(gg[q], 0) + 1
            sel[f"a_penalty_lambda{lam:g}"] = ch
        # (b) budget
        for cap in BUDGETS:
            cnt, ch = {}, []
            for q in range(len(order)):
                if not capok(cnt, q):
                    continue
                if ch:
                    m = avg_pair(Cm[np.ix_(ch + [q], ch + [q])])
                    if np.isfinite(m) and m > cap:
                        continue
                ch.append(q)
                cnt[gg[q]] = cnt.get(gg[q], 0) + 1
                if len(ch) == 10:
                    break
            sel[f"b_budget_{cap:.2f}"] = ch
        for k, ch in sel.items():
            if ch:
                W[k][r, order[ch]] = 1.0 / len(ch)
                bc[k][r] = avg_pair(Cm[np.ix_(ch, ch)])
        # (c) inverse-correlation weights on the base picks
        if base:
            Cb = Cm[np.ix_(base, base)].copy()
            np.fill_diagonal(Cb, np.nan)
            mc = np.array([np.nanmean(x) if np.isfinite(x).any() else 0.0 for x in Cb])
            w = 1.0 / (1.0 + mc)
            W["c_inverse_corr_weights"][r, order[base]] = w / w.sum()
            bc["c_inverse_corr_weights"][r] = bc["base"][r]
        if r % 100 == 0:
            print("picks day", r, "/", len(S), f"{time.time() - t0:.0f}s", flush=True)
    out = dict(W={k: pd.DataFrame(v, index=S.index, columns=cols) for k, v in W.items()},
               bc={k: pd.Series(v, index=S.index) for k, v in bc.items()})
    if CACHE:
        pd.to_pickle(out, CACHE)
    return out


def setup():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    S = blend(cols, days[-1])
    grp, th = S166.industry_map(cols)
    K = make_picks(P, cols, days, S, grp)
    return P, cols, days, S, grp, th, K


# ------------------------------------------------------------------ fetch
def fetch():
    P, cols, days, S, grp, th, K = setup()
    del P
    need = pd.concat([S166.legs(W, days) for W in K["W"].values()], ignore_index=True)
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    if os.path.exists(AUC173):
        have = pd.read_parquet(AUC173)
        have["date"] = pd.to_datetime(have.date)
    else:
        have = pd.DataFrame(columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)
    S94.AUC_FN = AUC173
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
    if os.path.exists(AUC173):
        S94.refetch_missing()


# ------------------------------------------------------------------ 173
def tstat(d):
    d = pd.Series(d).dropna()
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan


def study173(P, cols, days, th, K):
    Wb = K["W"]["base"]
    Rx, miss, C = S166.returns(P, cols, days, Wb.index)
    nets, rows = {}, []
    thv = th.reindex(cols).fillna(False).values
    for k, W in K["W"].items():
        nets[k] = bt.run(W, Rx, C).net.where(W.sum(axis=1) > 0).dropna()
    base = nets["base"]
    print("base Sharpe", {p: round(ann_stats(base.loc[a:b])["sharpe"], 2) for p, a, b in PER},
          "mean bp", round(1e4 * base.mean(), 1), "(study 135: 2.65 / 2.23 / 2.45, 38.2 bp)", flush=True)
    Hb = Wb > 0
    for k, W in K["W"].items():
        H = W > 0
        nth = (H & thv).sum(axis=1)
        nn = H.sum(axis=1)
        ov = (H & Hb).sum(axis=1) / Hb.sum(axis=1).replace(0, np.nan)
        ms = (miss & H).sum(axis=1) / nn.replace(0, np.nan)
        wmax = W.max(axis=1)
        for p, a, b in PERQ:
            x = nets[k].loc[a:b]
            s = ann_stats(x)
            d = x - base.reindex(x.index)
            rows.append(dict(table="variants", variant=k, period=p, n=s["n"], sharpe=s["sharpe"],
                             net_bp=1e4 * x.mean(), maxdd=s["maxdd"], worst_bp=1e4 * x.min(),
                             share_le_2pct=(x <= -0.02).mean(), book_corr=K["bc"][k].loc[a:b].mean(),
                             techhw=nth.loc[a:b].mean(), names=nn.loc[a:b].mean(), overlap_base=ov.loc[a:b].mean(),
                             max_weight=wmax.loc[a:b].mean(), diff_bp=1e4 * d.mean(), t_vs_base=tstat(d),
                             legs_at_panel=ms.loc[a:b].mean()))
    df = pd.DataFrame(rows)
    first = df[(df.period == "2024-25H1") & (df.variant != "base")].sort_values("sharpe", ascending=False)
    best = first.variant.iloc[0]
    for p in ["2024-25H1", "2025H2-26", "2026Q3"]:
        r = df[(df.variant == best) & (df.period == p)].iloc[0].to_dict()
        r.update(table="choice", variant=f"chosen on 2024-25H1: {best}")
        rows.append(r)
    rows += live_inverse_corr(P, cols, days)
    return pd.DataFrame(rows), base


def live_inverse_corr(P, cols, days):
    """(c) on the live nights: inverse-correlation weights on the actual picks with their returns at the prints."""
    v = pd.read_csv(os.path.join(PAPER, "virtual.csv"), parse_dates=["day"])
    vd = pd.read_csv(os.path.join(PAPER, "virtual_days.csv"), parse_dates=["day"]).set_index("day")
    on = (P["o"][cols] / P["c"][cols].shift(1) - 1).values.astype("float32")
    ci = {t: j for j, t in enumerate(cols)}
    out = {}
    bcs = {}
    for d, g in v.groupby("day"):
        g = g.copy()
        g["net_name"] = g.ret - g.cost
        js = [ci.get(t.replace(".", "-")) for t in g.symbol]
        if d not in days or any(j is None for j in js):
            out[d] = np.nan
            continue
        Cm = corr_matrix(on, days.get_loc(d), np.array(js))
        bcs[d] = avg_pair(Cm)
        np.fill_diagonal(Cm, np.nan)
        mc = np.array([np.nanmean(x) if np.isfinite(x).any() else 0.0 for x in Cm])
        w = 1.0 / (1.0 + mc)
        out[d] = float((w / w.sum() * g.net_name.values).sum())
    inv = pd.Series(out).sort_index()
    act = vd.net.reindex(inv.index)
    eq = act.reindex(inv.index)
    return [dict(table="live_inverse_corr", variant="c_inverse_corr_weights", period="live 2026-09-29..10-15",
                 n=int(inv.notna().sum()), net_bp=1e4 * inv.mean(), total=(1 + inv.fillna(0)).prod() - 1,
                 base_net_bp=1e4 * eq.mean(), base_total=(1 + eq).prod() - 1, diff_bp=1e4 * (inv - act).mean(),
                 book_corr=pd.Series(bcs).mean(), worst_bp=1e4 * inv.min(), base_worst_bp=1e4 * act.min())]


# ------------------------------------------------------------------ 175
def run_rule(r, kind, par):
    """r: shadow nightly returns (Series). Returns (traded series, paused flags)."""
    x = r.values
    n = len(x)
    paused = np.zeros(n, bool)
    tr = np.zeros(n)
    sh_eq = np.cumprod(1 + x)
    if kind == "dd":
        thr, resume = par
        eq, peak, left, on_pause = 1.0, 1.0, 0, False
        for t in range(n):
            if on_pause:
                # resume check uses information through night t-1
                if resume == "high20":
                    if t - 1 >= 20 and sh_eq[t - 1] > sh_eq[t - 21:t - 1].max():
                        on_pause = False
                else:
                    if left == 0:
                        on_pause = False
                if not on_pause:
                    peak = eq
            elif eq / peak - 1 <= -thr:
                on_pause, left = True, (resume if resume != "high20" else 0)
            if on_pause:
                paused[t] = True
                if resume != "high20":
                    left -= 1
            else:
                tr[t] = x[t]
                eq *= 1 + x[t]
                peak = max(peak, eq)
    elif kind == "mean20":
        thr = par
        m = pd.Series(x).rolling(20).mean().shift(1).values   # mean of nights t-20..t-1
        on_pause = False
        for t in range(n):
            if on_pause and np.isfinite(m[t]) and m[t] > 0:
                on_pause = False
            elif not on_pause and np.isfinite(m[t]) and m[t] < thr:
                on_pause = True
            paused[t] = on_pause
            tr[t] = 0.0 if on_pause else x[t]
    elif kind == "streak":
        k = par
        loss = x < 0
        for t in range(n):
            for j in range(1, k + 1):          # a 3-night losing run that ended at night t-j
                e = t - j
                if e >= 2 and loss[e] and loss[e - 1] and loss[e - 2]:
                    paused[t] = True
                    break
            tr[t] = 0.0 if paused[t] else x[t]
    return pd.Series(tr, index=r.index), pd.Series(paused, index=r.index)


RULES = ([(f"a_dd{int(th * 100)}_resume_{'%dn' % rs if rs != 'high20' else 'shadow20high'}", "dd", (th, rs))
          for th in [0.08, 0.12] for rs in [5, 10, "high20"]] +
         [("b_mean20_below_0", "mean20", 0.0), ("b_mean20_below_-20bp", "mean20", -0.002)] +
         [("c_3losses_pause1", "streak", 1), ("c_3losses_pause3", "streak", 3)])


def study175(base):
    vd = pd.read_csv(os.path.join(PAPER, "virtual_days.csv"), parse_dates=["day"]).set_index("day").net
    full = pd.concat([base, vd[vd.index > base.index[-1]]])
    nl = len(vd)
    rows = []
    for name, kind, par in [("base (no pause)", None, None)] + RULES:
        if kind is None:
            tr, pz = full, pd.Series(False, index=full.index)
            trl, pzl = vd, pd.Series(False, index=vd.index)
        else:
            tr, pz = run_rule(full, kind, par)
            trl, pzl = run_rule(vd, kind, par)
        for p, a, b in PERQ:
            x, xb, pp = tr.loc[base.index].loc[a:b], base.loc[a:b], pz.loc[base.index].loc[a:b]
            s = ann_stats(x)
            rows.append(dict(table="backtest", rule=name, period=p, n=s["n"], sharpe=s["sharpe"], total=s["total"],
                             maxdd=s["maxdd"], share_paused=pp.mean(), nights_paused=int(pp.sum()),
                             net_bp=1e4 * x.mean(), diff_bp=1e4 * (x - xb).mean(), t_vs_base=tstat(x - xb)))
        xl = tr.loc[vd.index]
        rows.append(dict(table="live", rule=name, period="live 2026-09-29..10-15 (state carried from backtest)", n=nl,
                         total=(1 + xl).prod() - 1, base_total=(1 + vd).prod() - 1,
                         change_pts=((1 + xl).prod() - (1 + vd).prod()), nights_paused=int(pz.loc[vd.index].sum()),
                         share_paused=pz.loc[vd.index].mean(), net_bp=1e4 * xl.mean(),
                         maxdd=ann_stats(xl)["maxdd"] if xl.std() > 0 else np.nan,
                         paused_days=" ".join(str(d.date()) for d in pz.loc[vd.index][pz.loc[vd.index]].index)))
        rows.append(dict(table="live", rule=name, period="live 2026-09-29..10-15 (fresh state)", n=nl,
                         total=(1 + trl).prod() - 1, base_total=(1 + vd).prod() - 1,
                         change_pts=((1 + trl).prod() - (1 + vd).prod()), nights_paused=int(pzl.sum()),
                         share_paused=pzl.mean(), net_bp=1e4 * trl.mean(),
                         maxdd=ann_stats(trl)["maxdd"] if trl.std() > 0 else np.nan,
                         paused_days=" ".join(str(d.date()) for d in pzl[pzl].index)))
    return pd.DataFrame(rows)


def show(df, cols_):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df[[c for c in cols_ if c in df]].round(4).to_string(index=False))


def main():
    P, cols, days, S, grp, th, K = setup()
    df173, base = study173(P, cols, days, th, K)
    del P
    df173.to_csv(OUT173, index=False)
    show(df173, ["table", "variant", "period", "n", "sharpe", "net_bp", "maxdd", "worst_bp", "share_le_2pct",
                 "book_corr", "techhw", "names", "overlap_base", "diff_bp", "t_vs_base", "legs_at_panel", "total",
                 "base_total"])
    df175 = study175(base)
    df175.to_csv(OUT175, index=False)
    show(df175, ["table", "rule", "period", "n", "sharpe", "total", "base_total", "change_pts", "maxdd",
                 "share_paused", "nights_paused", "net_bp", "diff_bp", "t_vs_base", "paused_days"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
