"""Studies 181, 182 and 183: is the weakness of the overnight blend since July 2026 specific to the models or market-wide?

Background: the live blend lost about 12.5% over 12 paper nights 2026-09-29..10-15; the backtest at official crosses
(study135_137_expectations.build(), study 94 method) was -3 bp a night 2026-07..09 against +42 bp before. Study 178
found about half of the live gap is a market-wide October reversal of high-beta / tech hardware stocks overnight; the
rest is unexplained, and a similar unexplained shortfall shows in 2026Q3.

181  Input drift. Question: did the model inputs (data/ml_frame.parquet, every non-y_ column) change distribution in
     2026Q3 against 2024-25, and do the drifting inputs matter to the models? Did any input's missing share jump
     (study 158 found days_to_e missing in Aug-Sep 2026)?
182  Families. Question: did every score of the blend (five LightGBM members, their ensemble, p_jump, p_drop, p_jump -
     p_drop, the live blend) weaken from July 2026, or only some?
183  Simple rules. Question: did simple overnight rules with no model (top 10 by 60-day beta, by 20-day volatility,
     10 worst and 10 best day returns among larger names) also weaken in July-October 2026?

Data: data/ml_frame.parquet (inputs as of the close of day t; ends 2026-09-25). core.load_panel (split- and
dividend-adjusted Yahoo o/c, rawc, dv; ends 2026-10-15). Member and ensemble predictions results/study23_pred.parquet,
p_jump / p_drop results/study33_pred.parquet (out of sample, quarterly refits, 2024-01-02..2026-09-25). Gain
importance (normalized to 100 per model) from results/study150_pick_traits.csv (table importance; models night =
pooled member, ens_pooled_beh, ens_pooled_com, jump_jump, jump_drop; 2026Q1 and 2026Q4 fits averaged) and the
days_to_e gain shares of the 2026Q3 fits from results/study158_earnings_gap.csv. Industries data/store/sectors.parquet.
Official crosses: every cached data/local/auctions*.parquet (studies 88-173, read via study 166's cached_crosses) plus
data/local/auctions182.parquet fetched here for the legs of the new picks of 182 and 183 with study 94's rule
(listing-venue 'O' / '6' print in widening windows; second pass: the largest 'O' / '6' print of any listing exchange),
about 60 Alpaca requests a minute. The night 2026-10-15 -> 10-16 (not in the panel) for study 183 uses the Alpaca SIP
daily bars cached by study 176 (data/local/study176_bars.parquet, liquid names on 10-15) and crosses where found.
Liquid universe (all studies): stocks (no ETFs) with raw close > $5 and 20-day median dollar volume > $5M on day t
(as study 174).

Design 181: for each input, reference = all rows of 2024-01..2025-12 in the group; 10 bins at the reference deciles
(fewer when the input has ties, e.g. dow, tom, days_to_e; values outside the reference range go to the end bins);
PSI = sum (a - e) ln(a / e) over bins of the non-missing values (shares floored at 1e-4), per month 2024-01..2026-09
and pooled for 2026Q1, 2026Q2 and 2026Q3 (July-September, the ranking period). Missing share per month separately;
a jump is flagged when a month's missing share differs from the reference share by more than 2 points and by more
than half of it. Groups: the liquid universe, and the blend's top 50 candidates each day (uncapped top 50 of
S = (2 rank(ensemble) + rank(p_jump - p_drop)) / 3 on the live universe; names with all five members and p_jump).
Rule of thumb: PSI < 0.1 small, 0.1-0.25 moderate, > 0.25 large. Months inside the reference have PSI near the
sampling floor, so 2026H1 is the fair out-of-reference baseline for 2026Q3. Market-level inputs (spy*, vix*, iwm_spy5,
dow, tom) take one value per day, so their monthly PSI reflects the market regime of a month (about 21 values)
rather than a change in the stock cross-section; they are marked kind = market. Importance: ens_gain = mean gain share
of night, ens_pooled_beh and ens_pooled_com; jump_gain = mean of jump_jump and jump_drop; blend_gain = 2/3 ens_gain +
1/3 jump_gain (the blend weights); weighted = blend_gain x PSI 2026Q3 (top-50 group), as a rough 'drift that
matters' score. The per-group members (per_beh, per_com) and the group-feature extras (g_*) are not in ml_frame.
Design 182: each score's own top 10 per day with at most 3 per industry (study 166's capped_topk, the rule of
paper_overnight.cap_by_industry), all on the live blend's universe, equal weight, buy at the closing cross of t, sell
at the opening cross of t+1 (study 166 returns(): panel night return x open cross / panel open on t+1 / (close cross
/ panel close on t); legs without a usable cross at the panel price), costs core.exec_cost_bps(P, "auction") + 2.5
bp per side, 2024-01..2026-09-25 (last night 09-25 -> 09-28). Monthly mean net bp a night; periods 2024-01..2026-06
(base), 2026-07..09 (recent) with the difference and a Welch t; correlation of each score's nightly net with the
live blend's.
Design 183: rules with no model on the liquid universe, selected on data known at the close of t, no industry cap:
beta60 = top 10 by 60-day beta of daily close returns to SPY (study 90 roll_beta), vol20 = top 10 by 20-day sd of
daily close returns, worst1 / best1 = the 10 lowest / highest close-to-close returns of day t among raw close > $10
and 20-day median dollar volume > $20M. Buy at the close of t, sell at the open of t+1, equal weight, same costs.
Two pricings: panel (Yahoo close and open; the open is the first trade, which study 94 found is about +1-2 bp above
the opening cross for liquid names and +5-11 bp for volatile, gapped names, so panel returns are flattered) and
cross (official crosses fetched for every pick, panel price where no cross). Monthly net 2024-01..2026-10 (October to
the 10-15 entry night); periods base 2024-01..2026-06, 2026Q3, 2026-10, and 2026-07..10, with Welch t against base and
each period's mean placed as a percentile among all rolling windows of the same length in 2024-01..2026-10.

Output: results/study181_drift.csv (table = monthly: month, group, feature, psi, miss, n; table = rank: feature,
group, kind, psi_2026q1/q2/q3, psi_ref_months (mean monthly PSI in 2024-25), miss_ref, miss_q3, miss_flag_months,
gains, weighted, rank_q3), results/study182_families.csv (table = monthly / period: score, family, month or period,
nights, net_bp, sharpe, diff_bp, t_vs_base, corr_live, fallback), results/study183_simple_rules.csv (table =
monthly / period: rule, pricing, month or period, nights, net_bp, sharpe, t_vs_base, pct_window, corr_blend).

    python src/study181_183_decay_checks.py fetch   # crosses for the picks of 182 and 183 (resumable)
    python src/study181_183_decay_checks.py         # all three studies (or: 181 / 182 / 183)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ALPACA_PER_MIN", "60")
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402
import study166_167_blend_parts as S166   # noqa: E402  (sets A.RL to ALPACA_PER_MIN)
import study94_cross_rescore as S94   # noqa: E402
from study174_178_recent_edge import roll_beta   # noqa: E402

A = S166.A
A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "60")))
LOC = os.path.join(DATA, "local")
AUC182 = os.path.join(LOC, "auctions182.parquet")
BARS176 = os.path.join(LOC, "study176_bars.parquet")
ML = os.path.join(DATA, "ml_frame.parquet")
OUT181 = os.path.join(RES, "study181_drift.csv")
OUT182 = os.path.join(RES, "study182_families.csv")
OUT183 = os.path.join(RES, "study183_simple_rules.csv")
MEMBERS = S166.MEMBERS
MARKET = {"spy1", "spy5", "spy20", "vix", "vix_chg5", "vix_term", "iwm_spy5", "dow", "tom"}
EXTRA = pd.Timestamp("2026-10-15")       # entry day of the night after the panel's last day
EXTRA_EXIT = pd.Timestamp("2026-10-16")
FAMILY = {"pooled": "ensemble member", "pooled_beh": "ensemble member", "per_beh": "ensemble member",
          "pooled_com": "ensemble member", "per_com": "ensemble member", "ensemble": "ensemble",
          "p_jump": "jump model", "p_drop": "jump model", "jump_minus_drop": "jump model", "live": "blend"}


def liquid(P, cols):
    adv = P["dv"][cols].rolling(20, min_periods=15).median()
    return (P["rawc"][cols] > 5) & (adv > 5e6), adv


def pct_rank(dist, v):
    dist = np.asarray(dist)
    dist = dist[~np.isnan(dist)]
    return (dist < v).mean() + 0.5 * (dist == v).mean()


def welch_t(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def sh(x):
    x = pd.Series(x).dropna()
    return ann_stats(x)["sharpe"] if len(x) >= 5 else np.nan


# ------------------------------------------------------------------ scores (182) and blend top 50 (181)
def family_scores(cols, last_day):
    """Score name -> wide frame on 2024-01..2026-09, all on the live blend's universe (all members and p_jump)."""
    m = pd.read_parquet(f"{RES}/study23_pred.parquet")
    m = m[m.index.get_level_values(0) < last_day]
    W = {k: m[k].unstack().reindex(columns=cols).astype("float32") for k in MEMBERS + ["ensemble"]}
    del m
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    idx = W["ensemble"].index
    W["p_jump"] = pred.p_jump.unstack().reindex(index=idx, columns=cols).astype("float32")
    W["p_drop"] = pred.p_drop.unstack().reindex(index=idx, columns=cols).astype("float32")
    del pred
    W["jump_minus_drop"] = W["p_jump"] - W["p_drop"]
    ok = W["ensemble"].notna() & W["p_jump"].notna()
    for k in MEMBERS:
        ok &= W[k].notna()
    sl = slice("2024-01", "2026-09")
    out = {k: v.where(ok).loc[sl] for k, v in W.items()}
    re_ = out["ensemble"].rank(axis=1, pct=True)
    rj = out["jump_minus_drop"].rank(axis=1, pct=True)
    out["live"] = (2 * re_ + rj) / 3
    return out


def picks182(P, cols, days):
    grp, _ = S166.industry_map(cols)
    S = family_scores(cols, days[-1])
    Wd = {}
    for k, s in S.items():
        Wd[k] = S166.capped_topk(s, grp)
        if k == "live":
            S166.check_cap(s, Wd[k])
    return Wd


# ------------------------------------------------------------------ simple rules (183)
def rule_picks(P, cols):
    c = P["c"][cols]
    r1 = (c / c.shift(1) - 1).astype("float32")
    spy = P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1
    liq, adv = liquid(P, cols)
    big = liq & (P["rawc"][cols] > 10) & (adv > 2e7)
    vol20 = r1.rolling(20, min_periods=15).std().replace(0, np.nan)
    beta60 = roll_beta(r1, spy)
    sl = slice("2024-01", None)
    sc = {"beta60": beta60.where(liq), "vol20": vol20.where(liq), "worst1": -r1.where(big), "best1": r1.where(big)}
    Wd = {}
    for k, s in sc.items():
        s = s.loc[sl]
        rk = s.rank(axis=1, ascending=False, method="first")
        H = rk <= 10
        Wd[k] = H.div(H.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return Wd


# ------------------------------------------------------------------ fetch
def fetch():
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    Wall = list(picks182(P, cols, days).values()) + list(rule_picks(P, cols).values())
    need = pd.concat([S166.legs(W, days) for W in Wall], ignore_index=True)
    # the extra night 10-15 -> 10-16 of the rules (outside the panel calendar)
    for W in rule_picks(P, cols).values():
        if W.index[-1] == EXTRA:
            t = W.columns[W.iloc[-1] > 0]
            need = pd.concat([need, pd.DataFrame({"ticker": t, "date": EXTRA, "kind": "close"}),
                              pd.DataFrame({"ticker": t, "date": EXTRA_EXIT, "kind": "open"})], ignore_index=True)
    del P, Wall
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    if os.path.exists(AUC182):
        have = pd.read_parquet(AUC182)
        have["date"] = pd.to_datetime(have.date)
    else:
        have = pd.DataFrame(columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    mm = todo.merge(have[["ticker", "date", "kind"]], how="left", indicator=True)
    todo = mm[mm._merge == "left_only"].drop(columns="_merge")
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), "est. requests", sum(-(-len(j[2]) // 40) for j in jobs), flush=True)
    if "dry" in sys.argv:
        return
    S94.AUC_FN = AUC182
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
    if os.path.exists(AUC182):
        S94.refetch_missing()


# ------------------------------------------------------------------ 181
def psi_bins(ref, vals):
    """Edges from the reference deciles (unique); returns bin shares of ref and of each array in vals."""
    q = np.unique(np.nanquantile(ref, np.linspace(0.1, 0.9, 9)))
    def shares(x):
        b = np.searchsorted(q, x, side="right")
        return np.bincount(b, minlength=len(q) + 1) / max(len(x), 1)
    return shares(ref), [shares(v) for v in vals]


def psi(e, a, eps=1e-4):
    e, a = np.maximum(e, eps), np.maximum(a, eps)
    return float(np.sum((a - e) * np.log(a / e)))


def study181(P, cols):
    days = P["c"].index
    f = pq.ParquetFile(ML)
    feats = [n for n in f.schema_arrow.names if n not in ("date", "ticker") and not n.startswith("y_")]
    X = pd.read_parquet(ML, columns=feats, filters=[("date", ">=", pd.Timestamp("2024-01-01"))])
    print("ml_frame rows", len(X), "inputs", len(feats), flush=True)
    liq, _ = liquid(P, cols)
    d = X.index.get_level_values(0)
    t = X.index.get_level_values(1)
    L = liq.stack()
    L = L[L]
    in_liq = pd.MultiIndex.from_arrays([d, t]).isin(L.index)
    S = family_scores(cols, days[-1])["live"]
    rk = S.rank(axis=1, ascending=False, method="first")
    T = (rk <= 50).stack()
    T = T[T]
    in_top = pd.MultiIndex.from_arrays([d, t]).isin(T.index)
    del S, rk, L
    month = d.to_period("M").astype(str).values
    groups = {"liquid": in_liq, "top50": in_top}
    rows, rk_rows = [], []
    months = sorted(set(month))
    qmask = {"2026q1": (d >= "2026-01-01") & (d < "2026-04-01"), "2026q2": (d >= "2026-04-01") & (d < "2026-07-01"),
             "2026q3": (d >= "2026-07-01") & (d < "2026-10-01")}
    refm = d < "2026-01-01"
    for g, gm in groups.items():
        print("group", g, "rows", int(gm.sum()), flush=True)
        mon = month[gm]
        for k in feats:
            v = X[k].values[gm]
            ref = v[refm[gm] & ~np.isnan(v)]
            masks = [mon == mm for mm in months] + [qmask[q][gm] for q in qmask]
            sub = [v[mk] for mk in masks]
            e, sh_ = psi_bins(ref, [s[~np.isnan(s)] for s in sub])
            miss = [float(np.isnan(s).mean()) if len(s) else np.nan for s in sub]
            for i, mm in enumerate(months):
                rows.append(dict(table="monthly", month=mm, group=g, feature=k, psi=psi(e, sh_[i]), miss=miss[i],
                                 n=len(sub[i])))
            nm = len(months)
            mref = float(np.isnan(v[refm[gm]]).mean())
            mon_miss = pd.Series(miss[:nm], index=months)
            flag = mon_miss[(mon_miss - mref).abs() > np.maximum(0.02, 0.5 * mref)]
            rk_rows.append(dict(table="rank", group=g, feature=k, kind="market" if k in MARKET else "stock",
                                psi_2026q1=psi(e, sh_[nm]), psi_2026q2=psi(e, sh_[nm + 1]),
                                psi_2026q3=psi(e, sh_[nm + 2]),
                                psi_ref_months=float(np.mean([psi(e, sh_[i]) for i, mm in enumerate(months)
                                                              if mm < "2026-01"])),
                                psi_2026h1_months=float(np.mean([psi(e, sh_[i]) for i, mm in enumerate(months)
                                                                 if "2026-01" <= mm < "2026-07"])),
                                miss_ref=mref, miss_2026h1=float(mon_miss.loc["2026-01":"2026-06"].mean()),
                                miss_q3=miss[nm + 2], miss_flag_months=" ".join(f"{a}:{b:.3f}" for a, b in flag.items())))
    del X
    R = pd.DataFrame(rk_rows)
    imp = pd.read_csv(f"{RES}/study150_pick_traits.csv")
    imp = imp[(imp.table == "importance") & imp.period.isin(["2026Q1", "2026Q4"])]
    G = imp.pivot_table(index="item", columns="model", values="mean", aggfunc="mean")
    gains = pd.DataFrame({"gain_night": G["night"], "gain_ens": G[["night", "ens_pooled_beh", "ens_pooled_com"]].mean(axis=1),
                          "gain_jump": G[["jump_jump", "jump_drop"]].mean(axis=1)})
    gains["gain_blend"] = 2 / 3 * gains.gain_ens + 1 / 3 * gains.gain_jump
    R = R.merge(gains, left_on="feature", right_index=True, how="left")
    R["weighted"] = R.gain_blend / 100 * R.psi_2026q3
    R["psi_q3_minus_h1"] = R.psi_2026q3 - R[["psi_2026q1", "psi_2026q2"]].mean(axis=1)
    R["rank_q3"] = R.groupby("group").psi_2026q3.rank(ascending=False)
    s158 = pd.read_csv(f"{RES}/study158_earnings_gap.csv")
    s158 = s158[(s158.table == "importance") & (s158.period == "2026Q3")][["variant", "gain_share_days_to_e"]]
    print("study 158, days_to_e gain share in the 2026Q3 fits:\n", s158.round(4).to_string(index=False), flush=True)
    df = pd.concat([pd.DataFrame(rows), R], ignore_index=True)
    df.to_csv(OUT181, index=False)
    pd.set_option("display.width", 250)
    show = ["feature", "kind", "psi_ref_months", "psi_2026h1_months", "psi_2026q1", "psi_2026q2", "psi_2026q3",
            "miss_ref", "miss_q3", "gain_ens", "gain_jump", "weighted"]
    for g in groups:
        x = R[R.group == g].sort_values("psi_2026q3", ascending=False)
        print(f"\n181 {g}: inputs by PSI 2026Q3\n", x[show].round(3).head(25).to_string(index=False))
        print(f"\n181 {g}: by weighted\n", x.sort_values("weighted", ascending=False)[show].round(3).head(12)
              .to_string(index=False))
        fl = x[x.miss_flag_months != ""][["feature", "miss_ref", "miss_q3", "miss_flag_months"]]
        print(f"\n181 {g}: missing-share jumps\n", fl.to_string(index=False))
    M = pd.DataFrame(rows)
    top = R[R.group == "top50"].sort_values("weighted", ascending=False).feature.head(8).tolist()
    print("\n181 monthly PSI, top50, top weighted inputs\n",
          M[(M.group == "top50") & M.feature.isin(top) & (M.month >= "2025-07")]
          .pivot(index="month", columns="feature", values="psi").round(3).to_string())


# ------------------------------------------------------------------ net series helpers
def nets(Wd, Rx, C, miss):
    out, fb = {}, {}
    for k, W in Wd.items():
        out[k] = bt.run(W, Rx, C).net.where(W.sum(axis=1) > 0).dropna()
        h = W > 0
        fb[k] = float((miss.reindex_like(W) & h).sum().sum() / h.sum().sum())
    return out, fb


def monthly(x):
    g = x.groupby(x.index.to_period("M"))
    return pd.DataFrame({"nights": g.size(), "net_bp": 1e4 * g.mean(), "sharpe": g.apply(sh)})


# ------------------------------------------------------------------ 182
def study182(P, cols):
    days = P["c"].index
    Wd = picks182(P, cols, days)
    index = Wd["live"].index
    Rx, miss, C = S166.returns(P, cols, days, index)
    N, fb = nets(Wd, Rx, C, miss)
    import study135_137_expectations as S135
    ref = S135.build()["nets"]["capped"]
    print("live here vs study 135 build(): corr", round(N["live"].corr(ref), 4), "mean bp here / build",
          round(1e4 * N["live"].mean(), 2), round(1e4 * ref.mean(), 2), flush=True)
    rows = []
    for k, x in N.items():
        for p, r in monthly(x).iterrows():
            rows.append(dict(table="monthly", score=k, family=FAMILY[k], month=str(p), **r.to_dict()))
        base, rec = x.loc["2024-01":"2026-06"], x.loc["2026-07":"2026-09"]
        for nm, s in [("2024-01..2026-06", base), ("2026-07..09", rec), ("2024-01..2025-06", x.loc["2024-01":"2025-06"]),
                      ("2025-07..2026-06", x.loc["2025-07":"2026-06"])]:
            rows.append(dict(table="period", score=k, family=FAMILY[k], period=nm, nights=len(s), net_bp=1e4 * s.mean(),
                             sharpe=sh(s), diff_bp=1e4 * (s.mean() - base.mean()) if nm == "2026-07..09" else np.nan,
                             t_vs_base=welch_t(s, base) if nm == "2026-07..09" else np.nan,
                             corr_live=x.corr(N["live"]), fallback=fb[k]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT182, index=False)
    pd.set_option("display.width", 250)
    per = df[df.table == "period"]
    print("\n182 periods\n", per.pivot(index="score", columns="period", values="net_bp").round(1).to_string())
    print(per[per.period == "2026-07..09"][["score", "family", "nights", "net_bp", "sharpe", "diff_bp", "t_vs_base",
                                            "corr_live", "fallback"]].round(3).to_string(index=False))
    mm = df[(df.table == "monthly")].pivot(index="month", columns="score", values="net_bp")
    print("\n182 monthly net bp\n", mm.loc["2025-07":].round(0).to_string())
    return N["live"]


# ------------------------------------------------------------------ 183
def extra_night_returns(Wd, cols):
    """Night 10-15 -> 10-16 per rule: (panel-style return from Alpaca daily bars, cross return, fallback share)."""
    if not os.path.exists(BARS176):
        return {}
    B = pd.read_parquet(BARS176)
    bc = B[B.date == EXTRA].set_index("ticker").c
    bo = B[B.date == EXTRA_EXIT].set_index("ticker").o
    X = S166.cached_crosses().dropna(subset=["px"])
    X["ticker"] = X.ticker.str.replace(".", "-", regex=False)
    xc = X[(X.date == EXTRA) & (X.kind == "close")].set_index("ticker").px
    xo = X[(X.date == EXTRA_EXIT) & (X.kind == "open")].set_index("ticker").px
    out = {}
    for k, W in Wd.items():
        if W.index[-1] != EXTRA:
            continue
        t = W.columns[W.iloc[-1] > 0]
        rp = (bo.reindex(t) / bc.reindex(t) - 1)
        rx = (xo.reindex(t) / xc.reindex(t) - 1)
        okx = rx.notna() & ((rx - rp).abs() < 0.2)
        rxx = rx.where(okx, rp)
        out[k] = (rp.fillna(0).mean(), rxx.fillna(0).mean(), 1 - okx.mean(), t)
    return out


def study183(P, cols, live_net=None):
    days = P["c"].index
    Wd = rule_picks(P, cols)
    index = Wd["beta60"].index
    Rx, miss, C = S166.returns(P, cols, days, index)
    Rp = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex(index=index)
    Cx = C.copy()
    ex = extra_night_returns(Wd, cols)
    series = {}
    for k, W in Wd.items():
        W = W.loc[:days[-2]]           # panel nights end with entry on the day before the panel's last day
        for pricing, R in [("panel", Rp), ("cross", Rx)]:
            x = bt.run(W, R, Cx).net.where(W.sum(axis=1) > 0).dropna()
            if k in ex:
                t = ex[k][3]
                cst = 2 * float(C.loc[EXTRA, t].fillna(100).mean()) / 1e4
                x.loc[EXTRA] = (ex[k][0] if pricing == "panel" else ex[k][1]) - cst
            series[(k, pricing)] = x
        h = W > 0
        print(k, "cross fallback share", round(float((miss.reindex_like(W) & h).sum().sum() / h.sum().sum()), 4),
              "extra night fallback", round(float(ex[k][2]), 3) if k in ex else None, flush=True)
    if live_net is not None:
        series[("blend_capped", "cross")] = live_net
    rows = []
    PERS = [("2024-01..2026-06", "2024-01", "2026-06"), ("2026Q3", "2026-07", "2026-09"),
            ("2026-10", "2026-10", "2026-10"), ("2026-07..10", "2026-07", "2026-10"),
            ("2024-01..2025-06", "2024-01", "2025-06"), ("2025-07..2026-06", "2025-07", "2026-06")]
    for (k, pricing), x in series.items():
        x = x.sort_index()
        for p, r in monthly(x).iterrows():
            rows.append(dict(table="monthly", rule=k, pricing=pricing, month=str(p), **r.to_dict()))
        base = x.loc["2024-01":"2026-06"]
        full = x.loc["2024-01":]
        for nm, a, b in PERS:
            s = x.loc[a:b]
            n = len(s)
            roll = full.rolling(n).mean().dropna() if n > 1 else full
            rows.append(dict(table="period", rule=k, pricing=pricing, period=nm, nights=n, net_bp=1e4 * s.mean(),
                             sharpe=sh(s), t_vs_base=welch_t(s, base) if nm != "2024-01..2026-06" else np.nan,
                             pct_window=pct_rank(roll.values, s.mean()) if nm != "2024-01..2026-06" else np.nan,
                             corr_blend=x.corr(live_net) if live_net is not None else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(OUT183, index=False)
    pd.set_option("display.width", 250)
    per = df[df.table == "period"]
    print("\n183 periods (net bp a night)\n",
          per.pivot_table(index=["rule", "pricing"], columns="period", values="net_bp", sort=False).round(1).to_string())
    print(per[per.period.isin(["2026Q3", "2026-10", "2026-07..10"])][["rule", "pricing", "period", "nights", "net_bp",
                                                                       "t_vs_base", "pct_window", "corr_blend"]]
          .round(3).to_string(index=False))
    mm = df[(df.table == "monthly")].pivot_table(index="month", columns=["rule", "pricing"], values="net_bp")
    print("\n183 monthly net bp\n", mm.loc["2025-07":].round(0).to_string())


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    P = load_panel()
    cols = stock_cols(P)
    if which in ("all", "181"):
        study181(P, cols)
    live = None
    if which in ("all", "182"):
        live = study182(P, cols)
    if which in ("all", "183"):
        study183(P, cols, live)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        main()
