"""Studies 90, 91 and 92: risk controls for the overnight blend (study 33; top 10 by 2 x ensemble rank + jump-minus-drop
rank at 15:45, closing-auction buy, opening-auction sell, auction costs + 2.5 bp per side), 2024-01..2026-09.
Prompted by the night of 2026-10-06 (all 10 picks fell, average -2.4%, five of them chip/storage/hardware names).

Baseline exactly as src/study69_blend_variants.py (uncapped), and the live job's version with at most 3 names per
industry (src/paper_overnight.py cap_by_industry; names without an industry are their own group).
Periods: rules are chosen on 2024-01..2025-06 ('selection') and reported on 2025-07..2026-09 ('holdout'), plus 2024-26.
Every variant that was run is in the output. Paired t = mean / se of the daily difference variant - baseline.
A bad night = net return of the uncapped baseline <= -2%.

90  Bad-night filter. Inputs known by 15:45 (the daily close stands in for the 15:45 price, a small look-ahead of the
    last 15 minutes): SPY/QQQ/IWM/SMH return today and over 5 days, SPY realized vol 5/20 days, ^VIX level and 1/5-day
    change (panel close), the picks' average 60-day beta to SPY (daily close returns) and 60-day overnight beta, their
    average 20-day vol, concentration (max names per sector / industry, names in the tech hardware group, average
    pairwise 60-day correlation among the 10 picks), picks reporting earnings that night (store 'earnings': date = t
    and not pre-market, or date = next trading day and not after-hours; most rows have no time so both count), a
    CPI/NFP release at 08:30 the next morning or an FOMC decision today (data/store/macro_events.csv), weekday, the
    last night's blend return and the trailing 20-night blend vol.
    (a) quintile tables of next-night net return and bad-night share per input (cut points from the selection period).
    (b) pre-registered rule family: halve the book (or skip the night) when one input is in its top or bottom
        selection quintile; the single rule with the best selection Sharpe is the chosen one, all are reported.
    (c) walk-forward logistic regression (sklearn, standardized inputs, monthly refit on all earlier nights, first
        prediction 2024-07) for P(bad night); halve or skip when P is above the 80th percentile of its training fit.
91  Volatility-scaled sizes for the same 10 picks, gross 100%: weights 1/vol20 (daily close returns), 1/sqrt(vol20),
    1/vol of 60-day overnight returns, equal risk contribution from the 60-day daily-return covariance; book scaling
    min(1, target / trailing 20-night blend vol) and min(1, target / ex-ante book vol from the picks' 60-day overnight
    covariance), target = median of that vol over the selection period.
92  Concentration caps (greedy down the blend ranking): at most 2/3/4 names per sector, at most 2/3 names in a tech
    hardware group (industries listed in TECH_HW below; NASDAQ classification), tech hardware 2 plus industry 3, and a
    correlation cap (skip a name whose 60-day daily-return correlation with an already chosen pick exceeds 0.6/0.7).
    sectors.parquet is a current snapshot (classification look-ahead, small).
Output: results/study90_bad_nights.csv, results/study91_vol_sizing.csv, results/study92_concentration.csv
"""
import numpy as np
import pandas as pd
import store
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, ROOT
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < days[-1]]
pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
per = [("selection", "2024-01", "2025-06"), ("holdout", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
idx = S.index
SEL = slice("2024-01", "2025-06")

# ---------------------------------------------------------------- point-in-time inputs (all known at the close of t)
c = P["c"][cols]
r1 = (c / c.shift(1) - 1)
on = (P["o"][cols] / c.shift(1) - 1)                      # overnight return ending at today's open
spy = (P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1)
spy_on = P["o"]["SPY"] / P["c"]["SPY"].shift(1) - 1


def roll_beta(x, m, n=60, mp=40):
    xm = x.rolling(n, min_periods=mp).mean()
    mm = m.rolling(n, min_periods=mp).mean()
    cov = x.mul(m, axis=0).rolling(n, min_periods=mp).mean() - xm.mul(mm, axis=0)
    return cov.div(m.rolling(n, min_periods=mp).var(ddof=0), axis=0)


beta60 = roll_beta(r1, spy).reindex(idx)
obeta60 = roll_beta(on, spy_on).reindex(idx)
vol20 = r1.rolling(20, min_periods=15).std().reindex(idx).replace(0, np.nan)   # 0 = stale price
ovol60 = on.rolling(60, min_periods=40).std().reindex(idx).replace(0, np.nan)
pos = days.get_indexer(idx)                               # row of each signal day in the full panel
r1v = r1.values
onv = on.values
ci = {t: j for j, t in enumerate(cols)}

sec = pd.read_parquet(f"{ROOT}/data/store/sectors.parquet").drop_duplicates("ticker").set_index("ticker")
industry = sec.industry.reindex(cols)
sector = sec.sector.reindex(cols)
# tech hardware: semiconductors, semiconductor equipment, storage / computer hardware, electronic components.
# NASDAQ puts semi equipment (LRCX, ASML, VECO) and VRT in 'Industrial Machinery/Components' and APH/CLS/JBL/FLEX in
# 'Electrical Products', so those two count only within the Technology sector.
TECH_HW = ["Semiconductors", "Electronic Components", "Computer peripheral equipment", "Computer Manufacturing",
           "Computer Communications Equipment", "Retail: Computer Software & Peripheral Equipment"]
TECH_HW_TECHSECTOR = ["Electrical Products", "Industrial Machinery/Components"]
techhw = (industry.isin(TECH_HW) | (industry.isin(TECH_HW_TECHSECTOR) & (sector == "Technology"))).values


def groups(lab):
    """integer group code per column; missing / blank labels are their own group (as cap_by_industry)"""
    lab = lab.fillna("").astype(str).str.strip()
    lab = lab.where(lab != "", "__" + pd.Series(cols, index=lab.index))
    return pd.factorize(lab)[0]


g_ind, g_sec = groups(industry), groups(sector)
g_tech = np.where(techhw, -1, np.arange(len(cols)))      # tech hardware = one group (-1), others unconstrained


def corr_block(rows_end, js, n=60, mp=40):
    """correlation matrix of daily returns over the n days ending at panel row rows_end for columns js"""
    X = r1v[rows_end - n + 1:rows_end + 1][:, js].astype("float64")
    good = np.isfinite(X).sum(0) >= mp
    X = X - np.nanmean(X, 0)
    X = np.where(np.isfinite(X), X, 0.0)
    sd = np.sqrt((X ** 2).sum(0))
    sd[sd == 0] = np.nan
    Cm = (X.T @ X) / np.outer(sd, sd)
    Cm[~good, :] = np.nan
    Cm[:, ~good] = np.nan
    return Cm


def select_greedy(k=10, caps=(), corr_cap=None, depth=150):
    """walk down each day's blend ranking; caps = [(group codes, max per group)]; corr_cap skips a name whose
    60-day correlation with an already chosen name exceeds it (names without 40 days of history are not blocked)"""
    Sv = S.values
    W = np.zeros_like(Sv, dtype="float32")
    for i in range(len(idx)):
        s = Sv[i]
        okj = np.where(np.isfinite(s))[0]
        order = okj[np.argsort(-s[okj], kind="stable")][:depth]
        Cm = corr_block(pos[i], order) if corr_cap is not None else None
        cnt = [dict() for _ in caps]
        chosen = []
        for a, j in enumerate(order):
            if any(cnt[q].get(g[j], 0) >= m for q, (g, m) in enumerate(caps)):
                continue
            if Cm is not None and chosen and np.nanmax(np.r_[Cm[a, [b for b, _ in chosen]], -1]) > corr_cap:
                continue
            for q, (g, m) in enumerate(caps):
                cnt[q][g[j]] = cnt[q].get(g[j], 0) + 1
            chosen.append((a, j))
            if len(chosen) == k:
                break
        if chosen:
            W[i, [j for _, j in chosen]] = 1.0 / len(chosen)
    return pd.DataFrame(W, index=idx, columns=cols)


W = bt.select_topk(S, S.notna(), 10)
base = bt.run(W, R, C).net
Wcap = select_greedy(caps=[(g_ind, 3)])
capped = bt.run(Wcap, R, C).net
bad = base <= -0.02


def stats(name, net, gross_w=None, extra=None):
    """one row per period: Sharpe, mean, maxdd, worst night, bad nights, paired t vs both baselines"""
    out = []
    for p, a, b in per:
        x = net.loc[a:b].dropna()
        st = ann_stats(x)
        row = dict(variant=name, period=p, n=len(x), sharpe=st["sharpe"], ann=st["ann_ret"], net_bp=1e4 * x.mean(),
                   maxdd=st["maxdd"], worst_bp=1e4 * x.min(), n_le_2pct=int((x <= -0.02).sum()),
                   avg_gross=np.nan if gross_w is None else gross_w.loc[a:b].mean())
        for bn, bs in [("base", base), ("capped", capped)]:
            d = (x - bs.loc[a:b]).dropna()
            row[f"diff_bp_vs_{bn}"] = 1e4 * d.mean()
            row[f"t_vs_{bn}"] = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() > 0 else np.nan
        # scaled books: compare with the baseline held at the same average gross (a constant scale-down keeps the
        # Sharpe and shrinks the mean), so a gain here is timing, not just less exposure
        g = 1.0 if gross_w is None else gross_w.loc[x.index].mean()
        d = (x - g * base.loc[x.index]).dropna()
        row["diff_bp_vs_base_same_gross"] = 1e4 * d.mean()
        row["t_vs_base_same_gross"] = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() > 0 else np.nan
        if extra is not None:
            for k_, v in extra.items():
                row[k_] = v.loc[a:b].mean()
        out.append(row)
    return out


def show(df, cols_=("sharpe", "net_bp", "maxdd", "worst_bp", "n_le_2pct", "t_vs_base", "avg_gross", "t_vs_base_same_gross")):
    pd.set_option("display.width", 250)
    print(df.pivot_table(index="variant", columns="period", values=list(cols_), sort=False)
          .reindex(columns=[x for x in ["selection", "holdout", "2024-26"]], level=1).round(2).to_string())


def name_contrib(Wv, label, n=6):
    """data check: largest single name-night contributions to the variant - baseline difference"""
    D = ((Wv - W) * R.fillna(0)).stack()
    D = D[D != 0]
    top = D.reindex(D.abs().sort_values(ascending=False).index[:n])
    print(f"  {label}: total diff {1e4 * D.sum() / len(idx):.2f} bp/night; largest name-nights (bp of book):",
          ", ".join(f"{d.date()} {t} {1e4 * v:+.0f}" for (d, t), v in top.items()))


print(f"picks with |R| > 40%: {int(((W > 0) & (R.abs() > 0.4)).sum().sum())} (data check)")
gross1 = pd.Series(1.0, index=idx)
base_rows = stats("baseline (uncapped)", base, gross1) + stats("baseline industry cap 3 (live)", capped, gross1)

# ================================================================ study 90: bad-night filter
pk = [np.where(W.values[i] > 0)[0] for i in range(len(idx))]
F = pd.DataFrame(index=idx)
for e in ["SPY", "QQQ", "IWM", "SMH"]:
    F[f"{e}_r1"] = (P["c"][e] / P["c"][e].shift(1) - 1).reindex(idx)
    F[f"{e}_r5"] = (P["c"][e] / P["c"][e].shift(5) - 1).reindex(idx)
F["SPY_vol5"] = spy.rolling(5).std().reindex(idx)
F["SPY_vol20"] = spy.rolling(20).std().reindex(idx)
vix = P["c"]["^VIX"]
F["VIX"] = vix.reindex(idx)
F["VIX_chg1"] = (vix - vix.shift(1)).reindex(idx)
F["VIX_chg5"] = (vix - vix.shift(5)).reindex(idx)
F["pick_beta60"] = (beta60 * (W > 0)).sum(axis=1) / 10
F["pick_onbeta60"] = (obeta60 * (W > 0)).sum(axis=1) / 10
F["pick_vol20"] = (vol20 * (W > 0)).sum(axis=1) / 10
F["max_per_sector"] = [pd.Series(g_sec[j]).value_counts().max() for j in pk]
F["max_per_industry"] = [pd.Series(g_ind[j]).value_counts().max() for j in pk]
F["n_techhw"] = [int(techhw[j].sum()) for j in pk]
F["pick_corr60"] = [np.nanmean(corr_block(pos[i], j)[np.triu_indices(len(j), 1)]) for i, j in enumerate(pk)]
E = store.read("earnings", start="2023-12")
E["date"] = pd.to_datetime(E.date)
nxt = pd.Series(days[np.minimum(days.get_indexer(idx) + 1, len(days) - 1)], index=idx)
e_today = set(zip(E.date[E.time != "time-pre-market"], E.symbol[E.time != "time-pre-market"]))
e_next = set(zip(E.date[E.time != "time-after-hours"], E.symbol[E.time != "time-after-hours"]))
F["n_earn"] = [sum(((d, cols[j]) in e_today) or ((nxt[d], cols[j]) in e_next) for j in pk[i])
               for i, d in enumerate(idx)]
M = pd.read_csv(f"{ROOT}/data/store/macro_events.csv", parse_dates=["date"])
F["cpi_nfp_next"] = nxt.isin(M.date[M.event.isin(["CPI", "NFP"])]).astype(int).values
F["fomc_today"] = idx.isin(M.date[M.event.str.startswith("FOMC")]).astype(int)
F["weekday"] = idx.weekday
F["prev_night"] = base.shift(1)
F["blend_vol20"] = base.rolling(20, min_periods=15).std().shift(1)
DISCRETE = ["max_per_sector", "max_per_industry", "n_techhw", "n_earn", "cpi_nfp_next", "fomc_today", "weekday"]
CONT = [f for f in F.columns if f not in DISCRETE]

print("\nbaselines:")
show(pd.DataFrame(base_rows))
print(f"bad nights (<= -2%): selection {int(bad.loc[SEL].sum())}/{len(bad.loc[SEL])}, holdout "
      f"{int(bad.loc['2025-07':'2026-09'].sum())}/{len(bad.loc['2025-07':'2026-09'])}")

# (a) bucket tables
rows90 = []
qcut = {}
for f in F.columns:
    x = F[f]
    if f in DISCRETE:
        b = x.clip(upper=x.loc[SEL].quantile(0.95)) if f.startswith(("max_", "n_")) else x
    else:
        edges = np.unique(x.loc[SEL].quantile([0, .2, .4, .6, .8, 1]).values)
        edges[0], edges[-1] = -np.inf, np.inf
        qcut[f] = edges
        b = pd.cut(x, edges, labels=False)
    for p, a, z in per[:2]:
        net = base.loc[a:z]
        bb = b.loc[a:z]
        for lev, v in net.groupby(bb):
            rows90.append(dict(part="buckets", feature=f, bucket=lev, period=p, n=len(v), net_bp=1e4 * v.mean(),
                               bad_share=(v <= -0.02).mean(), feat_mean=x.loc[v.index].mean(),
                               t=v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 2 else np.nan))
B = pd.DataFrame(rows90)
print("\n90a: by bucket, net bp and bad-night share (%), selection | holdout")
for f in F.columns:
    t_ = B[B.feature == f].pivot_table(index="bucket", columns="period", values=["net_bp", "bad_share"])
    print(f"  {f:16s} " + "  ".join(f"[{q:g}] {t_.loc[q, ('net_bp', 'selection')]:.0f}/{t_.loc[q].get(('net_bp', 'holdout'), np.nan):.0f}bp "
                                    f"{100 * t_.loc[q, ('bad_share', 'selection')]:.0f}/{100 * t_.loc[q].get(('bad_share', 'holdout'), np.nan):.0f}%"
                                    for q in t_.index))

# (b) one-input rules: halve / skip when the input is in its top or bottom selection quintile
rule_rows = []
scales = {}
for f in F.columns:
    x = F[f]
    if f in DISCRETE:
        if f in ("weekday",):
            opts = {f"{f}=={v}": x == v for v in range(5)}
        elif f in ("cpi_nfp_next", "fomc_today"):
            opts = {f"{f}==1": x == 1}
        else:
            hi = x.loc[SEL].quantile(0.8)
            opts = {f"{f}>={hi:g}": x >= max(hi, 1)}
    else:
        e = qcut[f]
        opts = {f"{f} top quintile": x > e[-2], f"{f} bottom quintile": x <= e[1]}
    for nm, m in opts.items():
        for act, s in [("halve", 0.5), ("skip", 0.0)]:
            sc = pd.Series(np.where(m.fillna(False), s, 1.0), index=idx)
            scales[f"90b: {act} if {nm}"] = sc
rule_stats = []
for nm, sc in scales.items():
    rule_stats += stats(nm, base * sc, sc, extra={"flag_share": (sc < 1).astype(float)})
RS = pd.DataFrame(rule_stats)
sel = RS[RS.period == "selection"].set_index("variant")
chosen = {act: sel[sel.index.str.contains(f": {act} if")].sharpe.idxmax() for act in ["halve", "skip"]}
RS["chosen"] = RS.variant.isin(chosen.values())
print(f"\n90b: {len(scales)} rules tried; chosen on selection: {chosen}")
hold = RS[RS.period == "holdout"]
print(f"  holdout Sharpe across all rules: median {hold.sharpe.median():.2f}, baseline "
      f"{ann_stats(base.loc['2025-07':'2026-09'])['sharpe']:.2f}; share of rules beating baseline "
      f"{(hold.sharpe > ann_stats(base.loc['2025-07':'2026-09'])['sharpe']).mean():.2f}")
top_sel = sel.sort_values("sharpe", ascending=False).head(8).index
show(RS[RS.variant.isin(list(top_sel) + list(chosen.values()))], ("sharpe", "net_bp", "maxdd", "n_le_2pct", "flag_share", "t_vs_base_same_gross"))
print("  holdout top 8 by Sharpe (not chosen; for reference):")
show(RS[RS.variant.isin(hold.sort_values("sharpe", ascending=False).head(8).variant)], ("sharpe", "net_bp", "maxdd", "n_le_2pct", "flag_share", "t_vs_base_same_gross"))

# (c) walk-forward logistic P(bad night)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = F.copy()
X["weekday"] = X["weekday"].astype(float)
y = bad.astype(int)
pbad = pd.Series(np.nan, index=idx)
thr = pd.Series(np.nan, index=idx)
months = pd.period_range("2024-07", idx[-1].to_period("M"), freq="M")
for mth in months:
    tr = idx < mth.start_time
    te = (idx >= mth.start_time) & (idx <= mth.end_time)
    if te.sum() == 0:
        continue
    tr = tr & np.asarray(base.notna())
    mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
    Xtr = ((X[tr] - mu) / sd).fillna(0).values
    Xte = ((X[te] - mu) / sd).fillna(0).values
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, y[tr].values)
    pbad[te] = lr.predict_proba(Xte)[:, 1]
    thr[te] = np.quantile(lr.predict_proba(Xtr)[:, 1], 0.8)
for p, a, z in per:
    m = pbad.loc[a:z].notna()
    yy, pp = y.loc[a:z][m], pbad.loc[a:z][m]
    print(f"90c: logistic AUC {p} (from 2024-07): {roc_auc_score(yy, pp):.3f}  n={m.sum()}")
lr_rows = []
flag = (pbad > thr)
for act, s in [("halve", 0.5), ("skip", 0.0)]:
    sc = pd.Series(np.where(flag, s, 1.0), index=idx)
    nm = f"90c: logistic {act} if P(bad) > train 80th pct"
    scales[nm] = sc
    lr_rows += stats(nm, (base * sc).where(pbad.notna()), sc.where(pbad.notna()))
lr_rows += stats("baseline (uncapped), from 2024-07", base.where(pbad.notna()), gross1)
LR = pd.DataFrame(lr_rows)
show(LR)
aucs = {p: roc_auc_score(y.loc[a:z][pbad.loc[a:z].notna()], pbad.loc[a:z].dropna()) for p, a, z in per}
LR["auc"] = LR.period.map(aucs)
LR["flag_share"] = LR.period.map({p: flag.loc[a:z][pbad.loc[a:z].notna()].mean() for p, a, z in per})
out90 = pd.concat([pd.DataFrame(base_rows).assign(part="baseline"), RS.assign(part="rule"), LR.assign(part="logistic"),
                   B], ignore_index=True)
out90.to_csv(f"{RES}/study90_bad_nights.csv", index=False)

# ================================================================ study 91: volatility-scaled sizes
Wv = W.values
pick = Wv > 0


def norm_rows(A):
    A = np.where(pick, A, 0.0)
    A = np.where(np.isfinite(A), A, 0.0)
    # a pick with no vol estimate gets the median weight of the others
    for i in np.where((pick & (A == 0)).any(1))[0]:
        z = pick[i] & (A[i] == 0)
        A[i, z] = np.median(A[i, pick[i] & (A[i] > 0)]) if (pick[i] & (A[i] > 0)).any() else 1.0
    return pd.DataFrame(A / A.sum(1, keepdims=True), index=idx, columns=cols)


def cov_block(rows_end, js, src, n=60, mp=40):
    Xm = src[rows_end - n + 1:rows_end + 1][:, js].astype("float64")
    Xm = Xm - np.nanmean(Xm, 0)
    good = np.isfinite(Xm).sum(0) >= mp
    Xm = np.where(np.isfinite(Xm), Xm, 0.0)
    Cv = Xm.T @ Xm / (n - 1)
    # names with short history: diagonal with the median variance of the others, no correlation
    if (~good).any():
        mv = np.median(np.diag(Cv)[good]) if good.any() else 1e-4
        Cv[~good, :] = 0
        Cv[:, ~good] = 0
        Cv[~good, ~good] = mv
    return Cv


def erc(Cv, it=300):
    w = 1 / np.sqrt(np.diag(Cv))
    w /= w.sum()
    for _ in range(it):
        rc = w * (Cv @ w)
        w = w * np.sqrt(rc.mean() / np.maximum(rc, 1e-18))
        w /= w.sum()
    return w


Werc = np.zeros_like(Wv, dtype="float64")
exante = pd.Series(np.nan, index=idx)
for i in range(len(idx)):
    js = pk[i]
    Werc[i, js] = erc(cov_block(pos[i], js, r1v))
    Co = cov_block(pos[i], js, onv)
    w = np.full(len(js), 0.1)
    exante.iloc[i] = np.sqrt(w @ Co @ w)
var91 = {"91: 1/vol20": norm_rows(1 / vol20.values),
         "91: 1/sqrt(vol20)": norm_rows(1 / np.sqrt(vol20.values)),
         "91: 1/vol overnight 60d": norm_rows(1 / ovol60.values),
         "91: equal risk contribution (60d cov)": pd.DataFrame(Werc, index=idx, columns=cols)}
rows91 = []
print("\n91: data check and stats")
for nm, Wx in var91.items():
    rows91 += stats(nm, bt.run(Wx, R, C).net, gross1, extra={"max_weight": Wx.max(axis=1)})
    name_contrib(Wx, nm)
trail = base.rolling(20, min_periods=15).std().shift(1)
for nm, v in [("91: book min(1, target / trailing 20-night blend vol)", trail),
              ("91: book min(1, target / ex-ante 60d overnight book vol)", exante)]:
    tgt = v.loc[SEL].median()
    sc = (tgt / v).clip(upper=1).fillna(1.0)
    rows91 += stats(nm, base * sc, sc)
    print(f"  {nm}: target {1e4 * tgt:.0f} bp, scale < 1 on {(sc < 1).mean():.0%} of nights")
D91 = pd.DataFrame(base_rows + rows91)
sel91 = D91[(D91.period == "selection") & D91.variant.str.startswith("91")].set_index("variant").sharpe
D91["chosen"] = D91.variant == sel91.idxmax()
show(D91)
print("  chosen on selection:", sel91.idxmax())
D91.to_csv(f"{RES}/study91_vol_sizing.csv", index=False)

# ================================================================ study 92: concentration caps
print("\n92: tech hardware industries:", TECH_HW, "+ within Technology sector:", TECH_HW_TECHSECTOR)
print("  tech hardware names among picks (most frequent):",
      pd.Series([cols[j] for i in range(len(idx)) for j in pk[i] if techhw[j]]).value_counts().head(25).to_dict())
var92 = {"92: industry cap 3 (live)": Wcap,
         "92: sector cap 2": select_greedy(caps=[(g_sec, 2)]),
         "92: sector cap 3": select_greedy(caps=[(g_sec, 3)]),
         "92: sector cap 4": select_greedy(caps=[(g_sec, 4)]),
         "92: tech hardware cap 2": select_greedy(caps=[(g_tech, 2)]),
         "92: tech hardware cap 3": select_greedy(caps=[(g_tech, 3)]),
         "92: tech hardware cap 2 + industry cap 3": select_greedy(caps=[(g_tech, 2), (g_ind, 3)]),
         "92: correlation cap 0.6": select_greedy(corr_cap=0.6),
         "92: correlation cap 0.7": select_greedy(corr_cap=0.7)}


def conc(Wx):
    pkx = [np.where(Wx.values[i] > 0)[0] for i in range(len(idx))]
    return {"n_sectors": pd.Series([len(set(g_sec[j])) for j in pkx], index=idx),
            "max_per_sector": pd.Series([pd.Series(g_sec[j]).value_counts().max() for j in pkx], index=idx),
            "n_techhw": pd.Series([int(techhw[j].sum()) for j in pkx], index=idx),
            "pick_corr60": pd.Series([np.nanmean(corr_block(pos[i], j)[np.triu_indices(len(j), 1)])
                                      for i, j in enumerate(pkx)], index=idx)}


rows92 = stats("baseline (uncapped)", base, gross1, extra=conc(W))
for nm, Wx in var92.items():
    rows92 += stats(nm, bt.run(Wx, R, C).net, gross1, extra=conc(Wx))
    name_contrib(Wx, nm)
D92 = pd.DataFrame(rows92)
sel92 = D92[(D92.period == "selection") & D92.variant.str.startswith("92")].set_index("variant").sharpe
D92["chosen"] = D92.variant == sel92.idxmax()
show(D92, ("sharpe", "net_bp", "maxdd", "worst_bp", "n_le_2pct", "t_vs_base", "t_vs_capped", "n_sectors",
           "n_techhw"))
print("  chosen on selection:", sel92.idxmax())
D92.to_csv(f"{RES}/study92_concentration.csv", index=False)

# the worst baseline nights: which inputs were unusual (context for 90)
print("\nworst 12 baseline nights 2024-26 with inputs:")
wn = base.loc["2024-01":"2026-09"].nsmallest(12).index
print(pd.concat([1e4 * base.loc[wn].rename("net_bp"), F.loc[wn, ["SPY_r1", "SMH_r1", "VIX", "VIX_chg1", "pick_beta60",
                 "pick_corr60", "max_per_sector", "n_techhw", "n_earn"]]], axis=1).round(3).to_string())
