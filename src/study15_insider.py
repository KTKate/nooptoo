"""Study 15: a thorough test of insider open-market purchases (SEC Form 4) as a signal, 2020-01 .. 2026-03
(the 2026q2 data set was not yet published on 2026-09-28, so there is no fresh holdout beyond study 12's).

Data: SEC Form 3/4/5 structured data sets, 2016q1 .. latest quarter, re-downloaded because study 12's file lacks the
officer title, the 10b5-1 flag (AFF10B5ONE, filed since 2023-04), shares owned after the trade and the trade date.
Per quarter the parsed rows are cached as parquet in $STUDY15_CACHE (default /tmp/study15_sec), not in the repo.
Also a footnote flag: any footnote of the filing mentions Rule 10b5-1 without "not" right before it (all years).

Timing: a filing dated d is usable from the next trading day t (decision 15:45, buy in the closing auction of t).
Events: (ticker, t) with at least one purchase filing that becomes usable on t. The event attributes come from the
filings that become usable on t, except n_buyers = distinct purchasers of the ticker over the 21 trading days up to t.
Portfolio per variant: W1 = event mask / max(#events, 10) (at most 1/10 of capital per name), overlapping cohorts
held h = 1, 5, 20, 60 trading days (bt.hold_k_days, 1/h of capital per cohort, cash otherwise), close to close,
cost per side exec_cost_bps(P, "auction") + 2.5 bp.
Tiers (lagged 20d median dollar volume and price): L > $50M (px > 5), M $5-50M (px > 5), S $1-5M (px > 2),
ALL = union. Market adjustment: OLS of daily net returns on SPY (and on SPY + IWM), within each period.
Outputs: results/study15_variants.csv (all variants x periods), results/study15_events.csv (event counts and
composition per year), results/study15_car.csv (event-level mean excess returns over SPY), results/study15_best.csv.
"""
import io
import os
import re
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import requests

os.environ.setdefault("OMP_NUM_THREADS", "1")
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, \
    block_bootstrap_sharpe, RES
import bt

CACHE = os.environ.get("STUDY15_CACHE", "/tmp/study15_sec")
UA = {"User-Agent": "nooptoo research incarnadins@gmail.com"}      # owner's approved SEC contact
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
PLAN_RE = re.compile(r"10b5-?1|10b-5-1|10b5\s1", re.I)
NOT_RE = re.compile(r"not\s+(?:made\s+|effected\s+|executed\s+|entered\s+into\s+|done\s+)?(?:pursuant|under|in\s+accordance)"
                    r"[^.]{0,40}(?:10b5-?1|10b-5-1)", re.I)


# ---------------------------------------------------------------- data
def fetch_quarter(q):
    fn = os.path.join(CACHE, f"{q}.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    time.sleep(0.5)                                                    # well under 5 requests/second
    r = requests.get(URL.format(q=q), headers=UA, timeout=600)
    if r.status_code != 200:
        print(q, "http", r.status_code, flush=True)
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))

    def rd(f, cols):
        return pd.read_csv(z.open(f), sep="\t", usecols=lambda c: c in cols, dtype=str, on_bad_lines="skip",
                           quoting=3)

    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
    sub = sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])]
    tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                   "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
                                   "DIRECT_INDIRECT_OWNERSHIP"])
    tr = tr[tr.TRANS_CODE.isin(["P", "S"])]
    ow = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    ow = ow.drop_duplicates("ACCESSION_NUMBER")
    d = tr.merge(sub, on="ACCESSION_NUMBER").merge(ow, on="ACCESSION_NUMBER", how="left")
    pacc = set(d.loc[d.TRANS_CODE == "P", "ACCESSION_NUMBER"])
    fnt = rd("FOOTNOTES.tsv", ["ACCESSION_NUMBER", "FOOTNOTE_TXT"])
    fnt = fnt[fnt.ACCESSION_NUMBER.isin(pacc) & fnt.FOOTNOTE_TXT.fillna("").str.contains(PLAN_RE)]
    fnt = fnt[~fnt.FOOTNOTE_TXT.str.contains(NOT_RE)]
    d["fn_plan"] = d.ACCESSION_NUMBER.isin(set(fnt.ACCESSION_NUMBER))
    if "AFF10B5ONE" not in d:
        d["AFF10B5ONE"] = None
    out = pd.DataFrame({
        "acc": d.ACCESSION_NUMBER, "filed": pd.to_datetime(d.FILING_DATE, format="%d-%b-%Y", errors="coerce"),
        "tdate": pd.to_datetime(d.TRANS_DATE, format="%d-%b-%Y", errors="coerce"),
        "ticker": d.ISSUERTRADINGSYMBOL.str.upper().str.strip(), "issuer": d.ISSUERCIK, "code": d.TRANS_CODE,
        "ad": d.TRANS_ACQUIRED_DISP_CD, "shares": pd.to_numeric(d.TRANS_SHARES, errors="coerce"),
        "price": pd.to_numeric(d.TRANS_PRICEPERSHARE, errors="coerce"),
        "after": pd.to_numeric(d.SHRS_OWND_FOLWNG_TRANS, errors="coerce"), "di": d.DIRECT_INDIRECT_OWNERSHIP,
        "owner": d.RPTOWNERCIK, "role": d.RPTOWNER_RELATIONSHIP, "title": d.RPTOWNER_TITLE,
        "aff": d.AFF10B5ONE.map({"1": 1.0, "0": 0.0, "true": 1.0, "false": 0.0}), "fn_plan": d.fn_plan})
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(fn, index=False)
    print(q, len(out), flush=True)
    return out


def load_raw():
    qs = [f"{y}q{k}" for y in range(2016, 2027) for k in range(1, 5) if f"{y}q{k}" <= "2026q2"]
    parts = [x for x in (fetch_quarter(q) for q in qs) if x is not None]
    return pd.concat(parts, ignore_index=True).dropna(subset=["filed", "owner"])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    d = load_raw()
    print(d.groupby(d.filed.dt.year).code.value_counts().unstack())
    sys.exit(0)

CEO_RE = re.compile(r"\bCEO\b|\bCFO\b|chief\s+exec|chief\s+financ|principal\s+exec|principal\s+financ|"
                    r"\bC\.E\.O\b|\bC\.F\.O\b", re.I)


def build_filings(raw):
    """One row per purchase filing (accession): value, shares, holdings before, role class, plan and routine flags."""
    raw = raw.copy()
    raw["tdate"] = raw.tdate.fillna(raw.filed)
    raw.loc[raw.tdate > raw.filed, "tdate"] = raw.filed
    # routine (Cohen, Malloy, Pomorski 2012): the insider traded (P or S) this issuer in the same calendar month
    # in each of the 3 previous calendar years
    ym = raw[["owner", "issuer"]].assign(y=raw.tdate.dt.year, m=raw.tdate.dt.month).drop_duplicates()
    key = set(zip(ym.owner, ym.issuer, ym.y, ym.m))
    x = raw[(raw.code == "P") & (raw.ad.fillna("A") == "A") & (raw.shares > 0) & (raw.price > 0)].copy()
    x = x.drop_duplicates(["owner", "ticker", "tdate", "shares", "price"])        # amendments repeating a trade
    x["value"] = x.shares * x.price
    x = x[x.value < 1e9]                                                          # obvious unit errors
    x["plan_aff"] = x.aff == 1
    g = x.sort_values(["acc", "tdate"]).groupby("acc")
    f = g.agg(filed=("filed", "first"), tdate=("tdate", "first"), ticker=("ticker", "first"),
              issuer=("issuer", "first"), owner=("owner", "first"), role=("role", "first"), title=("title", "first"),
              value=("value", "sum"), shares=("shares", "sum"), after=("after", "max"),
              aff=("aff", "max"), fn_plan=("fn_plan", "max")).reset_index()
    prior = f.after - f.shares
    f["rel"] = np.where(f.after.notna(), np.where(prior > 0, f.shares / prior.where(prior > 0), np.inf), np.nan)
    r, t = f.role.fillna(""), f.title.fillna("")
    f["cls"] = np.select([t.str.contains(CEO_RE) & (r.str.contains("Officer") | r.str.contains("Director")),
                          r.str.contains("Officer"), r.str.contains("Director"), r.str.contains("TenPercent")],
                         ["ceo_cfo", "officer", "director", "tenpct"], "other")
    f["plan"] = (f.aff == 1) | f.fn_plan.astype(bool)
    y, m = f.tdate.dt.year, f.tdate.dt.month
    f["routine"] = np.array([all((o, i, yy - k, mm) in key for k in (1, 2, 3))
                             for o, i, yy, mm in zip(f.owner, f.issuer, y, m)])
    return f


def build_events(f, days, cols, P):
    f = f[f.ticker.isin(set(cols))].copy()
    f["di"] = days.searchsorted(f.filed, side="right")               # first trading day strictly after the filing
    f = f[f.di < len(days)]
    # distinct purchasers of the ticker over the 21 trading days up to and including t
    od = f[["ticker", "owner", "di"]].drop_duplicates()
    ev = f.groupby(["ticker", "di"]).agg(
        val=("value", "sum"), rel=("rel", "max"), n_new=("owner", "nunique"),
        ceo_cfo=("cls", lambda s: (s == "ceo_cfo").any()),
        officer=("cls", lambda s: s.isin(["ceo_cfo", "officer"]).any()),
        director_only=("cls", lambda s: (s == "director").all()),
        tenpct_only=("cls", lambda s: s.isin(["tenpct", "other"]).all()),
        opp=("routine", lambda s: (~s).any()), routine=("routine", "all"),
        noplan=("plan", lambda s: (~s).any()), plan_any=("plan", "any"),
        aff_known=("aff", lambda s: s.notna().any())).reset_index()
    m = ev[["ticker", "di"]].merge(od, on="ticker", suffixes=("", "_o"))
    m = m[(m.di_o <= m.di) & (m.di_o > m.di - 21)]
    ev = ev.merge(m.groupby(["ticker", "di"]).owner.nunique().rename("nb").reset_index(), on=["ticker", "di"])
    ev["day"] = days[ev.di.values]
    c = P["c"]
    ci = {t: i for i, t in enumerate(c.columns)}
    j = ev.ticker.map(ci).values
    cv = c.values
    ev["mom"] = cv[ev.di.values - 1, j] / cv[np.maximum(ev.di.values - 22, 0), j] - 1   # 20d return up to t-1
    return f, ev


VARIANTS = {
    "any": lambda e: e.nb >= 1,
    "nb1": lambda e: e.nb == 1, "nb2": lambda e: e.nb == 2, "nb3p": lambda e: e.nb >= 3,
    "cluster2p": lambda e: e.nb >= 2,
    "ceo_cfo": lambda e: e.ceo_cfo, "officer": lambda e: e.officer,
    "director_only": lambda e: e.director_only, "tenpct_only": lambda e: e.tenpct_only,
    "val_lt50k": lambda e: e.val < 5e4, "val_50_250k": lambda e: (e.val >= 5e4) & (e.val < 2.5e5),
    "val_250k_1m": lambda e: (e.val >= 2.5e5) & (e.val < 1e6), "val_gt1m": lambda e: e.val >= 1e6,
    "rel_lt5": lambda e: e.rel < 0.05, "rel_5_25": lambda e: (e.rel >= 0.05) & (e.rel < 0.25),
    "rel_gt25": lambda e: e.rel >= 0.25,
    "opportunistic": lambda e: e.opp, "routine": lambda e: e.routine,
    "noplan": lambda e: e.noplan,
    "mom_down20": lambda e: e.mom < -0.2, "mom_down": lambda e: (e.mom >= -0.2) & (e.mom < 0),
    "mom_up": lambda e: e.mom >= 0,
    "cluster_opp": lambda e: (e.nb >= 2) & e.opp, "officer_opp": lambda e: e.officer & e.opp,
    "cluster_down": lambda e: (e.nb >= 2) & (e.mom < 0),
}
HORIZONS = [1, 5, 20, 60]
PERIODS = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-01..2025-06", "2024-01-01", "2025-06-30"),
           ("2025-07..2026-03", "2025-07-01", "2026-03-31"),
           ("full 2020-01..2026-03", "2020-01-01", "2026-03-31")] + \
          [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2020, 2026)] + [("2026Q1", "2026-01-01", "2026-03-31")]


def mkt_stats(r, spy, iwm, ew):
    """OLS of daily net returns on SPY, on SPY + IWM, and on SPY + IWM + the tier's equal-weight stock return
    (same survivorship as the signal): alpha (bp/day), beta(s), hedged Sharpe."""
    df = pd.concat([r, spy, iwm, ew], axis=1, keys=["r", "s", "i", "e"], sort=True).dropna()
    out = {}
    if len(df) < 20 or df.r.std() == 0:
        return out
    X1 = np.c_[np.ones(len(df)), df.s]
    b1, *_ = np.linalg.lstsq(X1, df.r.values, rcond=None)
    h1 = df.r - b1[1] * df.s
    e1 = df.r.values - X1 @ b1
    se1 = np.sqrt(e1.var(ddof=2) / len(df))
    X2 = np.c_[np.ones(len(df)), df.s, df.i]
    b2, *_ = np.linalg.lstsq(X2, df.r.values, rcond=None)
    h2 = df.r - b2[1] * df.s - b2[2] * df.i
    out.update(beta_spy=b1[1], alpha_spy_bpd=1e4 * b1[0], alpha_spy_ann=252 * b1[0], alpha_spy_t=b1[0] / se1,
               hsharpe_spy=h1.mean() / h1.std() * np.sqrt(252),
               beta2_spy=b2[1], beta2_iwm=b2[2], alpha2_bpd=1e4 * b2[0], alpha2_ann=252 * b2[0],
               hsharpe_spy_iwm=h2.mean() / h2.std() * np.sqrt(252))
    X3 = np.c_[np.ones(len(df)), df.s, df.i, df.e]
    b3, *_ = np.linalg.lstsq(X3, df.r.values, rcond=None)
    h3 = df.r - X3[:, 1:] @ b3[1:]
    e3 = df.r.values - X3 @ b3
    out.update(beta3_ew=b3[3], alpha3_bpd=1e4 * b3[0], alpha3_ann=252 * b3[0],
               alpha3_t=b3[0] / np.sqrt(e3.var(ddof=4) / len(df)), hsharpe_ew=h3.mean() / h3.std() * np.sqrt(252))
    return out


def main():
    raw = load_raw()
    print("raw rows", len(raw), flush=True)
    f = build_filings(raw)
    del raw
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    f, ev = build_events(f, days, cols, P)
    ev = ev[(ev.day >= "2020-01-01") & (ev.day <= "2026-06-30")]
    # tiers at t (lagged information, as study 12)
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
    px = traded_close(P)[cols].shift(1).fillna(P["rawc"][cols].shift(1))
    ci = {t: i for i, t in enumerate(cols)}
    j, di = ev.ticker.map(ci).values, ev.di.values
    a, p = adv.values[di, j], px.values[di, j]
    ev["adv"], ev["px"] = a, p
    ev["tier"] = np.select([(a > 5e7) & (p > 5), (a > 5e6) & (a <= 5e7) & (p > 5), (a > 1e6) & (a <= 5e6) & (p > 2)],
                           ["L", "M", "S"], "none")
    ev["year"] = ev.day.dt.year
    print("events", len(ev), ev.tier.value_counts().to_dict(), flush=True)

    # ---------------- composition per year (filings and events)
    fy = f[(f.filed >= "2019-12-31") & (f.di < len(days))].copy()
    fy["day"] = days[fy.di.values]
    fy = fy[(fy.day >= "2020-01-01") & (fy.day <= "2026-06-30")]
    fy["year"] = fy.day.dt.year
    comp = fy.groupby("year").agg(filings=("acc", "size"), owners=("owner", "nunique"),
                                  tickers=("ticker", "nunique"), med_value=("value", "median"),
                                  sh_aff_known=("aff", lambda s: s.notna().mean()),
                                  sh_plan_aff=("aff", lambda s: (s == 1).mean()),
                                  sh_plan_footnote=("fn_plan", "mean"), sh_routine=("routine", "mean"),
                                  sh_ceo_cfo=("cls", lambda s: (s == "ceo_cfo").mean()),
                                  sh_officer=("cls", lambda s: (s == "officer").mean()),
                                  sh_director=("cls", lambda s: (s == "director").mean()),
                                  sh_tenpct=("cls", lambda s: s.isin(["tenpct", "other"]).mean()))
    comp.insert(0, "what", "filings (all panel tickers)")
    rows_c = []
    for tier in ["L", "M", "S", "ALL"]:
        e = ev[ev.tier != "none"] if tier == "ALL" else ev[ev.tier == tier]
        cnt = pd.DataFrame({v: e[fn(e).fillna(False).astype(bool)].groupby("year").size() for v, fn in VARIANTS.items()})
        cnt.insert(0, "what", f"events tier {tier}")
        rows_c.append(cnt)
    comp_ev = pd.concat(rows_c)
    comp = pd.concat([comp.reset_index(), comp_ev.reset_index()], ignore_index=True)
    comp.to_csv(f"{RES}/study15_events.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(comp.round(3).to_string(), flush=True)

    # ---------------- portfolios
    tick = sorted(set(ev.ticker))
    tpos = {t: i for i, t in enumerate(tick)}
    sel = slice("2020-01-02", "2026-09-25")
    D = days[(days >= "2020-01-02")]
    c = P["c"][tick].loc[sel]
    Rcc = (c.shift(-1) / c - 1).astype("float32")
    spy = (P["c"]["SPY"].shift(-1) / P["c"]["SPY"] - 1).loc[sel]
    iwm = (P["c"]["IWM"].shift(-1) / P["c"]["IWM"] - 1).loc[sel]
    cost = (exec_cost_bps(P, "auction")[tick] + 2.5).loc[sel]
    # equal-weight return of every eligible stock in each tier (benchmark with the same survivorship)
    cA = P["c"][cols].loc[sel]
    RA = cA.shift(-1) / cA - 1
    TM = {"L": (adv > 5e7) & (px > 5), "M": (adv > 5e6) & (adv <= 5e7) & (px > 5),
          "S": (adv > 1e6) & (adv <= 5e6) & (px > 2)}
    TM["ALL"] = TM["L"] | TM["M"] | TM["S"]
    ew = {k: RA.where(m.loc[sel]).mean(axis=1) for k, m in TM.items()}
    del adv, px, cA, RA, TM
    rows, series = [], {}
    car_rows = []
    fwd = {h: (P["c"][tick].shift(-h) / P["c"][tick] - 1) for h in HORIZONS}
    fwd_spy = {h: P["c"]["SPY"].shift(-h) / P["c"]["SPY"] - 1 for h in HORIZONS}
    for tier in ["ALL", "L", "M", "S"]:
        e0 = ev[ev.tier != "none"] if tier == "ALL" else ev[ev.tier == tier]
        for v, fn in VARIANTS.items():
            e = e0[fn(e0).fillna(False).astype(bool).values]
            jj = e.ticker.map(tpos).values.astype(int)
            A = np.zeros((len(D), len(tick)), dtype="float32")
            A[D.searchsorted(e.day.values), jj] = 1.0
            M = pd.DataFrame(A, index=D, columns=tick)
            n_day = M.sum(axis=1)
            W1 = M.div(n_day.clip(lower=10), axis=0).astype("float32")
            # event-level excess returns over SPY (gross and net of 2 x cost)
            for h in HORIZONS:
                Wh = bt.hold_k_days(W1, h)
                r = bt.run(Wh, Rcc, cost, roundtrip=False)
                series[(tier, v, h)] = r.net
                if len(e):
                    fr = fwd[h].values[e.di.values, jj] - fwd_spy[h].values[e.di.values]
                    cst = 2 * cost.values[D.searchsorted(e.day.values), jj] / 1e4
                    ex = pd.DataFrame({"day": e.day.values, "x": fr, "xn": fr - cst}).dropna()
                for per, a_, b_ in PERIODS:
                    x = r.loc[a_:b_]
                    st = ann_stats(x.net)
                    ne = int(((e.day >= a_) & (e.day <= b_)).sum())
                    row = dict(tier=tier, variant=v, hold=h, period=per, n_events=ne, sharpe=st["sharpe"],
                               ann_ret=st["ann_ret"], maxdd=st["maxdd"], net_bpd=1e4 * x.net.mean(),
                               gross_bpd=1e4 * x.gross.mean(), invested=float(Wh.loc[a_:b_].sum(axis=1).mean()),
                               names=float(x.n.mean()))
                    row.update(mkt_stats(x.net, spy, iwm, ew[tier]))
                    rows.append(row)
                    if len(e):
                        xe = ex[(ex.day >= a_) & (ex.day <= b_)]
                        if len(xe) >= 10:
                            mo = xe.groupby(xe.day.dt.to_period("M"))[["x", "xn"]].mean()   # month-clustered
                            car_rows.append(dict(tier=tier, variant=v, hold=h, period=per, n_events=len(xe),
                                                 car_bps=1e4 * xe.x.mean(), car_net_bps=1e4 * xe.xn.mean(),
                                                 t_month=mo.x.mean() / mo.x.std() * np.sqrt(len(mo))
                                                 if len(mo) > 2 else np.nan, hit=(xe.x > 0).mean()))
            print(tier, v, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/study15_variants.csv", index=False)
    pd.DataFrame(car_rows).to_csv(f"{RES}/study15_car.csv", index=False)

    # ---------------- multiple testing: best variants, deflated Sharpe, bootstrap
    n_var = len(VARIANTS) * len(HORIZONS) * 4
    n_trials = n_var + 270
    best = []
    main_p = ["2020-23", "2024-01..2025-06", "2025-07..2026-03", "full 2020-01..2026-03"]
    for crit_per, crit in [("full 2020-01..2026-03", "sharpe"), ("full 2020-01..2026-03", "hsharpe_spy_iwm"),
                           ("full 2020-01..2026-03", "hsharpe_ew"),
                           ("2020-23", "sharpe"), ("2025-07..2026-03", "sharpe")]:
        k = df[df.period == crit_per].sort_values(crit, ascending=False).iloc[0]
        key = (k.tier, k.variant, int(k.hold))
        r = series[key].loc["2020-01-01":"2026-03-31"]
        spy_f = spy.reindex(r.index)
        iwm_f = iwm.reindex(r.index)
        X = np.c_[np.ones(len(r)), spy_f.fillna(0), iwm_f.fillna(0), ew[k.tier].reindex(r.index).fillna(0)]
        b, *_ = np.linalg.lstsq(X, r.values, rcond=None)
        hr = pd.Series(r.values - X[:, 1:] @ b[1:], index=r.index)
        out = dict(selected_on=f"{crit} in {crit_per}", tier=k.tier, variant=k.variant, hold=int(k.hold),
                   n_variants=n_var, n_trials=n_trials)
        for nm, s in [("raw", r), ("hedged", hr)]:
            sr = s.mean() / s.std()
            out[f"{nm}_sharpe_full"] = sr * np.sqrt(252)
            out[f"{nm}_dsr"] = deflated_sharpe(sr, n_trials, len(s), skew=s.skew(), kurt=s.kurt() + 3)
            lo, md, hi = block_bootstrap_sharpe(s, block=20, n=1000)
            out[f"{nm}_ci_lo"], out[f"{nm}_ci_hi"] = lo, hi
        for per in main_p:
            q = df[(df.tier == k.tier) & (df.variant == k.variant) & (df.hold == k.hold) & (df.period == per)].iloc[0]
            out[f"sharpe {per}"] = q.sharpe
            out[f"hsharpe {per}"] = q.hsharpe_spy_iwm
            out[f"hsharpe_ew {per}"] = q.hsharpe_ew
        best.append(out)
    best = pd.DataFrame(best)
    best.to_csv(f"{RES}/study15_best.csv", index=False)
    print(best.T.to_string())
    typ = df[df.period.isin(main_p)].groupby(["period", "hold"])[["sharpe", "hsharpe_spy", "hsharpe_spy_iwm",
                                                                    "alpha2_bpd", "hsharpe_ew", "alpha3_bpd"]].median()
    print("median over variants\n", typ.round(2).to_string())


if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1] != "fetch"):
    main()
