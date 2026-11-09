"""Study 211: study 15's insider-cluster rule (mid caps, exactly 2 buyers, 60-day hold) rescored on data filed after
study 15, a $10k long-only account design, and a paper-shadow specification.

Question: does the frozen study 15 winner keep working on Form 4 filings that study 15 never saw (filed 2026-04-01 or
later), and what would it look like run in a $10k cash account (no shorting)?

Frozen rule (results/study211_frozen.json, written before any new download or scoring; identical to study 15's
best row, tier M, variant nb2, hold 60):
  * Data: SEC Form 3/4/5 structured data sets, Form 4 and 4/A, non-derivative code P (open-market purchase),
    acquired, shares > 0, price > 0, amendment duplicates dropped; one row per accession (first reporting owner).
  * Timing: a filing with FILING_DATE d is usable on t = the first trading day strictly after d; buy in the closing
    auction of t (decision at 15:45 of t).
  * Cluster: nb = distinct insiders with purchase filings of that ticker usable on trading days t-20..t. The rule
    needs at least one new purchase filing usable on t and nb == 2 EXACTLY (study 15's '2+ buyers' variant,
    cluster2p, was weaker; the report row's '2+ buyers' wording is loose).
  * Mid cap (tier M): 20-day median dollar volume through t-1 in ($5M, $50M] and prior close > $5. No size filter.
  * Hold 60 trading days, sell in the closing auction of t+60. Costs per side: auction 1.3 + 0.1 x half-spread
    (core.half_spread_model "15:45") + 2.5 bp.
  * Study 15's 'hedged' Sharpe is an in-sample OLS on SPY and IWM daily returns, not a tradable fixed short.

Data: SEC quarterly data sets 2019q4..2026q3 (2026q2 and 2026q3 moved to /files/datastandardsinnovation/...; the
study 15 URL returns 404 for them, so study 15 actually stopped at filings of 2026-03-31). Filings after 2026-09-30
are not in any data set yet (the Q4 set appears in early 2027); they are not needed to score complete 60-day holds
(the last complete entry is about 60 trading days before the last price date). Fetch via study 15's fetch_quarter
(User-Agent "nooptoo research incarnadins@gmail.com", one request per 0.5 s), cache in $STUDY211_CACHE.
Prices: daily return = Alpaca d1 official closing-cross return (alpaca_data.read("d1"), ~1100 liquid names, to
2026-10-30) plus the Yahoo dividend part (adjusted minus raw-close return); the Yahoo adjusted return from the daily
store where d1 has no bar, after 2026-10-30, or where d1 and Yahoo price returns differ by > 10 points (split jumps).
Most mid caps are not in d1, so most event returns come from the daily store. BIL (T-bill ETF) from
data/local/etf_long. Survivorship: the daily panel holds only tickers alive in 2026, which flatters 60-day returns
before 2026; for the new period (2026-04..) the bias is small but not zero.

Parts:
  A. replication: the study 15 book (W1 = 1/max(#events,10), 60 overlapping cohorts) on 2020-01..2026-03 with the
     re-downloaded data, vs study 15's numbers.
  B. new data: events with all triggering filings filed after 2026-03-31. Per event 60-day net return (2 x cost),
     raw (long-only), minus SPY, minus IWM, minus the tier M equal-weight average (survivorship-matched); mean with
     a month-clustered t. Also the study 15 book's daily returns over 2026-04..latest (raw and OLS-hedged Sharpe).
     No pass threshold: report whether the sign and size agree with study 15 (199..639 bp vs SPY per period).
  C. $10k long-only account 2020-01..latest: fixed fraction of equity per new event, max concurrent names = 1/fraction,
     whole shares at the traded close, idle cash in 0% cash, BIL or SPY (SPY flows pay 3.5 bp per side). Events that
     arrive while all slots are full are skipped; same-day ties are taken in order of larger total insider purchase
     value (the frozen rule has no ranking; a 200-run random-priority check shows how much luck that adds).
  D. by year and events per month.

Paper-shadow specification (no orders are placed by this study or by any code it adds):
  1. Each trading evening after 22:00 ET, read that day's Form 4 and 4/A filings from EDGAR (daily form index
     https://www.sec.gov/Archives/edgar/daily-index/ form.YYYYMMDD.idx, or the "Latest filings" Form 4 feed), parse
     the XML for non-derivative transactions with code P, acquired, shares > 0, price > 0. Keep the issuer ticker,
     reporting owner CIK, filing date, value. Same User-Agent, under 10 requests per second.
  2. Next trading day t (the first trading day after the filing date) at 15:45 ET: for each ticker with a purchase
     filing usable on t, count distinct purchasing owners whose filings became usable on t-20..t. Keep tickers with
     exactly 2. Keep tier M: 20-day median dollar volume through t-1 between $5M and $50M and close of t-1 > $5.
  3. Order (shadow only): market-on-close buy (closing auction) on t, notional = 10% of shadow equity (whole shares,
     round down), only if fewer than 10 names are held; ties by larger total insider purchase value. Record the
     official closing price of t as the fill.
  4. Exit: market-on-close sell on the 60th trading day after t (t+60), whatever happened in between (no stop, no
     re-entry: a new cluster in a held name is ignored). Record the official close.
  5. Idle cash: BIL (or SPY), adjusted when names enter or leave. Compare monthly with SPY and with this study's
     backtest expectation. Log: filing accession numbers, nb, ADV, prior close, fill, exit, cost estimate.

Results (run 2026-11-09, prices to 2026-11-05):
  A. Replication exact: 1674 / 475 / 232 events, Sharpe 1.11 / 1.20 / 1.66, OLS-hedged 1.55 / 1.42 / 1.33.
  B. New data (filed 2026-04-01..2026-09-30): 153 events, 97 with a complete 60-day hold (entries 2026-04-02..08-12,
     5 months, 69 tickers). Mean per event net of costs: raw -26 bp (t -0.9), vs SPY -395 bp (t -2.3), vs IWM
     -269 bp (t -2.4), vs tier-M equal weight -585 bp; median vs SPY -4 bp, hit 49%. Without MNTS (3 events,
     -47% to -78% after a real 85M-share spike) -192 bp vs SPY (t -1.5); first event per ticker only -92 bp (t -1.3).
     Study 15: +199..+639 bp per period vs SPY. Opposite sign and outside the earlier range; 2026Q1 (in study 15's
     data) was already -972 bp. The study 15 book on 2026-04..11: Sharpe 0.06 raw, -0.75 OLS-hedged.
  C. $10k account, 10% per name, max 10, idle BIL: CAGR -2.1% (deterministic tie order), 200 random tie orders
     median +2.4% (5-95%: -2.8% to +9.5%), max drawdown about -50%; SPY buy and hold +14.2%, max DD -34%, $24,814.
     Other sizes range from -3% to +25% CAGR: which ~3 of ~31 monthly events fill the slots is mostly luck.
     Whole-share rounding leaves about 1% of a $1,000 position unspent (negligible).
  D. 31 events a month on average (median 27, 3 to 136); a 10-slot account enters about 3.3 a month.
  Verdict: Reject. The new-period result has the wrong sign (t about -2), and even in-sample a $10k long-only
  version trails SPY with a deeper drawdown because the edge was small relative to mid-cap single-name risk.

Outputs: results/study211_frozen.json, results/study211_replication.csv, results/study211_newdata_events.csv,
results/study211_newdata.csv, results/study211_account.csv, results/study211_account_years.csv,
results/study211_events_month.csv, results/study211_events_year.csv, results/study211_account_mc.csv.
    python src/study211_insider_clusters.py fetch     # SEC downloads only
    python src/study211_insider_clusters.py           # everything else
"""
import os
import sys

os.environ.setdefault("STUDY15_CACHE", os.environ.get("STUDY211_CACHE", "/tmp/study211_sec"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import study15_insider as S15
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, RES, DATA
import bt

QUARTERS = [f"{y}q{k}" for y in range(2019, 2027) for k in range(1, 5) if "2019q4" <= f"{y}q{k}" <= "2026q3"]
URL_OLD = S15.URL
URL_NEW = "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{q}_form345.zip"
NEW_FROM = "2026-03-31"          # new data = filed after this date
HOLD = 60


def load_raw():
    parts = []
    for q in QUARTERS:
        S15.URL = URL_NEW if q >= "2026q2" else URL_OLD
        x = S15.fetch_quarter(q)
        if x is not None:
            parts.append(x)
    S15.URL = URL_OLD
    return pd.concat(parts, ignore_index=True).dropna(subset=["filed", "owner"])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    d = load_raw()
    print(d.groupby(d.filed.dt.to_period("Q")).code.value_counts().unstack())
    sys.exit(0)


# ------------------------------------------------------------------ helpers
def tstat_month(day, x):
    s = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(day)).dropna()
    mo = s.groupby(s.index.to_period("M")).mean()
    return mo.mean() / mo.std() * np.sqrt(len(mo)) if len(mo) > 2 else np.nan


def returns_panel(P, tick):
    """Daily total return close(t-1) -> close(t): Alpaca d1 official-close return + Yahoo dividend part; Yahoo
    adjusted return where d1 is missing or disagrees with the Yahoo price return by more than 10 points."""
    import alpaca_data as A
    days = P["c"].index
    Ry = P["c"][tick].astype("float64").pct_change(fill_method=None)
    Rpx = P["rawc"][tick].astype("float64").pct_change(fill_method=None)
    d = A.read("d1", start="2019-12", tickers=tick, columns=["ts", "ticker", "c"])
    d["date"] = d.ts.dt.normalize()
    C = d.drop_duplicates(["date", "ticker"], keep="last").pivot(index="date", columns="ticker", values="c")
    C = C.reindex(index=days, columns=tick).astype("float64")
    Rd = C.pct_change(fill_method=None)
    div = (Ry - Rpx).fillna(0.0).clip(-0.02, 0.2)
    ok = Rd.notna() & ((Rd - Rpx).abs() <= 0.10)
    ok = ok | (Rd.notna() & Rpx.isna())
    R = Ry.where(~ok, Rd + div)
    print("d1 share of event-ticker return days:", round(float(ok.values.sum() / max(R.notna().values.sum(), 1)), 3),
          "| d1 days replaced by Yahoo:", int((Rd.notna() & ~ok).values.sum()), flush=True)
    return R


def fwd_ret(R, di, jj, h):
    """Compounded return over closes di -> di+h (NaN daily returns count as 0); NaN if di+h is past the data."""
    G = np.cumprod(1 + np.nan_to_num(R), axis=0)
    out = np.full(len(di), np.nan)
    ok = di + h < len(G)
    out[ok] = G[di[ok] + h, jj[ok]] / G[di[ok], jj[ok]] - 1
    return out


def simulate(ev, R, PX, C, idle_r, idle_cost, frac=None, fixed=None, maxn=10, start_cap=1e4, seed=None, D0=0):
    """Daily $10k long-only account. ev: events with di, j (column in R), val. R, PX, C: arrays (days x tickers);
    idle_r: daily return of idle cash. Positions enter at the close of t, exit at the close of t+HOLD."""
    T = R.shape[0]
    e = ev.copy()
    if seed is None:
        e = e.sort_values(["di", "val"], ascending=[True, False])
    else:
        e = e.assign(k=np.random.default_rng(seed).random(len(e))).sort_values(["di", "k"])
    by_day = {d: g for d, g in e.groupby("di")}
    cash, pos = start_cap, []                    # pos: [j, value, exit_di]
    eq = np.full(T, np.nan)
    names = np.zeros(T)
    inv = np.zeros(T)
    taken, skipped_full, skipped_cash = [], 0, 0
    trades_cost = 0.0
    for i in range(D0, T):
        if i > D0:
            cash *= 1 + idle_r[i]
            for p in pos:
                r = R[i, p[0]]
                p[1] *= 1 + (0.0 if np.isnan(r) else r)
        # exits at the close
        keep = []
        for p in pos:
            if p[2] == i:
                c = C[i, p[0]] / 1e4 if not np.isnan(C[i, p[0]]) else 0.005
                cash += p[1] * (1 - c) * (1 - idle_cost)
                trades_cost += p[1] * (c + idle_cost)
            else:
                keep.append(p)
        pos = keep
        equity = cash + sum(p[1] for p in pos)
        if i in by_day:
            held = {p[0] for p in pos}
            for _, x in by_day[i].iterrows():
                if x.j in held:
                    continue
                if len(pos) >= maxn:
                    skipped_full += 1
                    continue
                px = PX[i, x.j]
                if not (px > 0):
                    continue
                target = fixed if fixed is not None else frac * equity
                sh = np.floor(target / px)
                c = C[i, x.j] / 1e4 if not np.isnan(C[i, x.j]) else 0.005
                need = sh * px * (1 + c) * (1 + idle_cost)
                if sh < 1 or need > cash:
                    skipped_cash += 1
                    continue
                cash -= need
                trades_cost += sh * px * (c + idle_cost)
                pos.append([x.j, sh * px, i + HOLD])
                held.add(x.j)
                taken.append((i, x.j))
        eq[i] = cash + sum(p[1] for p in pos)
        names[i] = len(pos)
        inv[i] = (eq[i] - cash) / eq[i]
    return eq, names, inv, taken, skipped_full, skipped_cash, trades_cost


def eq_stats(eq, idx, rf=None):
    s = pd.Series(eq, index=idx).dropna()
    r = s.pct_change().dropna()
    yrs = len(r) / 252
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    return dict(final=s.iloc[-1], cagr=(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, vol=r.std() * np.sqrt(252),
                sharpe=r.mean() / r.std() * np.sqrt(252), sharpe_xs_tbill=ex.mean() / ex.std() * np.sqrt(252),
                maxdd=(s / s.cummax() - 1).min())


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    raw = load_raw()
    f = S15.build_filings(raw)
    del raw
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    f, ev = S15.build_events(f, days, cols, P)
    fmin = f.groupby(["ticker", "di"]).filed.min().rename("filed_min").reset_index()
    ev = ev.merge(fmin, on=["ticker", "di"])
    ev = ev[ev.day >= "2020-01-01"].copy()
    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
    px = traded_close(P)[cols].shift(1).fillna(P["rawc"][cols].shift(1))
    ci = {t: i for i, t in enumerate(cols)}
    j, di = ev.ticker.map(ci).values, ev.di.values
    a, p = adv.values[di, j], px.values[di, j]
    ev["adv"], ev["px"] = a, p
    ev["tier"] = np.select([(a > 5e7) & (p > 5), (a > 5e6) & (a <= 5e7) & (p > 5), (a > 1e6) & (a <= 5e6) & (p > 2)],
                           ["L", "M", "S"], "none")
    TM = (adv > 5e6) & (adv <= 5e7) & (px > 5)
    del adv, px
    rule = ev[(ev.tier == "M") & (ev.nb == 2)].copy().reset_index(drop=True)
    rule["new"] = rule.filed_min > NEW_FROM
    print("rule events", len(rule), "new", int(rule.new.sum()), "last event day", rule.day.max().date(), flush=True)
    last = days[-1]

    tick = sorted(set(rule.ticker) | {"SPY", "IWM"})
    tpos = {t: i for i, t in enumerate(tick)}
    R = returns_panel(P, tick)
    Ry = P["c"][tick].astype("float64").pct_change(fill_method=None)
    cost = (exec_cost_bps(P, "auction")[tick] + 2.5).astype("float64")
    cA = P["c"][cols]
    ewM = (cA / cA.shift(1) - 1).where(TM).mean(axis=1)
    del cA, TM
    rule["j"] = rule.ticker.map(tpos).astype(int)
    di, jj = rule.di.values, rule.j.values
    Rv = R.values
    rule["cost_rt"] = 2 * cost.values[di, jj] / 1e4
    rule["r60"] = fwd_ret(Rv, di, jj, HOLD)
    rule["r60_yahoo"] = fwd_ret(Ry.values, di, jj, HOLD)
    rule["spy60"] = fwd_ret(Rv, di, np.full(len(di), tpos["SPY"]), HOLD)
    rule["iwm60"] = fwd_ret(Rv, di, np.full(len(di), tpos["IWM"]), HOLD)
    rule["ew60"] = fwd_ret(ewM.values[:, None], di, np.zeros(len(di), int), HOLD)
    rule["net"] = rule.r60 - rule.cost_rt
    rule["x_spy"] = rule.net - rule.spy60
    rule["x_iwm"] = rule.net - rule.iwm60
    rule["x_ew"] = rule.net - rule.ew60
    rule["complete"] = rule.di + HOLD < len(days)
    # mark-to-latest for incomplete holds
    G = np.cumprod(1 + np.nan_to_num(Rv), axis=0)
    rule["r_to_date"] = G[-1, jj] / G[di, jj] - 1
    rule["spy_to_date"] = G[-1, tpos["SPY"]] / G[di, tpos["SPY"]] - 1

    # ---------------- A. replication of study 15 (Yahoo returns, same book)
    sel = slice("2020-01-02", "2026-03-31")
    D = days[days >= "2020-01-02"]
    spyY = Ry["SPY"].shift(-1)       # study 15 convention: row t holds close t -> t+1
    iwmY = Ry["IWM"].shift(-1)

    def book(evs, Rret):
        A_ = np.zeros((len(D), len(tick)), dtype="float32")
        np.add.at(A_, (D.searchsorted(evs.day.values), evs.j.values), 1.0)
        M = pd.DataFrame(np.minimum(A_, 1), index=D, columns=tick)
        W1 = M.div(M.sum(axis=1).clip(lower=10), axis=0)
        Wh = bt.hold_k_days(W1, HOLD)
        r = bt.run(Wh, Rret.shift(-1).reindex(D), cost.reindex(D), roundtrip=False)
        r["inv"] = Wh.sum(axis=1)
        return r
    rep_rows = []
    s15 = pd.read_csv(f"{RES}/study15_variants.csv")
    s15 = s15[(s15.tier == "M") & (s15.variant == "nb2") & (s15.hold == 60)].set_index("period")
    ev15 = rule[rule.day <= "2026-03-31"]
    bY = book(ev15, Ry)
    for per, a_, b_ in S15.PERIODS[:4]:
        x = bY.loc[a_:b_]
        st = ann_stats(x.net)
        ms = S15.mkt_stats(x.net, spyY, iwmY, ewM.shift(-1))
        rep_rows.append(dict(period=per, n_events=int(((ev15.day >= a_) & (ev15.day <= b_)).sum()),
                             n_events_s15=s15.loc[per, "n_events"], sharpe=st["sharpe"], sharpe_s15=s15.loc[per, "sharpe"],
                             hsharpe_spy_iwm=ms.get("hsharpe_spy_iwm"), hsharpe_s15=s15.loc[per, "hsharpe_spy_iwm"],
                             ann_ret=st["ann_ret"], ann_ret_s15=s15.loc[per, "ann_ret"], invested=x.inv.mean()))
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(f"{RES}/study211_replication.csv", index=False)
    print(rep.round(3).to_string(), flush=True)

    # ---------------- B. new data
    nw = rule[rule.new].copy()
    nw.drop(columns=["j"]).to_csv(f"{RES}/study211_newdata_events.csv", index=False)
    rows = []
    for nm, sub in [("new, complete 60d", nw[nw.complete]), ("new, all (marked to latest if incomplete)", nw),
                    ("study 15 period 2020-01..2026-03", rule[(rule.day <= "2026-03-31") & rule.complete]),
                    ("2025-07..2026-03 (study 15 last period)", rule[(rule.day >= "2025-07-01") & (rule.day <= "2026-03-31")])]:
        if nm.startswith("new, all"):
            net = np.where(sub.complete, sub.net, sub.r_to_date - sub.cost_rt)
            xs = np.where(sub.complete, sub.x_spy, sub.r_to_date - sub.cost_rt - sub.spy_to_date)
            rows.append(dict(sample=nm, n=len(sub), months=sub.day.dt.to_period("M").nunique(),
                             first=sub.day.min().date(), last=sub.day.max().date(),
                             net_raw_bps=1e4 * np.mean(net), t_raw=tstat_month(sub.day, net),
                             x_spy_bps=1e4 * np.mean(xs), t_spy=tstat_month(sub.day, xs)))
            continue
        r = dict(sample=nm, n=len(sub), months=sub.day.dt.to_period("M").nunique(), first=sub.day.min().date(),
                 last=sub.day.max().date())
        for k, c in [("net_raw", "net"), ("x_spy", "x_spy"), ("x_iwm", "x_iwm"), ("x_ew", "x_ew")]:
            r[f"{k}_bps"] = 1e4 * sub[c].mean()
            r[f"t_{k.split('_')[-1]}"] = tstat_month(sub.day, sub[c])
        r["median_x_spy_bps"] = 1e4 * sub.x_spy.median()
        r["hit_x_spy"] = (sub.x_spy > 0).mean()
        r["net_raw_yahoo_bps"] = 1e4 * (sub.r60_yahoo - sub.cost_rt).mean()
        rows.append(r)
    # study-15 book on days after 2026-03-31 (all cohorts) and the new-events-only book
    bR = book(rule, R)
    bN = book(nw, R)
    spyR, iwmR = R["SPY"].shift(-1), R["IWM"].shift(-1)
    lastbook = days[-2]
    for nm, b in [("book, all cohorts, 2026-04-01..latest", bR), ("book, new events only, 2026-04-01..latest", bN)]:
        x = b.loc["2026-04-01":lastbook]
        st = ann_stats(x.net)
        ms = S15.mkt_stats(x.net, spyR, iwmR, ewM.shift(-1))
        rows.append(dict(sample=nm, n=len(x), net_raw_bps=1e4 * x.net.mean(), sharpe_raw=st["sharpe"],
                         ann_ret=st["ann_ret"], maxdd=st["maxdd"], hsharpe_spy=ms.get("hsharpe_spy"),
                         hsharpe_spy_iwm=ms.get("hsharpe_spy_iwm"), beta_spy=ms.get("beta_spy"),
                         alpha2_ann=ms.get("alpha2_ann"), invested=x.inv.mean(), names=x.n.mean()))
    newres = pd.DataFrame(rows)
    newres.to_csv(f"{RES}/study211_newdata.csv", index=False)
    print(newres.round(4).T.to_string(), flush=True)
    nc = nw[nw.complete]
    print("new complete events by entry month:\n",
          nc.groupby(nc.day.dt.to_period("M")).agg(n=("x_spy", "size"), x_spy=("x_spy", "mean"),
                                                     raw=("net", "mean")).round(4).to_string(), flush=True)

    # ---------------- C. $10k account
    i0 = int(days.searchsorted(pd.Timestamp("2020-01-02")))
    PXt = traded_close(P)[tick].fillna(P["rawc"][tick]).astype("float64").values
    Cv = cost.values
    etf = pd.read_parquet(os.path.join(DATA, "local", "etf_long", "etf_daily.parquet"))
    adj = etf.pivot(index="date", columns="ticker", values="adj")
    adj.index = pd.to_datetime(adj.index).normalize()
    bil = adj["BIL"].reindex(days).ffill().pct_change(fill_method=None).fillna(0).values
    spy_r = np.nan_to_num(R["SPY"].values)
    idles = {"cash 0%": (np.zeros(len(days)), 0.0), "BIL": (bil, 3.5e-4), "SPY": (spy_r, 3.5e-4)}
    designs = [("10% of equity, max 10", dict(frac=0.10, maxn=10)), ("15%, max 6", dict(frac=0.15, maxn=6)),
               ("20%, max 5", dict(frac=0.20, maxn=5)), ("fixed $1,000, max 10", dict(fixed=1000, maxn=10)),
               ("fixed $2,000, max 5", dict(fixed=2000, maxn=5))]
    evs = rule[["di", "j", "val", "day"]]
    D2 = days[i0:]
    rf = pd.Series(bil, index=days)
    acc_rows, yr_rows, taken_prim = [], {}, None
    for dn, kw in designs:
        for iname, (ir, icost) in idles.items():
            eq, names, inv, taken, sf, sc, tc = simulate(evs, Rv, PXt, Cv, ir, icost, D0=i0, **kw)
            st = eq_stats(eq[i0:], D2, rf)
            acc_rows.append(dict(design=dn, idle=iname, **st, avg_names=names[i0:].mean(), invested=inv[i0:].mean(),
                                 trades=len(taken), skipped_full=sf, skipped_cash=sc, costs_paid=tc,
                                 start=D2[0].date(), end=D2[-1].date()))
            s = pd.Series(eq[i0:], index=D2)
            yr_rows[(dn, iname)] = s.resample("YE").last().pct_change().fillna(s.resample("YE").last().iloc[0] / 1e4 - 1)
            yr_rows[(dn, iname)].index = yr_rows[(dn, iname)].index.year
            if dn.startswith("10%") and iname == "BIL":
                taken_prim = taken
    for nm, rr in [("SPY buy and hold", spy_r), ("BIL buy and hold", bil)]:
        eq = 1e4 * np.cumprod(1 + rr[i0:]) / (1 + rr[i0])
        st = eq_stats(eq, D2, rf)
        acc_rows.append(dict(design=nm, idle="", **st, avg_names=np.nan, invested=1.0, start=D2[0].date(),
                             end=D2[-1].date()))
        s = pd.Series(eq, index=D2)
        yr_rows[(nm, "")] = s.resample("YE").last().pct_change().fillna(s.resample("YE").last().iloc[0] / 1e4 - 1)
        yr_rows[(nm, "")].index = yr_rows[(nm, "")].index.year
    # study-15 book (fractional, 1/max(n,10) per cohort), long-only, idle at 0% and in BIL
    bk = bR.net.loc[D2[0]:days[-2]]
    for iname, ir in [("cash 0%", 0 * bil), ("BIL", bil)]:
        irs = pd.Series(ir, index=days).shift(-1).reindex(bk.index).fillna(0)
        r = bk + (1 - bR.inv.reindex(bk.index)) * irs
        eq = 1e4 * (1 + r).cumprod()
        st = eq_stats(eq.values, eq.index, rf)
        acc_rows.append(dict(design="study 15 book (fractional, ~15% invested)", idle=iname, **st,
                             avg_names=bR.n.loc[bk.index].mean(), invested=bR.inv.loc[bk.index].mean(),
                             start=bk.index[0].date(), end=bk.index[-1].date()))
        yr = (1 + r).groupby(r.index.year).prod() - 1
        yr_rows[("study 15 book", iname)] = yr
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(f"{RES}/study211_account.csv", index=False)
    print(acc.round(3).to_string(), flush=True)
    yrs = pd.DataFrame({f"{a} | {b}": v for (a, b), v in yr_rows.items()}).T
    yrs.to_csv(f"{RES}/study211_account_years.csv")
    print(yrs.round(3).to_string(), flush=True)
    # random same-day priority (luck of which events fill the slots)
    mc = []
    for iname in ["BIL", "SPY"]:
        ir, icost = idles[iname]
        for sd in range(200):
            eq, names, inv, taken, sf, sc, tc = simulate(evs, Rv, PXt, Cv, ir, icost, D0=i0, frac=0.10, maxn=10, seed=sd)
            mc.append(dict(idle=iname, seed=sd, **eq_stats(eq[i0:], D2, rf)))
    mc = pd.DataFrame(mc)
    mc.to_csv(f"{RES}/study211_account_mc.csv", index=False)
    print(mc.groupby("idle")[["final", "cagr", "sharpe", "maxdd"]].describe(percentiles=[.05, .5, .95]).T.round(3)
          .to_string(), flush=True)

    # ---------------- D. events per month and by year
    m = rule.groupby(rule.day.dt.to_period("M")).agg(events=("ticker", "size"), tickers=("ticker", "nunique"))
    tk = pd.Series(1, index=pd.PeriodIndex([days[i].to_period("M") for i, _ in taken_prim])).groupby(level=0).sum()
    m["taken_10pct_account"] = tk.reindex(m.index).fillna(0).astype(int)
    m.index = m.index.astype(str)
    m.to_csv(f"{RES}/study211_events_month.csv")
    print("events per month: mean", round(m.events.mean(), 1), "median", m.events.median(), "min", m.events.min(),
          "max", m.events.max(), "| account entries per month", round(m.taken_10pct_account.mean(), 2), flush=True)
    rc = rule[rule.complete]
    by = rc.groupby(rc.day.dt.year).agg(events=("x_spy", "size"), net_raw_bps=("net", "mean"),
                                        x_spy_bps=("x_spy", "mean"), x_iwm_bps=("x_iwm", "mean"),
                                        x_ew_bps=("x_ew", "mean"), hit_x_spy=("x_spy", lambda s: (s > 0).mean()))
    for c in ["net_raw_bps", "x_spy_bps", "x_iwm_bps", "x_ew_bps"]:
        by[c] *= 1e4
    by["t_x_spy"] = [tstat_month(g.day, g.x_spy) for _, g in rc.groupby(rc.day.dt.year)]
    by.to_csv(f"{RES}/study211_events_year.csv")
    print(by.round(2).to_string(), flush=True)


if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1] != "fetch"):
    main()
