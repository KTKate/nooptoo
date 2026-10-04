"""Study 79: buy insider purchases in the extended session right after an after-close Form 4 filing.

Study 75 found that after-close Form 4 open-market BUY filings are followed by a +70..160 bp overnight jump (close d
-> open d+1, not tradable at the auctions because the filing comes after the closing auction) and +14..20 bp more in
the next day session. Here the jump is priced where it could actually be caught: a limit buy at the SIP NBBO ask
5, 15 or 30 minutes after the EDGAR acceptance time.

Data:
  Form 4 structured data sets 2019q3..2026q1 (2026q2+ not published yet) and EDGAR acceptance timestamps
  (data.sec.gov/submissions of every issuer with an open-market purchase), downloaded with study 75's code into
  $STUDY75_CACHE (step `fetch_accept` here fetches the acceptance times of buy issuers only).
  Events: buy filings (code P, acquired) accepted on a trading day d between 16:00 and 19:45 ET; one event per
  (ticker, d) with t0 = first such acceptance; subsets from study 75's build_insider (CEO/CFO, value, first buy by
  the owner in 12 months), plus buy value / 20-day median dollar volume. Only events in the liquid universe of
  study 75 (U5: traded price > $5, 20d median $ volume > $5M before d), 2020-01 .. 2026-03 (Alpaca request budget).
  Quotes: last SIP NBBO quote (bid > 0, ask > 0, timestamp >= 16:00 of d) at or before T = t0 + 5 / 15 / 30 minutes
  (see quotes_one), trades from t0 - 30 min to t0 + 30 min (count, volume, last trade before t0 and T).
Returns (traded basis -> adjusted basis with the day-d factor C_adj[d] / traded_close[d], so splits at d+1 do not
matter): entry = ask_T x (1 + 2.5 bp fees); exits = next official open (P['o'][d+1], adjusted) with the auction cost
(core.exec_cost_bps 'auction' + 2.5 bp), or the next close (same cost). Excess vs the liquid universe (traded price
> $5, 20d median $ volume > $5M) equal-weight mean of close d -> open d+1 (or close d -> close d+1), and vs SPY.
Reference columns: the same trade from the closing auction of d (impossible: filing is after the close) and from the
last trade before t0. Events with |log(ask / traded close d)| > 0.5 or a crossed quote are dropped as bad prints.
t-stats clustered by date. Book: each night at most 10 names (first accepted first), 10% each, cash otherwise.
Output: results/study79_insider_afterhours.csv (sections coverage, events, book).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = os.path.dirname(SRC)
CACHE = os.environ.get("STUDY75_CACHE", "/tmp/study75")
QFN = os.path.join(CACHE, "study79_quotes.parquet")
OUT = os.path.join(ROOT, "results", "study79_insider_afterhours.csv")
UA = {"User-Agent": "nooptoo research incarnadins@gmail.com"}      # sec.gov only
DELAYS = (5, 15, 30)


# ------------------------------------------------------------------ fetch: acceptance times of buy issuers
def fetch_accept():
    import threading
    import requests
    from concurrent.futures import ThreadPoolExecutor
    lock, last = threading.Lock(), [0.0]

    def sec_get(url):
        for k in range(5):
            with lock:
                w = 0.16 - (time.time() - last[0])                  # <= 6.25 requests/second over all threads
                if w > 0:
                    time.sleep(w)
                last[0] = time.time()
            try:
                r = requests.get(url, headers=UA, timeout=60)
            except requests.RequestException as e:
                print("err", url, e, flush=True)
                time.sleep(2 + 3 * k)
                continue
            if r.status_code in (429, 503):
                time.sleep(5 + 10 * k)
                continue
            return r
        return None

    d4 = os.path.join(CACHE, "form4")
    f4 = pd.concat([pd.read_parquet(os.path.join(d4, f)) for f in sorted(os.listdir(d4))], ignore_index=True)
    tick = set(pd.read_pickle(os.path.join(ROOT, "data", "panel.pkl"))["c"].columns)
    b = f4[(f4.code == "P") & f4.ticker.isin(tick)].dropna(subset=["issuer"])
    iss = [int(c) for c in b.groupby("issuer").size().sort_values(ascending=False).index]
    fn = os.path.join(CACHE, "accept", "accept.parquet")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    parts, done = [], set()
    if os.path.exists(fn):
        old = pd.read_parquet(fn)
        parts.append(old)
        done = set(old.issuer.unique())
    todo = [c for c in iss if c not in done]
    print("issuers", len(iss), "todo", len(todo), flush=True)

    def one(c):
        r = sec_get(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
        if r is None or r.status_code != 200:
            return c, None, []
        j = r.json()
        extra = []
        for f in j["filings"].get("files", []):
            if f.get("filingTo", "9999") >= "2019-07-01":
                r2 = sec_get("https://data.sec.gov/submissions/" + f["name"])
                if r2 is not None and r2.status_code == 200:
                    extra.append(r2.json())
        return c, j, extra

    batch = []

    def flush():
        if batch:
            parts.append(pd.concat(batch, ignore_index=True))
            batch.clear()
            pd.concat(parts, ignore_index=True).to_parquet(fn, index=False)

    with ThreadPoolExecutor(3) as pool:
        for k, (c, j, extra) in enumerate(pool.map(one, todo)):
            if j is None:
                continue
            rows = []
            for blk in [j["filings"]["recent"]] + extra:
                df = pd.DataFrame({"acc": blk["accessionNumber"], "form": blk["form"],
                                   "accepted": blk["acceptanceDateTime"]})
                rows.append(df[df.form.isin(["4", "4/A"])])
            x = pd.concat(rows, ignore_index=True)
            x["issuer"] = c
            batch.append(x)
            if k % 200 == 0:
                flush()
                print(k, len(todo), flush=True)
    flush()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch_accept":
    fetch_accept()
    sys.exit(0)

# ================================================================== events (study 75 code)
import study75_disclosed_trades as S75                             # noqa: E402 (loads the panel)
import alpaca_data as A                                            # noqa: E402
from core import ann_stats                                         # noqa: E402
H = S75.H
DAYS, ND, COLS = S75.DAYS, S75.ND, S75.COLS
C, O, PX = S75.C, S75.O, S75.PX                                    # adjusted close/open, traded close (study41)
COSTV, U5, ADVL = S75.COSTV, S75.U5, S75.ADVL
# traded close of d: Alpaca raw close (2023-12+), else Yahoo raw close with later splits multiplied back
# (s6162_common.traded_price; study 41's traded_px misses splits inside 2020-23)
from s6162_common import traded_price                              # noqa: E402
TCL = H.TC.fillna(traded_price(H.P)[H.cols]).astype("float64").values
PERIODS = (("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31"))


def build_events():
    f = S75.build_insider()
    b = f[(f.code == "P") & (f.timing == "post") & f.ts.notna()].copy()
    hm = b.ts.dt.hour * 60 + b.ts.dt.minute
    b = b[(hm >= 16 * 60) & (hm < 19 * 60 + 45)]
    b = b[(b.fdi >= 21) & (b.fdi < ND - 2)]
    # earlier buy filings of the same ticker that were public before t0 on the same day (info, not a filter)
    g = b.sort_values("ts").groupby(["ci", "fdi"])
    ev = g.agg(ticker=("ticker", "first"), t0=("ts", "min"), n_filings=("acc", "nunique"),
               n_owners=("owner", "nunique"), value=("value", "sum"),
               ceo=("cls", lambda s: (s == "ceo_cfo").any()), first12=("first12", "any"),
               plan_all=("plan", "all"), tdate=("tdate", "min")).reset_index()
    i, c = ev.fdi.values, ev.ci.values
    ev["date"] = DAYS[i]
    ev["u5"] = U5[i, c]                                             # liquid universe known before day d
    ev["adv"] = ADVL[i, c]
    ev["value_adv"] = ev.value / ev.adv
    ev["tclose"] = TCL[i, c]                                        # traded close of d
    return ev


# ------------------------------------------------------------------ quotes and trades
def _utc(ts):
    return ts.tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _quotes(s, a, b, limit, sort):
    try:
        j = A.get("quotes", dict(symbols=s, start=_utc(a), end=_utc(b), limit=limit, feed="sip", sort=sort))
    except RuntimeError as e:
        print("quote err", s, b, str(e)[:100], flush=True)
        return None, False
    q = (j.get("quotes") or {}).get(s, [])
    return [x for x in q if x.get("bp", 0) > 0 and x.get("ap", 0) > 0], j.get("next_page_token") is not None


def quotes_one(sym, t0, d):
    """NBBO standing at t0 + 5/15/30 min (latest quote with bid, ask > 0 at or before T, not before 16:00 of d) and
    trades t0 - 30 min .. t0 + 30 min. Requests: quotes t0 - 2 min .. t0 + 30 min ascending (one page of 10,000);
    a descending lookup from 16:00 only when no quote precedes T in that window or the page was truncated."""
    s = sym.replace("-", ".")
    out = dict(ticker=sym, t0=t0)
    t16 = pd.Timestamp(d) + pd.Timedelta(hours=16)
    q, trunc = _quotes(s, max(t16, t0 - pd.Timedelta(minutes=2)), t0 + pd.Timedelta(minutes=30), 10000, "asc")
    out["q_n"], out["q_trunc"] = (len(q) if q is not None else np.nan), trunc
    qt = pd.to_datetime([x["t"] for x in q], utc=True, format="ISO8601").tz_convert("America/New_York") \
        .tz_localize(None) if q else pd.DatetimeIndex([])
    for k in DELAYS:
        T = t0 + pd.Timedelta(minutes=k)
        best = None
        if q:
            m = np.where(qt <= T)[0]
            if len(m) and not (trunc and qt[-1] < T):
                best = q[m[-1]]
        if best is None:
            q2, _ = _quotes(s, t16, T, 50, "desc")
            best = q2[0] if q2 else None
        if best is not None:
            out.update({f"bid{k}": best["bp"], f"ask{k}": best["ap"], f"asz{k}": best.get("as"), f"qt{k}": best["t"]})
    try:
        j = A.get("trades", dict(symbols=s, start=_utc(t0 - pd.Timedelta(minutes=30)),
                                 end=_utc(t0 + pd.Timedelta(minutes=30)), limit=10000, feed="sip"))
        tr = (j.get("trades") or {}).get(s, [])
        out["tr_trunc"] = j.get("next_page_token") is not None
    except RuntimeError as e:
        print("trade err", sym, t0, str(e)[:100], flush=True)
        tr = None
    if tr is not None:
        if tr:
            td = pd.DataFrame(tr)
            td["t"] = pd.to_datetime(td.t, utc=True, format="ISO8601").dt.tz_convert("America/New_York").dt.tz_localize(None)
            pre = td[td.t <= t0]
            out["last_pre"] = pre.p.iloc[-1] if len(pre) else np.nan
            for k in DELAYS:
                w = td[(td.t > t0) & (td.t <= t0 + pd.Timedelta(minutes=k))]
                upto = td[td.t <= t0 + pd.Timedelta(minutes=k)]
                out[f"ntr{k}"] = len(w)
                out[f"vol{k}"] = float(w.s.sum())
                out[f"dvol{k}"] = float((w.s * w.p).sum())
                out[f"last{k}"] = upto.p.iloc[-1] if len(upto) else np.nan
        else:
            for k in DELAYS:
                out[f"ntr{k}"] = 0
                out[f"dvol{k}"] = 0.0
    return out


def fetch_quotes(ev):
    from concurrent.futures import ThreadPoolExecutor
    have = pd.read_parquet(QFN) if os.path.exists(QFN) else pd.DataFrame(columns=["ticker", "t0"])
    key = set(zip(have.ticker, pd.to_datetime(have.t0)))
    todo = ev[[(t, x) not in key for t, x in zip(ev.ticker, ev.t0)]].sort_values("t0", ascending=False)
    print("quote events todo", len(todo), "of", len(ev), flush=True)
    rows, parts = [], [have] if len(have) else []
    with ThreadPoolExecutor(int(os.environ.get("STUDY79_THREADS", "2"))) as ex:
        for n, r in enumerate(ex.map(lambda a: quotes_one(*a), zip(todo.ticker, todo.t0, todo.date))):
            rows.append(r)
            if n % 250 == 249:
                parts.append(pd.DataFrame(rows))
                rows = []
                pd.concat(parts, ignore_index=True).to_parquet(QFN, index=False)
                print("quotes", n + 1, len(todo), flush=True)
    if rows:
        parts.append(pd.DataFrame(rows))
    q = pd.concat(parts, ignore_index=True)
    q.to_parquet(QFN, index=False)
    return q


# ------------------------------------------------------------------ analysis
def cl_t(x, g):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(g)[ok]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    m = x.mean()
    e = pd.Series(x - m).groupby(g).sum().values
    G = len(e)
    se = np.sqrt((e ** 2).sum() * G / max(G - 1, 1)) / n
    return m, (m / se if se > 0 else np.nan), n


def returns(ev):
    i, c = ev.fdi.values, ev.ci.values
    fac = C[i, c] / ev.tclose.values                                # traded -> adjusted (day d)
    on_b = S75.bench_vec("on", 0)[i + 1]                            # universe close d -> open d+1
    cc_b = S75.bench_vec("c", 1)[i]                                 # universe close d -> close d+1
    spy = H.SPY.values
    spy_on = H.P["o"]["SPY"].values[i + 1] / spy[i] - 1
    spy_cc = spy[i + 1] / spy[i] - 1
    o1, c1 = O[i + 1, c], C[i + 1, c]
    ev["cost_x"] = np.nan_to_num(COSTV[i, c], nan=0.002)            # auction exit cost incl. 2.5 bp
    ev["r_close_on"] = o1 / C[i, c] - 1                            # reference: from the closing auction (impossible)
    ev["r_close_cc"] = c1 / C[i, c] - 1
    lpr = ev.last_pre.values
    lp = np.where(np.abs(np.log(lpr / ev.tclose.values)) < 0.5, lpr * fac, np.nan)     # drop bad prints
    ev["r_lastpre_on"] = o1 / lp - 1
    ev["on_b"], ev["cc_b"], ev["spy_on"], ev["spy_cc"] = on_b, cc_b, spy_on, spy_cc
    for k in DELAYS:
        a, b_ = ev[f"ask{k}"].values, ev[f"bid{k}"].values
        ok = np.isfinite(a) & (a > 0) & (b_ > 0) & (a >= b_) & (np.abs(np.log(a / ev.tclose.values)) < 0.5)
        ev[f"ok{k}"] = ok
        ev[f"spread{k}_bp"] = np.where(ok, 1e4 * (a - b_) / ((a + b_) / 2), np.nan)
        ev[f"ask_vs_close{k}_bp"] = np.where(ok, 1e4 * (a / ev.tclose.values - 1), np.nan)
        ea = np.where(ok, a * fac * (1 + 2.5e-4), np.nan)              # entry, adjusted basis, incl. fees
        ev[f"g_on{k}"] = o1 / ea - 1                                     # gross of the exit cost
        ev[f"n_on{k}"] = ev[f"g_on{k}"] - ev.cost_x
        ev[f"n_cc{k}"] = c1 / ea - 1 - ev.cost_x
        ev[f"x_on{k}"] = ev[f"n_on{k}"] - on_b
        ev[f"x_cc{k}"] = ev[f"n_cc{k}"] - cc_b
        ev[f"s_on{k}"] = ev[f"n_on{k}"] - spy_on
        ev[f"s_cc{k}"] = ev[f"n_cc{k}"] - spy_cc
        # same events, close-auction reference
        ev[f"ref_on{k}"] = np.where(ok, ev.r_close_on - on_b, np.nan)
        # optimistic entries (fill not guaranteed): at the quote midpoint, or at the last trade at or before T
        em = np.where(ok, (a + b_) / 2 * fac * (1 + 2.5e-4), np.nan)
        lt = ev[f"last{k}"].values
        el = np.where(ok & (lt > 0) & (np.abs(np.log(lt / ev.tclose.values)) < 0.5), lt * fac * (1 + 2.5e-4), np.nan)
        ev[f"mid_on{k}"] = o1 / em - 1 - ev.cost_x - on_b
        ev[f"last_on{k}"] = o1 / el - 1 - ev.cost_x - on_b
        ev[f"mid_cc{k}"] = c1 / em - 1 - ev.cost_x - cc_b
    return ev


def subsets(ev):
    return {"all": ev, "ceo_cfo": ev[ev.ceo], "value_gt_250k": ev[ev.value > 2.5e5],
            "buy_gt_5pct_adv": ev[ev.value_adv > 0.05], "first12": ev[ev.first12],
            "ceo_and_250k": ev[ev.ceo & (ev.value > 2.5e5)], "u5_liquid": ev[ev.u5],
            "u5_value_gt_250k": ev[ev.u5 & (ev.value > 2.5e5)],
            "accepted_16_17": ev[ev.t0.dt.hour == 16], "accepted_17_1945": ev[ev.t0.dt.hour >= 17]}


def book(ev, col, max_names=10, w=0.10):
    """Each night: up to max_names events by acceptance time, w each; daily series on the exit day."""
    x = ev[np.isfinite(ev[col])].sort_values("t0")
    x = x.groupby("fdi").head(max_names)
    r = (w * x[col]).groupby(x.fdi + 1).sum()
    n = x.groupby(x.fdi + 1).size()
    df = pd.DataFrame({"net": r, "n": n}).reindex(range(ND)).fillna(0.0)
    df.index = DAYS
    return df


def main():
    ev = build_events()
    n_all = len(ev)
    ev = ev[ev.u5 & (ev.date >= "2020-01-01")].reset_index(drop=True)
    print("all after-close buy events", n_all, "-> liquid (U5: traded price > $5, ADV > $5M) 2020+:", len(ev))
    print("after-close buy events (16:00-19:45):", len(ev), "u5", int(ev.u5.sum()),
          "by year", ev.date.dt.year.value_counts().sort_index().to_dict(), flush=True)
    q = pd.read_parquet(QFN) if os.environ.get("STUDY79_NOFETCH") else fetch_quotes(ev)   # NOFETCH: dry run
    q["t0"] = pd.to_datetime(q.t0)
    ev = ev.merge(q, on=["ticker", "t0"], how="inner" if os.environ.get("STUDY79_NOFETCH") else "left")
    ev = returns(ev)
    rows = []
    for p, a, b in PERIODS + (("all", "2020-01-01", "2026-12-31"),):
        z = ev[(ev.date >= a) & (ev.date <= b)]
        for k in DELAYS:
            rows.append(dict(section="coverage", subset="all", period=p, delay=k, n=len(z),
                             quoted_share=z[f"ok{k}"].mean(), traded_share=(z[f"ntr{k}"] > 0).mean(),
                             med_spread_bp=z[f"spread{k}_bp"].median(), mean_spread_bp=z[f"spread{k}_bp"].mean(),
                             med_ask_vs_close_bp=z[f"ask_vs_close{k}_bp"].median(),
                             med_dvol_usd=z[f"dvol{k}"].median(), med_ask_size=z[f"asz{k}"].median()))
    for name, s in subsets(ev).items():
        for p, a, b in PERIODS + (("all", "2020-01-01", "2026-12-31"),):
            z = s[(s.date >= a) & (s.date <= b)]
            if len(z) < 20:
                continue
            m0, t0_, n0 = cl_t(z.r_close_on - z.on_b, z.date)
            ml, tl, nl = cl_t(z.r_lastpre_on - z.on_b, z.date)
            for k, filt in [(k, f) for k in DELAYS for f in ("any", "le50", "le100")]:
                y = z[z[f"ok{k}"]]
                if filt != "any":                                   # quoted spread at T (known when ordering)
                    y = y[y[f"spread{k}_bp"] <= float(filt[2:])]
                if len(y) < 10:
                    continue
                d = dict(section="events", subset=name, period=p, delay=k, spread_filter=filt, n=len(z),
                         n_quoted=len(y),
                         quoted_share=len(y) / len(z), med_spread_bp=y[f"spread{k}_bp"].median(),
                         med_ask_vs_close_bp=y[f"ask_vs_close{k}_bp"].median(),
                         ref_close_on_xs_bp=1e4 * m0, ref_close_on_t=t0_,
                         ref_lastpre_on_xs_bp=1e4 * ml, ref_lastpre_on_t=tl,
                         ref_close_on_xs_same_bp=1e4 * cl_t(y[f"ref_on{k}"], y.date)[0])
                for col in ("g_on", "n_on", "x_on", "s_on", "n_cc", "x_cc", "s_cc", "mid_on", "last_on", "mid_cc"):
                    m, t, n = cl_t(y[f"{col}{k}"], y.date)
                    d[f"{col}_bp"], d[f"{col}_t"] = 1e4 * m, t
                d["hit_n_on"] = (y[f"n_on{k}"] > 0).mean()
                rows.append(d)
    # books
    spy_r = H.SPY.pct_change()
    for name in ("all", "value_gt_250k", "ceo_cfo", "first12", "u5_liquid"):
        s = subsets(ev)[name]
        for k, filt in [(k, f) for k in DELAYS for f in ("any", "le50")]:
            sk = s if filt == "any" else s[s[f"spread{k}_bp"] <= 50]
            for col in ("n_on", "n_cc"):
                df = book(sk, f"{col}{k}")
                for p, a, b in PERIODS:
                    b = min(pd.Timestamp(b), s.date.max() + pd.Timedelta(days=5))
                    x = df.loc[a:b]
                    st = ann_stats(x.net)
                    sp = ann_stats(spy_r.loc[x.index])
                    rows.append(dict(section="book", subset=name, period=p, delay=k, spread_filter=filt, exit=col[2:], days=len(x),
                                     nights_active=int((x.n > 0).sum()), avg_names=x.n[x.n > 0].mean(),
                                     ann_ret=st["ann_ret"], sharpe=st["sharpe"], maxdd=st["maxdd"],
                                     t=st["tstat"], spy_ann=sp["ann_ret"], spy_sharpe=sp["sharpe"],
                                     corr_spy=float(x.net.corr(spy_r.loc[x.index]))))
    R = pd.DataFrame(rows)
    R.to_csv(OUT if not os.environ.get("STUDY79_NOFETCH") else OUT.replace(".csv", "_dry.csv").replace(
        os.path.join(ROOT, "results"), CACHE), index=False)
    ev.to_parquet(os.path.join(CACHE, "study79_events.parquet"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 40)
    print(R[R.section == "coverage"].dropna(axis=1, how="all").round(3).to_string())
    e = R[R.section == "events"]
    print(e[["subset", "period", "delay", "spread_filter", "n", "n_quoted", "med_spread_bp", "med_ask_vs_close_bp",
             "ref_close_on_xs_bp", "ref_close_on_xs_same_bp", "g_on_bp", "n_on_bp", "n_on_t", "x_on_bp", "x_on_t",
             "n_cc_bp", "n_cc_t", "x_cc_bp", "x_cc_t", "mid_on_bp", "mid_on_t", "last_on_bp"]].round(1).to_string())
    bk = R[R.section == "book"]
    print(bk[["subset", "period", "delay", "spread_filter", "exit", "nights_active", "avg_names", "ann_ret", "sharpe", "maxdd",
              "spy_ann", "spy_sharpe", "corr_spy"]].round(3).to_string())
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
