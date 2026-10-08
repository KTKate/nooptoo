"""Study 78: do point-in-time insider trading features (Form 4) add to the day-session long model (study 60) and to
the overnight blend (studies 23/33/69)?

Question: study 75 found that insider buys disclosed after the close earn +14..20 bp in the next session (t 3-5),
too thin to trade alone after costs. Do the same filings help as inputs: (1) in the study-60 day-session LightGBM
(open-to-close, inputs known before 09:28), (2) as a filter or tilt on the overnight blend score (study 69 base:
2 x ensemble rank + jump-minus-drop rank, top 10), (3) as extra inputs to one pooled LightGBM overnight rank model
(the 'pooled' member of src/ensemble.py)?

Data:
  form4   SEC Form 3/4/5 structured data sets 2019q3..2026q3 (2026q2+ under /files/datastandardsinnovation/...),
          open-market P (acquired) and S (disposed) of non-derivative stock, Form 4 and 4/A; same parsing as study 75
          (owner, relationship, officer title, 10b5-1 checkbox AFF10B5ONE from 2023-04 and footnote 10b5-1 mentions).
  accept  EDGAR acceptance timestamps (data.sec.gov/submissions of every issuer whose ticker is in the panel, incl.
          older pages back to 2019-07), joined by accession number. UTC in the JSON, converted to ET.
  Both cached in $STUDY75_CACHE (default /tmp/study75; not in the repo):  python src/study78_insider_inputs.py fetch
SEC requests: User-Agent 'nooptoo research incarnadins@gmail.com', at most 6.5 requests/second.

Point-in-time rule: a filing is usable at cutoff hh:mm of trading day t if it was accepted before hh:mm on t or
on any earlier calendar day. Filings without a timestamp (issuer list miss): usable from the next trading day after
the filing date (filing date d means acceptance before 17:30 of d). Cutoffs: 09:28 for the day-session model,
15:45 for the overnight decision. Windows of n trading days count the filings that became usable on the last n
trading days up to t (a filing accepted after the 15:45 cutoff of day t-1 and before 09:28 of t is usable at both
cutoffs of t).
Cleaning: shares > 0, price > 0, duplicate rows (owner, ticker, code, trade date, shares, price) dropped, value
< $2bn per transaction, transaction price within a factor of 3 of the panel close of the trade date (else dropped,
catches unit errors and wrong tickers), filings whose issuer ticker is not a panel column dropped.
Features per (stock, day), at each cutoff:
  ib_any1/5/20    any open-market purchase (P) usable in the last 1 / 5 / 20 trading days (0/1)
  ib_val20        purchase value in 20 days / 20-day median dollar volume (known before the day), clipped at 5
  ib_n20          distinct buyers in 20 days
  ib_off20        an officer (incl. CEO/CFO) bought in 20 days; ib_dir20: a director bought in 20 days
  ib_clu10        >= 2 distinct buyers in 10 days (cluster)
  is_val20        sales value in 20 days / ADV, excluding sales flagged 10b5-1 (checkbox or footnote), clipped at 5
  is_n20          distinct non-plan sellers in 20 days
Design:
  Day session: study 60's 'all' inputs and walk-forward (LightGBM params, quarters 2024Q3..2026Q3, training from
  2024-01, 5-day embargo, top 10 / 20, price > $5, ADV > $5M, cost column of study 25), with and without the
  insider features at 09:28. The study-60 frame is rebuilt with its code and cached in $STUDY75_CACHE.
  Overnight (a): blend score S of study 69 (lines 22-46), base top 10. Variants: drop names with a 20-day non-plan
  sale value > 0.5 x ADV; restrict to / tilt toward names with a recent purchase (S + w x ib_any5, w = 0.05, 0.10,
  0.20, and S + 0.1 x ib_clu10); avoid recent heavy sellers plus tilt. Features at 15:45 of the decision day.
  Overnight (b): the pooled member of src/ensemble.py (same params at num_threads=2, 53 study-3 inputs of
  data/ml_frame.parquet, rank target of close -> next open, training from 2020-01, 10-day embargo, quarters
  2024Q1..2026Q3) trained with and without the insider features (15:45). Scored on the close-based frame (not the
  15:45 substitution of study 23), so the absolute Sharpe is not the live one; only the paired difference matters.
  Top 10, price > $5, ADV > $5M, closing-auction buy, opening-auction sell, auction cost + 2.5 bp per side.
  Paired t: daily net return difference (variant - base), mean / (sd / sqrt(n)).
Live feed check (step `live`): a few requests to the EDGAR getcurrent Atom feed (type 4) and the daily index;
reports the delay between acceptance and visibility.
Output: results/study78_insider_inputs.csv (section, variant, k, period, sharpe, net_bp, diff_bp, t_diff, ...).
"""
import io
import json
import os
import re
import sys
import threading
import time
import zipfile

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
ROOT = os.path.dirname(SRC)
CACHE = os.environ.get("STUDY75_CACHE", "/tmp/study75")
OUT = os.path.join(ROOT, "results", "study78_insider_inputs.csv")
UA = {"User-Agent": "nooptoo research incarnadins@gmail.com"}      # owner's approved SEC contact (sec.gov only)
F345 = "https://www.sec.gov/files/{path}/data/insider-transactions-data-sets/{q}_form345.zip"
PLAN_RE = re.compile(r"10b5-?1|10b-5-1|10b5\s1", re.I)
NOT_RE = re.compile(r"not\s+(?:made\s+|effected\s+|executed\s+|entered\s+into\s+|done\s+)?(?:pursuant|under|in\s+accordance)"
                    r"[^.]{0,40}(?:10b5-?1|10b-5-1)", re.I)
CEO_RE = re.compile(r"\bCEO\b|\bCFO\b|chief\s+exec|chief\s+financ|principal\s+exec|principal\s+financ|"
                    r"\bC\.E\.O\b|\bC\.F\.O\b", re.I)
_last, _lock = [0.0], threading.Lock()


def sec_get(url, timeout=300):
    """GET from sec.gov at <= 6.5 requests/second over all threads, with retries."""
    import requests
    for k in range(5):
        with _lock:
            w = 0.155 - (time.time() - _last[0])
            if w > 0:
                time.sleep(w)
            _last[0] = time.time()
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
        except requests.RequestException as e:
            print("err", url, e, flush=True)
            time.sleep(2 + 3 * k)
            continue
        if r.status_code in (429, 503):
            time.sleep(5 + 10 * k)
            continue
        return r
    return None


def cpath(*a):
    p = os.path.join(CACHE, *a)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ------------------------------------------------------------------ fetch: Form 4 data sets (study 75 parsing)
def fetch_form4_quarter(q):
    fn = cpath("form4", f"{q}.parquet")
    if os.path.exists(fn):
        return
    r = None
    for path in ("structureddata", "datastandardsinnovation"):
        r = sec_get(F345.format(path=path, q=q), timeout=900)
        if r is not None and r.status_code == 200:
            break
    if r is None or r.status_code != 200:
        print(q, "http", None if r is None else r.status_code, flush=True)
        return
    z = zipfile.ZipFile(io.BytesIO(r.content))

    def rd(f, cols):
        return pd.read_csv(z.open(f), sep="\t", usecols=lambda c: c in cols, dtype=str, on_bad_lines="skip", quoting=3)

    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
    sub = sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])]
    tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                   "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS"])
    tr = tr[tr.TRANS_CODE.isin(["P", "S"])]
    ow = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    ow = ow.drop_duplicates("ACCESSION_NUMBER")
    d = tr.merge(sub, on="ACCESSION_NUMBER").merge(ow, on="ACCESSION_NUMBER", how="left")
    accs = set(d.ACCESSION_NUMBER)
    fnt = rd("FOOTNOTES.tsv", ["ACCESSION_NUMBER", "FOOTNOTE_TXT"])
    fnt = fnt[fnt.ACCESSION_NUMBER.isin(accs) & fnt.FOOTNOTE_TXT.fillna("").str.contains(PLAN_RE)]
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
        "owner": d.RPTOWNERCIK, "role": d.RPTOWNER_RELATIONSHIP, "title": d.RPTOWNER_TITLE,
        "aff": d.AFF10B5ONE.map({"1": 1.0, "0": 0.0, "true": 1.0, "false": 0.0}), "fn_plan": d.fn_plan})
    out.to_parquet(fn, index=False)
    print(q, len(out), flush=True)


def load_form4():
    d4 = os.path.join(CACHE, "form4")
    parts = [pd.read_parquet(os.path.join(d4, f)) for f in sorted(os.listdir(d4))]
    return pd.concat(parts, ignore_index=True).dropna(subset=["filed", "ticker"])


# ------------------------------------------------------------------ fetch: acceptance times (all panel issuers)
def fetch_accept():
    from concurrent.futures import ThreadPoolExecutor
    tick = set(pd.read_pickle(os.path.join(ROOT, "data", "panel.pkl"))["c"].columns)
    f4 = load_form4()
    f4 = f4[f4.ticker.isin(tick)].dropna(subset=["issuer"])
    nb = f4[f4.code == "P"].groupby("issuer").size()
    iss = [int(c) for c in nb.sort_values(ascending=False).index]          # issuers with purchases first
    iss += sorted(set(f4.issuer.astype(int)) - set(iss))
    fn = cpath("accept", "accept.parquet")
    parts, done = [], set()
    if os.path.exists(fn):
        old = pd.read_parquet(fn)
        parts.append(old)
        done = set(old.issuer.unique())
    todo = [c for c in iss if c not in done]
    print("issuers", len(iss), "todo", len(todo), flush=True)

    def one(c):
        r = sec_get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", timeout=60)
        if r is None or r.status_code != 200:
            return c, None, []
        j = r.json()
        extra = []
        for f in j["filings"].get("files", []):
            if f.get("filingTo", "9999") >= "2019-07-01":
                r2 = sec_get("https://data.sec.gov/submissions/" + f["name"], timeout=60)
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


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    for y in range(2019, 2027):
        for k in range(1, 5):
            if "2019q3" <= f"{y}q{k}" <= "2026q3":
                fetch_form4_quarter(f"{y}q{k}")
    fetch_accept()
    sys.exit(0)


# ------------------------------------------------------------------ live feed check (a few requests)
def live_check():
    """EDGAR getcurrent Atom feed (Form 4) and the daily index: how soon after acceptance a filing is visible."""
    from email.utils import parsedate_to_datetime
    rows = []
    atom = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include"
            "&start=0&count=100&output=atom")
    for rep in range(3):
        t0 = pd.Timestamp.now(tz="America/New_York")
        r = sec_get(atom, timeout=60)
        if r is None or r.status_code != 200:
            rows.append(dict(section="live", variant="getcurrent_atom", note=f"http {None if r is None else r.status_code}"))
            continue
        upd = pd.to_datetime(re.findall(r"<updated>([^<]+)</updated>", r.text)[1:], utc=True)
        upd = upd.tz_convert("America/New_York")
        accs = re.findall(r"accession-number=(\d{10}-\d{2}-\d{6})", r.text)
        srv = r.headers.get("Date")
        now = pd.Timestamp(parsedate_to_datetime(srv)).tz_convert("America/New_York") if srv else t0
        lag = (now - upd.max()).total_seconds() if len(upd) else np.nan
        rows.append(dict(section="live", variant="getcurrent_atom", period=str(now.floor("s").tz_localize(None)),
                         n=len(upd), latest_accept=str(upd.max().tz_localize(None)) if len(upd) else "",
                         lag_sec=lag, span_min=(upd.max() - upd.min()).total_seconds() / 60 if len(upd) else np.nan,
                         note=f"{len(set(accs))} accessions; lag = server time - newest acceptance in feed"))
        print(rows[-1], flush=True)
        if rep < 2:
            time.sleep(60)
    d = pd.Timestamp.now(tz="America/New_York")
    q = (d.month - 1) // 3 + 1
    for day in (d - pd.Timedelta(days=1), d):
        u = f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx"
        r = sec_get(u, timeout=60)
        st = None if r is None else r.status_code
        n4 = len(re.findall(r"^4\s", r.text, re.M)) if st == 200 else 0
        lm = r.headers.get("Last-Modified", "") if r is not None else ""
        rows.append(dict(section="live", variant="daily_index", period=f"{day:%Y-%m-%d}", n=n4,
                         note=f"http {st}; last-modified {lm}"))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "live":
    lv = live_check()
    lv.to_csv(cpath("parts", "live.csv"), index=False)
    sys.exit(0)


# ================================================================== analysis
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402

P = load_panel()
cols = stock_cols(P)
COLS = pd.Index(cols)
DAYS = P["c"].index
ND, NC = len(DAYS), len(COLS)
ADV = P["dv"][cols].rolling(20).median().shift(1)                  # 20d median $ volume known before day t
PERIODS_DAY = (("2024H2-25H1", "2024-07", "2025-06"), ("2025H2-26", "2025-07", "2026-09"))
PERIODS_NIGHT = (("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"))
FEATS = ["ib_any1", "ib_any5", "ib_any20", "ib_val20", "ib_n20", "ib_off20", "ib_dir20", "ib_clu10", "is_val20",
         "is_n20"]
CHECKS = []


def traded_px():
    """Traded close where known (2023-12+), else Yahoo close x (traded/Yahoo ratio on the first traded-close day)."""
    tc = traded_close(P)[cols].astype("float64")
    raw = P["rawc"][cols].astype("float64")
    ratio = (tc / raw.where(raw > 0)).bfill().iloc[0].fillna(1.0)
    return tc.fillna(raw * ratio)


def build_filings():
    raw = load_form4()
    n0 = len(raw)
    raw["tdate"] = raw.tdate.fillna(raw.filed)
    raw.loc[raw.tdate > raw.filed, "tdate"] = raw.filed
    raw = raw[((raw.code == "P") & (raw.ad.fillna("A") == "A")) | ((raw.code == "S") & (raw.ad.fillna("D") == "D"))]
    raw = raw[(raw.shares > 0) & (raw.price > 0)]
    raw = raw.drop_duplicates(["owner", "ticker", "code", "tdate", "shares", "price"])
    raw["value"] = raw.shares * raw.price
    n1 = len(raw)
    big = raw[raw.value >= 2e9]
    raw = raw[(raw.value < 2e9) & raw.ticker.isin(set(COLS))].copy()
    n2 = len(raw)
    # price sanity: transaction price vs the traded close of the trade date (unit errors, wrong tickers)
    tpx = traded_px().values
    ti = np.clip(DAYS.searchsorted(raw.tdate.values, "right") - 1, 0, ND - 1)
    ref = tpx[ti, COLS.get_indexer(raw.ticker)]
    ratio = raw.price.values / ref
    bad = np.isfinite(ratio) & ((ratio > 3) | (ratio < 1 / 3))
    CHECKS.append(dict(section="check", variant="rows", note=f"form4 rows {n0}; P/S valid dedup {n1}; >= $2bn dropped "
                       f"{len(big)}; in panel {n2}; price off by > 3x vs traded close dropped {int(bad.sum())} "
                       f"(P {int((bad & (raw.code.values == 'P')).sum())}); no reference price {int((~np.isfinite(ratio)).sum())}"))
    raw = raw[~bad]
    f = raw.sort_values(["acc", "tdate"]).groupby(["acc", "code"]).agg(
        filed=("filed", "first"), tdate=("tdate", "min"), ticker=("ticker", "first"), issuer=("issuer", "first"),
        owner=("owner", "first"), role=("role", "first"), title=("title", "first"), value=("value", "sum"),
        aff=("aff", "max"), fn_plan=("fn_plan", "max")).reset_index()
    r, t = f.role.fillna(""), f.title.fillna("")
    f["off"] = (t.str.contains(CEO_RE) | r.str.contains("Officer"))
    f["dir"] = r.str.contains("Director") & ~f.off
    f["plan"] = (f.aff == 1) | f.fn_plan.astype(bool)
    a = pd.read_parquet(os.path.join(CACHE, "accept", "accept.parquet")).drop_duplicates("acc")
    a["ts"] = pd.to_datetime(a.accepted, errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    f = f.merge(a[["acc", "ts"]], on="acc", how="left")
    f["ci"] = COLS.get_indexer(f.ticker)
    CHECKS.append(dict(section="check", variant="timestamps",
                       note=f"filings {len(f)} (P {int((f.code == 'P').sum())}); with acceptance time "
                            f"{f.ts.notna().mean():.3f} (P {f.ts[f.code == 'P'].notna().mean():.3f}); filing date "
                            f"= acceptance date {(f.ts.dt.normalize() == f.filed)[f.ts.notna()].mean():.3f}"))
    return f


def avail_index(f, cut_min):
    """First trading day index whose cutoff (minutes after midnight ET) comes after the filing became public."""
    ts = f.ts
    d = ts.dt.normalize().values
    mins = (ts.dt.hour * 60 + ts.dt.minute).values
    di = DAYS.searchsorted(d)
    is_td = (di < ND) & (DAYS[np.minimum(di, ND - 1)].values == d)
    a_ts = np.where(is_td & (mins >= cut_min), di + 1, di)
    a_nots = DAYS.searchsorted(f.filed.values, "right")               # no timestamp: next trading day after filing date
    return np.where(ts.notna().values, a_ts, a_nots)


def roll_sum(a, w):
    """Sum over the last w rows including the current one."""
    cs = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
    out = cs[1:].copy()
    out[w:] -= cs[1:-w]
    return out


def daily(ai, ci, v=None):
    a = np.zeros((ND, NC))
    m = ai < ND
    np.add.at(a, (ai[m], ci[m]), 1.0 if v is None else v[m])
    return a


def distinct(ai, ci, own, w):
    """Number of distinct owners with an event in the last w trading days (interval union per owner)."""
    df = pd.DataFrame(dict(ai=ai, ci=ci, o=own)).drop_duplicates().sort_values(["ci", "o", "ai"])
    nxt = df.groupby(["ci", "o"]).ai.shift(-1).fillna(10 ** 9).values
    end = np.minimum(np.minimum(df.ai.values + w, nxt), ND).astype(int)
    m = df.ai.values < ND
    d = np.zeros((ND + 1, NC))
    np.add.at(d, (df.ai.values[m], df.ci.values[m]), 1)
    np.add.at(d, (end[m], df.ci.values[m]), -1)
    return np.cumsum(d, axis=0)[:ND]


def features(f, cut_min):
    ai = avail_index(f, cut_min)
    adv = ADV.values
    out = {}
    b = (f.code == "P").values
    s = ((f.code == "S") & ~f.plan).values
    ci = f.ci.values
    cb = daily(ai[b], ci[b])
    for w in (1, 5, 20):
        out[f"ib_any{w}"] = (roll_sum(cb, w) > 0).astype("float32")
    vb = roll_sum(daily(ai[b], ci[b], f.value.values[b]), 20)
    out["ib_val20"] = np.clip(vb / adv, 0, 5).astype("float32")
    out["ib_n20"] = distinct(ai[b], ci[b], f.owner.values[b], 20).astype("float32")
    for k in ("off", "dir"):
        m = b & f[k].values
        out[f"ib_{k}20"] = (roll_sum(daily(ai[m], ci[m]), 20) > 0).astype("float32")
    out["ib_clu10"] = (distinct(ai[b], ci[b], f.owner.values[b], 10) >= 2).astype("float32")
    vs = roll_sum(daily(ai[s], ci[s], f.value.values[s]), 20)
    out["is_val20"] = np.clip(vs / adv, 0, 5).astype("float32")
    out["is_n20"] = distinct(ai[s], ci[s], f.owner.values[s], 20).astype("float32")
    return {k: pd.DataFrame(v, index=DAYS, columns=COLS) for k, v in out.items()}


def get_features():
    fn = cpath("study78_feats.pkl")
    if os.path.exists(fn):
        return pd.read_pickle(fn)
    f = build_filings()
    # data checks: largest purchases vs ADV, purchase counts per year
    b = f[f.code == "P"].copy()
    b["fi"] = DAYS.searchsorted(b.filed.values)
    b["adv"] = ADV.values[np.minimum(b.fi.values, ND - 1), b.ci.values]
    b["v_adv"] = b.value / b.adv
    top = b.sort_values("value", ascending=False).head(8)
    for r in top.itertuples():
        CHECKS.append(dict(section="check", variant="largest_buys", period=str(r.filed.date()),
                           note=f"{r.ticker} ${r.value / 1e6:.1f}m, {r.v_adv:.2f} x ADV, role {r.role}"))
    CHECKS.append(dict(section="check", variant="buy_filings_per_year",
                       note=str(b.groupby(b.filed.dt.year).size().to_dict())))
    CHECKS.append(dict(section="check", variant="buy_value_over_adv",
                       note=f"median {b.v_adv.median():.4f}, p99 {b.v_adv.quantile(.99):.2f}, > 5 x ADV "
                            f"{int((b.v_adv > 5).sum())} filings (clipped at 5)"))
    hrs = (b.ts.dt.hour + b.ts.dt.minute / 60)
    CHECKS.append(dict(section="check", variant="buy_accept_hour_ET",
                       note=f"before 09:28 {(hrs < 9 + 28 / 60).mean():.3f}, 09:28-15:45 "
                            f"{((hrs >= 9 + 28 / 60) & (hrs < 15.75)).mean():.3f}, after 15:45 {(hrs >= 15.75).mean():.3f}, "
                            f"no time {hrs.isna().mean():.3f}"))
    F = {"0928": features(f, 9 * 60 + 28), "1545": features(f, 15 * 60 + 45)}
    for k, v in F.items():
        CHECKS.append(dict(section="check", variant=f"coverage_{k}",
                           note="; ".join(f"{n} mean {float(np.nanmean(p.loc['2024':].values)):.4f}" for n, p in v.items())))
    F["checks"] = CHECKS
    pd.to_pickle(F, fn)
    return F


def paired(a, b):
    """Paired t of the daily difference a - b (aligned on common days)."""
    d = (a - b).dropna()
    if len(d) < 10 or d.std() == 0:
        return np.nan, np.nan
    return 1e4 * d.mean(), d.mean() / (d.std() / np.sqrt(len(d)))


def save_part(name, df):
    df.to_csv(cpath("parts", f"{name}.csv"), index=False)
    parts = [pd.read_csv(os.path.join(CACHE, "parts", x)) for x in sorted(os.listdir(os.path.join(CACHE, "parts")))]
    order = ["checks", "live", "uni", "day", "blend", "model"]
    parts = sorted(zip(sorted(os.listdir(os.path.join(CACHE, "parts"))), parts),
                   key=lambda z: order.index(z[0][:-4]) if z[0][:-4] in order else 99)
    pd.concat([p for _, p in parts], ignore_index=True).to_csv(OUT, index=False)


# ------------------------------------------------------------------ day session (study 60 'all')
LGB_DAY = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
               bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2)


def frame60():
    """Study 60's frame with the 'all' inputs (same code), cached."""
    fn = cpath("study60_frame.parquet")
    if os.path.exists(fn):
        x = pd.read_parquet(fn)
        return x, [c for c in x.columns if c.startswith("f_")]
    import pickle
    import analyst_features as AF
    x = pd.read_pickle(f"{DATA}/local/m5pre/_study25_features.pkl")
    x["date"] = pd.to_datetime(x.date)
    x = x[x.R.notna() & x.ticker.isin(cols) & (x.R.abs() < 1)].reset_index(drop=True)
    pre = [c for c in x.columns if c not in ("ticker", "date", "R", "cost", "o930", "ygap", "prevc", "pm_first",
                                              "pm_last", "pm_hi", "pm_lo", "px")]
    prev = pd.Series(DAYS[:-1], index=DAYS[1:])
    x["prev"] = x.date.map(prev)
    M = pd.read_parquet(f"{DATA}/ml_frame.parquet", filters=[("date", ">=", pd.Timestamp("2023-12-01"))])
    close_f = [c for c in M.columns if not c.startswith("y_") and c not in ("date", "ticker")]
    M = M.reset_index(drop=True) if "date" in M.columns else M.reset_index()
    M = M[["date", "ticker"] + close_f].rename(columns={"date": "prev", **{c: "c_" + c for c in close_f}})
    x = x.merge(M, on=["prev", "ticker"], how="left")
    del M
    new = []

    def put(name, panel, key):
        s = panel.reindex(columns=cols).stack(future_stack=True)
        x[name] = s.reindex(pd.MultiIndex.from_arrays([x[key], x.ticker])).values.astype("float32")
        new.append(name)

    A = AF.load()
    for k in ["pt_n", "pt_up", "pt_dn", "pt_net20", "pt_gap", "pt_firms", "pt_gap_chg20", "ev_guid_up20",
              "ev_guid_dn20", "ev_buyback20", "ev_exec20"]:
        put("an_" + k, A[k], "prev")
    del A
    FP = pickle.load(open(f"{DATA}/local/fundamentals_panels.pkl", "rb"))
    stale = FP["days_since_filing"] > 200
    for k in ["market_cap", "ev_sales", "pe_ttm", "fcf_yield", "sbc_to_revenue", "op_margin", "revenue_growth_yoy"]:
        put("fu_" + k, FP[k].where(~stale).rank(axis=1, pct=True), "date")
    del FP, stale
    feat = pre + ["c_" + c for c in close_f] + new
    keep = x[["date", "ticker", "R", "cost", "px", "adv20"]].copy()
    for c in feat:
        keep["f_" + c] = x[c].astype("float32")
    keep.to_parquet(fn, index=False)
    return keep, ["f_" + c for c in feat]


def run_day(F):
    x, base = frame60()
    print("day frame", x.shape, flush=True)
    for k, p in F["0928"].items():
        s = p.stack(future_stack=True)
        x["f_" + k] = s.reindex(pd.MultiIndex.from_arrays([x.date, x.ticker])).values.astype("float32")
    ins = ["f_" + k for k in FEATS]
    x["y"] = x.groupby("date").R.rank(pct=True) - 0.5
    sets = {"all": base, "all+insider": base + ins}
    import lightgbm as lgb
    fn = cpath("study78_day_pred.parquet")
    if os.path.exists(fn):
        pr = pd.read_parquet(fn)
        for k in sets:
            x["p_" + k] = pr["p_" + k].values
    else:
        for k in sets:
            x["p_" + k] = np.nan
        for q in pd.period_range("2024Q3", "2026Q3", freq="Q"):
            cut = DAYS[max(0, DAYS.searchsorted(q.start_time) - 6)]
            tr = (x.date < cut).values
            te = ((x.date >= q.start_time) & (x.date <= q.end_time)).values
            if te.sum() == 0:
                continue
            for k, f in sets.items():
                m = lgb.train(LGB_DAY, lgb.Dataset(x.loc[tr, f], x.y[tr]), num_boost_round=300)
                x.loc[te, "p_" + k] = m.predict(x.loc[te, f])
                if k == "all+insider":
                    imp = pd.Series(m.feature_importance("gain"), index=f)
                    print(q, "insider gain share", round(imp[ins].sum() / imp.sum(), 4), flush=True)
            print(q, flush=True)
        x[["p_" + k for k in sets]].to_parquet(fn)
    liq = (x.px > 5) & (x.adv20 > 5e6)
    rows, nets = [], {}
    for k in sets:
        for n in [10, 20]:
            sel = x[liq & x["p_" + k].notna()].sort_values("p_" + k, ascending=False).groupby("date").head(n)
            g = sel.groupby("date")
            nets[(k, n)] = g.R.mean() - 2 * g.cost.mean() / 1e4
            nets[(k, n, "ins")] = g.apply(lambda d: (d.f_ib_any20 > 0).mean(), include_groups=False)
    for k in sets:
        for n in [10, 20]:
            net = nets[(k, n)]
            for p, a, b in PERIODS_DAY:
                dbp, t = paired(net.loc[a:b], nets[("all", n)].loc[a:b])
                rows.append(dict(section="day_session", variant=k, k=n, period=p,
                                 sharpe=ann_stats(net.loc[a:b])["sharpe"], net_bp=1e4 * net.loc[a:b].mean(),
                                 days=len(net.loc[a:b]), diff_bp=dbp, t_diff=t,
                                 note=f"share of picks with a buy in 20d {nets[(k, n, 'ins')].loc[a:b].mean():.3f}"))
    # univariate: open-to-close excess (vs the liquid-universe daily mean) of rows with a purchase usable at 09:28
    u = x[liq & (x.date >= "2024-07-01")].copy()
    u["ex"] = u.R - u.groupby("date").R.transform("mean")
    for k in ["ib_any1", "ib_any5", "ib_clu10"]:
        for p, a, b in PERIODS_DAY:
            s = u[(u["f_" + k] > 0) & (u.date >= a) & (u.date <= pd.Period(b).end_time)]
            dm = s.groupby("date").ex.mean()
            rows.append(dict(section="day_univariate", variant=k, period=p, n=len(s), days=len(dm),
                             net_bp=1e4 * dm.mean(), t_diff=dm.mean() / (dm.std() / np.sqrt(len(dm))),
                             note="mean open-to-close excess (gross) of rows with the flag, date-clustered t"))
    for k in ["is_val20"]:
        for p, a, b in PERIODS_DAY:
            s = u[(u["f_" + k] > 0.5) & (u.date >= a) & (u.date <= pd.Period(b).end_time)]
            dm = s.groupby("date").ex.mean()
            rows.append(dict(section="day_univariate", variant="is_val20>0.5", period=p, n=len(s), days=len(dm),
                             net_bp=1e4 * dm.mean(), t_diff=dm.mean() / (dm.std() / np.sqrt(len(dm))),
                             note="mean open-to-close excess (gross) of rows with the flag, date-clustered t"))
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(), flush=True)
    save_part("day", df)


# ------------------------------------------------------------------ overnight (a): filter / tilt on the blend
def night_inputs():
    import bt
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < DAYS[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex_like(S)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S)
    return bt, S, R, C


def run_blend(F):
    bt, S, R, C = night_inputs()
    G = {k: v.reindex_like(S) for k, v in F["1545"].items()}
    base_w = bt.select_topk(S, S.notna(), 10)
    base = bt.run(base_w, R, C).net
    seller = G["is_val20"] > 0.5
    variants = {
        "base: blend top 10": S,
        "filter: no 20d non-plan sales > 0.5 x ADV": S.where(~seller),
        "filter: no 20d non-plan sales > 0.2 x ADV": S.where(~(G["is_val20"] > 0.2)),
        "only names with a buy in 20d (top 10 of them)": S.where(G["ib_any20"] > 0),
        "tilt: S + 0.05 x buy in 5d": S + 0.05 * G["ib_any5"],
        "tilt: S + 0.10 x buy in 5d": S + 0.10 * G["ib_any5"],
        "tilt: S + 0.20 x buy in 5d": S + 0.20 * G["ib_any5"],
        "tilt: S + 0.10 x buy in 1d": S + 0.10 * G["ib_any1"],
        "tilt: S + 0.10 x cluster 10d": S + 0.10 * G["ib_clu10"],
        "tilt: S + 0.10 x officer buy 20d": S + 0.10 * G["ib_off20"],
        "tilt 0.10 x buy 5d + no sellers > 0.5 x ADV": (S + 0.10 * G["ib_any5"]).where(~seller),
    }
    rows = []
    for name, s in variants.items():
        W = bt.select_topk(s, s.notna() & S.notna(), 10)
        net = bt.run(W, R, C).net
        chg = ((W > 0) != (base_w > 0)).sum(axis=1) / 2
        for p, a, b in PERIODS_NIGHT:
            st = ann_stats(net.loc[a:b])
            dbp, t = paired(net.loc[a:b], base.loc[a:b])
            rows.append(dict(section="night_blend", variant=name, k=10, period=p, sharpe=st["sharpe"],
                             net_bp=1e4 * net.loc[a:b].mean(), days=st["n"], diff_bp=dbp, t_diff=t,
                             note=f"mean names changed vs base per night {chg.loc[a:b].mean():.2f}; days with a "
                                  f"change {(chg.loc[a:b] > 0).mean():.3f}"))
    # univariate overnight: close -> next open excess of universe rows with the flag at 15:45
    univ = S.notna()
    ex = R.where(univ).sub(R.where(univ).mean(axis=1), axis=0)
    for k, thr in [("ib_any1", 0), ("ib_any5", 0), ("ib_clu10", 0), ("ib_off20", 0), ("is_val20", 0.5)]:
        for p, a, b in PERIODS_NIGHT:
            dm = ex.where(G[k] > thr).loc[a:b].mean(axis=1).dropna()
            n = int((G[k] > thr).where(univ, False).loc[a:b].sum().sum())
            rows.append(dict(section="night_univariate", variant=f"{k}>{thr}", period=p, n=n, days=len(dm),
                             net_bp=1e4 * dm.mean(), t_diff=dm.mean() / (dm.std() / np.sqrt(len(dm))),
                             note="mean close->open excess (gross) of blend-universe rows with the flag, "
                                  "date-clustered t"))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(), flush=True)
    save_part("blend", df)


# ------------------------------------------------------------------ overnight (b): pooled LightGBM +/- insider
def run_model(F):
    import bt
    import lightgbm as lgb
    import ensemble
    fn = cpath("study78_night_pred.parquet")
    sets = ("pooled", "pooled+insider")
    if not os.path.exists(fn):
        X = pd.read_parquet(f"{DATA}/ml_frame.parquet", filters=[("date", ">=", pd.Timestamp("2020-01-01"))])
        feat = [k for k in X.columns if not k.startswith("y_")]
        y = X["y_night"]
        X = X[feat]
        dates = X.index.get_level_values(0)
        ri = DAYS.get_indexer(dates)
        cj = COLS.get_indexer(X.index.get_level_values(1))
        okc = (ri >= 0) & (cj >= 0)
        ins = []
        for k, p in F["1545"].items():
            v = np.full(len(X), np.nan, dtype="float32")
            v[okc] = p.values[ri[okc], cj[okc]]
            X["ins_" + k] = v
            ins.append("ins_" + k)
        yr = (y.groupby(level=0).rank(pct=True) - 0.5).values
        ok = y.notna().values
        prm = dict(ensemble.PARAMS, num_threads=2)
        out = pd.DataFrame(index=X.index)
        for s in sets:
            out[s] = np.nan
        for q in pd.period_range("2024Q1", "2026Q3", freq="Q"):
            cut = DAYS[max(0, DAYS.searchsorted(q.start_time) - 11)]
            tr = ok & (dates < cut)
            te = (dates >= q.start_time) & (dates <= q.end_time)
            for s, f in zip(sets, (feat, feat + ins)):
                m = lgb.train(prm, lgb.Dataset(X.loc[tr, f], yr[tr]), num_boost_round=300)
                out.loc[te, s] = m.predict(X.loc[te, f])
                if s == "pooled+insider":
                    imp = pd.Series(m.feature_importance("gain"), index=f)
                    print(q, "insider gain share", round(imp[ins].sum() / imp.sum(), 4), flush=True)
            print(q, flush=True)
        out.dropna(how="all").to_parquet(fn)
        del X
    pr = pd.read_parquet(fn)
    tpx = traded_px()
    univ = (tpx > 5) & (P["dv"][cols].rolling(20).median() > 5e6)
    R = P["o"][cols].shift(-1) / P["c"][cols] - 1
    C = exec_cost_bps(P, "auction")[cols] + 2.5
    nets = {}
    for s in sets:
        S = pr[s].unstack().reindex(columns=cols)
        S = S.loc[S.index < DAYS[-1]]
        E = S.notna() & univ.reindex_like(S).fillna(False)
        W = bt.select_topk(S, E, 10)
        nets[s] = bt.run(W, R.reindex_like(W), C.reindex_like(W)).net
    # the blend with the pooled+insider rank added as a third leg (2 x ens + jump-drop + pooled+insider) / 4
    _, Sb, Rb, Cb = night_inputs()
    pi = pr["pooled+insider"].unstack().reindex(index=Sb.index, columns=cols)
    pb = pr["pooled"].unstack().reindex(index=Sb.index, columns=cols)
    base = bt.run(bt.select_topk(Sb, Sb.notna(), 10), Rb, Cb).net
    for nm, add in [("blend + pooled rank (no insider)", pb), ("blend + pooled+insider rank", pi)]:
        s = (3 * Sb + add.where(Sb.notna()).rank(axis=1, pct=True)) / 4
        nets[nm] = bt.run(bt.select_topk(s, s.notna(), 10), Rb, Cb).net
    rows = []
    for s, net in nets.items():
        ref = nets["pooled"] if s.startswith("pooled") else base
        for p, a, b in PERIODS_NIGHT:
            st = ann_stats(net.loc[a:b])
            dbp, t = paired(net.loc[a:b], ref.loc[a:b])
            rows.append(dict(section="night_model", variant=s, k=10, period=p, sharpe=st["sharpe"],
                             net_bp=1e4 * net.loc[a:b].mean(), days=st["n"], diff_bp=dbp, t_diff=t,
                             note="close-based frame; diff vs " + ("pooled" if s.startswith("pooled") else
                                                                    "blend base (2.85)")))
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(), flush=True)
    save_part("model", df)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    F = get_features()
    save_part("checks", pd.DataFrame(F["checks"]))
    if step in ("all", "day"):
        run_day(F)
    if step in ("all", "blend"):
        run_blend(F)
    if step in ("all", "model"):
        run_model(F)
