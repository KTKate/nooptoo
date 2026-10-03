"""Study 75: do disclosed trades (corporate insiders, members of Congress, activist 13D filers) predict returns over
short horizons (overnight, next day session, 1-5 trading days), on their own or around earnings?

Data (downloaded by the fetch steps, cached in $STUDY75_CACHE, default /tmp/study75; not in the repo):
  form4   SEC Form 3/4/5 structured data sets 2019q3.. (open-market P/S of non-derivative stock, Form 4 and 4/A):
          accession, filing date, trade date, issuer CIK, ticker, owner, relationship, officer title, shares, price,
          shares owned after, 10b5-1 checkbox (AFF10B5ONE, from 2023-04) and a footnote 10b5-1 mention.
  accept  EDGAR acceptance timestamps (data.sec.gov/submissions/CIK*.json of each issuer, incl. older pages), joined to
          Form 4 filings by accession number. Gives the exact public time of each filing.
  idx     EDGAR full-index form.idx per quarter: Schedule 13D / 13G (and amendments) with filing date. Each filing is
          listed under both the subject company and the filer; the subject is the CIK that maps to a panel ticker
          (data/local/sec/company_tickers.json); filings where both map are resolved from the filing header.
  house   House Clerk financial disclosure index (disclosures-clerk.house.gov, yearly ZIP with XML), periodic
          transaction reports (PTR) parsed from the electronically filed PDFs (ticker in parentheses, type P/S,
          trade date, notification date, amount range). Scanned (hand-written) PTRs cannot be parsed and are skipped.
          Senate eFD needs an interactive terms-acceptance session; House/Senate Stock Watcher S3 buckets return 403.
SEC requests: User-Agent 'nooptoo research incarnadins@gmail.com', at most 6.5 requests/second (shared limiter).

Timing (no look-ahead):
  Form 4: event time = acceptance timestamp (ET). Accepted before 09:30 on trading day d -> first tradable price is the
          open of d; accepted 09:30-16:00 -> the close of d (no intraday bars for most names in 2020-23); accepted
          after 16:00 -> open of d+1. Without a timestamp: filing date, assume after the close.
          Returns measured from that entry point E: to the next close (day session if E is an open), to close +1, +2,
          +5 trading days; also the overnight right after a post-close filing (close d -> open d+1, not tradable, it is
          the 'news' reaction) and the auction-to-auction version from the close of d (filing evening).
  13D/13G: event = filing date d (filings accepted after 17:30 get the next business day's date, and those accepted
          16:00-17:30 carry date d), so the first safe entry is the open of d+1. Reaction (close d-1 -> open d+1) is
          reported but is not tradable.
  Congress: event = PTR notification (filing) date d, assumed public after the close; entry open of d+1.
Excess return = stock return minus the equal-weight mean of the liquid universe over the same window (same entry
type: open-to-close windows vs open-to-close mean, etc.). t-stats: per event date means, Newey-West lag h
(date-clustered), plus calendar-time portfolios (study41.ct_portfolio) for the tradable rules.
Universe: traded price > $5 and 20d median dollar volume > $5M on the day before entry (also > $20M subset).
Survivorship: the panel only holds tickers alive in 2026; excess vs the universe mean carries the same bias.
Costs: horizon_lib COST (auction exec_cost_bps + 2.5 bp) per side.
Output: results/study75_disclosed_trades.csv (section, source, variant, horizon, period, n, mean, t, ...).
"""
import io
import json
import os
import re
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import requests

os.environ.setdefault("OMP_NUM_THREADS", "2")
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
CACHE = os.environ.get("STUDY75_CACHE", "/tmp/study75")
UA = {"User-Agent": "nooptoo research incarnadins@gmail.com"}      # owner's approved SEC contact (sec.gov only)
F345 = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
PLAN_RE = re.compile(r"10b5-?1|10b-5-1|10b5\s1", re.I)
NOT_RE = re.compile(r"not\s+(?:made\s+|effected\s+|executed\s+|entered\s+into\s+|done\s+)?(?:pursuant|under|in\s+accordance)"
                    r"[^.]{0,40}(?:10b5-?1|10b-5-1)", re.I)
_last = [0.0]
import threading
_lock = threading.Lock()


def sec_get(url, timeout=300, stream=False):
    """GET from sec.gov at <= 5 requests/second with retries."""
    for k in range(5):
        with _lock:
            w = 0.155 - (time.time() - _last[0])                      # <= 6.5 requests/second over all threads
            if w > 0:
                time.sleep(w)
            _last[0] = time.time()
        try:
            r = requests.get(url, headers=UA, timeout=timeout, stream=stream)
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


# ------------------------------------------------------------------ fetch: Form 4
def fetch_form4_quarter(q):
    fn = cpath("form4", f"{q}.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    r = sec_get(F345.format(q=q), timeout=900)
    if r is None or r.status_code != 200:
        print(q, "http", None if r is None else r.status_code, flush=True)
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))

    def rd(f, cols):
        return pd.read_csv(z.open(f), sep="\t", usecols=lambda c: c in cols, dtype=str, on_bad_lines="skip",
                           quoting=3)

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
        "after": pd.to_numeric(d.SHRS_OWND_FOLWNG_TRANS, errors="coerce"),
        "owner": d.RPTOWNERCIK, "role": d.RPTOWNER_RELATIONSHIP, "title": d.RPTOWNER_TITLE,
        "aff": d.AFF10B5ONE.map({"1": 1.0, "0": 0.0, "true": 1.0, "false": 0.0}), "fn_plan": d.fn_plan})
    out.to_parquet(fn, index=False)
    print(q, len(out), flush=True)
    return out


def fetch_form4():
    qs = [f"{y}q{k}" for y in range(2019, 2027) for k in range(1, 5) if "2019q3" <= f"{y}q{k}" <= "2026q3"]
    for q in qs:
        fetch_form4_quarter(q)


def load_form4():
    parts = [pd.read_parquet(os.path.join(CACHE, "form4", f)) for f in sorted(os.listdir(os.path.join(CACHE, "form4")))]
    return pd.concat(parts, ignore_index=True).dropna(subset=["filed", "ticker"])


# ------------------------------------------------------------------ fetch: acceptance times
def fetch_accept():
    """Acceptance timestamps of all filings 2019-07+ of each issuer whose ticker is in the panel."""
    tick = set(pd.read_pickle(os.path.join(ROOT, "data", "panel.pkl"))["c"].columns)
    f4 = load_form4()
    f4 = f4[f4.ticker.isin(tick)].dropna(subset=["issuer"])
    nb = f4[f4.code == "P"].groupby("issuer").size()
    iss = [int(c) for c in nb.sort_values(ascending=False).index]          # issuers with purchases first
    iss += sorted(set(f4.issuer.astype(int)) - set(iss))
    print("issuers", len(iss), flush=True)
    fn = cpath("accept", "accept.parquet")
    done = set()
    parts = []
    if os.path.exists(fn):
        old = pd.read_parquet(fn)
        parts.append(old)
        done = set(old.issuer.unique())
    batch = []

    def flush():
        if batch:
            parts.append(pd.concat(batch, ignore_index=True))
            batch.clear()
            pd.concat(parts, ignore_index=True).to_parquet(fn, index=False)

    from concurrent.futures import ThreadPoolExecutor

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

    todo = [c for c in iss if c not in done]
    pool = ThreadPoolExecutor(3)
    for k, (c, j, extra) in enumerate(pool.map(one, todo)):
        if j is None:
            continue
        blocks = [j["filings"]["recent"]] + extra
        rows = []
        for b in blocks:
            df = pd.DataFrame({"acc": b["accessionNumber"], "form": b["form"], "accepted": b["acceptanceDateTime"]})
            rows.append(df[df.form.isin(["4", "4/A", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
                                         "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A"])])
        x = pd.concat(rows, ignore_index=True)
        x["issuer"] = c
        batch.append(x)
        if k % 200 == 0:
            flush()
            print(k, len(iss), flush=True)
    flush()


# ------------------------------------------------------------------ fetch: 13D / 13G from full-index
SCH = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A", "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A")


def fetch_idx():
    for y in range(2019, 2027):
        for k in range(1, 5):
            if not ("2019q4" <= f"{y}q{k}" <= "2026q3"):
                continue
            fn = cpath("idx", f"{y}q{k}.parquet")
            if os.path.exists(fn):
                continue
            r = sec_get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{k}/form.idx", timeout=900)
            if r is None or r.status_code != 200:
                print(y, k, "http", None if r is None else r.status_code, flush=True)
                continue
            rows = []
            for line in r.text.splitlines():
                if not (line.startswith("SC 13") or line.startswith("SCHEDULE 13")):
                    continue
                m = re.match(r"^(.*?)\s{2,}(.*?)\s{2,}(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$", line)
                if m and m.group(1).strip() in SCH:
                    rows.append((m.group(1).strip(), m.group(2).strip(), int(m.group(3)), m.group(4), m.group(5)))
            df = pd.DataFrame(rows, columns=["form", "company", "cik", "date", "file"])
            df.to_parquet(fn, index=False)
            print(y, k, len(df), flush=True)


def resolve_13d():
    """Subject company of each schedule filing: the listed CIK that maps to a panel ticker; ambiguous ones from the
    filing header (<SUBJECT-COMPANY> block)."""
    fn = cpath("idx", "resolved.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    tick = set(pd.read_pickle(os.path.join(ROOT, "data", "panel.pkl"))["c"].columns)
    ct = json.load(open(os.path.join(ROOT, "data", "local", "sec", "company_tickers.json")))
    cik2t = {}
    for v in ct.values():                                  # first listed ticker per CIK (the main class)
        if v["ticker"] in tick and v["cik_str"] not in cik2t:
            cik2t[v["cik_str"]] = v["ticker"]
    d = pd.concat([pd.read_parquet(os.path.join(CACHE, "idx", f)) for f in sorted(os.listdir(os.path.join(CACHE, "idx")))
                   if f[:4].isdigit()], ignore_index=True)
    d["ticker"] = d.cik.map(cik2t)
    g = d[d.ticker.notna()].groupby("file")
    one = g.filter(lambda x: x.cik.nunique() == 1).drop_duplicates("file")
    amb = d[d.ticker.notna() & d.file.isin(set(d.file) - set(one.file))]
    print("schedule filings:", d.file.nunique(), " single mapped:", len(one), " ambiguous:", amb.file.nunique(), flush=True)
    res = []
    for k, (f, x) in enumerate(amb.groupby("file")):
        acc = os.path.basename(f).replace(".txt", "")
        cik0 = f.split("/")[2]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik0}/{acc.replace('-', '')}/{acc}-index-headers.html"
        r = sec_get(url, timeout=60)
        if r is None or r.status_code != 200:
            continue
        m = re.search(r"SUBJECT COMPANY:.*?CENTRAL INDEX KEY:\s*(\d+)", r.text, re.S)
        if m and int(m.group(1)) in cik2t:
            row = x[x.cik == int(m.group(1))].iloc[:1]
            res.append(row)
        if k % 200 == 0:
            print("hdr", k, flush=True)
    out = pd.concat([one] + res, ignore_index=True)
    out.to_parquet(fn, index=False)
    return out


# ------------------------------------------------------------------ fetch: House PTRs
HOUSE = "https://disclosures-clerk.house.gov/public_disc"
PTR_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,6})\)\s*(?:\[([A-Za-z]{2})\])?\s*(P|S \(partial\)|S|E)\s+"
                    r"(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(\$[\d,]+(?:\s*-\s*\$[\d,]+)?|Over \$[\d,]+)")


def fetch_house():
    """Yearly House Clerk index (XML) and electronically filed PTR PDFs (DocID not starting with 8 = paper scan)."""
    import pypdf
    idx = []
    for y in range(2020, 2027):
        fz = cpath("house", f"{y}FD.zip")
        if not os.path.exists(fz):
            r = requests.get(f"{HOUSE}/financial-pdfs/{y}FD.zip", timeout=120)
            open(fz, "wb").write(r.content)
            time.sleep(1)
        x = pd.read_xml(zipfile.ZipFile(fz).open(f"{y}FD.xml"))
        idx.append(x[x.FilingType == "P"].assign(year=y))
    idx = pd.concat(idx, ignore_index=True)
    idx.to_parquet(cpath("house", "index.parquet"), index=False)
    fn = cpath("house", "ptr.parquet")
    old = pd.read_parquet(fn) if os.path.exists(fn) else pd.DataFrame(columns=["doc"])
    old = old[old.ticker.notna()] if len(old) else old                   # re-parse documents without hits
    done = set(old.doc.astype(str))
    rows = []
    todo = idx[~idx.DocID.astype(str).str.startswith("8")]
    print("PTRs", len(idx), "electronic", len(todo), "done", len(done), flush=True)
    todo = todo[~todo.DocID.astype(str).isin(done)]

    def one(rw):
        doc = str(rw.DocID)
        time.sleep(1.0)                                               # 3 threads -> about 1-2 requests/second
        try:
            r = requests.get(f"{HOUSE}/ptr-pdfs/{rw.year}/{doc}.pdf", timeout=60)
            txt = " ".join((p.extract_text() or "") for p in pypdf.PdfReader(io.BytesIO(r.content)).pages)
        except Exception:
            return [dict(doc=doc, ticker=None)]
        txt = re.sub(r"\s+", " ", txt.replace("\x00", ""))
        hits = PTR_RE.findall(txt)
        if not hits:
            return [dict(doc=doc, ticker=None)]
        return [dict(doc=doc, ticker=t, asset=asset.upper(), type=typ, tdate=td, ndate=nd, amount=amt)
                for t, asset, typ, td, nd, amt in hits]

    from concurrent.futures import ThreadPoolExecutor
    for k, res in enumerate(ThreadPoolExecutor(3).map(one, list(todo.itertuples()))):
        rows.extend(res)
        if k % 100 == 0:
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_parquet(fn, index=False)
            print(k, len(todo), flush=True)
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_parquet(fn, index=False)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch_house":
    fetch_house()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1].startswith("fetch"):
    step = sys.argv[1]
    if step in ("fetch", "fetch_form4"):
        fetch_form4()
    if step in ("fetch", "fetch_idx"):
        fetch_idx()
    if step in ("fetch", "fetch_accept"):
        fetch_accept()
    if step in ("fetch", "fetch_13d"):
        resolve_13d()
    sys.exit(0)


# ================================================================== analysis
import horizon_lib as H                                              # noqa: E402  (loads the panel)
from study41_event_drift import traded_px, nw_t                      # noqa: E402
from core import RES                                                 # noqa: E402
import store                                                         # noqa: E402

PERIODS = (("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-12-31"))
HS = (0, 1, 2, 5)            # closes after the entry day: 0 = same-day close (only for open entries)
DAYS = H.days
ND = len(DAYS)
COLS = pd.Index(H.cols)
O = H.P["o"][H.cols].astype("float64").values
C = H.C.astype("float64").values
PX = traded_px().values
ADVL = H.ADV.shift(1).values                                         # 20d median $ volume known before day i
PXL = np.vstack([np.full((1, PX.shape[1]), np.nan), PX[:-1]])
U5 = (PXL > 5) & (ADVL > 5e6) & np.isfinite(C) & np.isfinite(O)
U20 = U5 & (ADVL > 20e6)
COSTV = H.COST.values.astype("float64")
MCAP = pd.read_pickle(os.path.join(ROOT, "data", "local", "fundamentals_panels.pkl"))["market_cap"] \
    .reindex(index=DAYS, columns=COLS).ffill(limit=10).shift(1).values
VARIANTS = []                # every (source, variant) tried, for the count


def _bench(num_kind, i0, h):
    """Universe (U5) EW mean of the window starting at i0 with entry kind 'o' (open of i0) or 'c' (close of i0)."""
    j = min(i0 + h, ND - 1)
    start = O[i0] if num_kind == "o" else C[i0]
    r = C[j] / start - 1
    return np.nanmean(np.where(U5[i0 if num_kind == "o" else min(i0 + 1, ND - 1)], r, np.nan))


_BC = {}


def bench_vec(kind, h):
    """Vector over entry index i of the universe EW return for entry kind at i, horizon h."""
    key = (kind, h)
    if key not in _BC:
        out = np.full(ND, np.nan)
        for i in range(1, ND - h - 1):
            j = i + h
            if kind == "o":
                r, u = C[j] / O[i] - 1, U5[i]
            elif kind == "c":
                r, u = C[j] / C[i] - 1, U5[i + 1] if i + 1 < ND else U5[i]
            elif kind == "on":                                        # overnight close i-1 -> open i
                r, u = O[i] / C[i - 1] - 1, U5[i]
            out[i] = np.nanmean(np.where(u & np.isfinite(r), r, np.nan))
        _BC[key] = out
    return _BC[key]


def event_returns(ev):
    """ev: DataFrame with ci (column index), oi (entry day index for an OPEN entry, i.e. the first open after the
    event is public). Adds excess returns: on (close oi-1 -> open oi), o{h} (open oi -> close oi+h), c{h}
    (close oi -> close oi+h; the first close after a pre-open/after-close event, skipping the day session),
    plus universe flags and costs."""
    ev = ev[(ev.oi >= 1) & (ev.oi < ND - 6)].copy()
    i, c = ev.oi.values, ev.ci.values
    ev["u5"] = U5[i, c]
    ev["u20"] = U20[i, c]
    ev["on"] = O[i, c] / C[i - 1, c] - 1 - bench_vec("on", 0)[i]
    for h in HS:
        ev[f"o{h}"] = C[i + h, c] / O[i, c] - 1 - bench_vec("o", h)[i]
        if h > 0:
            ev[f"c{h}"] = C[i + h, c] / C[i, c] - 1 - bench_vec("c", h)[i]
    ev["cost"] = COSTV[i - 1, c]
    ev["date"] = DAYS[i]
    ev["ret20"] = C[i - 1, c] / C[np.maximum(i - 21, 0), c] - 1      # stock return over the 20 days before entry
    ev["mcap"] = MCAP[i, c]
    ev["adv"] = ADVL[i, c]
    return ev


def summarize(ev, source, variant, rows, cols=("on", "o0", "o1", "o2", "o5", "c1", "c2", "c5"), univ="u5"):
    VARIANTS.append((source, variant, univ))
    e = ev[ev[univ]]
    for p, a, b in PERIODS + (("all", "2020-01-01", "2026-12-31"),):
        s = e[(e.date >= a) & (e.date <= b)]
        for col in cols:
            x = s[[col, "date"]].dropna()
            if len(x) < 20:
                continue
            h = int(col[1:]) if col[1:].isdigit() else 0
            m, t, nd = nw_t(x.groupby("date")[col].mean(), h + 1)
            rows.append(dict(section="event", source=source, variant=variant, univ=univ, ret=col, period=p,
                             n=len(x), n_dates=nd, mean_bp=1e4 * x[col].mean(), date_mean_bp=1e4 * m, t=t,
                             hit=(x[col] > 0).mean(), cost_rt_bp=2e4 * s.cost.mean()))


# ------------------------------------------------------------------ calendar-time tradable portfolio
def ct_trades(ev, col, max_names=20):
    """Each event = one trade: enter at the open of oi ('o' cols) or the close of oi ('c' cols), exit at the close of
    oi+h. Daily equal-weight portfolio of open trades (at most max_names, first come), cash otherwise. Costs: COST per
    side (entry, exit). Returns daily DataFrame: n, gross, net, bench (same legs of the U5 universe), excess_net."""
    kind, h = col[0], int(col[1:])
    ev = ev.sort_values("oi")
    legs = {}       # day -> list of (gross leg, bench leg, cost)
    held = np.zeros(ND, dtype=int)
    rv = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
    ds = C / O - 1
    bc = np.nanmean(np.where(U5, rv, np.nan), axis=1)
    bo = np.nanmean(np.where(U5, ds, np.nan), axis=1)
    taken = 0
    for i, c, cst in zip(ev.oi.values, ev.ci.values, ev.cost.values):
        a = i if kind == "o" else i + 1                              # first day whose return the trade earns
        b = i + h
        if b >= ND or a > b:
            continue
        if held[a:b + 1].max() >= max_names:
            continue
        held[a:b + 1] += 1
        taken += 1
        cst = 0.0002 if not np.isfinite(cst) else cst
        for d in range(a, b + 1):
            if d == a and kind == "o":
                g, bm = ds[d, c], bo[d]
            else:
                g, bm = rv[d, c], bc[d]
            k = (cst if d == a else 0.0) + (cst if d == b else 0.0)
            legs.setdefault(d, []).append((np.nan_to_num(g), bm, k))
    out = np.zeros((ND, 4))
    for d, L in legs.items():
        a = np.array(L)
        out[d] = (len(L), a[:, 0].mean(), np.nanmean(a[:, 1]), a[:, 2].mean())
    df = pd.DataFrame(out, index=DAYS, columns=["n", "gross", "bench", "cost"])
    df["net"] = df.gross - df.cost
    df["excess_net"] = (df.net - df.bench).where(df.n > 0)
    df.attrs["trades"] = taken
    return df


def ct_stats(df, a, b):
    x = df.loc[a:b]
    yrs = len(x) / 252
    net = x.net
    sr = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else np.nan
    xe = x.excess_net.dropna()
    m, t, _ = nw_t(xe, 5)
    return dict(days=len(x), invested=(x.n > 0).mean(), net_ann=(1 + net).prod() ** (1 / yrs) - 1 if yrs else np.nan,
                net_sharpe=sr, xs_net_bp_day=1e4 * m if np.isfinite(m) else np.nan, xs_t=t,
                xs_sharpe=xe.mean() / xe.std() * np.sqrt(252) if len(xe) > 20 else np.nan,
                spy_ann=(1 + H.SPY.pct_change().loc[a:b]).prod() ** (1 / yrs) - 1 if yrs else np.nan,
                trades_yr=x.n.diff().clip(lower=0).sum() / yrs if yrs else np.nan)


# ------------------------------------------------------------------ insiders
CEO_RE = re.compile(r"\bCEO\b|\bCFO\b|chief\s+exec|chief\s+financ|principal\s+exec|principal\s+financ|"
                    r"\bC\.E\.O\b|\bC\.F\.O\b", re.I)


def accept_times():
    a = pd.read_parquet(os.path.join(CACHE, "accept", "accept.parquet")).drop_duplicates("acc")
    # data.sec.gov acceptanceDateTime is UTC ('Z'; the hour histogram peaks at 20-21 UTC = 16-17 ET); convert to ET
    a["ts"] = pd.to_datetime(a.accepted, errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return a


def entry_index(ts, filed):
    """First open after the event is public, and the entry kind. ts = acceptance time (ET, naive) or NaT."""
    ts = pd.Series(ts)
    d = ts.dt.normalize().fillna(pd.Series(filed))
    mins = (ts.dt.hour * 60 + ts.dt.minute).fillna(24 * 60).values
    di = DAYS.searchsorted(d.values)                                 # first trading day >= calendar date
    is_td = (di < ND) & (DAYS[np.minimum(di, ND - 1)].values == d.values)
    pre = is_td & (mins < 9 * 60 + 30)
    intra = is_td & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    post_td = is_td & ~pre & ~intra
    oi = np.where(pre, di, np.where(post_td | intra, di + 1, di))   # non-trading day: next trading day's open
    timing = np.select([pre, intra, post_td], ["pre", "intra", "post"], "nontd")
    return oi, timing, np.where(is_td, di, -1)


def build_insider():
    raw = load_form4()
    raw["tdate"] = raw.tdate.fillna(raw.filed)
    raw.loc[raw.tdate > raw.filed, "tdate"] = raw.filed
    raw = raw[((raw.code == "P") & (raw.ad.fillna("A") == "A")) | ((raw.code == "S") & (raw.ad.fillna("D") == "D"))]
    raw = raw[(raw.shares > 0) & (raw.price > 0)]
    raw = raw.drop_duplicates(["owner", "ticker", "code", "tdate", "shares", "price"])
    raw["value"] = raw.shares * raw.price
    raw = raw[raw.value < 2e9]
    f = raw.sort_values(["acc", "tdate"]).groupby(["acc", "code"]).agg(
        filed=("filed", "first"), tdate=("tdate", "min"), ticker=("ticker", "first"), issuer=("issuer", "first"),
        owner=("owner", "first"), role=("role", "first"), title=("title", "first"), value=("value", "sum"),
        shares=("shares", "sum"), after=("after", "min"), aff=("aff", "max"), fn_plan=("fn_plan", "max")).reset_index()
    r, t = f.role.fillna(""), f.title.fillna("")
    f["cls"] = np.select([t.str.contains(CEO_RE) & r.str.contains("Officer|Director"), r.str.contains("Officer"),
                          r.str.contains("Director"), r.str.contains("TenPercent")],
                         ["ceo_cfo", "officer", "director", "tenpct"], "other")
    f["plan"] = (f.aff == 1) | f.fn_plan.astype(bool)
    a = accept_times()
    f = f.merge(a[["acc", "ts"]], on="acc", how="left")
    f = f[f.ticker.isin(set(COLS))].copy()
    f["ci"] = COLS.get_indexer(f.ticker)
    f["oi"], f["timing"], f["fdi"] = entry_index(f.ts.values, f.filed.values)
    f["delay"] = (f.filed - f.tdate).dt.days
    # first buy by this owner in this issuer in 12 months
    f = f.sort_values("filed")
    f["prev_same"] = f.groupby(["owner", "ticker", "code"]).filed.shift(1)
    f["first12"] = f.prev_same.isna() | ((f.filed - f.prev_same).dt.days > 365)
    return f


def insider_events(f, code):
    """One event per (ticker, entry open index): filings of one side that become public before that open."""
    x = f[f.code == code].copy()
    g = x.groupby(["ci", "oi"])
    ev = g.agg(ticker=("ticker", "first"), n_filings=("acc", "nunique"), n_owners=("owner", "nunique"),
               value=("value", "sum"), filed=("filed", "max"), tdate=("tdate", "min"),
               ceo=("cls", lambda s: (s == "ceo_cfo").any()), officer=("cls", lambda s: s.isin(["ceo_cfo", "officer"]).any()),
               director=("cls", lambda s: (s == "director").any()), tenpct=("cls", lambda s: (s == "tenpct").all()),
               plan_all=("plan", "all"), plan_any=("plan", "any"), first12=("first12", "any"),
               timing=("timing", lambda s: "intra" if (s == "intra").any() else ("post" if (s == "post").any()
                                                     else ("pre" if (s == "pre").all() else "nontd"))),
               has_ts=("ts", lambda s: s.notna().all()), delay=("delay", "min")).reset_index()
    # distinct owners buying this ticker in the 10 trading days up to the entry (known filings only)
    x = x.sort_values("oi")
    cl = []
    for ci, grp in x.groupby("ci"):
        oi_arr, own = grp.oi.values, grp.owner.values
        for o in np.unique(oi_arr):
            m = (oi_arr <= o) & (oi_arr > o - 10)
            cl.append((ci, o, len(set(own[m]))))
    cl = pd.DataFrame(cl, columns=["ci", "oi", "n_cluster"])
    ev = ev.merge(cl, on=["ci", "oi"], how="left")
    return event_returns(ev)


def earnings_table():
    E = store.read("earnings")
    E = E[E.symbol.isin(set(COLS)) & (E.date >= "2019-10-01")].copy()
    E["ti"] = DAYS.searchsorted(E.date.values.astype("datetime64[ns]"))
    E = E[(E.ti >= 2) & (E.ti + 6 < ND)].drop_duplicates(["symbol", "ti"])
    E["ci"] = COLS.get_indexer(E.symbol)
    return E[["symbol", "ci", "ti", "date", "time"]]


def run_insiders(rows, ctrows):
    f = build_insider()
    cov = f.groupby([f.filed.dt.year, "code"]).agg(n=("acc", "size"), ts=("ts", lambda s: s.notna().mean()))
    print(cov.unstack().round(3), flush=True)
    print("timing share (buys):", f[f.code == "P"].timing.value_counts(normalize=True).round(3).to_dict())
    for k, v in cov.iterrows():
        rows.append(dict(section="coverage", source="insider", variant=f"filings_{k[1]}", period=str(k[0]),
                         n=v.n, mean_bp=np.nan, t=np.nan, hit=v.ts))
    B = insider_events(f, "P")
    S = insider_events(f, "S")
    print("buy events", len(B), "in U5", B.u5.sum(), "sell events", len(S), "in U5", S.u5.sum(), flush=True)
    rel = B.value / B.mcap
    reladv = B.value / B.adv
    split = {
        "buy_all": B, "buy_ceo_cfo": B[B.ceo], "buy_officer": B[B.officer], "buy_director_only": B[B.director & ~B.officer],
        "buy_tenpct_only": B[B.tenpct], "buy_cluster3": B[B.n_cluster >= 3], "buy_cluster2": B[B.n_cluster >= 2],
        "buy_big_vs_mcap": B[rel > rel.quantile(0.8)], "buy_small_vs_mcap": B[rel < rel.quantile(0.2)],
        "buy_big_vs_adv": B[reladv > 0.05], "buy_value_gt_250k": B[B.value > 2.5e5], "buy_first12": B[B.first12],
        "buy_after_drop": B[B.ret20 < -0.10], "buy_after_rise": B[B.ret20 > 0.10],
        "buy_filed_pre": B[B.timing == "pre"], "buy_filed_intra": B[B.timing == "intra"],
        "buy_filed_post": B[B.timing == "post"], "buy_fast_delay01": B[B.delay <= 1],
        "buy_ceo_first12_drop": B[B.ceo & B.first12 & (B.ret20 < -0.05)],
        "sell_all": S, "sell_plan": S[S.plan_all], "sell_nonplan": S[~S.plan_any],
        "sell_nonplan_ceo": S[~S.plan_any & S.ceo], "sell_nonplan_big": S[~S.plan_any & (S.value > 1e6)],
        "sell_cluster3_nonplan": S[~S.plan_any & (S.n_cluster >= 3)], "sell_first12_nonplan": S[~S.plan_any & S.first12],
        "sell_after_rise_nonplan": S[~S.plan_any & (S.ret20 > 0.10)],
    }
    for k, e in split.items():
        summarize(e, "insider", k, rows)
        if k in ("buy_all", "buy_cluster3", "buy_ceo_cfo", "sell_nonplan", "sell_all"):
            summarize(e, "insider", k, rows, univ="u20")
    # earnings link: insider activity in the 30 calendar days before the report (filings public before the close
    # of t-1), and returns around the report (timing unknown: window close t-1 -> close t+1)
    E = earnings_table()
    ti, ci = E.ti.values, E.ci.values
    E["u5"] = U5[ti - 1, ci]
    E["u20"] = U20[ti - 1, ci]
    E["oi"] = ti                     # for summarize's date column
    E["date"] = DAYS[ti]
    E["cost"] = COSTV[ti - 2, ci]
    rv = lambda a, b: a / b - 1
    E["on"] = rv(O[ti, ci], C[ti - 1, ci]) - bench_vec("on", 0)[ti]
    E["o0"] = rv(C[ti, ci], O[ti, ci]) - bench_vec("o", 0)[ti]
    E["on1"] = rv(O[ti + 1, ci], C[ti, ci]) - bench_vec("on", 0)[ti + 1]
    E["d1"] = rv(C[ti + 1, ci], O[ti + 1, ci]) - bench_vec("o", 0)[ti + 1]
    E["c2"] = rv(C[ti + 1, ci], C[ti - 1, ci]) - bench_vec("c", 2)[ti - 1]       # close t-1 -> close t+1 (tradable)
    E["c5"] = rv(C[ti + 4, ci], C[ti - 1, ci]) - bench_vec("c", 5)[ti - 1]
    E["p1"] = rv(C[ti + 6, ci], C[ti + 1, ci]) - bench_vec("c", 5)[ti + 1]       # 5 days after the window
    f2 = f[f.fdi >= 0].copy()
    f2["pub_i"] = np.where(f2.timing == "post", f2.fdi + 1, f2.fdi)              # first close at/after publication
    acts = {}
    for name, m in [("b", f2.code == "P"), ("s", (f2.code == "S") & ~f2.plan)]:
        x = f2[m]
        key = x.groupby("ci")
        acts[name] = {c: np.sort(g.pub_i.values) for c, g in key}
    def count(name):
        d = acts[name]
        out = []
        for c, t, dt in zip(E.ci.values, E.ti.values, E.date.values):
            arr = d.get(c)
            if arr is None:
                out.append(0)
                continue
            lo = DAYS.searchsorted(pd.Timestamp(dt) - pd.Timedelta(days=30))
            out.append(int(((arr >= lo) & (arr <= t - 1)).sum()))
        return np.array(out)
    E["nb"], E["ns"] = count("b"), count("s")
    ecols = ("on", "o0", "on1", "d1", "c2", "c5", "p1")
    for k, e in {"earn_all": E, "earn_buy30": E[E.nb > 0], "earn_sell30_nonplan": E[E.ns > 0],
                 "earn_none30": E[(E.nb == 0) & (E.ns == 0)], "earn_buy_only": E[(E.nb > 0) & (E.ns == 0)],
                 "earn_sell_only": E[(E.ns > 0) & (E.nb == 0)]}.items():
        summarize(e, "insider_earn", k, rows, cols=ecols)
    # long-short within reports: buy30 minus sell30, per report date pooled by month
    return dict(B=B, S=S, E=E, split=split)


# ------------------------------------------------------------------ 13D / 13G
def run_13d(rows):
    d = resolve_13d()
    d["date"] = pd.to_datetime(d.date)
    d["acc"] = d.file.str.extract(r"(\d{10}-\d{2}-\d{6})")[0]
    try:
        a = accept_times()
        d = d.merge(a[["acc", "ts"]], on="acc", how="left")
    except Exception:
        d["ts"] = pd.NaT
    d["ci"] = COLS.get_indexer(d.ticker)
    d = d[d.ci >= 0]
    d["base"] = d.form.str.replace("SCHEDULE", "SC").str.replace("/A", "")
    d["amend"] = d.form.str.endswith("/A")
    # conservative timing: without a timestamp, the filing date may include 16:00-17:30 acceptances -> next open
    oi_ts, timing, _ = entry_index(d.ts.values, d.date.values)
    oi_nots = DAYS.searchsorted(d.date.values, side="right")
    d["oi"] = np.where(d.ts.notna(), oi_ts, oi_nots)
    d["timing"] = np.where(d.ts.notna(), timing, "date_only")
    cov = d.groupby([d.date.dt.year, "form"]).size().unstack()
    print(cov, flush=True)
    print("13D/G with timestamp:", d.ts.notna().mean().round(3), flush=True)
    for (y, fm), n in cov.stack().items():
        rows.append(dict(section="coverage", source="13dg", variant=fm, period=str(y), n=n))
    out = {}
    for name, m in [("13D_initial", (d.base == "SC 13D") & ~d.amend), ("13D_amend", (d.base == "SC 13D") & d.amend),
                    ("13G_initial", (d.base == "SC 13G") & ~d.amend), ("13G_amend", (d.base == "SC 13G") & d.amend)]:
        x = d[m].groupby(["ci", "oi"]).agg(ticker=("ticker", "first"), timing=("timing", "first")).reset_index()
        ev = event_returns(x)
        out[name] = ev
        summarize(ev, "13dg", name, rows)
        if name == "13D_initial":
            summarize(ev, "13dg", name, rows, univ="u20")
            summarize(ev[ev.timing == "post"], "13dg", "13D_initial_post_close", rows)
            summarize(ev[ev.timing == "pre"], "13dg", "13D_initial_pre_open", rows)
            summarize(ev[ev.timing == "intra"], "13dg", "13D_initial_intraday", rows)
            summarize(ev[ev.ret20 < -0.10], "13dg", "13D_initial_after_drop", rows)
    return out


# ------------------------------------------------------------------ Congress (House)
def amount_lo(s):
    m = re.search(r"\$([\d,]+)", s or "")
    return float(m.group(1).replace(",", "")) if m else np.nan


def run_house(rows):
    p = pd.read_parquet(os.path.join(CACHE, "house", "ptr.parquet"))
    idx = pd.read_parquet(os.path.join(CACHE, "house", "index.parquet"))
    idx["doc"] = idx.DocID.astype(str)
    n_docs, n_parsed = p.doc.nunique(), p[p.ticker.notna()].doc.nunique()
    p = p[p.ticker.notna()].merge(idx[["doc", "Last", "First", "StateDst", "FilingDate"]], on="doc", how="left")
    p["fdate"] = pd.to_datetime(p.FilingDate, format="%m/%d/%Y", errors="coerce")
    p["tdate"] = pd.to_datetime(p.tdate, format="%m/%d/%Y", errors="coerce")
    p["delay"] = (p.fdate - p.tdate).dt.days
    p["amt"] = p.amount.map(amount_lo)
    p["side"] = np.where(p.type == "P", "buy", np.where(p.type.str.startswith("S"), "sell", "other"))
    p["member"] = p.First.str.split().str[0] + " " + p.Last
    p = p[p.asset.isin(["ST"]) | p.asset.isna()]
    p = p[(p.fdate >= "2020-01-01") & p.ticker.isin(set(COLS))]
    print("House PTR docs fetched", n_docs, "with parsed rows", n_parsed, "stock rows in panel", len(p), flush=True)
    rows.append(dict(section="coverage", source="house", variant="docs_fetched", period="all", n=n_docs))
    rows.append(dict(section="coverage", source="house", variant="docs_parsed", period="all", n=n_parsed))
    for (y, s), n in p.groupby([p.fdate.dt.year, "side"]).size().items():
        rows.append(dict(section="coverage", source="house", variant=f"rows_{s}", period=str(y), n=n))
    p["ci"] = COLS.get_indexer(p.ticker)
    p["oi"] = DAYS.searchsorted(p.fdate.values, side="right")         # filed date assumed public after the close
    top = p.groupby("member").size().nlargest(10).index
    out = {}
    for side in ("buy", "sell"):
        x = p[p.side == side]
        base = x.groupby(["ci", "oi"]).agg(ticker=("ticker", "first"), amt=("amt", "max"), delay=("delay", "min"),
                                           top=("member", lambda s: s.isin(top).any()),
                                           n_members=("member", "nunique")).reset_index()
        ev = event_returns(base)
        out[side] = ev
        summarize(ev, "house", f"{side}_all", rows)
        summarize(ev[ev.amt >= 15001], "house", f"{side}_amt_ge15k", rows)
        summarize(ev[ev.amt >= 50001], "house", f"{side}_amt_ge50k", rows)
        summarize(ev[ev.delay <= 14], "house", f"{side}_delay_le14d", rows)
        summarize(ev[ev.top], "house", f"{side}_top10_members", rows)
        summarize(ev[ev.n_members >= 2], "house", f"{side}_2plus_members", rows)
    return out


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    rows, ctrows = [], []
    res = {}
    res["ins"] = run_insiders(rows, ctrows)
    res["13d"] = run_13d(rows)
    if os.path.exists(os.path.join(CACHE, "house", "ptr.parquet")):
        res["house"] = run_house(rows)
    R = pd.DataFrame(rows)
    pd.to_pickle(res, os.path.join(CACHE, "events.pkl"))
    R.to_csv(os.path.join(CACHE, "event_rows.csv"), index=False)
    ev = R[R.section == "event"]
    pv = ev[ev.period != "all"].pivot_table(index=["source", "variant", "univ", "ret"], columns="period",
                                            values=["date_mean_bp", "t", "n"])
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(pv.round(1).to_string())
    print("variants tried (source x subset x universe):", len(VARIANTS), " x return windows")
