"""Study 15: a thorough test of insider open-market purchases (SEC Form 4) as a signal, 2020-01 .. 2026-03
(plus 2026-04..06, which the first pass (study 12) never saw, as a fresh holdout).

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
#PART2
