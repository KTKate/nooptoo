"""Study 210: copy the "best ideas" of skilled concentrated managers from SEC Form 13F filings.

Theory: Cohen, Polk and Silli (2010) "Best Ideas": a manager's highest-conviction positions (largest overweight vs the
market) outperform; 13F cloning (Anton and Polk; Agarwal et al.). Filings arrive up to 45 days after quarter end, so
every trade is placed after the filing date.

Question: buying the best ideas most commonly held by managers with past skill, each time a filing arrives, and
holding until the next update, does it beat SPY after costs?

Data
  SEC Form 13F structured data sets (sec.gov, quarterly zips by filing date, 2013-04 .. 2026-08), cached under
  data/local/13f/zips, parsed one file at a time into data/local/13f/parsed. Only original 13F-HR holdings or
  combination reports (no amendments, no 13F-NT), only share positions (SH) without put/call. VALUE is in $1000 before
  2023-01-03 and in dollars after; a filing whose implied prices are 1000x off the cross-filer median is rescaled.
  Prices of record at each quarter end: the median VALUE/SHARES across all filers of that CUSIP (survivorship-free,
  used only for manager skill). A split is detected when the implied price ratio is far from 1 while price x total
  reported shares is about unchanged.
  Market weights: each CUSIP's share of the aggregate 13F equity value of the previous period (all filers), a proxy
  for market-cap weight that is known before the filing.
  Ticker mapping: 13F issuer names (no CUSIP master is free) matched to Alpaca asset names (active and inactive
  US-exchange symbols) after normalisation; ambiguous candidates resolved by the agreement of their quarterly returns
  with the 13F implied returns; a mapping is accepted only when that agreement is close (median abs gap < 6%).
  Daily prices (total return): Alpaca 1Day bars (adjustment=all; close = official closing cross) from 2019-12, yfinance
  (auto_adjust) before; each fills the other's gaps (Alpaca has inactive symbols from 2016, which keeps many delisted
  names). Picks without any price are dropped and counted.
Design
  Manager = filer CIK. Eligible filing: 13F equity value >= $500M and <= 100 positions.
  Holdings return of the report for quarter q over q -> q+1 (implied prices, weights from the report), minus the
  aggregate 13F value-weighted return. Skill at q = mean of the prior 8 such quarterly excess returns (report q-8 .. q-1,
  realised by the end of q, at least 6 present).
  Skilled = top X% by skill among managers whose previous filing (q-1) was eligible; their filing for q must also be
  eligible. Best ideas of a filing: the 3 positions with the largest weight minus market weight (ETF-like names
  excluded). A filing becomes active at the close 2 trading days after its filing date (auction buy) and stays
  active until the manager's next filing becomes active (or 200 days after its period end).
  Each trading day the book holds the N CUSIPs named most often across the active skilled filings (tie: larger summed
  overweight), equal weight; it changes only when the active set changes. Daily returns close to close.
  Benchmarks: SPY (same price sources) and the equal-weight average of all active best ideas of all eligible
  managers regardless of skill.
  Costs per side: 1.3 + 0.1 x half-spread ("15:45" model) + 2.5 bp on traded notional (closing auction).
Protocol (pre-registered 2026-11-09): choose skill cut X in {10, 20, 33}% and N in {10, 20, 40} by net Sharpe on daily
returns 2014-2019 (filings 2014-2019; skill exists from the 2014Q3 reports), write results/study210_frozen.json, then
score 2020-01 .. 2026-10 once. Pass: net Sharpe above SPY's, monthly excess return over SPY t > 2, positive excess in
most calendar years. No changes after the confirmation run.
    python src/study210_13f_best_ideas.py fetch      # download + parse the 13F zips (sec.gov, < 1 request/s)
    python src/study210_13f_best_ideas.py build      # skill, best ideas, ticker map, prices
    python src/study210_13f_best_ideas.py discover   # grid on 2014-2019, freeze
    python src/study210_13f_best_ideas.py confirm    # 2020-2026 once
Output: results/study210_grid.csv, results/study210_frozen.json, results/study210_confirm.csv (by period and year),
        results/study210_daily.csv, results/study210_mapping.csv (match-rate summary), results/study210_picks.csv
"""
import glob
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core import RES, half_spread_model  # noqa: E402

D13 = os.path.join(ROOT, "data", "local", "13f")
ZIPS, PARSED = os.path.join(D13, "zips"), os.path.join(D13, "parsed")
UA = {"User-Agent": "nooptoo research incarnadins@gmail.com"}   # sec.gov only
INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
FROZEN = os.path.join(RES, "study210_frozen.json")
CUTS, NS = [0.10, 0.20, 0.33], [10, 20, 40]
MIN_AUM, MAX_POS = 500e6, 100
ETF_RE = re.compile(r"ISHARES|SPDR|VANGUARD|POWERSHARES|PROSHARES|DIREXION|WISDOMTREE|INVESCO QQQ|SELECT SECTOR|"
                    r"\bETF\b|\bETN\b|\bFUND\b|\bFD\b|\bINDEX\b|GRAYSCALE|ISHS|\bTR UNIT\b|VANECK|MARKET VECTORS|"
                    r"\bETP\b|ULTRA\b|\bS&P 500\b|GLOBAL X|FIRST TR|SCHWAB STRATEGIC|JPMORGAN CHASE & CO ALERIAN|"
                    r"EXCHANGE TRADED|\bETFS\b|ETF TR|COVINGTON|KRANESHARES|NUVEEN|ABERDEEN|PACER F|INNOVATOR|"
                    r"SERIES TRUST|SIMPLIFY EXCH|\bMUN\b|\bMUNI|AMPLIFY ETF|TIDAL|\bTR II\b|EXCH TRADED|ALPS ETF|"
                    r"WESTERN ASSET|BLACKROCK .*(TR|FD|INCOME)|PIMCO|EATON VANCE .*(FD|TR|FUND)|ADVISORSHARES|"
                    r"ISHARES|SPROTT PHYSICAL|BITCOIN|ETHEREUM|PHYSICAL GOLD|GOLD TR|SILVER TR")


# ------------------------------------------------------------------ fetch + parse (one zip at a time)
def fetch():
    os.makedirs(ZIPS, exist_ok=True)
    os.makedirs(PARSED, exist_ok=True)
    html = requests.get(INDEX, headers=UA, timeout=60).text
    urls = sorted(set(re.findall(r'href="([^"]*13f[^"]*\.zip)"', html)))
    for u in urls:
        fn = os.path.join(ZIPS, os.path.basename(u))
        if not os.path.exists(fn):
            r = requests.get("https://www.sec.gov" + u, headers=UA, timeout=600)
            if r.status_code != 200:
                print("FAIL", u, r.status_code)
                continue
            open(fn, "wb").write(r.content)
            time.sleep(1.0)
    for fn in sorted(glob.glob(os.path.join(ZIPS, "*.zip"))):
        out = os.path.join(PARSED, os.path.basename(fn).replace(".zip", "_filings.parquet"))
        if not os.path.exists(out):
            parse(fn)


def parse(fn):
    tag = os.path.basename(fn).replace(".zip", "")
    z = zipfile.ZipFile(fn)
    mem = {os.path.basename(n): n for n in z.namelist()}      # some zips hold the files in a subfolder
    zo = lambda f: z.open(mem[f])
    rd = lambda f, cols: pd.read_csv(zo(f), sep="\t", usecols=cols, dtype=str, on_bad_lines="skip",
                                     quoting=3, keep_default_na=False)
    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    cov = rd("COVERPAGE.tsv", ["ACCESSION_NUMBER", "ISAMENDMENT", "REPORTTYPE", "FILINGMANAGER_NAME"])
    f = sub.merge(cov, on="ACCESSION_NUMBER")
    f = f[(f.SUBMISSIONTYPE == "13F-HR") & (f.ISAMENDMENT != "Y")
          & f.REPORTTYPE.str.contains("HOLDINGS|COMBINATION", case=False)]
    f = pd.DataFrame({"acc": f.ACCESSION_NUMBER, "cik": f.CIK.astype(int), "mgr": f.FILINGMANAGER_NAME,
                      "filed": pd.to_datetime(f.FILING_DATE, format="%d-%b-%Y", errors="coerce"),
                      "period": pd.to_datetime(f.PERIODOFREPORT, format="%d-%b-%Y", errors="coerce")}).dropna()
    keep = set(f.acc)
    parts = []
    cols = ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE",
            "PUTCALL"]
    for ch in pd.read_csv(zo("INFOTABLE.tsv"), sep="\t", usecols=cols, dtype=str, on_bad_lines="skip",
                          quoting=3, keep_default_na=False, chunksize=1_000_000):
        ch = ch[ch.ACCESSION_NUMBER.isin(keep) & (ch.SSHPRNAMTTYPE.str.upper() == "SH") & (ch.PUTCALL.str.strip() == "")]
        ch = pd.DataFrame({"acc": ch.ACCESSION_NUMBER, "cusip": ch.CUSIP.str.upper().str.strip().str.zfill(9).str[:9],
                           "name": ch.NAMEOFISSUER.str.upper().str.strip(), "title": ch.TITLEOFCLASS.str.upper(),
                           "value": pd.to_numeric(ch.VALUE, errors="coerce"),
                           "shares": pd.to_numeric(ch.SSHPRNAMT, errors="coerce")})
        parts.append(ch[(ch.value > 0) & (ch.shares > 0)])
    h = pd.concat(parts, ignore_index=True)
    del parts
    nm = h.groupby("cusip").agg(name=("name", lambda s: s.mode().iat[0]), title=("title", lambda s: s.mode().iat[0]),
                                nrow=("name", "size")).reset_index()
    h = h.groupby(["acc", "cusip"], as_index=False)[["value", "shares"]].sum()
    h = h.merge(f[["acc", "filed", "period"]], on="acc")
    h["value"] = h.value * np.where(h.filed < "2023-01-03", 1000.0, 1.0)
    # unit fix: a filing whose implied prices sit ~1000x away from the cross-filer median is rescaled
    lp = np.log10(h.value / h.shares)
    med = lp.groupby([h.period, h.cusip]).transform("median")
    off = (lp - med).groupby(h.acc).transform("median")
    h.loc[off > 2.5, "value"] /= 1000
    h.loc[off < -2.5, "value"] *= 1000
    h["px"] = h.value / h.shares
    tot = h.groupby("acc").agg(aum=("value", "sum"), npos=("cusip", "size")).reset_index()
    f = f.merge(tot, on="acc")
    agg = h.groupby(["period", "cusip"]).agg(value=("value", "sum"), shares=("shares", "sum"),
                                             lpx=("px", lambda s: np.log(s).median()), n=("px", "size")).reset_index()
    small = set(f.acc[(f.npos <= 250) & (f.aum >= 200e6)])
    hk = h[h.acc.isin(small)][["acc", "cusip", "value"]]
    f.to_parquet(os.path.join(PARSED, f"{tag}_filings.parquet"), index=False)
    hk.to_parquet(os.path.join(PARSED, f"{tag}_hold.parquet"), index=False)
    agg.to_parquet(os.path.join(PARSED, f"{tag}_agg.parquet"), index=False)
    nm.to_parquet(os.path.join(PARSED, f"{tag}_names.parquet"), index=False)
    print(tag, "filings", len(f), "kept", len(small), "rows", len(h), "kept rows", len(hk), flush=True)


# ------------------------------------------------------------------ build: skill and best ideas
def qn(ts):
    return ts.dt.year * 4 + (ts.dt.month - 1) // 3


def load_parsed(kind):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(PARSED, f"*_{kind}.parquet")))],
                     ignore_index=True)


def cusip_returns(agg):
    """Quarter q -> q+1 price return per CUSIP from the cross-filer median implied price, split-corrected."""
    agg = agg.assign(wl=agg.lpx * agg.n)
    a = agg.groupby(["q", "cusip"])[["value", "shares", "wl", "n"]].sum().reset_index()
    a["lpx"] = a.wl / a.n
    a = a.drop(columns="wl")
    nx = a[["q", "cusip", "lpx", "shares", "n"]].copy()
    nx["q"] -= 1
    a = a.merge(nx, on=["q", "cusip"], how="left", suffixes=("", "_1"))
    lr = a.lpx_1 - a.lpx
    lsr = np.log(a.shares_1 / a.shares)
    split = (lr.abs() > np.log(1.8)) & ((lr + lsr).abs() < np.log(1.3))
    lr = np.where(split, lr + lsr, lr)
    a["ret"] = np.clip(np.exp(lr) - 1, -0.95, 5.0)
    return a


def build_ideas():
    fil = load_parsed("filings")
    fil = fil[fil.period.dt.is_quarter_end & (fil.filed > fil.period)]
    fil["q"] = qn(fil.period)
    fil = fil.sort_values("filed").drop_duplicates(["cik", "q"])        # first original report per manager-quarter
    agg = load_parsed("agg")
    agg["q"] = qn(agg.period)
    a = cusip_returns(agg)
    del agg
    tot = a.groupby("q").value.sum()
    a["mw"] = a.value / a.q.map(tot)
    mret = a.dropna(subset=["ret"]).groupby("q").apply(lambda g: np.average(g.ret, weights=g.value),
                                                       include_groups=False)
    # holdings loaded one file at a time with categorical keys (memory)
    cats = pd.Index(sorted(a.cusip.unique()))
    ever = set(fil.cik[(fil.aum >= MIN_AUM) & (fil.npos <= MAX_POS)])   # only managers ever eligible need holdings
    accs = set(fil.acc[fil.cik.isin(ever)])
    acat = sorted(accs)
    parts = []
    for fn in sorted(glob.glob(os.path.join(PARSED, "*_hold.parquet"))):
        h = pd.read_parquet(fn)
        h = h[h.acc.isin(accs)]
        parts.append(pd.DataFrame({"acc": pd.Categorical(h.acc, categories=acat),
                                   "cusip": pd.Categorical(h.cusip, categories=cats),
                                   "value": h.value.values.astype("float32")}))
    hold = pd.concat(parts, ignore_index=True)
    del parts
    hold = hold.merge(fil[["acc", "cik", "q", "aum"]].astype({"acc": hold.acc.dtype}), on="acc")
    a["cusip"] = pd.Categorical(a.cusip, categories=cats)
    hold["w"] = hold.value / hold.aum
    hold = hold.merge(a[["q", "cusip", "ret"]], on=["q", "cusip"], how="left")
    ok = hold.ret.notna()
    g = hold[ok].assign(wr=lambda x: x.w * x.ret).groupby("acc").agg(wr=("wr", "sum"), wc=("w", "sum"))
    fil = fil.merge(g, left_on="acc", right_index=True, how="left")
    fil["hret"] = np.where(fil.wc >= 0.8, fil.wr / fil.wc, np.nan)
    fil["xret"] = (fil.hret - fil.q.map(mret)).clip(-0.5, 0.5)   # a quarter beyond +-50% is mostly data error
    # skill at q: mean xret of reports q-8 .. q-1 (each realised by the end of q)
    x = fil.pivot_table(index="q", columns="cik", values="xret")
    x = x.reindex(range(x.index.min(), int(fil.q.max()) + 1))
    sk = x.shift(1).rolling(8, min_periods=6).mean().stack().rename("skill").reset_index()
    fil = fil.merge(sk, on=["q", "cik"], how="left")
    fil["elig"] = (fil.aum >= MIN_AUM) & (fil.npos <= MAX_POS)
    # skill percentile at q among managers whose q-1 filing was eligible and whose skill exists at q
    # (both known at the end of q, before any filing for q arrives)
    base = fil[fil.elig][["cik", "q"]].assign(q=lambda d: d.q + 1).merge(sk, on=["q", "cik"])
    base["skill_pct"] = base.groupby("q").skill.rank(pct=True, ascending=False)
    fil = fil.merge(base[["cik", "q", "skill_pct"]], on=["cik", "q"], how="left")
    # best ideas: top 3 by weight minus previous-period market weight, ETF-like names excluded
    names = load_parsed("names").groupby(["cusip", "name"]).agg(title=("title", "first"), nrow=("nrow", "sum"))
    names = names.reset_index()
    etf = set(names.cusip[names.name.str.contains(ETF_RE) | names.title.str.contains(r"\bETF|\bETN|UNIT|WARR|RIGHT")])
    e = hold[hold.acc.isin(set(fil.acc[fil.elig])) & ~hold.cusip.isin(etf)].copy()
    pm = a[["q", "cusip", "mw"]].assign(q=lambda d: d.q + 1)
    e = e.merge(pm, on=["q", "cusip"], how="left")
    e["ow"] = e.w - e.mw.fillna(0)
    e = e.sort_values(["acc", "ow"], ascending=[True, False]).groupby("acc").head(3)
    ideas = e[["acc", "cusip", "ow", "w"]].merge(
        fil[["acc", "cik", "mgr", "q", "period", "filed", "skill", "skill_pct"]], on="acc")
    ideas = ideas.astype({"acc": str, "cusip": str})
    ideas.to_parquet(os.path.join(D13, "ideas.parquet"), index=False)
    fil.drop(columns=["wr", "wc"]).to_parquet(os.path.join(D13, "filings.parquet"), index=False)
    a[["q", "cusip", "ret", "lpx", "n"]].astype({"cusip": str}).to_parquet(os.path.join(D13, "cusip_ret.parquet"), index=False)
    names.to_parquet(os.path.join(D13, "names.parquet"), index=False)
    print("filings", len(fil), "eligible", int(fil.elig.sum()), "with skill pct", int(fil.skill_pct.notna().sum()),
          "ideas", len(ideas), "distinct cusips", ideas.cusip.nunique())
    print(fil[fil.elig].groupby(fil.period.dt.year).agg(n=("acc", "size"), skilled=("skill_pct", "count")))


# ------------------------------------------------------------------ ticker mapping
ABBR = {"INTL": "INTERNATIONAL", "INTERNATL": "INTERNATIONAL", "TECHNOLOGIES": "TECH", "TECHNOLOGY": "TECH",
        "TECHNOLOGS": "TECH", "TECHS": "TECH", "PHARMACEUTICALS": "PHARMA", "PHARMACEUTICAL": "PHARMA",
        "PHARMACEUTICA": "PHARMA", "PHARMS": "PHARMA", "SYS": "SYSTEMS", "SYSTEM": "SYSTEMS",
        "FINL": "FINANCIAL", "FIN": "FINANCIAL", "MGMT": "MANAGEMENT", "SVCS": "SERVICES", "SVC": "SERVICES",
        "COMMUNICATIONS": "COMM", "COMMUNICATION": "COMM", "COMMUNICATNS": "COMM", "ENTMT": "ENTERTAINMENT",
        "RES": "RESOURCES", "PPTYS": "PROPERTIES", "PPTY": "PROPERTIES", "INDS": "INDUSTRIES", "MFG": "MANUFACTURING",
        "AMER": "AMERICA", "MLS": "MILLS", "ENGR": "ENGINEERING", "RESH": "RESEARCH", "CTRS": "CENTERS", "STRS": "STORES", "AMERN": "AMERICAN", "NATL": "NATIONAL", "BANCORPORATION": "BANCORP", "BK": "BANK",
        "LABS": "LABORATORIES", "LAB": "LABORATORIES", "ENTERPRISES": "ENTERPRISE", "SOLUTIONS": "SOLUTION",
        "THERAPEUTICS": "THERAP", "THERAPEUTIC": "THERAP", "HLTH": "HEALTH", "HEALTHCARE": "HEALTH CARE",
        "ELEC": "ELECTRIC", "ELECTR": "ELECTRIC", "MTRS": "MOTORS", "BRANDS": "BRAND", "DEV": "DEVELOPMENT",
        "CHEM": "CHEMICAL", "CHEMICALS": "CHEMICAL", "PETE": "PETROLEUM", "PETE.": "PETROLEUM", "ENERGY": "ENERGY",
        "MED": "MEDICAL", "INSTRS": "INSTRUMENTS", "SEMICONDUCTOR": "SEMI", "SEMICONDUCTORS": "SEMI",
        "PRODS": "PRODUCTS", "PRODUCTS": "PRODUCTS", "GRP": "GROUP", "SCIENCES": "SCIENCE", "SCIENCS": "SCIENCE",
        "SOFTWARE": "SOFTWARE", "PARTNERS": "PARTNERS", "INVT": "INVESTMENT", "INVTS": "INVESTMENTS",
        "MINES": "MINING", "AUTOMOTIVE": "AUTO", "ENTERTAINMENT": "ENTERTAINMENT", "BIOSCIENCES": "BIOSCIENCE",
        "HOLDING": "", "HOLDINGS": "", "HLDGS": "", "HLDG": "", "HLDNGS": "", "GROUP": "", "INC": "", "CORP": "",
        "CORPORATION": "", "CO": "", "COMPANY": "", "COS": "", "LTD": "", "LIMITED": "", "PLC": "", "LP": "",
        "LLC": "", "THE": "", "NEW": "", "COM": "", "COMMON": "", "STOCK": "", "CLASS": "", "CL": "", "SHS": "",
        "ORD": "", "ORDINARY": "", "SA": "", "NV": "", "AG": "", "SE": "", "ADR": "", "ADS": "", "SPONSORED": "",
        "SPON": "", "AMERICAN DEPOSITARY": "", "DEPOSITARY": "", "SHARES": "", "SHARE": "", "REIT": "", "TRUST": "",
        "TR": "", "DEL": "", "PAR": "", "VALUE": "", "USD": "", "NPV": "", "N": "", "V": "", "A": "", "B": "",
        "C": "", "I": "", "II": "", "OF": "", "AND": "", "BERMUDA": "", "CDA": "", "NEW YORK": "", "MD": "",
        "LTD.": "", "WI": "", "WHEN": "", "ISSUED": "", "UNITS": "", "REPRESENTING": "", "EACH": "", "ONE": "",
        "VOTING": "", "SUB": "", "SUBORDINATE": "", "REDEEMABLE": "", "WARRANTS": ""}


def norm(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper().replace("&", " AND "))
    toks = []
    for t in s.split():
        t = ABBR.get(t, t)
        if t:
            toks.extend(t.split())
    return " ".join(toks)


def insider_names():
    """(ticker, issuer name) pairs from SEC Form 3/4/5 data sets 2013-2026 (EDGAR conformed issuer names with the
    trading symbol as filed at the time), cached; covers companies delisted since."""
    out = os.path.join(D13, "f345_names.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)
    parts = []
    for fn in sorted(glob.glob(os.path.join(D13, "f345", "*.zip"))):
        z = zipfile.ZipFile(fn)
        mem = {os.path.basename(n): n for n in z.namelist()}
        d = pd.read_csv(z.open(mem["SUBMISSION.tsv"]), sep="\t", usecols=["ISSUERNAME", "ISSUERTRADINGSYMBOL", "FILING_DATE"],
                        dtype=str, on_bad_lines="skip", quoting=3, keep_default_na=False)
        d["filed"] = pd.to_datetime(d.FILING_DATE, format="%d-%b-%Y", errors="coerce")
        d["symbol"] = d.ISSUERTRADINGSYMBOL.str.upper().str.strip().str.split().str[0].str.replace("-", ".")
        g = d.dropna(subset=["filed", "symbol"]).groupby(["symbol", "ISSUERNAME"]).agg(
            first=("filed", "min"), last=("filed", "max"), n=("filed", "size")).reset_index()
        parts.append(g)
    g = pd.concat(parts).groupby(["symbol", "ISSUERNAME"]).agg(first=("first", "min"), last=("last", "max"),
                                                                n=("n", "sum")).reset_index()
    g = g.rename(columns={"ISSUERNAME": "name"})
    g = g[g.symbol.str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?") & (g.n >= 2)]
    g.to_parquet(out, index=False)
    return g


def ticker_candidates(cusips, names):
    """Name-based candidates per CUSIP from every issuer name the CUSIP carried in 13F filings, matched to Alpaca
    asset names and Form 3/4/5 issuer names: exact, no-space, sorted-token, prefix, then close fuzzy matches."""
    import difflib
    a = pd.concat([pd.read_parquet(os.path.join(ROOT, "data", "local", f"alpaca_assets_{k}.parquet"))
                   for k in ["active", "inactive"]], ignore_index=True)
    a = a[a.exchange.isin(["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"]) & a.symbol.str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]
    a = a[~a.name.str.contains(r"\bETF\b|\bETN\b|Warrant|\bRights?\b|\bUnits?\b|Preferred|Notes due|%", case=False,
                               regex=True, na=False)]
    ins = insider_names()
    ref = pd.concat([a[["symbol", "name"]], ins[["symbol", "name"]]], ignore_index=True).drop_duplicates()
    ref["key"] = ref.name.map(norm)
    ref = ref[ref.key.str.len() > 0]
    ref["key0"] = ref.key.str.replace(" ", "")
    ref["keys"] = ref.key.map(lambda k: " ".join(sorted(k.split())))
    by = ref.groupby("key").symbol.apply(set).to_dict()
    by0 = ref.groupby("key0").symbol.apply(set).to_dict()
    bys = ref.groupby("keys").symbol.apply(set).to_dict()
    keys0 = np.array(sorted(by0))
    nm = names[names.cusip.isin(set(cusips))].sort_values("nrow", ascending=False)
    rows = []
    for c, g in nm.groupby("cusip", sort=False):
        cand, how = set(), None
        title = g.title.iat[0]
        for nme in g.name.drop_duplicates().head(4):
            k = norm(nme)
            k0 = k.replace(" ", "")
            ks = " ".join(sorted(k.split()))
            hit = by.get(k) or by0.get(k0) or bys.get(ks)
            h = "exact" if hit else None
            if not hit and len(k0) >= 6:
                i = np.searchsorted(keys0, k0)
                pre = set()
                while i < len(keys0) and keys0[i].startswith(k0) and len(pre) < 6:
                    pre |= by0[keys0[i]]
                    i += 1
                if not pre:
                    sh = [k0[:j] for j in range(len(k0) - 1, 6, -1) if k0[:j] in by0]
                    pre = by0[sh[0]] if sh else set()
                hit, h = pre, "prefix"
            if not hit and len(k0) >= 5:
                close = difflib.get_close_matches(k0, keys0, n=3, cutoff=0.85)
                hit = set().union(*[by0[x] for x in close]) if close else set()
                h = "fuzzy"
            if hit:
                cand |= hit
                how = how or h
        cand = sorted(cand)
        if len(cand) > 1 and isinstance(title, str):      # share classes
            m = re.search(r"\bCL(?:ASS)?\s*([A-C])\b", title)
            if m:
                cand = [x for x in cand if x.endswith("." + m.group(1))] or cand
        rows.append(dict(cusip=c, name=g.name.iat[0], how=how, cand=cand[:8]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ prices
def fetch_prices(tickers):
    """Daily total-return closes and volumes: Alpaca (adjustment=all, 2016+) and yfinance (2013+). Each batch of 100
    symbols is cached as its own file under data/local/13f/px (memory stays small)."""
    import alpaca_data as A
    import yfinance as yf
    pdir = os.path.join(D13, "px")
    os.makedirs(pdir, exist_ok=True)
    done = os.path.join(D13, "px_done.json")
    dn = json.load(open(done)) if os.path.exists(done) else {"a": [], "y": []}
    need_a = sorted(set(tickers) - set(dn["a"]))
    for i in range(0, len(need_a), 100):
        b = need_a[i:i + 100]
        d = A.bars(b, "1Day", "2016-01-01T00:00:00Z", "2026-11-01T00:00:00Z", adjustment="all")
        if len(d):
            pd.DataFrame({"date": d.ts.dt.normalize(), "ticker": d.ticker.astype(str), "c": d.c.astype("float32"),
                          "v": d.v.astype("float32")}).to_parquet(os.path.join(pdir, f"a_{len(dn['a']):06d}.parquet"))
        dn["a"].extend(b)
        json.dump(dn, open(done, "w"))
        print("alpaca", i, len(need_a), flush=True)
    need_y = sorted(set(tickers) - set(dn["y"]))
    for i in range(0, len(need_y), 100):
        b = need_y[i:i + 100]
        ys = [t.replace(".", "-") for t in b]
        try:
            x = yf.download(ys, start="2013-01-01", end="2026-11-01", auto_adjust=True, progress=False, threads=4,
                            group_by="column")
        except Exception as ex:   # noqa: BLE001
            print("yf err", ex)
            continue
        if len(x):
            c = x["Close"] if isinstance(x.columns, pd.MultiIndex) else x[["Close"]].set_axis(ys, axis=1)
            v = x["Volume"] if isinstance(x.columns, pd.MultiIndex) else x[["Volume"]].set_axis(ys, axis=1)
            d = c.stack().rename("c").to_frame().join(v.stack().rename("v")).reset_index()
            d.columns = ["date", "ticker", "c", "v"]
            d["ticker"] = d.ticker.astype(str).str.replace("-", ".")
            d.astype({"c": "float32", "v": "float32"}).to_parquet(
                os.path.join(pdir, f"y_{len(dn['y']):06d}.parquet"))
        dn["y"].extend(b)
        json.dump(dn, open(done, "w"))
        print("yf", i, len(need_y), flush=True)
        time.sleep(1)


def read_prices(src, tickers):
    parts = []
    for fn in sorted(glob.glob(os.path.join(D13, "px", f"{src}_*.parquet"))):
        d = pd.read_parquet(fn)
        d = d[d.ticker.isin(tickers)]
        d["ticker"] = d.ticker.astype("category")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["ticker"] = d.ticker.astype(str).astype("category")
    return d.drop_duplicates(["date", "ticker"])


def daily_returns(pa, py, tickers):
    """Wide daily return panel. Alpaca from 2019-12 (official closing cross), yfinance before; each fills gaps.
    Built in column chunks (float32) to stay well under the memory limit."""
    idx = pd.DatetimeIndex(sorted(set(pa.date.unique()) | set(py.date.unique())))
    idx = idx[idx.dayofweek < 5]
    cols = sorted(set(pa.ticker.unique()) | set(py.ticker.unique()))
    late = (idx >= "2019-12-02")[:, None]
    R, C, V = [], [], []
    for i in range(0, len(cols), 1000):
        cc = set(cols[i:i + 1000])
        w = lambda p, k: p[p.ticker.isin(cc)].astype({"ticker": str}).pivot_table(index="date", columns="ticker", values=k).reindex(
            index=idx, columns=sorted(cc)).astype("float32")
        ca, cy, va, vy = w(pa, "c"), w(py, "c"), w(pa, "v"), w(py, "v")
        ra, ry = ca / ca.ffill().shift(1) - 1, cy / cy.ffill().shift(1) - 1
        # a return across a gap of more than 5 trading days is not used (stale or reused symbol)
        ra = ra.where(ca.ffill(limit=5).shift(1).notna())
        ry = ry.where(cy.ffill(limit=5).shift(1).notna())
        r = pd.DataFrame(np.where(late, ra.fillna(ry), ry.fillna(ra)), index=idx, columns=ca.columns)
        R.append(r.where((r > -0.95) & (r < 3.0)).astype("float32"))
        C.append(pd.DataFrame(np.where(late, ca.fillna(cy), cy.fillna(ca)), index=idx, columns=ca.columns))
        V.append(pd.DataFrame(np.where(late, va.fillna(vy), vy.fillna(va)), index=idx, columns=ca.columns))
        del ca, cy, va, vy, ra, ry
    return pd.concat(R, axis=1), pd.concat(C, axis=1), pd.concat(V, axis=1)


def build_map():
    ideas = pd.read_parquet(os.path.join(D13, "ideas.parquet"))
    names = pd.read_parquet(os.path.join(D13, "names.parquet"))
    cr = pd.read_parquet(os.path.join(D13, "cusip_ret.parquet"))
    cus = sorted(ideas.cusip.unique())
    fc = os.path.join(D13, "candidates.parquet")
    old = pd.read_parquet(fc) if os.path.exists(fc) else pd.DataFrame(columns=["cusip", "name", "how", "cand"])
    miss = sorted(set(cus) - set(old.cusip))
    if miss:
        new = ticker_candidates(miss, names).assign(cand=lambda d: d.cand.map(" ".join))
        old = pd.concat([old, new], ignore_index=True)
        old.to_parquet(fc, index=False)
    m = old[old.cusip.isin(set(cus))].copy()
    m["cand"] = m.cand.fillna("").str.split()
    allc = sorted({t for c in m.cand for t in c} | {"SPY"})
    print("cusips", len(cus), "with candidates", int(m.cand.map(len).gt(0).sum()), "tickers", len(allc), flush=True)
    fetch_prices(allc)
    r, c, v = daily_returns(read_prices("a", set(allc)), read_prices("y", set(allc)), set(allc))
    print("panel", r.shape, flush=True)
    # quarter-end returns of each candidate vs 13F implied returns of the CUSIP
    qe = c.ffill(limit=5).resample("QE").last()
    qret = (qe.shift(-1) / qe - 1)
    qret.index = qn(pd.Series(qret.index)).values
    ideas_q = ideas.groupby("cusip").q.agg(["min", "max"])
    crs = cr.dropna(subset=["ret"]).set_index(["cusip", "q"]).ret
    best = []
    for _, row in m.iterrows():
        sc = []
        for t in row.cand:
            if t not in qret.columns:
                continue
            try:
                s = crs.loc[row.cusip]
            except KeyError:
                continue
            lo, hi = ideas_q.loc[row.cusip, "min"] - 4, ideas_q.loc[row.cusip, "max"] + 2
            s = s[(s.index >= lo) & (s.index <= hi)]
            j = pd.concat([s, qret[t]], axis=1, join="inner").dropna()
            if len(j) >= 2:
                sc.append((float((j.iloc[:, 0] - j.iloc[:, 1]).abs().median()), len(j), t))
        sc.sort()
        if sc:
            best.append(dict(cusip=row.cusip, ticker=sc[0][2], gap=sc[0][0], nq=sc[0][1]))
    b = pd.DataFrame(best)
    m = m.merge(b, on="cusip", how="left")
    m["mapped"] = m.gap < 0.06
    m["cand"] = m.cand.map(lambda x: " ".join(x))
    m.to_csv(os.path.join(D13, "ticker_map.csv"), index=False)
    r.astype("float32").to_parquet(os.path.join(D13, "ret_wide.parquet"))
    c.astype("float32").to_parquet(os.path.join(D13, "close_wide.parquet"))
    v.astype("float32").to_parquet(os.path.join(D13, "vol_wide.parquet"))
    # match rate by number and by idea count
    ideas = ideas.merge(m[["cusip", "mapped", "how"]], on="cusip", how="left")
    summ = pd.DataFrame([
        dict(measure="distinct best-idea CUSIPs", n=len(m), name_candidate=float(m.how.notna().mean()),
             mapped_and_price_checked=float(m.mapped.mean())),
        dict(measure="best-idea rows (filing x CUSIP)", n=len(ideas), name_candidate=float(ideas.how.notna().mean()),
             mapped_and_price_checked=float(ideas.mapped.fillna(False).mean())),
        dict(measure="skilled best-idea rows (top 20%)", n=int((ideas.skill_pct <= 0.2).sum()),
             name_candidate=float(ideas[ideas.skill_pct <= 0.2].how.notna().mean()),
             mapped_and_price_checked=float(ideas[ideas.skill_pct <= 0.2].mapped.fillna(False).mean()))])
    summ.to_csv(os.path.join(RES, "study210_mapping.csv"), index=False)
    print(summ.to_string(index=False))
    print(m.how.value_counts(dropna=False))


def build():
    build_ideas()
    build_map()


# ------------------------------------------------------------------ backtest
def load_bt():
    ideas = pd.read_parquet(os.path.join(D13, "ideas.parquet"))
    m = pd.read_csv(os.path.join(D13, "ticker_map.csv"))
    m = m[m.mapped == True]   # noqa: E712
    ideas = ideas.merge(m[["cusip", "ticker"]], on="cusip", how="left")
    keep = sorted(set(m.ticker) | {"SPY"})
    r = pd.read_parquet(os.path.join(D13, "ret_wide.parquet"), columns=keep)
    c = pd.read_parquet(os.path.join(D13, "close_wide.parquet"), columns=keep)
    v = pd.read_parquet(os.path.join(D13, "vol_wide.parquet"), columns=keep)
    spy = r["SPY"].copy()
    days = spy.dropna().index
    days = days[days >= "2013-06-01"]
    r, c, v = r.reindex(days), c.reindex(days), v.reindex(days)
    adv = (c * v).rolling(20, min_periods=10).median().shift(1)
    vol = np.log(c / c.shift(1)).rolling(20, min_periods=10).std().shift(1)
    hs = half_spread_model(adv.values, c.shift(1).values, np.nan_to_num(vol.clip(lower=1e-3).values, nan=0.02),
                           "15:45")
    cost = pd.DataFrame(1.3 + 0.1 * hs + 2.5, index=days, columns=c.columns).ffill().fillna(10.0)
    # filing becomes active at the close 2 trading days after the filing date
    pos = np.searchsorted(days.values, ideas.filed.values, side="right") - 1   # last trading day <= filed
    ideas["start_i"] = pos + 2
    return ideas, r, spy.reindex(days), cost, days


def active_sets(ideas, days):
    """For each filing: active from start_i until the same manager's next filing starts, or 200 days after period."""
    f = ideas.drop_duplicates("acc")[["acc", "cik", "start_i", "period"]].sort_values(["cik", "start_i"])
    f["end_i"] = f.groupby("cik").start_i.shift(-1)
    lim = np.searchsorted(days.values, (f.period + pd.Timedelta(days=200)).values)
    f["end_i"] = np.minimum(f.end_i.fillna(len(days)), lim).astype(int)
    return ideas.merge(f[["acc", "end_i"]], on="acc")


def run_book(ideas, r, cost, days, cut, N, skill=True):
    x = ideas if not skill else ideas[ideas.skill_pct <= cut]
    x = x[x.start_i < x.end_i]
    ev = sorted(set(x.start_i) | set(x.end_i))
    rv = r.values
    col = {t: j for j, t in enumerate(r.columns)}
    cv = cost.values
    gross = np.full(len(days), np.nan)
    tc = np.zeros(len(days))
    npos, nnp = np.zeros(len(days)), np.zeros(len(days))
    held, picks = [], []
    starts = x.start_i.values
    ends = x.end_i.values
    for k, e in enumerate(ev):
        if e >= len(days):
            break
        act = x[(starts <= e) & (ends > e)]
        if skill:
            s = act.groupby("cusip").agg(cnt=("acc", "size"), ow=("ow", "sum"), ticker=("ticker", "first"))
            s = s.sort_values(["cnt", "ow"], ascending=False).head(N)
        else:
            s = act.groupby("cusip").agg(cnt=("acc", "size"), ticker=("ticker", "first"))
        new = [col[t] for t in s.ticker.dropna() if t in col]
        nopx = len(s) - len(new)
        picks.append(dict(date=days[e], n=len(s), priced=len(new), tickers=" ".join(s.ticker.fillna("?").astype(str))))
        # trade at close of day e: cost on names entering and leaving (equal weight 1/len)
        old = set(held)
        nw = set(new)
        if nw != old:
            buy = sum(cv[e, j] for j in nw - old) / max(len(nw), 1)
            sell = sum(cv[e, j] for j in old - nw) / max(len(old), 1)
            tc[e] += buy + sell
        held = new
        nxt = ev[k + 1] if k + 1 < len(ev) else len(days) - 1
        for d in range(e + 1, min(nxt, len(days) - 1) + 1):
            if held:
                rr = rv[d, held]
                gross[d] = np.nanmean(rr) if np.isfinite(rr).any() else 0.0
            npos[d], nnp[d] = len(held), nopx
    o = pd.DataFrame({"gross": gross * 1e4, "cost": tc, "npos": npos, "nopx": nnp}, index=days)
    o["cost"] = o.cost.shift(1).fillna(0)  # cost paid at close of e, booked on the first holding day
    o["net"] = o.gross - o.cost
    return o.dropna(subset=["gross"]), pd.DataFrame(picks)


def stats(r, periods=252):
    r = r.dropna() / 1e4
    sd = r.std()
    return dict(n=len(r), ann_ret=float((1 + r).prod() ** (periods / max(len(r), 1)) - 1) if len(r) else np.nan,
                sharpe=float(r.mean() / sd * np.sqrt(periods)) if sd else np.nan)


def monthly_excess_t(a, b):
    ma = (1 + a / 1e4).groupby(a.index.to_period("M")).prod() - 1
    mb = (1 + b / 1e4).groupby(b.index.to_period("M")).prod() - 1
    ex = (ma - mb).dropna()
    return float(ex.mean() / ex.std() * np.sqrt(len(ex))), float(ex.mean() * 12), len(ex)


def summarize(o, spy, ew, tag, **kw):
    spy = spy.reindex(o.index) * 1e4
    ew = ew.reindex(o.index)
    row = dict(tag=tag, **kw, start=str(o.index.min().date()), end=str(o.index.max().date()))
    for k, s in [("net", o.net), ("gross", o.gross), ("spy", spy), ("ew_all", ew)]:
        st = stats(s)
        row.update({f"{k}_annret": st["ann_ret"], f"{k}_sharpe": st["sharpe"]})
    row["ex_spy_t"], row["ex_spy_ann"], row["months"] = monthly_excess_t(o.net, spy)
    row["ex_ew_t"], row["ex_ew_ann"], _ = monthly_excess_t(o.net, ew)
    row["cost_ann_bp"] = float(o.cost.sum() / max(len(o), 1) * 252)
    row["avg_names"] = float(o.npos.mean())
    row["avg_unpriced"] = float(o.nopx.mean())
    return row


def ew_all(ideas, r, cost, days):
    o, _ = run_book(ideas, r, cost, days, None, None, skill=False)
    return o.gross


def discover():
    ideas, r, spy, cost, days = load_bt()
    ideas = active_sets(ideas, days)
    ew = ew_all(ideas, r, cost, days)
    rows = []
    for cut in CUTS:
        for N in NS:
            o, _ = run_book(ideas, r, cost, days, cut, N)
            o = o[(o.index >= "2014-01-01") & (o.index <= "2019-12-31")]
            rows.append(summarize(o, spy, ew, "choose_2014_2019", cut=cut, N=N))
            print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(RES, "study210_grid.csv"), index=False)
    b = g.sort_values("net_sharpe", ascending=False).iloc[0]
    fz = dict(cut=float(b.cut), N=int(b.N), choice_net_sharpe=float(b.net_sharpe), choice_spy_sharpe=float(b.spy_sharpe),
              frozen_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
              pass_rule="confirmation 2020-01..2026-10: net Sharpe > SPY Sharpe, monthly excess over SPY t > 2, "
                        "positive excess in most calendar years")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    print(g.round(3).to_string(index=False))
    print(json.dumps(fz, indent=1))


def confirm():
    fz = json.load(open(FROZEN))
    ideas, r, spy, cost, days = load_bt()
    ideas = active_sets(ideas, days)
    ew = ew_all(ideas, r, cost, days)
    o, picks = run_book(ideas, r, cost, days, fz["cut"], fz["N"])
    o = o[o.index >= "2020-01-01"]
    rows = [summarize(o, spy, ew, "confirm_2020_2026", cut=fz["cut"], N=fz["N"])]
    for y, oy in o.groupby(o.index.year):
        rows.append(summarize(oy, spy, ew, f"confirm_{y}", cut=fz["cut"], N=fz["N"]))
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(RES, "study210_confirm.csv"), index=False)
    out = o.assign(spy=spy.reindex(o.index) * 1e4, ew_all=ew.reindex(o.index))
    out.to_csv(os.path.join(RES, "study210_daily.csv"))
    picks.to_csv(os.path.join(RES, "study210_picks.csv"), index=False)
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    {"fetch": fetch, "build": build, "build_ideas": build_ideas, "build_map": build_map, "discover": discover,
     "confirm": confirm}[sys.argv[1]]()
