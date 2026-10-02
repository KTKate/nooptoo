"""Point-in-time fundamentals panels from SEC EDGAR XBRL company facts.

Sources
  * https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip (all XBRL facts per filer, ~1.4 GB),
    saved to data/local/sec/companyfacts.zip; https://www.sec.gov/files/company_tickers.json (ticker -> CIK, as of
    today: renamed / delisted tickers are missed). Requests to sec.gov carry a contact User-Agent; only 2 requests.
  * Split events data/local/events/splits_events_used.csv + splits_yf.csv (src/fetch_events_splits.py) to put
    reported share counts on the same split basis as P['rawc'] (Yahoo close, split-adjusted to today).

Extraction (us-gaap / dei, units USD or shares, forms 10-Q, 10-K, 10-KT and their /A amendments only):
  revenue (max of RevenueFromContractWithCustomerExcl/InclAssessedTax, Revenues, SalesRevenueNet within a filing),
  NetIncomeLoss (else ProfitLoss), OperatingIncomeLoss, NetCashProvidedByUsedInOperatingActivities,
  PaymentsToAcquirePropertyPlantAndEquipment (else ...ProductiveAssets), ShareBasedCompensation (else
  AllocatedShareBasedCompensationExpense), ResearchAndDevelopmentExpense; instants CashCashEquivalentsAndShortTermInvestments, else
  CashAndCashEquivalents + ShortTermInvestments / MarketableSecuritiesCurrent / DebtSecuritiesCurrent),
  LongTermDebt (else LongTermDebtNoncurrent + Current),
  StockholdersEquity, Assets, shares = dei:EntityCommonStockSharesOutstanding (else us-gaap
  CommonStockSharesOutstanding, else WeightedAverageNumberOfDilutedSharesOutstanding).
  Filings are replayed in filed-date order; each (start, end) period keeps the value of the latest filing that
  reported it (restatements in later comparatives replace earlier values from their filed date on).
  Quarter = 3-month fact, else YTD(end) - YTD(previous quarter end, same start) (covers FY - 9M for Q4).
  TTM = 12-month fact, else YTD + last FY - prior-year YTD, else the sum of four derived quarters.
  revenue_growth_yoy = latest quarter / same quarter a year earlier - 1 (NaN if the base <= 0).
  A field whose latest period end lags the company's latest reported period end by > 200 days is set NaN (tag
  dropped / renamed), so stale values are not carried forever.

Point-in-time rule: the value on trading day t uses only filings with filed date strictly before t (a filing
filed on day t counts from the next trading day, as filings often arrive after the close); values are
forward-filled between filings. Price-based ratios use the raw close of t-1 (P['rawc'].shift(1)).

Share-count cleaning: reported counts < 100k (pre-IPO / spin-off shells) are dropped; shares are multiplied by all
split ratios dated after the share-count date. Then, per ticker, a filing-to-filing share change of > 3x (or < 1/3x)
that matches a known split within the window is undone (split already reflected by the filer), and an isolated > 3x
spike that reverts at the next filing is replaced by the previous value; an unconfirmed > 10x jump in the latest
filing is dropped. Persistent unexplained jumps are kept (real dilution / SPAC deals) and counted. Market caps > $6T
or > 200x total assets are masked, as are days where the implied market cap moves > 3x without a new filing (bad
price print).

Outputs
  data/local/fundamentals_panels.pkl: dict of float32 DataFrames on the P['c'] index x stock_cols(P) grid:
    ttm_revenue ttm_net_income ttm_op_income ttm_ocf ttm_capex ttm_fcf ttm_sbc ttm_rnd cash debt equity shares
    assets revenue_growth_yoy days_since_filing market_cap ev ev_sales pe_ttm fcf_yield sbc_to_revenue op_margin
    net_margin book_to_market cash_to_mcap
  results/fundamentals_coverage.csv: per year, share of liquid stock-days (20d median rawc*v > $5M) with
    non-NaN ttm_revenue / market_cap / both.
Usage: python src/fetch_fundamentals.py [--refresh]   (re-downloads the SEC files with --refresh)
"""
import os
import sys
import json
import time
import zipfile
import datetime as dt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, DATA, RES

SEC = os.path.join(DATA, "local", "sec")
ZIP = os.path.join(SEC, "companyfacts.zip")
TICK = os.path.join(SEC, "company_tickers.json")
OUT = os.path.join(DATA, "local", "fundamentals_panels.pkl")
EV = os.path.join(DATA, "local", "events")
UA = {"User-Agent": "nooptoo research incarnadins@gmail.com"}  # SEC fair-access contact header (sec.gov only)
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "10-KT/A"}
MIN_END = dt.date(2016, 6, 1).toordinal()
STALE = 200

# field -> (kind, [(namespace, concept, unit)], merge rule within one filing)
FIELDS = {
    "revenue": ("flow", [("us-gaap", c, "USD") for c in (
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax")], "max"),
    "net_income": ("flow", [("us-gaap", "NetIncomeLoss", "USD"), ("us-gaap", "ProfitLoss", "USD")], "first"),
    "op_income": ("flow", [("us-gaap", "OperatingIncomeLoss", "USD")], "first"),
    "ocf": ("flow", [("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),
                     ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "USD")], "first"),
    "capex": ("flow", [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
                       ("us-gaap", "PaymentsToAcquireProductiveAssets", "USD")], "first"),
    "sbc": ("flow", [("us-gaap", "ShareBasedCompensation", "USD"),
                     ("us-gaap", "AllocatedShareBasedCompensationExpense", "USD")], "first"),
    "rnd": ("flow", [("us-gaap", "ResearchAndDevelopmentExpense", "USD"),
                     ("us-gaap", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost", "USD")], "first"),
    "cash0": ("inst", [("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD"),
                       ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "USD")], "first"),
    "sti": ("inst", [("us-gaap", "ShortTermInvestments", "USD"), ("us-gaap", "MarketableSecuritiesCurrent", "USD"),
                     ("us-gaap", "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "USD"),
                     ("us-gaap", "DebtSecuritiesCurrent", "USD"), ("us-gaap", "AvailableForSaleSecuritiesCurrent", "USD")],
            "first"),
    "cash_sti": ("inst", [("us-gaap", "CashCashEquivalentsAndShortTermInvestments", "USD")], "first"),
    "ltd": ("inst", [("us-gaap", "LongTermDebt", "USD")], "first"),
    "ltd_nc": ("inst", [("us-gaap", "LongTermDebtNoncurrent", "USD")], "first"),
    "ltd_c": ("inst", [("us-gaap", "LongTermDebtCurrent", "USD")], "first"),
    "equity": ("inst", [("us-gaap", "StockholdersEquity", "USD"),
                        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD")],
               "first"),
    "assets": ("inst", [("us-gaap", "Assets", "USD")], "first"),
    "shares": ("inst", [("dei", "EntityCommonStockSharesOutstanding", "shares"),
                        ("us-gaap", "CommonStockSharesOutstanding", "shares"),
                        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares")], "first"),
}
FLOWS = [k for k, v in FIELDS.items() if v[0] == "flow"]
INSTS = [k for k, v in FIELDS.items() if v[0] == "inst"]
# per-event output columns (one row per distinct filed date of a company)
EVCOLS = ["filed", "rev_q", "revenue_growth_yoy"] + ["ttm_" + f for f in FLOWS] + \
         ["cash", "debt", "equity", "assets", "shares", "shares_end"]

_OD = {}


def od(s):
    v = _OD.get(s)
    if v is None:
        v = _OD[s] = dt.date.fromisoformat(s).toordinal()
    return v


# ---------------------------------------------------------------- download
def download(refresh=False):
    os.makedirs(SEC, exist_ok=True)
    for url, fn in [("https://www.sec.gov/files/company_tickers.json", TICK),
                    ("https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip", ZIP)]:
        if os.path.exists(fn) and not refresh:
            continue
        with requests.get(url, headers=UA, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(fn + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(fn + ".part", fn)
        time.sleep(1)  # far below the SEC limit of 10 requests/s


# ---------------------------------------------------------------- per-company extraction
def _field_records(facts, field):
    """{filed: [(start, end, val)]} after merging concepts within each filing (accession)."""
    kind, srcs, rule = FIELDS[field]
    best = {}
    for prio, (ns, concept, unit) in enumerate(srcs):
        c = facts.get(ns, {}).get(concept)
        if not c or unit not in c.get("units", {}):
            continue
        for r in c["units"][unit]:
            if r.get("form") not in FORMS or "filed" not in r or r.get("val") is None:
                continue
            e = od(r["end"])
            if e < MIN_END:
                continue
            s = od(r["start"]) if (kind == "flow" and "start" in r) else None
            if kind == "flow" and s is None:
                continue
            key = (r["accn"], s, e)
            v = float(r["val"])
            old = best.get(key)
            if old is None or (rule == "first" and prio < old[0]) or (rule == "max" and v > old[2]):
                best[key] = (prio, od(r["filed"]), v)
    out = {}
    for (accn, s, e), (prio, f, v) in best.items():
        out.setdefault(f, []).append((s, e, v))
    return out


def _near(d, x, lo, hi):
    """First key of dict d in [x+lo, x+hi], searched from the centre outward."""
    mid = (lo + hi) // 2
    for k in range(0, max(mid - lo, hi - mid) + 1):
        for y in (x + mid - k, x + mid + k):
            if lo <= y - x <= hi and y in d:
                return y
    return None


class Flow:
    """by_end[end][start] = value of the latest filing reporting that period."""
    def __init__(self):
        self.be = {}

    def update(self, recs):
        for s, e, v in recs:
            self.be.setdefault(e, {})[s] = v

    def q(self, E):
        d = self.be.get(E)
        if not d:
            return None
        for s, v in d.items():
            if 80 <= E - s <= 100:
                return v
        for s, v in sorted(d.items()):  # longest YTD first
            if E - s <= 100 or E - s > 380:
                continue
            Ep = _near(self.be, E, -105, -75)
            if Ep is not None and s in self.be[Ep]:
                return v - self.be[Ep][s]
        return None

    def fy(self, E):
        for s, v in self.be.get(E, {}).items():
            if 350 <= E - s <= 380:
                return v
        return None

    def ttm(self, E):
        v = self.fy(E)
        if v is not None:
            return v
        d = self.be.get(E, {})
        for s, ytd in sorted(d.items()):
            if not 80 <= E - s <= 300:
                continue
            Epy = _near(self.be, E, -375, -355)
            Efy = _near(self.be, s, -11, 9)  # last FY ends the day before the YTD start
            if Epy is None or Efy is None:
                continue
            spy = [x for x in self.be[Epy] if abs((s - x) - 365) <= 10]
            fy = self.fy(Efy)
            if spy and fy is not None:
                return ytd + fy - self.be[Epy][spy[0]]
        tot, e = 0.0, E
        for i in range(4):
            v = self.q(e)
            if v is None:
                return None
            tot += v
            if i < 3:
                e = _near(self.be, e, -105, -75)
                if e is None:
                    return None
        return tot

    def latest(self):
        ends = sorted((e for e, d in self.be.items() if any(80 <= e - s <= 380 for s in d)), reverse=True)
        return ends[:3]


def company_events(facts):
    recs = {f: _field_records(facts, f) for f in FIELDS}
    filed = sorted(set().union(*[set(r) for r in recs.values()]))
    flows = {f: Flow() for f in FLOWS}
    inst = {f: {} for f in INSTS}
    comp_end = -1
    rows = []
    for fd in filed:
        for f in FLOWS:
            for s, e, v in recs[f].get(fd, ()):
                if f in ("revenue", "op_income", "ocf", "net_income"):
                    comp_end = max(comp_end, e)
            flows[f].update(recs[f].get(fd, ()))
        for f in INSTS:
            for s, e, v in recs[f].get(fd, ()):
                inst[f][e] = v
                if f in ("cash0", "equity", "assets"):
                    comp_end = max(comp_end, e)
        if comp_end < 0:
            continue
        lim = comp_end - STALE
        row = {"filed": fd}
        for f in FLOWS:
            val = None
            for E in flows[f].latest():
                if E < lim:
                    break
                val = flows[f].ttm(E)
                if val is not None:
                    if f == "revenue":
                        qv = flows[f].q(E)
                        Epy = _near(flows[f].be, E, -375, -355)
                        qp = flows[f].q(Epy) if Epy is not None else None
                        row["rev_q"] = qv
                        row["revenue_growth_yoy"] = qv / qp - 1 if (qv is not None and qp and qp > 0) else None
                    break
            row["ttm_" + f] = val

        def last(f):
            d = inst[f]
            if not d:
                return None, None
            E = max(d)
            return (d[E], E) if E >= lim else (None, None)
        c0, E = last("cash0")
        if c0 is not None:
            row["cash"] = inst["cash_sti"].get(E, c0 + inst["sti"].get(E, 0.0))
        Ed = max(list(inst["ltd"]) + list(inst["ltd_nc"]) or [-1])
        if Ed >= lim:
            row["debt"] = inst["ltd"][Ed] if Ed in inst["ltd"] else inst["ltd_nc"][Ed] + inst["ltd_c"].get(Ed, 0.0)
        row["equity"] = last("equity")[0]
        row["assets"] = last("assets")[0]
        sh, se = last("shares")
        if sh is not None and sh >= 1e5:  # smaller counts are shell / pre-IPO placeholders (e.g. 1,000 shares)
            row["shares"], row["shares_end"] = sh, se
        rows.append([row.get(c) for c in EVCOLS])
    return rows


def _work(ciks):
    z = zipfile.ZipFile(ZIP)
    names = set(z.namelist())
    out = {}
    for cik in ciks:
        fn = f"CIK{cik:010d}.json"
        if fn not in names:
            continue
        facts = json.loads(z.read(fn)).get("facts", {})
        rows = company_events(facts)
        if rows:
            out[cik] = np.array(rows, dtype="float64")
    return out


# ---------------------------------------------------------------- panels
def load_splits():
    fr = []
    for fn in ["splits_events_used.csv", "splits_yf.csv"]:
        p = os.path.join(EV, fn)
        if os.path.exists(p):
            fr.append(pd.read_csv(p, usecols=["ticker", "date", "ratio"]))
    s = pd.concat(fr).drop_duplicates(["ticker", "date"])
    s = s[(s.ratio > 0) & np.isfinite(s.ratio) & (s.ratio != 1)]
    out = {}
    for t, g in s.groupby("ticker"):
        out[t] = [(pd.Timestamp(d).toordinal(), r) for d, r in zip(g.date, g.ratio)]
    return out


def clean_shares(ev, splits, stats):
    """ev: event array for one ticker. Adjusts EVCOLS 'shares' in place to today's split basis."""
    ish, ise = EVCOLS.index("shares"), EVCOLS.index("shares_end")
    sh = ev[:, ish]
    ok = np.isfinite(sh)
    for i in np.where(ok)[0]:
        for d, r in splits:
            if d > ev[i, ise]:
                sh[i] *= r
    idx = np.where(ok)[0]
    # collapse to share-count changes
    for a in range(1, len(idx)):
        i0, i1 = idx[a - 1], idx[a]
        r = sh[i1] / sh[i0]
        if 1 / 3 <= r <= 3:
            continue
        fixed = False
        nxt = idx[a + 1] if a + 1 < len(idx) else None
        for d, sr in splits:
            if ev[i0, ise] - 30 <= d <= ev[i1, ise] + 120:
                for m in (sr, 1 / sr):
                    if abs(np.log(r / m)) < np.log(1.25):
                        # one side was filed on the other split basis: fix the one its neighbours disagree with
                        if nxt is not None and 1 / 1.5 <= sh[nxt] / sh[i1] <= 1.5:
                            sh[i0] *= m
                        else:
                            sh[i1] /= m
                        fixed = True
                        break
            if fixed:
                break
        if fixed:
            stats["split_fixed"] += 1
            continue
        if nxt is not None and 1 / 1.5 <= sh[nxt] / sh[i0] <= 1.5:
            sh[i1] = sh[i0]
            stats["spike_dropped"] += 1
        elif nxt is None and not 0.1 <= r <= 10:
            sh[i1] = np.nan  # unconfirmed > 10x jump in the latest filing: likely a unit error
            stats["spike_dropped"] += 1
        else:
            stats["jump_kept"] += 1
    ev[:, ish] = sh


def step_panel(days_ord, events, col):
    """events: {ticker: array}; returns T x len(events) array, value from first day with day > filed."""
    T = len(days_ord)
    out = np.full((T, len(events)), np.nan, dtype="float32")
    ic = EVCOLS.index(col) if col != "filed" else 0
    for j, ev in enumerate(events.values()):
        pos = np.searchsorted(days_ord, ev[:, 0], side="right")
        keep = pos < T
        im = np.full(T, -1)
        im[pos[keep]] = np.where(keep)[0]
        im = np.maximum.accumulate(im)
        good = im >= 0
        out[good, j] = ev[im[good], ic]
    return out


def main():
    download("--refresh" in sys.argv)
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    days_ord = np.array([d.toordinal() for d in days])
    tmap = {}
    for v in json.load(open(TICK)).values():
        tmap.setdefault(v["ticker"].upper().replace(".", "-"), int(v["cik_str"]))
    tick = [t for t in cols if t in tmap]
    print(f"{len(cols)} stock columns, {len(tick)} mapped to a CIK ({len(tick) / len(cols):.1%})", flush=True)
    ciks = sorted({tmap[t] for t in tick})
    chunks = [ciks[i::24] for i in range(24)]
    t0 = time.time()
    res = {}
    with Pool(3, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, chunks)):
            res.update(r)
            print(f"  chunk {k + 1}/24, {len(res)} companies with events, {time.time() - t0:.0f}s", flush=True)
    splits = load_splits()
    stats = {"split_fixed": 0, "spike_dropped": 0, "jump_kept": 0}
    events = {}
    for t in tick:
        if tmap[t] in res:
            ev = res[tmap[t]].copy()
            clean_shares(ev, splits.get(t, []), stats)
            events[t] = ev
    print(f"{len(events)} tickers with filings; share cleaning {stats}", flush=True)

    tk = list(events)
    F = {}

    def panel(col):
        df = pd.DataFrame(step_panel(days_ord, events, col), index=days, columns=tk)
        return df.reindex(columns=cols).astype("float32")
    for c in ["revenue", "net_income", "op_income", "ocf", "capex", "sbc", "rnd"]:
        F["ttm_" + c] = panel("ttm_" + c)
    F["ttm_fcf"] = (F["ttm_ocf"] - F["ttm_capex"]).astype("float32")
    for c in ["cash", "debt", "equity", "shares", "assets", "revenue_growth_yoy"]:
        F[c] = panel(c)
    lastf = panel("filed")
    F["days_since_filing"] = (pd.DataFrame(np.repeat(days_ord[:, None], len(cols), 1), index=days, columns=cols)
                              - lastf).astype("float32")

    px = P["rawc"][cols].shift(1)
    mcap = F["shares"] * px
    # daily sanity: implied market cap jumping > 3x in a day without a filing that day (bad price print)
    jump = (mcap / mcap.shift(1))
    newf = lastf.ne(lastf.shift(1))
    bad = ((jump > 3) | (jump < 1 / 3)) & ~newf
    print(f"market-cap day jumps > 3x without a filing: {int(bad.values.sum())} stock-days masked", flush=True)
    implaus = (mcap > 6e12) | (mcap <= 0) | (mcap > 200 * F["assets"])
    print(f"implausible market caps (> $6T or > 200x assets): {int(implaus.values.sum())} stock-days masked")
    mcap = mcap.mask(bad | implaus)
    F["market_cap"] = mcap
    debt0 = F["debt"].where(F["debt"].notna() | F["cash"].isna(), 0.0)  # untagged debt treated as 0 when cash known
    F["ev"] = mcap + debt0 - F["cash"]
    rev = F["ttm_revenue"].where(F["ttm_revenue"] > 0)
    F["ev_sales"] = F["ev"] / rev
    F["pe_ttm"] = mcap / F["ttm_net_income"].where(F["ttm_net_income"] > 0)
    F["fcf_yield"] = F["ttm_fcf"] / mcap
    F["sbc_to_revenue"] = F["ttm_sbc"] / rev
    F["op_margin"] = F["ttm_op_income"] / rev
    F["net_margin"] = F["ttm_net_income"] / rev
    F["book_to_market"] = F["equity"] / mcap
    F["cash_to_mcap"] = F["cash"] / mcap
    for k in F:
        F[k] = F[k].replace([np.inf, -np.inf], np.nan).astype("float32")
    pd.to_pickle(F, OUT)
    print("saved", OUT, len(F), "panels", F["market_cap"].shape, flush=True)

    # coverage
    liq = (P["rawc"][cols] * P["v"][cols]).rolling(20, min_periods=10).median() > 5e6
    rows = []
    for y in sorted(set(days.year)):
        m = liq.loc[str(y)]
        n = m.values.sum()
        r = F["ttm_revenue"].loc[str(y)].notna() & m
        c = F["market_cap"].loc[str(y)].notna() & m
        rows.append(dict(year=y, liquid_stock_days=int(n), avg_liquid_names=round(n / len(m), 1),
                         ttm_revenue=r.values.sum() / n, market_cap=c.values.sum() / n,
                         both=(r & c).values.sum() / n))
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(RES, "fundamentals_coverage.csv"), index=False, float_format="%.4f")
    print(cov.to_string(index=False))

    for d in ["2025-09-30", "2026-09-29"]:
        for t in ["MDB", "AAPL", "NVDA", "GOOGL", "MSFT"]:
            if t in cols:
                print(d, t, f"mcap={F['market_cap'].at[d, t] / 1e9:.1f}B ev_sales={F['ev_sales'].at[d, t]:.2f} "
                      f"rev_yoy={F['revenue_growth_yoy'].at[d, t]:.3f} ttm_rev={F['ttm_revenue'].at[d, t] / 1e9:.2f}B "
                      f"pe={F['pe_ttm'].at[d, t]:.1f} days_since={F['days_since_filing'].at[d, t]:.0f}")


if __name__ == "__main__":
    main()
