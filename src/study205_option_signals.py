"""Study 205: option-implied signals and stock returns over 1-5 days.

Question: do option prices lead stock prices by enough to trade? Three published signals:
  ivs   implied-vol spread, mean(call IV - put IV) over matched strikes of one expiry (Cremers and Weinbaum 2010):
        expensive calls -> higher returns. Long the top N.
  skew  OTM put IV (K/S in 0.90-0.95, closest to 0.95) minus ATM call IV (K closest to S, within 5%)
        (Xing, Zhang and Zhao 2010): steep skew -> lower returns. Long the bottom N (flattest skew).
  datm  5-day change in ATM IV (mean of call and put IV at the strike closest to spot, same expiry both days)
        (An, Ang, Bali and Cakici 2014). The sign is not fixed by the brief: datm_up (long the largest rise) and
        datm_dn (long the largest fall) are both in the discovery grid.
Data: Alpaca options market data (read-only, data.alpaca.markets/v1beta1/options, daily bars; contract lists from the
read-only GET options/contracts reference endpoint), history from 2024-02. Universe: the 150 stocks of
data/local/hks_universe.csv with the highest 20-day median dollar volume at 2024-01-31 (Alpaca d1, before the sample, so
no look-ahead in the choice) that have listed options. For each stock and day the expiry is the nearest standard monthly
(third Friday, Thursday on a holiday) 20-60 calendar days out; strikes within +-10% of the stock close. IV from the option
daily-bar close (last trade) by Black-Scholes, r = 4.5%, no dividends, T = calendar days / 365, spot = the stock's
unadjusted official close the same day (the raw version of the d1 bar, since strikes are unadjusted). Contracts with no
trade that day or a close below $0.10 are dropped, as are IVs outside 1%-390%.
Timing: the option close is known only after 16:00, so a signal from day s is traded in the closing auction of the next
trading day t = s+1 and sold in the closing auction of t+K (K = 1 or 5). Returns from Alpaca d1 closes (split-adjusted).
Design: equal-weight top N = 10. Overlapping holds: K cohort chains with 1/K capital each; each chain's cohort replaces
the one it formed K days earlier, paying costs only on names that change. Daily book return = mean over chains of the
picks' daily returns. Excess = book minus the equal-weight daily return of all universe names with a signal (same day).
Also the gross top-minus-bottom decile spread of each signal (the test of the published effect), same 1/K bookkeeping.
Costs per side (bp): auction 1.3 + 0.1 x half-spread + 2.5, half-spread = core.half_spread_model at "15:45".
Protocol (pre-registered 2026-11-09, before any return was computed): choose signal in {ivs, skew, datm_up, datm_dn} and
K in {1, 5} by net excess Sharpe on trade days 2024-02..2024-12 only, write results/study205_frozen.json, then score
2025-01..2026-10 once. Pass: net excess Sharpe > 1, t > 2, and a positive net excess mean in both 2025 and 2026.
No changes after the confirmation run.
    python src/study205_option_signals.py fetch | build | discover | confirm
Output: results/study205_option_signals.csv (discovery grid and confirmation rows), results/study205_frozen.json,
        results/study205_daily.csv (confirmation daily book), data/local/options/s205_*.parquet (download cache)
"""
import json
import os
import sys

os.environ.setdefault("ALPACA_PER_MIN", "90")    # own ceiling; the shared file-lock limiter caps the account at 190
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
import alpaca_data as A  # noqa: E402
import fetch_options as F  # noqa: E402
from core import RES, ETFS, half_spread_model  # noqa: E402

OUT = os.path.join(A.LOCAL, "options")
FROZEN = os.path.join(RES, "study205_frozen.json")
CSV = os.path.join(RES, "study205_option_signals.csv")
NU, N, R = 150, 10, 0.045
SIGS = {"ivs": ("ivs", True), "skew": ("skew", False), "datm_up": ("datm", True), "datm_dn": ("datm", False)}
KS = [1, 5]
DATA_END = "2026-10-30"


# ------------------------------------------------------------------ fetch
def universe_candidates():
    d = A.read("d1", start="2023-12", end="2024-01", columns=["ts", "ticker", "c", "v"])
    d = d[d.ts <= "2024-01-31"]
    u = set(pd.read_csv(os.path.join(A.LOCAL, "hks_universe.csv")).ticker) - set(ETFS)
    d = d[d.ticker.isin(u)]
    d["dv"] = d.c * d.v
    last = d.sort_values("ts").groupby("ticker").tail(20)
    g = last.groupby("ticker").agg(dv=("dv", "median"), px=("c", "last"), n=("dv", "size"))
    g = g[(g.n >= 15) & (g.px > 5)]
    return g.sort_values("dv", ascending=False)


def monthly_expiries(ex):
    ex = pd.DatetimeIndex(sorted(set(ex)))
    ex = ex[(ex.day >= 15) & (ex.day <= 21) & (ex.day_of_week.isin([3, 4]))]
    s = pd.Series(ex, index=ex.to_period("M"))
    return pd.DatetimeIndex(s.groupby(level=0).max().values)    # Friday if listed, else the holiday Thursday


def contracts(u):
    fn = os.path.join(OUT, f"s205_contracts_{u}.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    rows = []
    for st in ["inactive", "active"]:
        p = dict(underlying_symbols=u, expiration_date_gte="2024-03-01", expiration_date_lte="2026-12-31", status=st,
                 limit=10000)
        while True:
            j = F.req(f"{F.TRADE}/options/contracts", p)     # read-only reference data, GET only
            rows.extend(j.get("option_contracts", []))
            tok = j.get("next_page_token")
            if not tok:
                break
            p["page_token"] = tok
    d = pd.DataFrame(rows)
    if len(d):
        d = d[["symbol", "type", "strike_price", "expiration_date", "size"]].drop_duplicates("symbol")
        d["strike_price"] = d.strike_price.astype(float)
        d["expiration_date"] = pd.to_datetime(d.expiration_date)
        # standard deliverable only: root == ticker (adjusted contracts get roots like AAPL1) and 100 shares
        d = d[d.symbol.str.match(rf"^{u}\d{{6}}[CP]\d{{8}}$") & (d["size"].astype(str) == "100")]
        d = d[d.expiration_date.isin(monthly_expiries(d.expiration_date))]
    else:
        d = pd.DataFrame(columns=["symbol", "type", "strike_price", "expiration_date", "size"])
    d.to_parquet(fn, index=False)
    return d


def raw_daily(tickers):
    fn = os.path.join(OUT, "s205_raw_daily.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    parts = []
    for i in range(0, len(tickers), 50):
        parts.append(A.bars(tickers[i:i + 50], "1Day", "2023-12-01T00:00:00Z", "2026-10-31T00:00:00Z",
                            adjustment="raw"))
    d = pd.concat(parts, ignore_index=True)
    d["date"] = d.ts.dt.normalize()
    d = d[["date", "ticker", "c"]].rename(columns={"c": "raw_c"})
    d.to_parquet(fn, index=False)
    return d


def fetch():
    os.makedirs(OUT, exist_ok=True)
    cand = universe_candidates()
    ufn = os.path.join(OUT, "s205_universe.csv")
    if os.path.exists(ufn):
        uni = pd.read_csv(ufn).ticker.tolist()
    else:
        uni = []
        for t in cand.index:
            c = contracts(t)
            ok = len(c) and c.expiration_date.nunique() >= 20
            print(t, len(c), c.expiration_date.nunique() if len(c) else 0, "ok" if ok else "skip", flush=True)
            if ok:
                uni.append(t)
            if len(uni) == NU:
                break
        pd.DataFrame({"ticker": uni, "dv_2024_01": cand.dv.reindex(uni).values}).to_csv(ufn, index=False)
    raw = raw_daily(uni)
    spot = raw.pivot(index="date", columns="ticker", values="raw_c").sort_index()
    allc = pd.concat([contracts(t).assign(ticker=t) for t in uni], ignore_index=True)
    last = pd.Timestamp(DATA_END)
    for E, g in allc.groupby("expiration_date"):
        a, b = E - pd.Timedelta(days=64), E - pd.Timedelta(days=19)
        if b < pd.Timestamp("2024-02-01") or a > last:
            continue
        fn = os.path.join(OUT, f"s205_bars_{E:%Y%m%d}.parquet")
        if os.path.exists(fn):
            continue
        w = spot[(spot.index >= a) & (spot.index <= b)]
        lo, hi = w.min() * 0.88, w.max() * 1.12
        g = g[(g.strike_price >= g.ticker.map(lo)) & (g.strike_price <= g.ticker.map(hi))]
        syms = g.symbol.tolist()
        end = min(b, last) + pd.Timedelta(days=1)
        bl = F.fetch_bars(syms, a.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) if syms else pd.DataFrame()
        if len(bl):
            bl["date"] = pd.to_datetime(bl.t).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            bl = bl[["symbol", "date", "c", "v", "n", "vw"]]
        else:
            bl = pd.DataFrame(columns=["symbol", "date", "c", "v", "n", "vw"])
        bl.to_parquet(fn, index=False)
        print(E.date(), "symbols", len(syms), "bars", len(bl), flush=True)


# ------------------------------------------------------------------ implied vols and signals
def bs_price(S, K, T, sig, call):
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (R + 0.5 * sig ** 2) * T) / sq
    d2 = d1 - sq
    c = S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return np.where(call, c, c - S + K * np.exp(-R * T))      # put by parity (European, no dividends)


def implied_vol(price, S, K, T, call):
    lo, hi = np.full(len(price), 0.005), np.full(len(price), 4.0)
    for _ in range(50):
        m = 0.5 * (lo + hi)
        up = bs_price(S, K, T, m, call) > price
        hi, lo = np.where(up, m, hi), np.where(up, lo, m)
    m = 0.5 * (lo + hi)
    return np.where((m > 0.01) & (m < 3.9), m, np.nan)


def build():
    fn = os.path.join(OUT, "s205_signals.parquet")
    uni = pd.read_csv(os.path.join(OUT, "s205_universe.csv")).ticker.tolist()
    raw = raw_daily(uni)
    meta = pd.concat([contracts(t).assign(ticker=t) for t in uni], ignore_index=True)
    meta = meta[["symbol", "ticker", "type", "strike_price", "expiration_date"]]
    files = sorted(f for f in os.listdir(OUT) if f.startswith("s205_bars_"))
    b = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in files], ignore_index=True)
    nb = len(b)
    b = b[(b.v > 0) & (b.c >= 0.10)].drop_duplicates(["symbol", "date"])
    b = b.merge(meta, on="symbol").merge(raw, on=["date", "ticker"])
    b = b.rename(columns={"strike_price": "K", "expiration_date": "E", "raw_c": "S"})
    b["T"] = (b.E - b.date).dt.days / 365.0
    b["m"] = b.K / b.S
    b = b[(b["T"] > 0) & (b.m >= 0.85) & (b.m <= 1.15)]
    call = (b.type == "call").values
    b["iv"] = implied_vol(b.c.values.astype(float), b.S.values, b.K.values, b["T"].values, call)
    print("option bars", nb, "used", len(b), "iv ok", b.iv.notna().mean().round(3), flush=True)
    b = b.dropna(subset=["iv"])
    # ATM IV per (ticker, date, expiry): call and put IV at the strike closest to spot (within 5%)
    a = b[(b.m - 1).abs() <= 0.05].copy()
    a["dist"] = (a.m - 1).abs()
    piv = a.pivot_table(index=["ticker", "date", "E", "K", "dist"], columns="type", values="iv").reset_index()
    piv["atm"] = piv[["call", "put"]].mean(axis=1)
    atm = piv.sort_values(["dist", "K"], kind="mergesort").groupby(["ticker", "date", "E"]).first().reset_index()
    atm_call = a[a.type == "call"].sort_values(["dist", "K"], kind="mergesort").groupby(["ticker", "date", "E"]).iv.first().rename("atm_call")
    # selected expiry per (ticker, date): nearest monthly with 20-60 days to go
    dte = (b.E - b.date).dt.days
    sel = b[(dte >= 20) & (dte <= 60)].groupby(["ticker", "date"]).E.min().rename("E").reset_index()
    x = b.merge(sel, on=["ticker", "date", "E"])
    w = x[(x.m - 1).abs() <= 0.10]
    pr = w.pivot_table(index=["ticker", "date", "K"], columns="type", values="iv").dropna()
    ivs = (pr.call - pr.put).groupby(level=[0, 1]).agg(["mean", "size"])
    ivs = ivs[ivs["size"] >= 2]["mean"].rename("ivs")
    p = x[(x.type == "put") & (x.m >= 0.90) & (x.m <= 0.95)].copy()
    p["d"] = (p.m - 0.95).abs()
    otm = p.sort_values(["d", "K"], kind="mergesort").groupby(["ticker", "date"]).iv.first().rename("otm_put")
    s = sel.set_index(["ticker", "date", "E"]).join(atm.set_index(["ticker", "date", "E"]).atm).join(atm_call)
    s = s.reset_index()
    # 5-trading-day change of ATM IV at the same expiry
    days = pd.DatetimeIndex(sorted(raw.date.unique()))
    lag = pd.Series(days[:-5], index=days[5:])
    s["date5"] = s.date.map(lag)
    a5 = atm.rename(columns={"date": "date5", "atm": "atm5"})[["ticker", "date5", "E", "atm5"]]
    s = s.merge(a5, on=["ticker", "date5", "E"], how="left")
    s["datm"] = s.atm - s.atm5
    s = s.set_index(["ticker", "date"]).join(ivs).join(otm)
    s["skew"] = s.otm_put - s.atm_call
    s = s.reset_index()[["ticker", "date", "E", "atm", "atm_call", "otm_put", "ivs", "skew", "datm"]]
    s.to_parquet(fn, index=False)
    print(s.describe().round(3).to_string())
    print("coverage by month (names with ivs / skew / datm):")
    cov = s.groupby(s.date.dt.to_period("M"))[["ivs", "skew", "datm"]].count() / s.groupby(s.date.dt.to_period("M")).date.nunique().values[:, None]
    print(cov.round(0).to_string())
    return s


# ------------------------------------------------------------------ backtest
def panel():
    uni = pd.read_csv(os.path.join(OUT, "s205_universe.csv")).ticker.tolist()
    d = A.read("d1", start="2023-11", tickers=uni, columns=["ts", "ticker", "c", "v"])
    d["date"] = d.ts.dt.normalize()
    d = d[d.date <= DATA_END].sort_values(["ticker", "date"])
    g = d.groupby("ticker")
    d["r"] = d.c / g.c.shift(1) - 1
    d.loc[d.r.abs() > 0.5, "r"] = np.nan     # data-error guard
    lr = np.log(d.c / g.c.shift(1))
    d["adv"] = (d.c * d.v).groupby(d.ticker).transform(lambda x: x.rolling(20, min_periods=10).median().shift(1))
    d["px"] = g.c.shift(1)
    d["vol"] = lr.groupby(d.ticker).transform(lambda x: x.rolling(20, min_periods=10).std().shift(1))
    hs = half_spread_model(d.adv, d.px, d.vol.clip(lower=1e-3), "15:45")
    d["side"] = (1.3 + 0.1 * hs + 2.5) / 1e4
    R_ = d.pivot(index="date", columns="ticker", values="r")
    C = d.pivot(index="date", columns="ticker", values="side")
    PX = d.pivot(index="date", columns="ticker", values="c")
    return R_, C, PX


def run(S, R_, C, PX, name, K):
    """Daily book (bp): K chains with 1/K capital each; chain i % K rebalances on trade day i (the signal of day i-1),
    paying side costs on the names that enter and leave. Costs are booked on the first return day after the trade."""
    col, top = SIGS[name]
    sig = S.pivot(index="date", columns="ticker", values=col).reindex(index=R_.index, columns=R_.columns)
    sig = sig.shift(1)                                   # known after the close of s, traded at the close of s+1
    sig = sig.where(PX > 5)
    days = R_.index
    chains = {}                                          # chain id -> (picks, top decile, bottom decile, universe)
    acc = {k: np.zeros(len(days)) for k in ["gross", "cost", "univ", "d10_d1", "n_chains"]}
    per_trade = []
    for i, t in enumerate(days[:-1]):
        s = sig.loc[t].dropna()
        ch = i % K
        prev = chains[ch][0] if ch in chains else pd.Index([])
        if len(s) >= 40:
            s = s if top else -s
            pick = s.nlargest(N).index
            q = s.rank(pct=True)
            new, out = pick.difference(prev), prev.difference(pick)
            cost = (C.loc[t, new].fillna(0.003).sum() + C.loc[t, out].fillna(0.003).sum()) / N
            chains[ch] = (pick, q[q > 0.9].index, q[q <= 0.1].index, s.index)
            fw = R_.iloc[i + 1:i + 1 + K]
            if len(fw) == K:
                gr = ((1 + fw[pick].fillna(0)).prod() - 1).mean()
                un = ((1 + fw[s.index].fillna(0)).prod() - 1).mean()
                per_trade.append(dict(date=days[i + 1], gross=gr, univ=un, cost=cost))
        else:                                            # no signal: the chain goes flat
            cost = C.loc[t, prev].fillna(0.003).sum() / N if len(prev) else 0.0
            chains.pop(ch, None)
        acc["cost"][i + 1] += cost / K
        r = R_.iloc[i + 1]
        for pk, hi, lo, un in chains.values():
            acc["gross"][i + 1] += r[pk].fillna(0).mean() / K
            acc["univ"][i + 1] += r[un].fillna(0).mean() / K
            acc["d10_d1"][i + 1] += (r[hi].fillna(0).mean() - r[lo].fillna(0).mean()) / K
            acc["n_chains"][i + 1] += 1
    book = pd.DataFrame(acc, index=days)
    book = book[book.n_chains > 0].copy()
    book["net"] = book.gross - book.cost
    book["excess_gross"] = book.gross - book.univ
    book["excess_net"] = book.net - book.univ
    book[["gross", "cost", "univ", "d10_d1", "net", "excess_gross", "excess_net"]] *= 1e4
    pt = pd.DataFrame(per_trade).set_index("date") * 1e4 if per_trade else pd.DataFrame()
    return book, pt


def stats(r):
    r = r.dropna()
    sd = r.std()
    return dict(n=len(r), mean_bp=r.mean(), sharpe=r.mean() / sd * np.sqrt(252) if sd else np.nan,
                t=r.mean() / sd * np.sqrt(len(r)) if sd else np.nan)


def summarize(b, pt, tag, name, K):
    row = dict(tag=tag, signal=name, K=K, days=len(b))
    for c in ["excess_net", "excess_gross", "net", "gross", "d10_d1"]:
        st = stats(b[c])
        row.update({f"{c}_{a}": v for a, v in st.items() if a != "n"})
    if len(pt):
        row["trade_excess_gross_bp"] = (pt.gross - pt.univ).mean()
        row["trade_excess_net_bp"] = (pt.gross - pt.univ - pt.cost).mean()
        row["trade_cost_bp"] = pt.cost.mean()      # round trip after netting names kept from the cohort K days back
        row["trades"] = len(pt)
    return row


def evaluate(a, z):
    S = pd.read_parquet(os.path.join(OUT, "s205_signals.parquet"))
    R_, C, PX = panel()
    out = {}
    for name in SIGS:
        for K in KS:
            b, pt = run(S, R_, C, PX, name, K)
            b = b[(b.index >= a) & (b.index <= z)]
            pt = pt[(pt.index >= a) & (pt.index <= z)] if len(pt) else pt
            out[(name, K)] = (b, pt)
    return out


def discover():
    res = evaluate("2024-02-01", "2024-12-31")
    rows = [summarize(b, pt, "discover_2024", n, K) for (n, K), (b, pt) in res.items()]
    g = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(g.round(2).to_string(index=False))
    best = g.sort_values("excess_net_sharpe", ascending=False).iloc[0]
    fz = dict(signal=best.signal, K=int(best.K), N=N, discovery_excess_net_sharpe=float(best.excess_net_sharpe),
              discovery_excess_net_bp_day=float(best.excess_net_mean_bp),
              frozen_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
              pass_rule="confirmation 2025-01..2026-10: net excess Sharpe > 1, t > 2, positive in 2025 and 2026")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(CSV, index=False)
    print(json.dumps(fz, indent=1))


def confirm():
    fz = json.load(open(FROZEN))
    res = evaluate("2025-01-01", DATA_END)
    rows = []
    for (n, K), (b, pt) in res.items():
        frozen = (n == fz["signal"] and K == fz["K"])
        tag = "confirm_frozen" if frozen else "confirm_other"   # other rows: information only, not the test
        rows.append(summarize(b, pt, f"{tag}_2025_2026", n, K))
        for y in [2025, 2026]:
            by = b[b.index.year == y]
            py = pt[pt.index.year == y] if len(pt) else pt
            rows.append(summarize(by, py, f"{tag}_{y}", n, K))
        if frozen:
            b.to_csv(os.path.join(RES, "study205_daily.csv"))
    c = pd.DataFrame(rows)
    old = pd.read_csv(CSV)
    old = old[~old.tag.str.startswith("confirm")]
    pd.concat([old, c]).to_csv(CSV, index=False)
    pd.set_option("display.width", 250)
    print(c.round(2).to_string(index=False))


if __name__ == "__main__":
    {"fetch": fetch, "build": build, "discover": discover, "confirm": confirm}[sys.argv[1]]()
