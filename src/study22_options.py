"""Study 22: options volatility risk premium for a $10k account (defined risk), from Alpaca option daily bars.

Data (src/fetch_options.py): data/local/options/{contracts,bars,underlying}_<U>.parquet, snap_quotes_*.parquet.
Fills: option daily-bar close (last trade of the day) at the entry date; underlying raw daily close for strikes,
deltas and expiry settlement. Costs per leg per transaction: half-spread (model below) + $0.05 fees.

Strategies (entry at the close; weekly = previous week's expiry day, 7 DTE; monthly = previous monthly expiry, ~28-35 DTE):
  PS   short put spread, short put at |delta| target, long put $5 lower (SPY, QQQ, IWM)
  IC   iron condor, both short strikes at the delta target, $5 wings (SPY, QQQ, weekly)
  CSP  cash-secured short put (XLF, TLT, HYG, KRE, SLV)
exit: hold to expiry, or close at the first close where both legs traded and mark <= 50% of the credit.
Returns are on capital at risk (max loss for spreads, strike*100 for CSP), compounding with all capital committed.

    python src/study22_options.py            # -> results/study22_trades.csv, study22_summary.csv, study22_vixbs.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT = os.path.join(ROOT, "data", "local", "options")
RES = os.path.join(ROOT, "results")
SPLIT = pd.Timestamp("2025-07-01")
R = 0.043
QDIV = dict(SPY=0.012, QQQ=0.006, IWM=0.012, XLF=0.015, TLT=0.04, HYG=0.058, KRE=0.03, SLV=0.0)
FEE = 0.05 / 100    # $0.05 per contract in option-price units; ORF ~0.023-0.027 + OCC 0.02-0.025 + TAF 0.003 (sells), per contract
COST_MULT = float(os.environ.get("COST_MULT", "1"))


# ------------------------------------------------------------------ pricing
def bs(S, K, T, sig, kind, q, r=R):
    sig = np.maximum(sig, 1e-6)
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1), -np.exp(-q * T) * norm.cdf(-d1)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), np.exp(-q * T) * norm.cdf(d1)


def iv(price, S, K, T, kind, q):
    lo = np.full(len(price), 0.005)
    hi = np.full(len(price), 4.0)
    for _ in range(60):
        m = 0.5 * (lo + hi)
        p, _ = bs(S, K, T, m, kind, q)
        up = p > price
        hi = np.where(up, m, hi)
        lo = np.where(up, lo, m)
    m = 0.5 * (lo + hi)
    return np.where((m > 0.01) & (m < 3.9), m, np.nan)


# ------------------------------------------------------------------ costs
def spread_model():
    fs = sorted(glob.glob(os.path.join(OPT, "snap_quotes_*.parquet")))
    d = pd.read_parquet(fs[-1])
    d = d[(d.bid > 0) & (d.ask > 0) & (d.ask >= d.bid)].copy()
    k = d.symbol.str[-8:].astype(int) / 1000
    typ = d.symbol.str[-9]
    d = d[((typ == "P") & (k < d.spot)) | ((typ == "C") & (k > d.spot))]      # OTM only
    d["mid"] = (d.bid + d.ask) / 2
    d["hs"] = (d.ask - d.bid) / 2
    edges = [0, 0.25, 0.5, 1, 2, 5, 1e9]
    d["b"] = pd.cut(d.mid, edges, labels=False)
    tab = d.groupby(["underlying", "b"]).hs.median().unstack()
    return tab, edges


TAB, EDGES = None, None


def half_spread(u, px):
    b = np.clip(np.searchsorted(EDGES, px, side="left") - 1, 0, len(EDGES) - 2)
    snap = TAB.loc[u].reindex(range(len(EDGES) - 1)).ffill().bfill().values[b]
    floor = 0.03 if u in ("SPY", "QQQ", "IWM") else 0.02   # SPY/QQQ/IWM: brief's $0.02-0.05 range, mid
    return COST_MULT * np.maximum(snap, floor)


# ------------------------------------------------------------------ data
def load(u):
    cs = pd.read_parquet(os.path.join(OPT, f"contracts_{u}.parquet"))
    b = pd.read_parquet(os.path.join(OPT, f"bars_{u}.parquet"))
    b = b[b.v > 0].merge(cs[["symbol", "type", "strike_price", "expiration_date"]], on="symbol")
    b = b.rename(columns={"strike_price": "K", "expiration_date": "E"})
    sp = pd.read_parquet(os.path.join(OPT, f"underlying_{u}.parquet")).set_index("date").c.astype(float).sort_index()
    sp.index = pd.to_datetime(sp.index)
    b["S"] = b.date.map(sp)
    b = b.dropna(subset=["S"])
    b["T"] = ((b.E - b.date).dt.days + 0.3) / 365.0
    out = []
    for kind, g in b.groupby("type"):
        g = g.copy()
        g["iv"] = iv(g.c.values, g.S.values, g.K.values, g["T"].values, kind, QDIV[u])
        g["delta"] = bs(g.S.values, g.K.values, g["T"].values, g.iv.fillna(0.3).values, kind, QDIV[u])[1]
        out.append(g)
    b = pd.concat(out, ignore_index=True)
    return b, sp


def expiries(b):
    ex = np.sort(b.E.unique())
    ex = pd.DatetimeIndex(ex)
    wk = ex.to_period("W")
    weekly = pd.Series(ex, index=wk).groupby(level=0).max().values
    weekly = pd.DatetimeIndex(weekly)
    # monthly: the weekly expiry that falls in days 15-21 of its month (third Friday, or Thursday before a holiday)
    monthly = weekly[(weekly.day >= 15) & (weekly.day <= 21)]
    return weekly, monthly


# ------------------------------------------------------------------ trade engine
def pick(chain, kind, target, S):
    c = chain[(chain.type == kind) & chain.iv.notna()]
    c = c[c.K < S] if kind == "put" else c[c.K > S]
    if c.empty:
        return None
    i = (c.delta.abs() - target).abs().idxmin()
    if abs(abs(c.delta[i]) - target) > 0.07:
        return None
    return c.loc[i]


def wing(chain, kind, Kshort, W):
    c = chain[chain.type == kind]
    want = Kshort - W if kind == "put" else Kshort + W
    c = c[(c.K < Kshort) if kind == "put" else (c.K > Kshort)]
    if c.empty:
        return None
    i = (c.K - want).abs().idxmin()
    if abs(c.K[i] - want) > W / 2:
        return None
    return c.loc[i]


def run(u, b, sp, strat, target, freq, take=None, W=5.0):
    weekly, monthly = expiries(b)
    sched = weekly if freq == "W" else monthly
    by_day = {k: g for k, g in b.groupby(["date", "E"])}
    px = b.pivot_table(index="date", columns="symbol", values="c")
    trades = []
    for e_prev, E in zip(sched[:-1], sched[1:]):
        t = e_prev
        if t not in sp.index or (t, E) not in by_day:
            continue
        S = sp[t]
        ch = by_day[(t, E)]
        legs = []   # (row, sign) sign=-1 short
        if strat in ("PS", "IC", "CSP"):
            sp_ = pick(ch, "put", target, S)
            if sp_ is None:
                continue
            legs.append((sp_, -1))
            if strat in ("PS", "IC"):
                lp = wing(ch, "put", sp_.K, W)
                if lp is None:
                    continue
                legs.append((lp, +1))
        if strat == "IC":
            sc = pick(ch, "call", target, S)
            if sc is None:
                continue
            lc = wing(ch, "call", sc.K, W)
            if lc is None:
                continue
            legs += [(sc, -1), (lc, +1)]
        credit = -sum(s * r.c for r, s in legs)
        if credit <= 0.02:
            continue
        cost_in = sum(half_spread(u, r.c) + FEE for r, s in legs)
        if strat == "CSP":
            risk = legs[0][0].K
        else:
            wp = legs[0][0].K - legs[1][0].K
            wc = (legs[3][0].K - legs[2][0].K) if strat == "IC" else 0
            risk = max(wp, wc) - credit
        if risk <= 0:
            continue
        # path
        days = sp.index[(sp.index > t) & (sp.index <= E)]
        syms = [r.symbol for r, s in legs]
        sub = px.reindex(columns=syms)
        marks = sub.reindex(sp.index[(sp.index >= t) & (sp.index <= E)]).ffill()
        fresh = sub.reindex(days).notna().all(axis=1)
        exit_d, exit_val, cost_out, how = None, None, 0.0, "expiry"
        path = []
        for d in days:
            val = -sum(s * marks.at[d, r.symbol] for r, s in legs)     # cost to buy back
            if strat != "CSP":
                val = min(max(val, 0.0), max(wp, wc if strat == "IC" else 0))
            if d == days[-1]:
                ST = sp[d]
                iv_ = 0.0
                for r, s in legs:
                    intr = max(r.K - ST, 0) if r.type == "put" else max(ST - r.K, 0)
                    iv_ += -s * intr
                val = iv_
                # assigned short (stock sold next day at ~1c half-spread/share)
                cost_out = 0.01 * sum(1 for r, s in legs if s < 0 and
                                      ((r.type == "put" and ST < r.K) or (r.type == "call" and ST > r.K)))
                exit_d, exit_val = d, val
                path.append((d, val))
                break
            path.append((d, val))
            if take is not None and fresh[d] and val <= (1 - take) * credit:
                exit_d, exit_val, how = d, val, "take"
                cost_out = sum(half_spread(u, marks.at[d, r.symbol]) + FEE for r, s in legs
                               if marks.at[d, r.symbol] > 0.0)
                break
        if exit_d is None:
            continue
        pnl = credit - exit_val - cost_in - cost_out
        trades.append(dict(u=u, strat=strat, target=target, freq=freq, take=take or 0, entry=t, expiry=E, exit=exit_d,
                           how=how, S=S, K=legs[0][0].K, iv_short=legs[0][0].iv, credit=credit, cost=cost_in + cost_out,
                           risk=risk, pnl=pnl, ret=pnl / risk,
                           path=[(d, (credit - v - cost_in) / risk) for d, v in path[:-1]] + [(exit_d, pnl / risk)]))
    return trades


def equity(trades, dates):
    """Daily P&L in units of the capital at risk, fixed (not compounded): each trade commits the same $10k as
    its max loss (spreads) or cash collateral (CSP). Equity = 1 + cumulative return on capital at risk."""
    eq = pd.Series(np.nan, index=dates)
    E = 1.0
    last = dates[0]
    for tr in sorted(trades, key=lambda x: x["entry"]):
        if tr["entry"] < last:
            continue
        eq[tr["entry"]] = E
        for d, r in tr["path"]:
            eq[d] = E + r
        E = E + tr["ret"]
        last = tr["exit"]
    return eq.ffill().fillna(1.0)


def stats(eq, trades, a, b):
    e = eq[(eq.index >= a) & (eq.index < b)]
    if len(e) < 20:
        return {}
    e = e - e.iloc[0] + 1.0
    r = e.diff().dropna()                     # daily return on capital at risk (fixed capital)
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    m = e.resample("ME").last().diff()
    m.iloc[0] = e.resample("ME").last().iloc[0] - 1
    tr = [t for t in trades if a <= t["entry"] < b]
    return dict(ann_ret=(e.iloc[-1] - 1) / yrs, sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                worst_month=m.min(), maxdd=(e - e.cummax()).min(), n=len(tr),
                win=np.mean([t["pnl"] > 0 for t in tr]) if tr else np.nan,
                avg_ret=np.mean([t["ret"] for t in tr]) if tr else np.nan,
                worst_trade=min([t["ret"] for t in tr]) if tr else np.nan,
                cost_pct_credit=np.sum([t["cost"] for t in tr]) / np.sum([t["credit"] for t in tr]) if tr else np.nan)


def spy_bh():
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "store", "daily", "20*.parquet")))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import store
    parts = []
    for f in fs:
        d = pd.read_parquet(f, filters=[("ticker", "in", ["SPY", "^VIX", "^VIX9D"])])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    try:
        d["f"] = store.adj_factor(d).astype(float)
    except Exception:  # noqa: BLE001
        d["f"] = 1.0
    d["date"] = pd.to_datetime(d.date)
    spy = d[d.ticker == "SPY"].set_index("date")
    spy = (spy.c * spy.f).sort_index()
    vix = d[d.ticker == "^VIX"].set_index("date").c.sort_index()
    v9 = d[d.ticker == "^VIX9D"].set_index("date").c.sort_index()
    return spy, vix, v9


PERIODS = [("val", pd.Timestamp("2024-02-01"), SPLIT), ("holdout", SPLIT, pd.Timestamp("2026-10-01")),
           ("all", pd.Timestamp("2024-02-01"), pd.Timestamp("2026-10-01"))]


# ------------------------------------------------------------------ VIX Black-Scholes secondary check
def vix_bs(spy_raw, vix, v9, skew_short, skew_long, target, freq, take=None, W_frac=5 / 550):
    """SPY put spreads priced with BS at sigma = VIX (VIX9D for weekly) x fitted skew ratios. MODEL, not market prices."""
    dates = spy_raw.index
    fri = dates[dates.dayofweek == 4]
    if freq == "M":
        fri = fri[(fri.day >= 15) & (fri.day <= 21)]
    rows = []
    for t, E in zip(fri[:-1], fri[1:]):
        S = spy_raw[t]
        v = (v9 if freq == "W" else vix).get(t, np.nan) / 100
        if not np.isfinite(v):
            continue
        T = ((E - t).days + 0.3) / 365
        W = max(1.0, round(W_frac * S))
        Ks = np.arange(np.floor(S * 0.7), np.floor(S), 1.0)
        _, dl = bs(S, Ks, T, v * skew_short, "put", 0.012)
        Ks_ = Ks[np.argmin(np.abs(np.abs(dl) - target))]
        Kl = Ks_ - W
        cr = bs(S, Ks_, T, v * skew_short, "put", 0.012)[0] - bs(S, Kl, T, v * skew_long, "put", 0.012)[0]
        cost = 2 * (0.03 + FEE)
        ST = spy_raw[E]
        payoff = max(Ks_ - ST, 0) - max(Kl - ST, 0)
        pnl = cr - payoff - cost - (0.01 if ST < Ks_ else 0)
        rows.append(dict(entry=t, expiry=E, credit=cr, pnl=pnl, ret=pnl / (W - cr)))
    return pd.DataFrame(rows)


def main():
    global TAB, EDGES
    TAB, EDGES = spread_model()
    spy, vix, v9 = spy_bh()
    allt, summ = [], []
    unds = [u for u in os.environ.get("UNDS", "SPY,QQQ,IWM,XLF,TLT,HYG,KRE,SLV").split(",")
            if os.path.exists(os.path.join(OPT, f"bars_{u}.parquet"))]
    ivfit = {}
    for u in unds:
        b, sp = load(u)
        dates = sp.index[(sp.index >= "2024-02-01")]
        cfgs = []
        if u in ("SPY", "QQQ", "IWM"):
            cfgs += [("PS", tg, fq, tk) for tg in (0.10, 0.16, 0.20, 0.30) for fq in ("W", "M") for tk in (None, 0.5)]
        if u in ("SPY", "QQQ"):
            cfgs += [("IC", tg, "W", tk) for tg in (0.10, 0.16, 0.20) for tk in (None, 0.5)]
        if u not in ("SPY", "QQQ", "IWM"):
            cfgs += [("CSP", tg, fq, tk) for tg in (0.20, 0.30) for fq in ("W", "M") for tk in (None, 0.5)]
        for strat, tg, fq, tk in cfgs:
            tr = run(u, b, sp, strat, tg, fq, tk)
            if not tr:
                continue
            eq = equity(tr, dates)
            for name, a, z in PERIODS:
                s = stats(eq, tr, a, z)
                if s:
                    summ.append(dict(u=u, strat=strat, target=tg, freq=fq, take=tk or 0, period=name, **s))
            allt += [{k: v for k, v in t.items() if k != "path"} for t in tr]
            print(u, strat, tg, fq, tk, len(tr), round(summ[-1]["ann_ret"], 3), round(summ[-1]["sharpe"], 2), flush=True)
        if u == "SPY":
            # skew ratios for the VIX model: short-leg and long-leg IV / VIX at entry, 16-delta monthly
            t = pd.DataFrame([x for x in allt if x["u"] == "SPY" and x["strat"] == "PS" and x["freq"] == "M"])
            ivfit["SPY"] = t
    for name, a, z in PERIODS:
        e = spy[(spy.index >= a) & (spy.index < z)]
        e = e / e.iloc[0]
        r = e.pct_change().dropna()
        m = e.resample("ME").last().pct_change().dropna()
        summ.append(dict(u="SPY", strat="buy&hold", period=name, ann_ret=e.iloc[-1] ** (365.25 / (e.index[-1] - e.index[0]).days) - 1,
                         sharpe=r.mean() / r.std() * np.sqrt(252), worst_month=m.min(), maxdd=(e / e.cummax() - 1).min()))
    S = pd.DataFrame(summ)
    tag = "" if COST_MULT == 1 else f"_cost{COST_MULT:g}"
    S.to_csv(os.path.join(RES, f"study22_summary{tag}.csv"), index=False)
    pd.DataFrame(allt).to_csv(os.path.join(RES, f"study22_trades{tag}.csv"), index=False)
    if COST_MULT != 1:
        return
    # VIX-BS check, skew fitted on 2024-26 SPY monthly short and long leg IVs vs VIX
    b, sp = load("SPY")
    b["vix"] = b.date.map(vix)
    ch = b[(b.type == "put") & b.iv.notna() & b.vix.notna() & (b["T"] > 20 / 365) & (b["T"] < 40 / 365)]
    ks = ch[(ch.delta.abs() - 0.16).abs() < 0.03]
    rshort = (ks.iv / (ks.vix / 100)).median()
    kl = ch[(ch.delta.abs() - 0.10).abs() < 0.03]
    rlong = (kl.iv / (kl.vix / 100)).median()
    raw = pd.read_parquet(os.path.join(OPT, "underlying_SPY.parquet")).set_index("date").c.astype(float)
    raw.index = pd.to_datetime(raw.index)
    rows = []
    for tg in (0.10, 0.16, 0.20, 0.30):
        for fq in ("W", "M"):
            d = vix_bs(raw, vix, v9, rshort, rlong, tg, fq)
            for name, a, z in [("2019-06..2024-01", pd.Timestamp("2019-06-01"), pd.Timestamp("2024-02-01")),
                               ("2024-02..2026-09", pd.Timestamp("2024-02-01"), pd.Timestamp("2026-10-01")),
                               ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
                               ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"))]:
                x = d[(d.entry >= a) & (d.entry < z)]
                if len(x):
                    eq = 1 + x.ret.cumsum()
                    per = 52 if fq == "W" else 12
                    rows.append(dict(model="VIX-BS", target=tg, freq=fq, period=name, n=len(x), mean_ret=x.ret.mean(),
                                     sharpe=x.ret.mean() / x.ret.std() * np.sqrt(per), win=(x.pnl > 0).mean(),
                                     worst=x.ret.min(), ann_ret=x.ret.mean() * per,
                                     maxdd=(eq - eq.cummax()).min(), skew_short=rshort, skew_long=rlong))
    # implied vs realized: VIX vs subsequent 21-day realized SPY vol
    rv = np.log(raw).diff().rolling(21).std().shift(-21) * np.sqrt(252) * 100
    x = pd.DataFrame({"vix": vix, "rv": rv}).dropna()
    for name, a, z in [("2019-06..2024-01", "2019-06-01", "2024-02-01"), ("2024-02..2026-09", "2024-02-01", "2026-10-01")]:
        y = x[(x.index >= a) & (x.index < z)]
        rows.append(dict(model="VIX-minus-RV21", period=name, n=len(y), mean_ret=(y.vix - y.rv).mean(),
                         win=(y.vix > y.rv).mean()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "study22_vixbs.csv"), index=False)


if __name__ == "__main__":
    main()
