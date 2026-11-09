"""Study 208: trend following across asset-class ETFs with volatility sizing (weeks-to-months holds).

Question: does a long-only time-series-momentum book over asset-class ETFs (Moskowitz, Ooi and Pedersen 2012;
Hurst, Ooi and Pedersen 2017; Faber 2007 10-month average) beat SPY buy-and-hold and a 60/40 SPY/IEF mix on
risk-adjusted terms, after costs, for a $10k long-only Alpaca account?
Data: Yahoo daily bars via yfinance (fetch step below, saved to data/local/etf_long/etf_daily.parquet; the shared
data store is not touched). Signals and returns use Adj Close (total return, dividends reinvested); whole-share
sizing uses the raw Close. Risk-free = 13-week T-bill yield (^IRX) for Sharpe.
Universe (risky): SPY QQQ IWM EFA EEM TLT IEF LQD HYG GLD DBC VNQ (each enters once it has 12 months + 60 days of
history: GLD late 2005, DBC 2007, HYG 2008). Cash = SHY (BIL starts only in 2007; SHY is used throughout).
Rule family (pre-registered 2026-11-09, 6 variants):
  signal  in {r12: 252-day return > 0, sma10m: price > mean of the last 10 month-end closes (current price as the
             10th), sma200: price > 200-day average}
  rebalance in {monthly: last trading day of the month, weekly: last trading day of the week}
  Sizing (fixed): ETFs with the signal on get weight ~ 1/vol60 (60-day stdev of daily returns), then the whole book is
  scaled to a 10% annual volatility target with the 60-day covariance matrix, capped at 100% invested (no leverage);
  the rest sits in SHY. No signal on -> 100% SHY.
  Timing: signal at the close of the rebalance day, trade at the NEXT day's close (one-day lag); weights drift with
  prices between rebalances.
Costs: 1 bp half-spread + 2 bp slippage = 3 bp per side on every dollar traded (SHY included); DBC 5 bp per side.
Benchmarks: SPY buy-and-hold; 60/40 SPY/IEF rebalanced monthly (same cost model).
Protocol: choose the variant with the best net Sharpe (excess of T-bill) on 2005-01..2018-12, write
results/study208_frozen.json, then score 2019-01..2026-10 once.
Pass: confirmation net Sharpe above both SPY's and 60/40's, max drawdown no worse than 60/40's, and a positive
return in at least 5 of the 8 calendar years 2019-2026. No changes after the confirmation run.
Dollars: the frozen rule is replayed from $10,000 on 2019-01 with whole shares (floor), leftover cash idle, to show
the rounding drag vs the fractional-share book (Alpaca supports fractional shares for most ETFs).
    python src/study208_trend_etf.py fetch | discover | confirm
Output: results/study208_trend.csv (grid, confirmation, yearly), results/study208_frozen.json,
        results/study208_daily.csv (confirmation daily returns), results/study208_dollars.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DIR = os.path.join(ROOT, "data", "local", "etf_long")
PANEL = os.path.join(DIR, "etf_daily.parquet")
FROZEN = os.path.join(RES, "study208_frozen.json")

RISKY = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"]
CASH = "SHY"
SECTOR = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "KRE", "ITB",
          "XME", "XOP", "IGV", "IBB", "KBE", "OIH", "XHB", "IYT", "IYR"]
ALL = sorted(set(RISKY + [CASH, "BIL", "PDBC"] + SECTOR))
SIDE_BP = {t: 3.0 for t in ALL}
SIDE_BP["DBC"] = 5.0
CHOOSE = ("2005-01-01", "2018-12-31")
CONFIRM = ("2019-01-01", "2026-10-31")
SIGNALS, REBALS = ["r12", "sma10m", "sma200"], ["monthly", "weekly"]
TARGET_VOL = 0.10


def fetch():
    import yfinance as yf
    os.makedirs(DIR, exist_ok=True)
    parts = []
    for grp in [ALL, ["^IRX"]]:
        for t in grp:                      # one ticker per request: a failed name does not sink the batch
            for k in range(4):
                try:
                    d = yf.download(t, start="2003-01-01", end="2026-11-08", auto_adjust=False, actions=False,
                                    progress=False)
                    if d is not None and len(d):
                        break
                except Exception as e:
                    print("yf err", t, e)
                time.sleep(3 * 2 ** k)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.reset_index().rename(columns={"Date": "date", "Close": "close", "Adj Close": "adj", "Volume": "v"})
            d["ticker"] = t
            parts.append(d[["date", "ticker", "close", "adj", "v"]].dropna(subset=["close"]))
            print(t, len(d), d.date.min().date(), flush=True)
    p = pd.concat(parts)
    p["date"] = pd.to_datetime(p.date).dt.tz_localize(None).dt.normalize()
    p.to_parquet(PANEL, index=False)


def load():
    p = pd.read_parquet(PANEL)
    adj = p.pivot(index="date", columns="ticker", values="adj").sort_index()
    raw = p.pivot(index="date", columns="ticker", values="close").sort_index()
    irx = adj.pop("^IRX").ffill()
    raw = raw.drop(columns="^IRX")
    days = adj.index[adj.SPY.notna()]
    adj, raw = adj.reindex(days), raw.reindex(days)
    rf = (irx.reindex(days).ffill() / 100 / 252).fillna(0.0)     # daily T-bill return
    return adj, raw, rf


def rebal_days(idx, how):
    s = pd.Series(idx, index=idx)
    key = s.dt.to_period("M") if how == "monthly" else s.dt.to_period("W-FRI")
    return set(s.groupby(key).max())


def signal_panel(adj, kind):
    if kind == "r12":
        return adj / adj.shift(252) - 1 > 0
    if kind == "sma200":
        return adj > adj.rolling(200, min_periods=200).mean()
    me = adj.groupby(adj.index.to_period("M")).tail(1)               # month-end closes
    out = pd.DataFrame(False, index=adj.index, columns=adj.columns)
    me_idx = me.index
    for t in adj.columns:
        m = me[t]
        prior9 = m.rolling(9, min_periods=9).sum()                    # sum of 9 month-ends up to each month-end
        pos = me_idx.searchsorted(adj.index, side="left") - 1         # last COMPLETED month-end before date
        pos_ok = pos >= 0
        s9 = np.full(len(adj), np.nan)
        s9[pos_ok] = prior9.values[pos[pos_ok]]
        # on a month-end date itself the current price is that month-end; use the 9 before it
        is_me = adj.index.isin(me_idx)
        s9_me = prior9.shift(1).reindex(adj.index).values
        s9 = np.where(is_me, s9_me, s9)
        avg = (s9 + adj[t].values) / 10
        out[t] = adj[t].values > avg
    return out & adj.notna()


def target_weights(adj, sig, d, risky, vol_target=TARGET_VOL):
    """Weights decided at the close of day d (inverse vol over signal-on ETFs, portfolio vol scaled, cap 1)."""
    i = adj.index.get_loc(d)
    if i < 260:
        return {CASH: 1.0}
    win = adj.iloc[i - 60:i + 1][risky].pct_change().iloc[1:]
    ok = [t for t in risky if win[t].notna().all() and adj[t].iloc[:i + 1].notna().sum() >= 312 and sig.at[d, t]]
    if not ok:
        return {CASH: 1.0}
    vol = win[ok].std()
    w = (1 / vol) / (1 / vol).sum()
    cov = win[ok].cov() * 252
    pv = float(np.sqrt(w.values @ cov.values @ w.values))
    k = min(1.0, vol_target / pv) if pv > 0 else 1.0
    out = (w * k).to_dict()
    out[CASH] = out.get(CASH, 0.0) + 1 - sum(out.values())
    return out


def simulate(adj, raw, rebal, weights_fn, side_bp=SIDE_BP, start=None, end=None):
    """Daily book: decide weights at the close of a rebalance day, trade at the next close, drift in between.
    Returns DataFrame(gross, cost, net, turnover) of daily returns."""
    idx = adj.index
    rets = adj.pct_change()
    lo = idx.searchsorted(pd.Timestamp(start)) if start else 261
    hi = idx.searchsorted(pd.Timestamp(end), side="right") if end else len(idx)
    w = pd.Series({CASH: 1.0})
    pending = None
    rows = []
    for i in range(261, hi):                      # always warm up from 2004 so the book is live at `start`
        d = idx[i]
        r = rets.iloc[i].reindex(w.index).fillna(0.0)
        gross = float((w * r).sum())
        w = w * (1 + r) / (1 + gross)
        cost = turn = 0.0
        if pending is not None:                       # trade at today's close
            new = pd.Series(pending)
            allk = w.index.union(new.index)
            dw = new.reindex(allk).fillna(0) - w.reindex(allk).fillna(0)
            turn = float(dw.abs().sum())
            cost = float(sum(abs(dw[t]) * side_bp.get(t, 3.0) for t in allk)) / 1e4
            w = new[new > 1e-9]
            pending = None
        if d in rebal:
            pending = weights_fn(d)
        if i >= lo:
            rows.append((d, gross, cost, gross - cost, turn))
    return pd.DataFrame(rows, columns=["date", "gross", "cost", "net", "turnover"]).set_index("date")


def stats(r, rf):
    r = r.dropna()
    ex = r - rf.reindex(r.index).fillna(0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = eq / eq.cummax() - 1
    yr = (1 + r).groupby(r.index.year).prod() - 1

    def ydd(y):
        x = r[r.index.year == y]
        if not len(x):
            return np.nan
        e = (1 + x).cumprod()
        return float((e / e.cummax().clip(lower=1) - 1).min())
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, vol=r.std() * np.sqrt(252),
                sharpe=ex.mean() / ex.std() * np.sqrt(252), maxdd=dd.min(), worst_year=yr.min(),
                worst_year_at=int(yr.idxmin()), pos_years=int((yr > 0).sum()), n_years=len(yr),
                dd2020=ydd(2020), dd2022=ydd(2022), n=len(r))


def benchmarks(adj, raw, start, end):
    idx = adj.index
    spy = simulate(adj, raw, set(), lambda d: None, start=start, end=end)
    spy["net"] = spy["gross"] = adj.SPY.pct_change().reindex(spy.index)
    six = simulate(adj, raw, rebal_days(idx, "monthly"), lambda d: {"SPY": 0.6, "IEF": 0.4}, start=start, end=end)
    return {"spy_buy_hold": spy, "spy60_ief40": six}


def run_variant(adj, raw, sigk, reb, start, end, sigs=None):
    sig = sigs[sigk] if sigs is not None else signal_panel(adj, sigk)
    return simulate(adj, raw, rebal_days(adj.index, reb), lambda d: target_weights(adj, sig, d, RISKY),
                    start=start, end=end)


def row(tag, o, rf, **kw):
    s = stats(o.net, rf)
    s["gross_sharpe"] = stats(o.gross, rf)["sharpe"]
    s["turnover_yr"] = o.turnover.sum() / (len(o) / 252) if "turnover" in o else np.nan
    s["cost_yr"] = o.cost.sum() / (len(o) / 252) if "cost" in o else np.nan
    return dict(tag=tag, **kw, **s)


def discover():
    adj, raw, rf = load()
    sigs = {k: signal_panel(adj, k) for k in SIGNALS}
    rows = []
    for sk in SIGNALS:
        for rb in REBALS:
            o = run_variant(adj, raw, sk, rb, *CHOOSE, sigs=sigs)
            rows.append(row("choose_2005_2018", o, rf, signal=sk, rebal=rb))
            print(rows[-1], flush=True)
    for k, o in benchmarks(adj, raw, *CHOOSE).items():
        rows.append(row("choose_2005_2018", o, rf, signal=k, rebal="-"))
    g = pd.DataFrame(rows)
    print(g.round(3).to_string(index=False))
    b = g[g.rebal != "-"].sort_values("sharpe", ascending=False).iloc[0]
    fz = dict(signal=b.signal, rebal=b.rebal, target_vol=TARGET_VOL, choose_net_sharpe=float(b.sharpe),
              universe=RISKY, cash=CASH, frozen_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
              pass_rule="2019-01..2026-10: net Sharpe (excess of T-bill) above SPY's and 60/40's, max drawdown no "
                        "worse than 60/40's, positive return in >= 5 of 8 calendar years")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(os.path.join(RES, "study208_trend.csv"), index=False)
    print(json.dumps(fz, indent=1))


def dollars(adj, raw, fz, start, end, cash0=10_000.0):
    """Whole-share replay of the frozen rule from $10k. Holdings value moves with adjusted (total) returns, so
    dividends accrue to the position; share counts are re-floored at each rebalance from the raw close."""
    sig = signal_panel(adj, fz["signal"])
    reb = rebal_days(adj.index, fz["rebal"])
    rets = adj.pct_change()
    idx = adj.index
    lo, hi = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end), side="right")
    val = {}                        # ticker -> dollar value
    cash = cash0
    pending = None
    rows, zero_lots, small = [], 0, 0
    for i in range(lo - 1, hi):
        d = idx[i]
        for t in list(val):
            val[t] *= 1 + (rets.at[d, t] if pd.notna(rets.at[d, t]) else 0.0)
        if pending is not None:
            eq = cash + sum(val.values())
            traded = 0.0
            new = {}
            for t, w in pending.items():
                px = raw.at[d, t]
                sh = np.floor(w * eq / px)
                if w * eq >= 1 and sh == 0:
                    zero_lots += 1
                if 0 < w * eq < 2 * px:
                    small += 1
                new[t] = sh * px
            for t in set(val) | set(new):
                traded += abs(new.get(t, 0.0) - val.get(t, 0.0))
            fee = sum(abs(new.get(t, 0.0) - val.get(t, 0.0)) * SIDE_BP.get(t, 3.0) / 1e4 for t in set(val) | set(new))
            cash = eq - sum(new.values()) - fee
            val = {t: v for t, v in new.items() if v > 0}
            pending = None
            rows.append(dict(date=d, kind="trade", equity=eq - fee, cash=cash, traded=traded, fee=fee,
                             n_pos=len(val), holdings=";".join(f"{t}:{v / (eq - fee):.3f}" for t, v in
                                                               sorted(val.items()))))
        if i == lo - 1 or d in reb:
            pending = target_weights(adj, sig, d, RISKY)
    eq = cash + sum(val.values())
    rows.append(dict(date=idx[hi - 1], kind="end", equity=eq, cash=cash, traded=0, fee=0, n_pos=len(val),
                     holdings=""))
    return pd.DataFrame(rows), zero_lots, small


def confirm():
    fz = json.load(open(FROZEN))
    adj, raw, rf = load()
    o = run_variant(adj, raw, fz["signal"], fz["rebal"], *CONFIRM)
    bm = benchmarks(adj, raw, *CONFIRM)
    rows = [row("confirm_2019_2026", o, rf, signal=fz["signal"], rebal=fz["rebal"])]
    for k, b in bm.items():
        rows.append(row("confirm_2019_2026", b, rf, signal=k, rebal="-"))
    for y in range(2019, 2027):
        for k, b in [("frozen", o)] + list(bm.items()):
            x = b[b.index.year == y]
            rows.append(dict(tag=f"year_{y}", signal=k, ret=float((1 + x.net).prod() - 1),
                             maxdd=float(((1 + x.net).cumprod() / (1 + x.net).cumprod().cummax().clip(lower=1) - 1).min())))
    # the other five variants on the confirmation period, for information only (choice is already frozen)
    for sk in SIGNALS:
        for rb in REBALS:
            if (sk, rb) != (fz["signal"], fz["rebal"]):
                rows.append(row("info_confirm_other_variants", run_variant(adj, raw, sk, rb, *CONFIRM), rf,
                                signal=sk, rebal=rb))
    c = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(RES, "study208_trend.csv"))
    old = old[old.tag.str.startswith("choose")]
    pd.concat([old, c]).to_csv(os.path.join(RES, "study208_trend.csv"), index=False)
    daily = pd.DataFrame({"frozen_net": o.net, "frozen_cost": o.cost, "frozen_turnover": o.turnover,
                          "spy": bm["spy_buy_hold"].net, "spy60_ief40": bm["spy60_ief40"].net, "rf": rf.reindex(o.index)})
    daily.to_csv(os.path.join(RES, "study208_daily.csv"))
    dl, zero_lots, small = dollars(adj, raw, fz, *CONFIRM)
    dl.to_csv(os.path.join(RES, "study208_dollars.csv"), index=False)
    frac_end = 10_000 * float((1 + o.net).prod())
    spy_end = 10_000 * float((1 + bm["spy_buy_hold"].net).prod())
    six_end = 10_000 * float((1 + bm["spy60_ief40"].net).prod())
    pd.set_option("display.width", 250)
    print(c[c.tag != "info_confirm_other_variants"].round(3).to_string(index=False))
    print(c[c.tag == "info_confirm_other_variants"].round(3).to_string(index=False))
    print(f"$10k whole shares end {dl.equity.iloc[-1]:,.0f}; fractional {frac_end:,.0f}; SPY {spy_end:,.0f}; "
          f"60/40 {six_end:,.0f}; positions rounded to 0 shares {zero_lots}; positions < 2 shares {small}; "
          f"trades {int((dl.kind == 'trade').sum())}; mean idle cash {dl.cash[dl.kind == 'trade'].mean():,.0f}")


if __name__ == "__main__":
    {"fetch": fetch, "discover": discover, "confirm": confirm}[sys.argv[1]]()
