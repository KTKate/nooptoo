"""Study 209: industry momentum (Moskowitz and Grinblatt 1999) entered on short-term dips, sector/industry ETF version.

Question: does holding the top-3 sector/industry ETFs by past 6- or 12-month return (skipping the last month) beat SPY
and an equal-weight basket of the same ETFs, and does waiting for a 5-day pullback before entering a new name help
(short-term reversal inside winners)?
Why ETFs, not stocks: ETF histories back to 2003 have no survivorship bias, so the choice period can be 2005-2018 and
the confirmation 2019-2026. The stock version (top 20 stocks in the top 20% of industries) was not run: the daily store
starts 2019-06 and lists only names alive in 2026, so it would need the short 2020-22 / 2023-26 split and excess-return
reporting; it is left as a follow-up if the ETF version shows an effect.
Data: Yahoo daily Adj Close via src/study208_trend_etf.py fetch (data/local/etf_long/etf_daily.parquet).
Universe (24): XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC SMH XBI KRE ITB XME XOP IGV IBB KBE OIH XHB IYT IYR.
An ETF is eligible once it has L + 1 months of history. Some pairs overlap (KBE/KRE, XHB/ITB, IBB/XBI, OIH/XOP,
IYR/XLRE); kept as listed, so the top 3 can hold two near-duplicates.
Rule family (pre-registered 2026-11-09, 4 variants):
  L in {6, 12} months: rank by the return from t-L*21 to t-21 trading days, at the last close of each month.
  entry in {now: buy at the next close; dip: a newly selected ETF is bought at the close after the first day (from the
            decision day on) whose close is below the close 5 trading days earlier; until then its slot sits in SHY}.
  Hold the top 3, 1/3 each, reset to 1/3 at each monthly rebalance; names that stay in the top 3 are kept.
  Exits: a name leaving the top 3 is sold at the next close.
Costs per side: 1 bp half-spread + 2 bp slippage for SPY, SHY and the 9 original SPDR sectors; 3 bp half-spread + 2 bp
for the other industry ETFs (wider quotes).
Benchmarks: SPY buy-and-hold; equal weight of all eligible universe ETFs, rebalanced monthly (the universe average).
Protocol: choose the variant with the best net Sharpe (excess of T-bill) on 2005-01..2018-12, write
results/study209_frozen.json, score 2019-01..2026-10 once.
Pass: confirmation net Sharpe above SPY's and the equal-weight basket's, max drawdown no worse than SPY's, positive
return in at least 5 of the 8 calendar years.
    python src/study209_industry_momentum.py discover | confirm
Output: results/study209_industry.csv (grid, confirmation, yearly), results/study209_frozen.json,
        results/study209_daily.csv (confirmation daily returns), results/study209_holdings.csv (picks per month)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from study208_trend_etf import load, rebal_days, stats, RES

UNIV = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "KRE", "ITB",
        "XME", "XOP", "IGV", "IBB", "KBE", "OIH", "XHB", "IYT", "IYR"]
CORE = {"SPY", "SHY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"}
SIDE_BP = {t: (3.0 if t in CORE else 5.0) for t in UNIV + ["SPY", "SHY"]}
CASH = "SHY"
CHOOSE = ("2005-01-01", "2018-12-31")
CONFIRM = ("2019-01-01", "2026-10-31")
LS, ENTRIES, TOP = [6, 12], ["now", "dip"], 3
FROZEN = os.path.join(RES, "study209_frozen.json")


def momentum(adj, L):
    m = adj.shift(21) / adj.shift(21 * L) - 1
    hist = adj.notna().cumsum()
    return m.where(hist > 21 * (L + 1))


def simulate(adj, L, entry, start, end, log=None):
    idx = adj.index
    rets = adj.pct_change()
    mom = momentum(adj[UNIV], L)
    dip = adj < adj.shift(5)
    reb = rebal_days(idx, "monthly")
    lo, hi = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end), side="right")
    w = pd.Series({CASH: 1.0})
    pending = None                 # full target weights to trade at today's close
    waiting = set()                # selected names not yet bought (dip variant)
    picks = []
    rows = []
    for i in range(260, hi):
        d = idx[i]
        r = rets.iloc[i].reindex(w.index).fillna(0.0)
        gross = float((w * r).sum())
        w = w * (1 + r) / (1 + gross)
        cost = turn = 0.0
        if pending is not None:
            new = pd.Series(pending)
            allk = w.index.union(new.index)
            dw = new.reindex(allk).fillna(0) - w.reindex(allk).fillna(0)
            turn = float(dw.abs().sum())
            cost = float(sum(abs(dw[t]) * SIDE_BP.get(t, 5.0) for t in allk)) / 1e4
            w = new[new > 1e-12]
            pending = None
        if d in reb:
            s = mom.loc[d].dropna()
            picks = list(s.sort_values(ascending=False).index[:TOP]) if len(s) >= TOP else []
            held = [t for t in picks if t in w.index]
            if entry == "now":
                waiting = set()
                buy = picks
            else:
                waiting = {t for t in picks if t not in held and not dip.at[d, t]}
                buy = [t for t in picks if t not in waiting]
            tgt = {t: 1 / TOP for t in buy}
            tgt[CASH] = 1 - sum(tgt.values())
            pending = tgt
            if log is not None and i >= lo:
                log.append(dict(date=d, L=L, entry=entry, picks=";".join(picks), waiting=";".join(sorted(waiting))))
        elif waiting:
            hit = {t for t in waiting if dip.at[d, t]}
            if hit:
                cur = w.to_dict()
                for t in hit:
                    add = min(1 / TOP, cur.get(CASH, 0.0))
                    cur[t] = cur.get(t, 0.0) + add
                    cur[CASH] = cur.get(CASH, 0.0) - add
                pending = cur
                waiting -= hit
        if i >= lo:
            rows.append((d, gross, cost, gross - cost, turn))
    return pd.DataFrame(rows, columns=["date", "gross", "cost", "net", "turnover"]).set_index("date")


def equal_weight(adj, start, end):
    """Equal weight of eligible universe ETFs (12 months of history), monthly rebalance, drift between."""
    idx = adj.index
    rets = adj[UNIV].pct_change()
    reb = rebal_days(idx, "monthly")
    hist = adj[UNIV].notna().cumsum()
    lo, hi = idx.searchsorted(pd.Timestamp(start)), idx.searchsorted(pd.Timestamp(end), side="right")
    w, pending, rows = pd.Series(dtype=float), None, []
    for i in range(260, hi):
        d = idx[i]
        r = rets.iloc[i].reindex(w.index).fillna(0.0)
        gross = float((w * r).sum()) if len(w) else 0.0
        if len(w):
            w = w * (1 + r) / (1 + gross)
        cost = 0.0
        if pending is not None:
            allk = w.index.union(pending.index)
            dw = pending.reindex(allk).fillna(0) - w.reindex(allk).fillna(0)
            cost = float(sum(abs(dw[t]) * SIDE_BP[t] for t in allk)) / 1e4
            w, pending = pending, None
        if d in reb:
            el = [t for t in UNIV if hist.at[d, t] > 252]
            pending = pd.Series(1 / len(el), index=el)
        if i >= lo:
            rows.append((d, gross, cost, gross - cost, 0.0))
    return pd.DataFrame(rows, columns=["date", "gross", "cost", "net", "turnover"]).set_index("date")


def benchmarks(adj, start, end):
    ew = equal_weight(adj, start, end)
    spy = ew.copy()
    spy["gross"] = spy["net"] = adj.SPY.pct_change().reindex(ew.index)
    spy["cost"] = 0.0
    return {"spy_buy_hold": spy, "equal_weight_universe": ew}


def row(tag, o, rf, **kw):
    s = stats(o.net, rf)
    s["gross_sharpe"] = stats(o.gross, rf)["sharpe"]
    s["turnover_yr"] = o.turnover.sum() / (len(o) / 252)
    s["cost_yr"] = o.cost.sum() / (len(o) / 252)
    return dict(tag=tag, **kw, **s)


def discover():
    adj, raw, rf = load()
    rows = []
    for L in LS:
        for e in ENTRIES:
            rows.append(row("choose_2005_2018", simulate(adj, L, e, *CHOOSE), rf, L=L, entry=e))
    for k, b in benchmarks(adj, *CHOOSE).items():
        rows.append(row("choose_2005_2018", b, rf, L=0, entry=k))
    g = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(g.round(3).to_string(index=False))
    b = g[g.L > 0].sort_values("sharpe", ascending=False).iloc[0]
    fz = dict(L=int(b.L), entry=b.entry, top=TOP, choose_net_sharpe=float(b.sharpe), universe=UNIV,
              frozen_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
              pass_rule="2019-01..2026-10: net Sharpe (excess of T-bill) above SPY's and the equal-weight universe's, "
                        "max drawdown no worse than SPY's, positive return in >= 5 of 8 calendar years")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(os.path.join(RES, "study209_industry.csv"), index=False)
    print(json.dumps(fz, indent=1))


def confirm():
    fz = json.load(open(FROZEN))
    adj, raw, rf = load()
    log = []
    o = simulate(adj, fz["L"], fz["entry"], *CONFIRM, log=log)
    bm = benchmarks(adj, *CONFIRM)
    rows = [row("confirm_2019_2026", o, rf, L=fz["L"], entry=fz["entry"])]
    for k, b in bm.items():
        rows.append(row("confirm_2019_2026", b, rf, L=0, entry=k))
    for y in range(2019, 2027):
        for k, b in [("frozen", o)] + list(bm.items()):
            x = (1 + b.net[b.index.year == y]).cumprod()
            rows.append(dict(tag=f"year_{y}", entry=k, ret=float(x.iloc[-1] - 1),
                             maxdd=float((x / x.cummax().clip(lower=1) - 1).min())))
    for L in LS:
        for e in ENTRIES:
            if (L, e) != (fz["L"], fz["entry"]):
                rows.append(row("info_confirm_other_variants", simulate(adj, L, e, *CONFIRM), rf, L=L, entry=e))
    c = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(RES, "study209_industry.csv"))
    old = old[old.tag.str.startswith("choose")]
    pd.concat([old, c]).to_csv(os.path.join(RES, "study209_industry.csv"), index=False)
    pd.DataFrame({"frozen_net": o.net, "frozen_cost": o.cost, "spy": bm["spy_buy_hold"].net,
                  "equal_weight": bm["equal_weight_universe"].net, "rf": rf.reindex(o.index)}
                 ).to_csv(os.path.join(RES, "study209_daily.csv"))
    pd.DataFrame(log).to_csv(os.path.join(RES, "study209_holdings.csv"), index=False)
    pd.set_option("display.width", 250)
    print(c.round(3).to_string(index=False))
    print(f"$10k end: frozen {1e4 * (1 + o.net).prod():,.0f}, SPY {1e4 * (1 + bm['spy_buy_hold'].net).prod():,.0f}, "
          f"EW {1e4 * (1 + bm['equal_weight_universe'].net).prod():,.0f}")


if __name__ == "__main__":
    {"discover": discover, "confirm": confirm}[sys.argv[1]]()
