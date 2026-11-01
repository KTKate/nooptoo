"""Studies 201-202: same-time-of-day repeat (Heston, Korajczyk and Sadka, Journal of Finance 2010).

Question: does a stock's average return in a fixed half hour over the past L days predict its return in the same half
hour today, by enough to pay for trading it?
  201 close trade: buy at 15:30 with a market order, sell in the closing auction.
  202 open trade:  buy in the opening auction, sell at 10:00 with a market order.
Data: Alpaca SIP 5-minute bars 09:30-10:00 and 15:30-16:00 (src/study201_fetch.py), Alpaca daily bars (close = official
closing cross). Universe on day t: the 500 stocks with the highest 20-day median dollar volume up to t-1, prior close
above $5 (ETFs excluded; delisted names included).
Signal: mean of the window's return over days t-L .. t-1 (at least 3/4 of them present). Long the top N, equal weight.
Prices: close trade 15:30 bar open -> daily close. Open trade 09:30 bar open (first trade) -> 09:55 bar close (10:00);
the confirmation also scores the open trade at the official opening cross.
Costs per side (bp): market order = quoted half-spread model (15:45 for 15:30; mean of 09:35 and 12:00 for 10:00)
+ 2.0 slippage + 0.3 fees; auction = 1.3 + 0.1 x half-spread + 2.5.
Protocol (pre-registered 2026-11-01): choose L in {10, 20, 40} and N in {10, 20} on 2020-2023 only by net Sharpe, write
results/study201_202_frozen.json and commit it, then score 2024-01..2026-10 once. Pass: net Sharpe > 1, t > 2 and a gain
in each of 2024, 2025, 2026. No changes after the confirmation run.
    python src/study201_202_hks.py discover | confirm
Output: results/study201_202_hks.csv (grid and confirmation), results/study201_202_frozen.json,
        results/study201_202_daily.csv (confirmation daily returns)
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import alpaca_data as A
from core import RES, half_spread_model

LS, NS = [10, 20, 40], [10, 20]
FROZEN = os.path.join(RES, "study201_202_frozen.json")


def load():
    fn = os.path.join(A.LOCAL, "hks_panel.parquet")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    d = A.read("d1", columns=["ts", "ticker", "o", "c", "v"])
    d["date"] = d.ts.dt.normalize()
    d = d.drop(columns="ts")
    m = A.read("m5win", columns=["ts", "ticker", "o", "c"])
    hm = m.ts.dt.hour * 100 + m.ts.dt.minute
    m["date"] = m.ts.dt.normalize()
    o930 = m[hm == 930].set_index(["date", "ticker"]).o.rename("o930")
    c955 = m[hm == 955].set_index(["date", "ticker"]).c.rename("c955")
    o1530 = m[hm == 1530].set_index(["date", "ticker"]).o.rename("o1530")
    p = d.set_index(["date", "ticker"]).join([o930, c955, o1530]).reset_index()
    p = p.sort_values(["ticker", "date"])
    g = p.groupby("ticker")
    lr = np.log(p.c / g.c.shift(1))
    p["adv"] = (p.c * p.v).groupby(p.ticker).transform(lambda x: x.rolling(20, min_periods=10).median().shift(1))
    p["px"] = g.c.shift(1)
    p["vol"] = lr.groupby(p.ticker).transform(lambda x: x.rolling(20, min_periods=10).std().shift(1))
    p["r_close"] = p.c / p.o1530 - 1
    p["r_open"] = p.c955 / p.o930 - 1
    for k in ["r_close", "r_open"]:          # data-error guard: a half hour beyond +-40% is dropped
        p.loc[p[k].abs() > 0.4, k] = np.nan
    p.to_parquet(fn, index=False)
    return p


def costs(p):
    v = p.vol.clip(lower=1e-3)
    h1545 = half_spread_model(p.adv, p.px, v, "15:45")
    h935 = half_spread_model(p.adv, p.px, v, "09:35")
    h1200 = half_spread_model(p.adv, p.px, v, "12:00")
    p["cost_close"] = (h1545 + 2.0 + 0.3) + (1.3 + 0.1 * h1545 + 2.5)
    p["cost_open"] = (1.3 + 0.1 * h935 + 2.5) + (0.5 * (h935 + h1200) + 2.0 + 0.3)
    return p


def signals(p):
    p = p.sort_values(["ticker", "date"]).copy()
    for k in ["r_close", "r_open"]:
        s = p.groupby("ticker")[k]
        for L in LS:
            p[f"{k}_s{L}"] = s.transform(lambda x: x.rolling(L, min_periods=int(0.75 * L)).mean().shift(1))
    p["rk"] = p.groupby("date").adv.rank(ascending=False)
    return p[(p.rk <= 500) & (p.px > 5)]


def book(u, k, L, N, ret=None):
    ret = ret or k
    s = f"{k}_s{L}"
    x = u.dropna(subset=[s, ret])
    pick = x[x.groupby("date")[s].rank(ascending=False, method="first") <= N]
    g = pick.groupby("date")
    out = pd.DataFrame({"gross": g[ret].mean() * 1e4, "cost": g[f"cost_{k[2:]}"].mean(),
                        "univ": x.groupby("date")[ret].mean().reindex(g.size().index) * 1e4})
    out["net"] = out.gross - out.cost
    # decile spread (gross, the published test of the effect itself)
    q = x.groupby("date")[s].rank(pct=True)
    out["d10_d1"] = (x[q > 0.9].groupby("date")[ret].mean() - x[q <= 0.1].groupby("date")[ret].mean()).reindex(out.index) * 1e4
    return out


def stats(r):
    r = r.dropna()
    sd = r.std()
    return dict(n=len(r), mean_bp=r.mean(), sharpe=r.mean() / sd * np.sqrt(252) if sd else np.nan,
                t=r.mean() / sd * np.sqrt(len(r)) if sd else np.nan)


def summarize(o, tag, **kw):
    row = dict(tag=tag, **kw)
    for c in ["gross", "net", "d10_d1"]:
        st = stats(o[c])
        row.update({f"{c}_{a}": b for a, b in st.items() if a != "n" or c == "net"})
    row["excess_gross_bp"] = (o.gross - o.univ).mean()
    row["cost_bp"] = o.cost.mean()
    return row


def discover():
    p = signals(costs(load()))
    p = p[(p.date >= "2020-01-01") & (p.date <= "2023-12-31")]
    rows = []
    for k in ["r_close", "r_open"]:
        for L in LS:
            for N in NS:
                rows.append(summarize(book(p, k, L, N), "discover_2020_2023", trade=k, L=L, N=N))
    g = pd.DataFrame(rows)
    print(g.round(2).to_string(index=False))
    fz = {}
    for k in ["r_close", "r_open"]:
        b = g[g.trade == k].sort_values("net_sharpe", ascending=False).iloc[0]
        fz[k] = dict(L=int(b.L), N=int(b.N), discovery_net_sharpe=float(b.net_sharpe))
    fz["frozen_on"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    fz["pass_rule"] = "confirmation 2024-01..2026-10: net Sharpe > 1, t > 2, gain in each of 2024, 2025, 2026"
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(os.path.join(RES, "study201_202_hks.csv"), index=False)
    print(json.dumps(fz, indent=1))


def confirm():
    fz = json.load(open(FROZEN))
    p = signals(costs(load()))
    p = p[p.date >= "2024-01-01"]
    rows, daily = [], []
    for k in ["r_close", "r_open"]:
        L, N = fz[k]["L"], fz[k]["N"]
        o = book(p, k, L, N)
        rows.append(summarize(o, "confirm_2024_2026", trade=k, L=L, N=N))
        for y, oy in o.groupby(o.index.year):
            rows.append(summarize(oy, f"confirm_{y}", trade=k, L=L, N=N))
        daily.append(o.assign(trade=k))
        if k == "r_open":                     # the same picks at the official opening cross
            s = f"{k}_s{L}"
            x = p.dropna(subset=[s, k])
            pick = x[x.groupby("date")[s].rank(ascending=False, method="first") <= N]
            ap = A.auction_prices(zip(pick.ticker, pick.date), workers=8)
            pick = pick.merge(ap[["ticker", "date", "open_off", "first_trade"]], on=["ticker", "date"], how="left")
            # bars are split-adjusted as of the fetch date and crosses are raw: move the adjusted first trade by the
            # raw cross / raw first-trade ratio from the same trades request
            ok = (pick.open_off > 0) & (pick.first_trade > 0)
            pick["r_cross"] = np.where(ok, pick.c955 / (pick.o930 * pick.open_off / pick.first_trade) - 1, np.nan)
            pick.loc[pick.r_cross.abs() > 0.4, "r_cross"] = np.nan
            g = pick.groupby("date")
            oc = pd.DataFrame({"gross": g.r_cross.mean() * 1e4, "cost": g.cost_open.mean(),
                               "univ": o.univ.reindex(g.size().index), "d10_d1": np.nan})
            oc["net"] = oc.gross - oc.cost
            rows.append(summarize(oc, "confirm_2024_2026_cross", trade=k, L=L, N=N,
                                  cross_coverage=float(pick.r_cross.notna().mean())))
            for y, oy in oc.groupby(oc.index.year):
                rows.append(summarize(oy, f"confirm_{y}_cross", trade=k, L=L, N=N))
            daily.append(oc.assign(trade="r_open_cross"))
    c = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(RES, "study201_202_hks.csv"))
    old = old[~old.tag.str.startswith("confirm")]
    pd.concat([old, c]).to_csv(os.path.join(RES, "study201_202_hks.csv"), index=False)
    pd.concat(daily).to_csv(os.path.join(RES, "study201_202_daily.csv"))
    print(c.round(2).to_string(index=False))


if __name__ == "__main__":
    {"discover": discover, "confirm": confirm}[sys.argv[1]]()
