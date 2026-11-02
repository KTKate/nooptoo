"""Study 203: the study 201 close trade with a resting limit buy at the bid instead of a market order.

Question: study 201's picks (top 10 by the average 15:30-to-close return over the past 10 days, among the 500 most
traded stocks) earned +4.9 bp a day gross in 2024-26 against 11 bp of costs, mostly the spread paid by the 15:30 market
buy. Does a limit buy at the 15:30 bid, which pays no spread but fills only when the price falls, leave a gain?
Data: Alpaca SIP NBBO quotes 15:29:55-15:30:00 for the picks (last quote before 15:30), 5-minute bars 15:30-16:00,
Alpaca daily close (official closing cross).
Design (fixed before the run): same picks as study 201 (L=10, N=10, settings frozen on 2020-23). Each pick gets a buy
limit at the 15:30 bid. Filled only if a 15:30-15:55 bar trades strictly below the bid (a trade at the bid may not reach
a new order in the queue). Unfilled slots stay in cash. Filled names are sold in the closing auction.
Return per slot = close / bid - 1; the day's return is the mean over the 10 slots (cash = 0).
Costs: entry 0.3 bp fees (no spread, no slippage); exit auction 1.3 + 0.1 x half-spread + 2.5 bp.
Sensitivity (not the test): fill when a bar trades at or below the bid.
Pass (as study 201): 2024-01..2026-10 net Sharpe > 1, t > 2, gain in each of 2024, 2025, 2026.
    python src/study203_limit_entry.py
Output: results/study203_limit_entry.csv, results/study203_daily.csv; quotes cached in data/local/quotes1530.parquet
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import alpaca_data as A
from core import RES, half_spread_model
import study201_202_hks as H

QFN = os.path.join(A.LOCAL, "quotes1530.parquet")


def quotes(pairs):
    have = pd.read_parquet(QFN) if os.path.exists(QFN) else pd.DataFrame(columns=["ticker", "date", "bid", "ask"])
    have["date"] = pd.to_datetime(have.date)
    done = set(zip(have.ticker, have.date))
    todo = {}
    for t, d in pairs:
        if (t, d) not in done:
            todo.setdefault(d, []).append(t)

    def one(item):
        d, ts = item
        a = pd.Timestamp(f"{d.date()} 15:29:55").tz_localize("America/New_York").tz_convert("UTC")
        b = a + pd.Timedelta(seconds=5)
        p = dict(symbols=",".join(ts), start=a.strftime("%Y-%m-%dT%H:%M:%SZ"), end=b.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 limit=10000, feed="sip")
        last = {}
        while True:
            j = A.get("quotes", p)
            for s, ql in (j.get("quotes") or {}).items():
                for q in ql:
                    if q["bp"] > 0 and q["ap"] >= q["bp"]:
                        last[s] = (q["bp"], q["ap"])
            if not j.get("next_page_token"):
                break
            p["page_token"] = j["next_page_token"]
        return [dict(ticker=t, date=d, bid=last.get(t, (np.nan, np.nan))[0], ask=last.get(t, (np.nan, np.nan))[1])
                for t in ts]
    items = sorted(todo.items())
    for i in range(0, len(items), 200):
        with ThreadPoolExecutor(8) as ex:
            rows = [r for rr in ex.map(one, items[i:i + 200]) for r in rr]
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_parquet(QFN, index=False)
        print("quotes", min(i + 200, len(items)), "/", len(items), flush=True)
    return have


def main():
    fz = json.load(open(H.FROZEN))["r_close"]
    L, N = fz["L"], fz["N"]
    p = H.signals(H.costs(H.load()))
    s = f"r_close_s{L}"
    x = p.dropna(subset=[s, "r_close"])
    pick = x[x.groupby("date")[s].rank(ascending=False, method="first") <= N].copy()
    q = quotes(zip(pick.ticker, pick.date))
    pick = pick.merge(q, on=["ticker", "date"], how="left")
    m = A.read("m5win", columns=["ts", "ticker", "l"], tickers=sorted(pick.ticker.unique()))
    m = m[m.ts.dt.hour * 100 + m.ts.dt.minute >= 1530]
    m["date"] = m.ts.dt.normalize()
    low = m.groupby(["date", "ticker"]).l.min().rename("low").reset_index()
    pick = pick.merge(low, on=["date", "ticker"], how="left")
    # quotes are raw, bars split-adjusted: put the bid on the bar scale through the quote midpoint ~ 15:30 open
    mid = (pick.bid + pick.ask) / 2
    pick["hsf"] = (pick.ask - pick.bid) / 2 / mid
    pick["bid_adj"] = pick.o1530 * (1 - pick.hsf)
    pick["fill"] = pick.low < pick.bid_adj - 1e-9
    pick["fill_touch"] = pick.low <= pick.bid_adj + 1e-9
    v = pick.vol.clip(lower=1e-3)
    h1545 = half_spread_model(pick.adv, pick.px, v, "15:45")
    pick["cost_lim"] = 0.3 + 1.3 + 0.1 * h1545 + 2.5
    pick["r_lim"] = (pick.c / pick.bid_adj - 1) * 1e4
    pick["r_mid"] = pick.r_close * 1e4
    ok = pick.bid_adj.notna() & pick.low.notna()
    rows, daily = [], []
    for rule in ["fill", "fill_touch"]:
        f = ok & pick[rule]
        pick["slot_gross"] = np.where(f, pick.r_lim, 0.0)
        pick["slot_net"] = np.where(f, pick.r_lim - pick.cost_lim, 0.0)
        g = pick[ok].groupby("date")
        o = pd.DataFrame({"gross": g.slot_gross.mean(), "net": g.slot_net.mean(), "fill_rate": g[rule].mean(),
                          "filled_ret": pick[f].groupby("date").r_lim.mean(),
                          "unfilled_ret_mid": pick[ok & ~pick[rule]].groupby("date").r_mid.mean()})
        daily.append(o.assign(rule=rule))
        for tag, oo in [("2020_2023", o[o.index < "2024-01-01"]), ("2024_2026", o[o.index >= "2024-01-01"])] + \
                [(str(y), oy) for y, oy in o[o.index >= "2024-01-01"].groupby(o[o.index >= "2024-01-01"].index.year)]:
            r = oo.net
            rows.append(dict(rule=rule, period=tag, days=len(oo), fill_rate=oo.fill_rate.mean(),
                             filled_name_bp=pick[f & pick.date.isin(oo.index)].r_lim.mean(),
                             unfilled_name_mid_bp=pick[ok & ~pick[rule] & pick.date.isin(oo.index)].r_mid.mean(),
                             gross_bp=oo.gross.mean(), net_bp=r.mean(), net_sharpe=r.mean() / r.std() * np.sqrt(252),
                             net_t=r.mean() / r.std() * np.sqrt(len(r))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "study203_limit_entry.csv"), index=False)
    pd.concat(daily).to_csv(os.path.join(RES, "study203_daily.csv"))
    print("quote coverage", ok.mean().round(3), "median half-spread bp", (pick.hsf * 1e4).median().round(2))
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
