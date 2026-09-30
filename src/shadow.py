"""Shadow tracking: record each day's picks of candidate strategies without trading them, then score them later
from actual prices. Runs in the entry job after the paper orders are sent (about 15:48 ET).

    python src/shadow.py record [--day=YYYY-MM-DD]   # writes logs/paper/shadow_<day>.json
    python src/shadow.py score                       # realized close -> next open returns, logs/paper/shadow_scores.csv

Candidates recorded:
  smallcap_nonews   10 small caps (price > $2, 20d median dollar volume $1-5M) with the largest loss from the open to
                    the latest price, among stocks with no news article since 15:45 ET of the previous trading day
                    (study 10: no-news losers rebound overnight)
  adr_loser         5 US-listed ADRs (data/store/adr_universe.csv, 20d median dollar volume > $5M, price > $2) with the
                    largest loss from the open (study 16)
  spy_meanrev       SPY and QQQ: enter at the close after 3 down days in a row (today's close taken as the latest
                    price), exit at the close when the price is above its 5-day average (study 27); state kept in
                    logs/paper/shadow_state.json, scored close to close
                    (logs/paper/shadow_meanrev.csv)
Not yet automated (need live data this runner does not have): behavior-cohort models (study 14, needs per-group
models saved for live use) and 60-day insider cluster buys (study 15, needs a daily Form 4 feed).
Prices at record time are the IEX last trade (or SIP 15:30 bar), as in paper_overnight.py; scoring uses Alpaca SIP
daily bars (raw): close of the record day and open of the next trading day, with the auction cost model.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
import paper_overnight as PO

NEWS = "https://data.alpaca.markets/v1beta1/news"


def news_symbols(since_utc):
    """Symbols tagged in any Alpaca/Benzinga article published after since_utc (1-3 symbol articles only)."""
    p = dict(start=since_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), limit=50, sort="asc", include_content="false")
    out = set()
    for _ in range(200):
        A.RL.wait()
        j = A._sess().get(NEWS, params=p, timeout=60).json()
        for n in j.get("news", []):
            s = n.get("symbols") or []
            if 1 <= len(s) <= 3:
                out.update(s)
        if not j.get("next_page_token"):
            break
        p["page_token"] = j["next_page_token"]
    return out


def daily_raw(tickers, start, end):
    # free plan: SIP requests must end more than 15 minutes in the past
    lim = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    end = min(pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end), lim)
    end = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    b = pd.concat([A.bars(tickers[i:i + 400], "1Day", start, end, adjustment="raw") for i in range(0, len(tickers), 400)])
    b["date"] = b.ts.dt.normalize()
    return b


def record(day=None):
    import ml_features as M
    P, cols = M.P, M.cols
    test = day is not None
    day = pd.Timestamp(day or dt.date.today())
    if P["c"].index[-1] >= day:
        P = {k: (x.loc[x.index < day] if isinstance(x, (pd.DataFrame, pd.Series)) else x) for k, x in P.items()}
    prev = P["c"].index[-1]
    small = PO.tradable(PO.smallcap_universe(P, cols))
    adr = pd.read_csv(os.path.join(A.ROOT, "data", "store", "adr_universe.csv"))
    adr = sorted(adr.ticker[adr.kind != "etf"].astype(str))
    # ADR liquidity from Alpaca daily bars (ADRs are not in the Yahoo store)
    a0 = (day - pd.Timedelta(days=45)).strftime("%Y-%m-%dT00:00:00Z")
    a1 = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:00:00Z")
    ad = daily_raw(adr, a0, a1)
    ad["dv"] = ad.c * ad.v
    g = ad.sort_values("ts").groupby("ticker")
    liq = g.dv.apply(lambda x: x.tail(20).median())
    lastpx = g.c.last()
    adr_ok = PO.tradable([t for t in adr if liq.get(t, 0) > 5e6 and lastpx.get(t, 0) > 2])
    intr = PO.today_intraday(sorted(set(small + adr_ok + ["SPY", "QQQ"])), day.date(), live=not test)
    since = (pd.Timestamp(f"{prev.date()} 15:45").tz_localize("America/New_York").tz_convert("UTC"))
    until = pd.Timestamp(f"{day.date()} 15:45").tz_localize("America/New_York").tz_convert("UTC")
    news = news_symbols(since) if not test else set()
    loss = (-(intr.p / intr.o - 1)).dropna()
    sc = loss.reindex(small).dropna()
    sc_nonews = sc[[t not in news for t in sc.index]].sort_values(ascending=False)
    ad_loss = loss.reindex(adr_ok).dropna().sort_values(ascending=False)
    # SPY / QQQ mean reversion (study 27), state machine across days
    st_fn = os.path.join(PO.LOG, "shadow_state.json")
    try:
        state = json.load(open(st_fn))
    except (OSError, ValueError):
        state = {}
    mr = {}
    for etf in ["SPY", "QQQ"]:
        hist = P["rawc"][etf].dropna().iloc[-10:]
        px = float(intr.p.get(etf, np.nan)) if etf in intr.index else np.nan
        if not np.isfinite(px):
            continue
        closes = pd.concat([hist, pd.Series([px], index=[day])])
        down3 = bool((closes.diff().iloc[-3:] < 0).all())
        above5 = bool(px > closes.iloc[-5:].mean())
        pos = state.get(etf, {}).get("in", False)
        action = "hold" if pos else "flat"
        if not pos and down3:
            action, pos = "enter", True
        elif pos and above5:
            action, pos = "exit", False
        state[etf] = {"in": pos, "day": str(day.date())}
        mr[etf] = dict(action=action, px=px, down3=down3, above5=above5)
    if not test:
        json.dump(state, open(st_fn, "w"), indent=1)
    rec = dict(day=str(day.date()), recorded_at=str(pd.Timestamp.now(tz="America/New_York")),
               news_window=[str(since), str(until)], n_news_symbols=len(news),
               smallcap_nonews={t: dict(loss=round(float(v), 4), px=float(intr.p[t])) for t, v in sc_nonews.head(10).items()},
               adr_loser={t: dict(loss=round(float(v), 4), px=float(intr.p[t])) for t, v in ad_loss.head(5).items()},
               spy_meanrev=mr, universe=dict(smallcap=len(small), smallcap_priced=len(sc), adr=len(adr_ok)))
    fn = os.path.join(PO.LOG, f"shadow_{day.date()}{'_replay' if test else ''}.json")
    json.dump(rec, open(fn, "w"), indent=1)
    print(json.dumps(rec, indent=1))


def score():
    """Realized returns of every recorded shadow day whose next open exists: close(day) -> open(next day) from
    Alpaca SIP daily bars (raw prices; a split between the two dates would show as a large move and is flagged)."""
    from core import half_spread_model
    files = sorted(f for f in os.listdir(PO.LOG) if f.startswith("shadow_2") and f.endswith(".json") and "replay" not in f)
    rows = []
    for f in files:
        r = json.load(open(os.path.join(PO.LOG, f)))
        day = pd.Timestamp(r["day"])
        for strat in ["smallcap_nonews", "adr_loser"]:
            tick = list(r.get(strat, {}))
            if not tick:
                continue
            b = daily_raw(tick, (day - pd.Timedelta(days=40)).strftime("%Y-%m-%dT00:00:00Z"),
                          (day + pd.Timedelta(days=6)).strftime("%Y-%m-%dT00:00:00Z"))
            for t in tick:
                x = b[b.ticker == t].set_index("date").sort_index()
                if day not in x.index or len(x.loc[x.index > day]) == 0:
                    continue
                c0, o1 = float(x.loc[day, "c"]), float(x.loc[x.index > day].iloc[0]["o"])
                hist = x.loc[x.index < day].tail(20)
                adv = float((hist.c * hist.v).median())
                vol = float(np.log(hist.c / hist.c.shift(1)).std())
                cost = 2 * (1.3 + 0.1 * float(half_spread_model(adv, c0, max(vol, 1e-3), "15:45")) + 2.5) / 1e4
                ret = o1 / c0 - 1
                rows.append(dict(day=r["day"], strategy=strat, ticker=t, close=c0, next_open=o1, ret=ret, cost=cost,
                                 net=ret - cost, flag_split=abs(np.log(o1 / c0)) > 0.4))
    score_meanrev(files)
    d = pd.DataFrame(rows)
    if not len(d):
        print("nothing to score yet")
        return
    d.to_csv(os.path.join(PO.LOG, "shadow_trades.csv"), index=False)
    s = d[~d.flag_split].groupby(["strategy", "day"]).net.mean().unstack(0)
    s.to_csv(os.path.join(PO.LOG, "shadow_scores.csv"))
    print(s.tail(10).round(4).to_string())
    print("mean net per night (bp):", (1e4 * s.mean()).round(1).to_dict(), "nights:", s.count().to_dict())


def score_meanrev(files):
    """SPY/QQQ mean reversion: one trade per enter .. exit pair of recorded actions, bought and sold at the official
    close (SIP daily bar) of those days, 1 bp cost per side. An open trade is marked at the latest close.
    Output: logs/paper/shadow_meanrev.csv"""
    acts = []
    for f in files:
        r = json.load(open(os.path.join(PO.LOG, f)))
        for etf, a in r.get("spy_meanrev", {}).items():
            if a["action"] in ("enter", "exit"):
                acts.append((r["day"], etf, a["action"]))
    if not acts:
        return
    first = min(a[0] for a in acts)
    b = daily_raw(["SPY", "QQQ"], f"{first}T00:00:00Z", (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"))
    rows = []
    for etf in ["SPY", "QQQ"]:
        x = b[b.ticker == etf].set_index("date").c.sort_index()
        if not len(x):
            continue
        entry = None
        for day, e, a in sorted(acts):
            if e != etf:
                continue
            d = pd.Timestamp(day)
            if a == "enter" and entry is None and d in x.index:
                entry = d
            elif a == "exit" and entry is not None and d in x.index:
                rows.append(dict(etf=etf, entry=entry.date(), exit=d.date(), ret=x[d] / x[entry] - 1 - 2e-4, open=False))
                entry = None
        if entry is not None:
            rows.append(dict(etf=etf, entry=entry.date(), exit=x.index[-1].date(),
                             ret=x.iloc[-1] / x[entry] - 1 - 1e-4, open=True))
    if rows:
        m = pd.DataFrame(rows)
        m.to_csv(os.path.join(PO.LOG, "shadow_meanrev.csv"), index=False)
        print(m.round(4).to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "record"
    day = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--day=")), None)
    {"record": lambda: record(day), "score": score}[what]()
