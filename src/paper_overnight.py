"""Paper-trading runner for the overnight strategies (Alpaca PAPER account only).

Strategies (--strategy=, default ml: the only one that survived the look-ahead-free tests, see reports/):
  ensemble  the five-model average of src/ensemble.py (study 14/23), same universe and trade as ml
  ml        rank the liquid universe (price > $5, ADV > $5M) with the latest quarterly overnight LightGBM model
  smallcap  small caps (price > $2, 20d median dollar volume $1-5M): largest loss from today's open to 15:45
            (s_intraday_loser in final_series.py)
  combo     half the equity in the top K/2 of each list (a name on both lists is bought once, double size)
  smallcap and combo are kept for comparison runs only: without the hindsight consistency filter their
  2024-26 Sharpe is about 1.0 (below SPY buy-and-hold), study 9 / results/final_summary.csv
At ~15:45 ET buy the picks in the closing auction (market-on-close), sell everything in the next opening
auction (market-on-open). Positions are held overnight only, so no day trades are created (PDT does not apply).

Steps (cron, America/New_York, trading days):
  15:45  python src/paper_overnight.py entry      # builds today's 15:45 features and the order list
         (Alpaca accepts market-on-close orders until 15:50; the run takes about 1-2 minutes)
  09:20  python src/paper_overnight.py exit       # market-on-open sells for every open position
  daily 17:30  python src/update_data.py daily    # keeps the Yahoo daily store current (used by the features)
  src/paper_job.sh wraps these for scheduled cloud runs (setup, calendar check, wait, run, commit the logs).
  python src/paper_overnight.py reconcile         # fills vs official auction prints, equity history
  quarterly     rm data/ml_frame.parquet; Q_START=<new quarter> Q_END=<new quarter> python src/study3_ml.py night

SAFETY: without --submit nothing is sent; the intended orders are written to logs/paper/. Orders go only to
paper-api.alpaca.markets and only when ALPACA_PAPER_KEY_ID is set and the account id matches
ALPACA_EXPECTED_PAPER_ACCOUNT_ID (if that variable is set). Do not run with --submit without the owner's go-ahead.

Live data on the free plan: SIP bars are available with a 15-minute delay, IEX in real time. Features use SIP
30-minute bars up to 15:30 for open/high/low/volume and the IEX last trade at 15:45 for the price (if that trade
is older than 15:40, which happens for thin small caps on IEX, the SIP 15:30 close is used instead).
Replay (--day=YYYY-MM-DD, never submits): the price is the SIP 15:40-15:45 bar close, as in the backtests.
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

K = 10
TRADE = "https://paper-api.alpaca.markets/v2"
LOG = os.path.join(A.ROOT, "logs", "paper")
os.makedirs(LOG, exist_ok=True)
H = {"APCA-API-KEY-ID": A.KEY or "", "APCA-API-SECRET-KEY": A.SEC or ""}


def account():
    a = requests.get(f"{TRADE}/account", headers=H, timeout=30).json()
    exp = os.environ.get("ALPACA_EXPECTED_PAPER_ACCOUNT_ID")
    if exp and a.get("id") != exp and a.get("account_number") != exp:
        raise SystemExit("account id does not match ALPACA_EXPECTED_PAPER_ACCOUNT_ID; refusing to trade")
    return a


def today_intraday(tickers, day, live=True):
    """SIP 30-minute bars 09:30-15:30 (15-minute delayed feed) and IEX last trades (real time); in a replay the
    price is the SIP 15:40-15:45 bar close."""
    ta = pd.Timestamp(f"{day} 09:30").tz_localize("America/New_York").tz_convert("UTC")
    tb = pd.Timestamp(f"{day} 15:30").tz_localize("America/New_York").tz_convert("UTC")
    b = pd.concat([A.bars(tickers[i:i + 400], "30Min", ta.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          tb.strftime("%Y-%m-%dT%H:%M:%SZ")) for i in range(0, len(tickers), 400)])
    g = b.groupby("ticker")
    agg = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "v": g.v.sum(), "c1530": g.c.last()})
    last = {}
    fresh = pd.Timestamp(f"{day} 15:40").tz_localize("America/New_York")
    for i in (range(0, len(tickers), 200) if live else []):
        j = A.get("snapshots", dict(symbols=",".join(tickers[i:i + 200]), feed="iex"))
        for t, s in j.items():
            if s and s.get("latestTrade") and pd.Timestamp(s["latestTrade"]["t"]) >= fresh:
                last[t] = s["latestTrade"]["p"]
    if not live:
        t1 = pd.Timestamp(f"{day} 15:40").tz_localize("America/New_York").tz_convert("UTC")
        t2 = pd.Timestamp(f"{day} 15:45").tz_localize("America/New_York").tz_convert("UTC")
        b5 = pd.concat([A.bars(tickers[i:i + 400], "5Min", t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               t2.strftime("%Y-%m-%dT%H:%M:%SZ")) for i in range(0, len(tickers), 400)])
        if len(b5):
            last = b5.groupby("ticker").c.last().to_dict()
    agg["p"] = pd.Series(last, dtype="float64")
    agg["p"] = agg.p.fillna(agg.c1530)
    agg["h"] = np.maximum(agg.h, agg.p)
    agg["l"] = np.minimum(agg.l, agg.p)
    return agg


def ml_scores(P, cols, intr, day, hist, use_ensemble=False):
    """Overnight LightGBM scores for day from the panels cut before day plus today's intraday row."""
    import lightgbm as lgb
    import ml_features as M
    from core import DATA
    idx = P["c"].index[hist].append(pd.DatetimeIndex([day]))

    def ext(k, row):
        x = P[k][cols].iloc[hist]
        return pd.concat([x, pd.DataFrame([row.reindex(cols)], index=[day])]).reindex(idx)
    adjf = (P["c"][cols] / P["rawc"][cols]).iloc[-1]          # no dividend adjustment known intraday: last factor
    o = ext("o", intr.o * adjf)
    h = ext("h", intr.h * adjf)
    l = ext("l", intr.l * adjf)
    c = ext("c", intr.p * adjf)
    raw = ext("rawc", intr.p)
    v = ext("v", intr.v)
    dv = ext("dv", intr.v * intr.p)
    spy = pd.concat([P["c"]["SPY"].iloc[hist], pd.Series([intr.p["SPY"]], index=[day])])
    iwm = pd.concat([P["c"]["IWM"].iloc[hist], pd.Series([intr.p["IWM"]], index=[day])])
    vix = pd.concat([P["c"]["^VIX"].iloc[hist], pd.Series([P["c"]["^VIX"].iloc[-1]], index=[day])])
    vix3 = pd.concat([P["c"]["^VIX3M"].iloc[hist], pd.Series([P["c"]["^VIX3M"].iloc[-1]], index=[day])])
    earn = M.earnings_features()
    earn = {k: (f if day in f.index else pd.concat([f, f.iloc[[-1]].set_axis([day])])) for k, f in earn.items()}
    F, mkt, adv20 = M.build(o, h, l, c, raw, v, dv, spy, vix, vix3, iwm, earn=earn)
    univ = ((raw > 5) & (adv20 > 5e6)).iloc[[-1]] & c.iloc[[-1]].notna()
    X = M.features_frame({k: f.iloc[[-1]] for k, f in F.items()}, mkt.iloc[[-1]], univ)
    models = sorted(f for f in os.listdir(os.path.join(DATA, "models")) if f.startswith("night_"))
    q = f"night_{pd.Period(day, freq='Q')}.txt"
    name = q if q in models else models[-1]
    mdl = lgb.Booster(model_file=os.path.join(DATA, "models", name))
    if use_ensemble:
        import ensemble
        ens = ensemble.predict(X, pd.Period(name.split("_")[1].split(".")[0], freq="Q"), pooled_model=mdl)
        pred = pd.Series(ens["ensemble"].values, index=X.index.get_level_values(1))
        # log every member's top 10 (the pooled member is the previous paper model) for comparison
        members = {m: pd.Series(ens[m].values, index=X.index.get_level_values(1)).nlargest(10).round(5).to_dict()
                   for m in ens.columns if m != "ensemble"}
        json.dump(dict(day=str(day.date()), members=members), open(os.path.join(LOG, f"members_{day.date()}.json"), "w"),
                  indent=1)
        return pred.sort_values(ascending=False), "ensemble5_" + name
    pred = pd.Series(mdl.predict(X[mdl.feature_name()]), index=X.index.get_level_values(1))
    return pred.sort_values(ascending=False), name


def cap_by_industry(pred, n, cap=3):
    """Top n names of pred (sorted descending) with at most cap names per industry (data/store/sectors.parquet;
    names without an industry count as their own group). Backtest effect on the ensemble: Sharpe 2.23 -> 2.20
    (2024-26), added to limit single-industry gap risk (for example several tankers on one night)."""
    fn = os.path.join(A.ROOT, "data", "store", "sectors.parquet")
    ind = pd.read_parquet(fn).drop_duplicates("ticker").set_index("ticker").industry if os.path.exists(fn) else pd.Series(dtype=str)
    out, cnt = [], {}
    for t in pred.index:
        g = ind.get(t)
        g = g if isinstance(g, str) and g else t
        if cnt.get(g, 0) >= cap:
            continue
        cnt[g] = cnt.get(g, 0) + 1
        out.append(t)
        if len(out) == n:
            break
    return pred.reindex(out)


def smallcap_universe(P, cols):
    """Small-cap tier known before the open: previous close > $2, 20d median dollar volume $1-5M."""
    adv = P["dv"][cols].iloc[-20:].median()
    px = P["rawc"][cols].iloc[-1]
    return [t for t in cols if 1e6 < adv[t] <= 5e6 and px[t] > 2]


def smallcap_scores(intr, tickers):
    """s_intraday_loser: loss from today's open (first SIP trade) to the 15:45 price; larger loss = higher score."""
    x = intr.reindex(tickers)
    return (-(x.p / x.o - 1)).dropna().sort_values(ascending=False)


def tradable(tickers):
    """Drop symbols Alpaca does not list as active and tradable (from the cached asset list)."""
    fn = os.path.join(A.LOCAL, "alpaca_assets_active.parquet")
    if not os.path.exists(fn):
        return list(tickers)
    a = pd.read_parquet(fn)
    if "tradable" in a.columns:
        a = a[a.tradable.astype(bool)]
    ok = set(a["symbol"])
    return [t for t in tickers if t in ok]


def entry(submit=False, day=None, strategy="ml"):
    import ml_features as M
    P, cols = M.P, M.cols
    test = day is not None
    day = pd.Timestamp(day or dt.date.today())
    if P["c"].index[-1] >= day:
        if not test:
            raise SystemExit("daily store already contains today; features must be built from 15:45 data")
        # dry-run replay of a past day: drop that day and later from the panels
        P = {k: (x.loc[x.index < day] if isinstance(x, (pd.DataFrame, pd.Series)) else x) for k, x in P.items()}
    hist = slice(len(P["c"]) - 260, len(P["c"]))
    liquid = [t for t in cols if P["dv"][t].iloc[-20:].median() > 5e6 and P["rawc"][t].iloc[-1] > 5]
    small = tradable(smallcap_universe(P, cols)) if strategy in ("smallcap", "combo") else []
    need = (liquid if strategy in ("ml", "combo", "ensemble") else []) + small + ["SPY", "IWM"]
    intr = today_intraday(sorted(set(need)), day.date(), live=not test)
    picks, rec = {}, dict(day=str(day.date()), strategy=strategy)
    if strategy in ("ml", "combo", "ensemble"):
        pred, model = ml_scores(P, cols, intr, day, hist, use_ensemble=strategy == "ensemble")
        pred = pred[pred.index.isin(tradable(pred.index))]
        n = K if strategy in ("ml", "ensemble") else K // 2
        top = cap_by_industry(pred, n)
        rec.update(model=model, ml_top=top.round(5).to_dict(), uncapped_top=pred.head(n).round(5).to_dict())
        for t in top.index:
            picks[t] = picks.get(t, 0.0) + 1.0 / K
    if strategy in ("smallcap", "combo"):
        sc = smallcap_scores(intr, small)
        n = K if strategy == "smallcap" else K // 2
        rec.update(smallcap_universe=len(small), smallcap_priced=int(sc.notna().sum()),
                   smallcap_top=sc.head(n).round(4).to_dict())
        for t in sc.head(n).index:
            picks[t] = picks.get(t, 0.0) + 1.0 / K
    acct = account() if (A.KEY and not test) else {"equity": "10000"}
    eq = float(acct["equity"])
    orders = []
    for t, w in picks.items():
        qty = int(eq * w // intr.p[t]) if intr.p.get(t, 0) > 0 else 0
        if qty >= 1:
            orders.append(dict(symbol=t, qty=qty, side="buy", type="market", time_in_force="cls"))
    rec.update(equity=eq, orders=orders, submitted=submit and not test,
               notional=round(sum(o["qty"] * float(intr.p[o["symbol"]]) for o in orders), 2))
    json.dump(rec, open(os.path.join(LOG, f"entry_{day.date()}{'_replay' if test else ''}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))
    if submit and not test:
        for od in orders:
            r = requests.post(f"{TRADE}/orders", headers=H, json=od, timeout=30)
            print(od["symbol"], r.status_code, r.text[:200])


def exit_(submit=False):
    pos = requests.get(f"{TRADE}/positions", headers=H, timeout=30).json()
    orders = [dict(symbol=p["symbol"], qty=abs(int(float(p["qty"]))), side="sell" if float(p["qty"]) > 0 else "buy",
                   type="market", time_in_force="opg") for p in pos]
    rec = dict(day=str(dt.date.today()), orders=orders, submitted=submit)
    json.dump(rec, open(os.path.join(LOG, f"exit_{dt.date.today()}.json"), "w"), indent=1)
    print(json.dumps(rec, indent=1))
    if submit:
        for od in orders:
            r = requests.post(f"{TRADE}/orders", headers=H, json=od, timeout=30)
            print(od["symbol"], r.status_code, r.text[:200])


def session_today():
    """Today's regular session from the Alpaca calendar: (open, close) as 'HH:MM', or None on a holiday."""
    d = dt.date.today().isoformat()
    cal = requests.get(f"{TRADE}/calendar", headers=H, params=dict(start=d, end=d), timeout=30).json()
    return (cal[0]["open"], cal[0]["close"]) if cal and cal[0]["date"] == d else None


def reconcile():
    """Fills of the last 10 days against the official auction prints; appends to logs/paper/fills.csv and
    writes the account equity history to logs/paper/equity.csv."""
    after = (dt.date.today() - dt.timedelta(days=10)).isoformat()
    od = requests.get(f"{TRADE}/orders", headers=H, timeout=30,
                      params=dict(status="closed", after=after + "T00:00:00Z", limit=500, direction="asc")).json()
    rows = []
    for o in od:
        if o.get("filled_at") and o.get("time_in_force") in ("cls", "opg"):
            day = pd.Timestamp(o["filled_at"]).tz_convert("America/New_York").strftime("%Y-%m-%d")
            try:
                off = A._official(o["symbol"], day, "close" if o["time_in_force"] == "cls" else "open")[0]
            except RuntimeError:
                off = np.nan
            px = float(o["filled_avg_price"])
            sign = 1 if o["side"] == "buy" else -1
            rows.append(dict(day=day, symbol=o["symbol"], side=o["side"], tif=o["time_in_force"],
                             qty=float(o["filled_qty"]), fill=px, official=off,
                             slip_bps=sign * 1e4 * (px / off - 1) if off == off and off else np.nan))
    fn = os.path.join(LOG, "fills.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(fn) and len(new):
        new = pd.concat([pd.read_csv(fn), new]).drop_duplicates(["day", "symbol", "side", "tif"], keep="last")
    if len(new):
        new.sort_values(["day", "tif", "symbol"]).to_csv(fn, index=False)
        print(new.tail(20).to_string(index=False))
        print("mean slippage vs official print (bp, positive = worse):", round(new.slip_bps.mean(), 2))
    ph = requests.get(f"{TRADE}/account/portfolio/history", headers=H, timeout=30,
                      params=dict(period="3M", timeframe="1D")).json()
    if ph.get("timestamp"):
        eq = pd.DataFrame({"date": pd.to_datetime(ph["timestamp"], unit="s").date, "equity": ph["equity"],
                           "pnl": ph["profit_loss"]})
        eq.to_csv(os.path.join(LOG, "equity.csv"), index=False)
        print(eq.tail(5).to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "entry"
    submit = "--submit" in sys.argv
    day = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--day=")), None)
    strategy = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--strategy=")), "ml")
    if strategy not in ("ml", "smallcap", "combo", "ensemble"):
        raise SystemExit("--strategy must be ml, ensemble, smallcap or combo")
    if what == "entry":
        entry(submit, day, strategy)
    elif what == "exit":
        exit_(submit)
    elif what == "session":
        print(session_today())
    elif what == "reconcile":
        reconcile()
