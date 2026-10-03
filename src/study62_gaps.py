"""Study 62: opening gaps in liquid stocks, fill or continuation?

Events: stock-days with |official open / previous close - 1| >= 2% (Yahoo adjusted prices) among stocks with
lagged 20-day median dollar volume > $20M and previous close > $5, 2020-01 .. 2026-09. The $5 filter uses the
traded price (Yahoo's raw close times later splits, s6162_common.traded_price); costs use the same price. Intraday prices from
5-minute SIP bars fetched for exactly these stock-days (data/local/m5s62, src/s6162_common.py gaps):
  premarket 07:00-09:25 (volume, last premarket trade), 09:30 bar (first trade, 09:35 price), 09:40 bar close
  (09:45 price), 10:25 bar close (10:30 price), 11:55 bar close (12:00 price). Close = official close (closing
  auction, Yahoo close in split-only units, the same scale as Alpaca's split-adjusted bars).

Splits of the events
  size        2-5%, 5-10%, >10%, gap up and gap down
  catalyst    earn : an earnings date (Nasdaq calendar, store 'earnings') on the previous trading day or the day itself
                     AND an earnings headline (EPS/earnings/results/revenue/sales) for the symbol between 16:00 the day
                     before and 09:30
              news : no such earnings event, >= 1 Benzinga article (any) for the symbol in [prev 16:00, 09:30)
              none : no article in that window and no earnings
  premarket   pm_lo / pm_hi : premarket volume 07:00-09:25 (bars starting before 09:25) as a fraction of 20-day average daily volume, below / above
              the median of 2020-23 events (threshold fixed on 2020-23, applied to 2024-26)
Entries: open  = opening auction (market-on-open). The official gap is not known when a MOO order must be sent (~09:28),
                 so this entry buckets events by the PREMARKET gap (last trade before 09:25 / previous close) instead;
                 events are still drawn from days whose official gap is >= 2% (small selection on the open).
         open_sip910 = the same with the last trade before 09:10, which is what 15-minute-delayed SIP shows at 09:25
                 (the free Alpaca plan; IEX real-time premarket trades are sparse).
         09:35 = close of the 09:30-09:35 bar, 09:45 = close of the 09:40-09:45 bar (official gap known by then).
Exits: 10:30, 12:00 (continuous market), close (closing auction).
Both sides for every event: long and short (for a gap up, long = continuation, short = fade).
Costs per side: auction = core.exec_cost_bps(P, "auction") + 2.5 bp; continuous = quote-model half-spread at that
time of day (09:35/12:00 calibration points, 09:45 and 10:30 interpolated) + 2 bp slippage + 0.3 bp fees + 2.5 bp.
Shorts: no borrow fee intraday, but locate availability (easy to borrow) is not knowable historically, and a stock
already >= 10% below its previous close is under the short-sale (uptick) rule -> short results are an upper bound.

Sanity filters: the Alpaca first trade must be within 3% of the official open (and ticker-months within 1% in the
median), and every regular-session price used must lie inside the Yahoo day's low-high range (+-1%).
Statistics per cell: events, mean gross/net bps per event, median net, hit rate, t of the mean computed on daily
averages (events on the same day are not independent), and the Sharpe of a portfolio that splits capital equally
among the day's events (cash on days without events), against SPY buy and hold.

    python src/study62_gaps.py            # writes results/study62_gaps.csv
    S62_EVENTS=/path/ev.parquet python src/study62_gaps.py              # also saves the event-level returns
    S62_EVENTS=/path/ev.parquet python src/study62_gaps.py checkopen    # official opening cross vs Yahoo open
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
import store
from core import load_panel, stock_cols, ann_stats, RES
from s6162_common import cont_cost_bps, auction_cost_bps, gap_mask, with_traded_price

PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-30"),
       ("val", "2024-01-01", "2025-06-30"), ("oos", "2025-07-01", "2026-09-30")]
P = load_panel()
PT = with_traded_price(P)            # real price levels for the $5 filter and the cost model
days = P["c"].index
cols = stock_cols(P)
F = P["c"] / P["rawc"]
PC_DIV = (P["rawc"].shift(1) * F.shift(1) / F)
O_S = P["o"] / F


def stack(df, name, idx):
    """Values of a wide panel at the (date, ticker) MultiIndex idx."""
    s = df.stack()
    s.index.names = ["date", "ticker"]
    return s.reindex(idx).rename(name)


def load_bars():
    parts = []
    d0 = os.path.join(A.LOCAL, "m5s62")
    for fn in sorted(os.listdir(d0)):
        if not fn.endswith(".parquet"):
            continue
        d = pd.read_parquet(os.path.join(d0, fn), columns=["ts", "ticker", "o", "c", "v"])
        hm = (d.ts.dt.hour * 100 + d.ts.dt.minute).values
        d["date"] = d.ts.dt.normalize()
        k = ["date", "ticker"]
        pre = d[hm < 925]                      # bars starting 07:00 .. 09:20: trades before 09:25
        gp = pre.groupby(k)
        x = pd.DataFrame({"pre_v": gp.v.sum(), "pre_last": gp.c.last()})
        pre2 = d[hm < 910]                     # trades before 09:10: what 15-minute-delayed SIP shows at 09:25
        x = x.join(pre2.groupby(k).c.last().rename("pre_last910"), how="outer")
        for h, col, fld in [(930, "a_o", "o"), (930, "p0935", "c"), (940, "p0945", "c"), (1025, "p1030", "c"),
                            (1155, "p1200", "c")]:
            x = x.join(d[hm == h].set_index(k)[fld].rename(col), how="outer")
        parts.append(x)
    x = pd.concat(parts)
    x = x[~x.index.duplicated(keep="last")]
    x = x.reset_index()
    x["ticker"] = x.ticker.str.replace(".", "-", regex=False)
    return x.set_index(["date", "ticker"])


def overnight_news(ev_idx):
    """Article counts and earnings-headline counts per event in [prev trading day 16:00, day 09:30)."""
    import news_features as NF
    d = NF.load_news()
    syms = set(ev_idx.get_level_values(1))
    d = d[d.sym.isin(syms)]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    cut = pd.DatetimeIndex(days) + pd.Timedelta(hours=9, minutes=30)
    start = pd.DatetimeIndex(np.r_[[pd.Timestamp("1990-01-01")], (pd.DatetimeIndex(days[:-1]) + pd.Timedelta(hours=16)).values])
    i = cut.searchsorted(ts.values, side="right")         # first day whose 09:30 cutoff is > ts
    ok = i < len(days)
    i = np.where(ok, i, 0)
    ok &= ts.values >= start[i]
    earn = d.headline.fillna("").str.contains(r"(?i)(?:\bEPS\b|earnings|results|revenue|sales)", regex=True).values
    dd = pd.DataFrame({"date": pd.DatetimeIndex(days)[i[ok]], "ticker": d.sym.values[ok], "earn_h": earn[ok]})
    g = dd.groupby(["date", "ticker"])
    out = pd.DataFrame({"n_news": g.size(), "n_earn_h": g.earn_h.sum()}).reindex(ev_idx).fillna(0)
    return out, ts.min()


def build_events():
    m = gap_mask(P)                  # the fetched set (price filter on Yahoo's split-adjusted close)
    m &= (PT["rawc"][cols].shift(1) > 5).reindex_like(m)    # traded price > $5 the day before
    print("events after the traded-price filter", int(m.sum().sum()), flush=True)
    s = m.stack()
    s = s[s]
    s.index.names = ["date", "ticker"]
    idx = s.index
    ev = pd.DataFrame(index=idx)
    ev["gap"] = stack(P["o"][cols] / P["c"][cols].shift(1) - 1, "gap", idx)
    ev["pc"] = stack(PC_DIV[cols], "pc", idx)
    ev["o_s"] = stack(O_S[cols], "o_s", idx)
    ev["close"] = stack(P["rawc"][cols], "close", idx)
    ev["lo_s"] = stack((P["l"] / F)[cols], "lo_s", idx)
    ev["hi_s"] = stack((P["h"] / F)[cols], "hi_s", idx)
    ev["advsh"] = stack(P["v"][cols].rolling(20, min_periods=10).mean().shift(1), "advsh", idx)
    for nm, panel in [("c_auc", auction_cost_bps(PT)), ("c_0935", cont_cost_bps(PT, "09:35")),
                      ("c_0945", cont_cost_bps(PT, "09:45")), ("c_1030", cont_cost_bps(PT, "10:30")),
                      ("c_1200", cont_cost_bps(PT, "12:00"))]:
        ev[nm] = stack(panel[cols], nm, idx)
    B = load_bars()
    ev = ev.join(B, how="left")
    n0 = len(ev)
    has = ev.a_o.notna() & ev.p0935.notna()
    print("events", n0, "with intraday bars", int(has.sum()), flush=True)
    # Alpaca / Yahoo consistency: first trade vs official open, and ticker-months with systematic disagreement
    ev["dif"] = (ev.a_o / ev.o_s - 1).abs()
    tm = ev.groupby([ev.index.get_level_values(0).to_period("M"), ev.index.get_level_values(1)]).dif.transform("median")
    ok = has & (ev.dif < 0.03) & (tm < 0.01) & ev.close.notna() & ev.pc.notna()
    print("dropped by consistency filter", int((has & ~ok).sum()), flush=True)
    # every regular-session price used must lie inside the Yahoo day range (+-1%): catches bad prints and
    # Alpaca/Yahoo symbol or split mismatches in the later bars
    inside = pd.Series(True, index=ev.index)
    for k in ["p0935", "p0945", "p1030", "p1200"]:
        inside &= ev[k].isna() | ((ev[k] >= 0.99 * ev.lo_s) & (ev[k] <= 1.01 * ev.hi_s))
    print("dropped: intraday price outside the Yahoo day range", int((ok & ~inside).sum()), flush=True)
    ok &= inside
    ev = ev[ok].copy()
    # catalysts
    E = store.read("earnings")
    E["date"] = pd.to_datetime(E.date).dt.normalize()
    di = pd.Series(np.arange(len(days)), index=days)
    E = E[E.date.isin(days)]
    E["di"] = di.reindex(E.date).values
    cal = set(zip(E.di, E.symbol)) | set(zip(E.di + 1, E.symbol))       # report day or the next trading day
    edi = di.reindex(ev.index.get_level_values(0)).values
    ev["earn_cal"] = [(a, t) in cal for a, t in zip(edi, ev.index.get_level_values(1))]
    nn, news_start = overnight_news(ev.index)
    ev = ev.join(nn)
    print("news archive starts", news_start, flush=True)
    ev["cat"] = np.where(ev.earn_cal & (ev.n_earn_h > 0), "earn", np.where(ev.n_news > 0, "news", "none"))
    print("earnings-calendar events without an overnight earnings headline (classified by news count):",
          int((ev.earn_cal & (ev.n_earn_h == 0)).sum()), flush=True)
    ev["pmrel"] = ev.pre_v.fillna(0) / ev.advsh
    thr = ev.loc[:"2023-12-31", "pmrel"].median()
    ev["pm"] = np.where(ev.pmrel > thr, "pm_hi", "pm_lo")
    print("premarket volume threshold (median 2020-23, fraction of ADV):", round(thr, 4), flush=True)
    ev["gpre"] = ev.pre_last / ev.pc - 1
    ev["gpre910"] = ev.pre_last910 / ev.pc - 1
    return ev


def size_bucket(g):
    a = g.abs()
    return np.select([a < 0.02, a < 0.05, a < 0.10], ["<2%", "2-5%", "5-10%"], ">10%")


def main():
    ev = build_events()
    spy = P["c"]["SPY"].pct_change()
    rows = []
    for per, a, b in PER:
        st = ann_stats(spy.loc[a:b])
        rows.append(dict(period=per, entry="bench", exit="SPY_buy_hold", side="long", direction="", size="",
                         split="", n=st["n"], sharpe=st["sharpe"], tstat=st["tstat"]))
    entries = {"open": ("o_s", "c_auc", "gpre"), "open_sip910": ("o_s", "c_auc", "gpre910"),
               "09:35": ("p0935", "c_0935", "gap"), "09:45": ("p0945", "c_0945", "gap")}
    exits = {"10:30": ("p1030", "c_1030"), "12:00": ("p1200", "c_1200"), "close": ("close", "c_auc")}
    out_events = []
    for en, (pin, cin, gcol) in entries.items():
        g = ev[gcol]
        sz = pd.Series(size_bucket(g.fillna(0)), index=ev.index)
        direc = np.where(g > 0, "up", "down")
        valid_g = g.notna() & (sz != "<2%")
        for ex, (pout, cout) in exits.items():
            r = ev[pout] / ev[pin] - 1
            cost = (ev[cin] + ev[cout]) / 1e4
            ok = valid_g & r.notna() & cost.notna()
            df = pd.DataFrame({"date": ev.index.get_level_values(0), "ticker": ev.index.get_level_values(1), "r": r.values, "cost": cost.values,
                               "size": sz.values, "direction": direc, "cat": ev["cat"].values, "pm": ev["pm"].values})[ok.values]
            df["entry"], df["exit"] = en, ex
            out_events.append(df)
            for side, sgn in [("long", 1), ("short", -1)]:
                df["gross"] = sgn * df.r
                df["net"] = df.gross - df.cost
                for direction in ["up", "down"]:
                    for size in ["2-5%", "5-10%", ">10%"]:
                        base = df[(df.direction == direction) & (df["size"] == size)]
                        splits = {"all": base}
                        for c in ["earn", "news", "none"]:
                            splits[c] = base[base.cat == c]
                        for c in ["pm_lo", "pm_hi"]:
                            splits[c] = base[base.pm == c]
                        for sp, x in splits.items():
                            for per, a, b in PER:
                                y = x[(x.date >= a) & (x.date <= b)]
                                row = dict(period=per, entry=en, exit=ex, side=side, direction=direction, size=size,
                                           split=sp, n=len(y))
                                if len(y) >= 10:
                                    dm = y.groupby("date").net.mean()
                                    port = dm.reindex(days[(days >= a) & (days <= b) & (days <= ev.index.get_level_values(0).max())]).fillna(0)
                                    st = ann_stats(port)
                                    row.update(gross_bps=1e4 * y.gross.mean(), net_bps=1e4 * y.net.mean(),
                                               median_net_bps=1e4 * y.net.median(), cost_bps=1e4 * y.cost.mean(),
                                               hit=(y.net > 0).mean(), n_days=len(dm),
                                               t_daily=dm.mean() / dm.std() * np.sqrt(len(dm)) if len(dm) > 2 else np.nan,
                                               sharpe=st["sharpe"], tstat=st["tstat"], ann_ret=st["ann_ret"],
                                               trim_net_bps=1e4 * y.net.clip(y.net.quantile(0.01), y.net.quantile(0.99)).mean())
                                rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "study62_gaps.csv"), index=False)
    E = pd.concat(out_events, ignore_index=True)
    if os.environ.get("S62_EVENTS"):
        E.to_parquet(os.environ["S62_EVENTS"], index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("\ncategory counts (09:35 entry, close exit):")
    x = E[(E.entry == "09:35") & (E.exit == "close")]
    print(pd.crosstab([x["size"], x.direction], [x.date.dt.year >= 2024, x.cat]))
    for en in ["open", "open_sip910", "09:35", "09:45"]:
        for ex in ["10:30", "12:00", "close"]:
            y = res[(res.entry == en) & (res.exit == ex) & (res.split == "all")]
            v = y.pivot_table(index=["direction", "size", "side"], columns="period", values=["net_bps", "t_daily", "sharpe"])
            print(f"\n== entry {en} exit {ex} (all events)")
            want = [(m, p) for p in ["2020-23", "2024-26"] for m in ["net_bps", "t_daily", "sharpe"]]
            print(v[[c for c in want if c in v.columns]].round(2).to_string())
    print(res[res.entry == "bench"][["period", "sharpe"]])


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def check_open(n_per=250, seed=0):
    """Is Yahoo's open the price a market-on-open order gets? For a random sample of premarket gap-ups > 10%
    (the one cell that looks profitable), fetch the primary-exchange opening cross from SIP trades (condition 'O')
    and recompute the open -> close short with it. Not cached (data/local/auctions.parquet belongs to other studies)."""
    import alpaca_data as A
    from s6162_common import traded_price
    ev_fn = os.environ["S62_EVENTS"]
    E = pd.read_parquet(ev_fn)
    x = E[(E.entry == "open") & (E.exit == "close") & (E.direction == "up") & (E["size"] == ">10%")].copy()
    rng = np.random.default_rng(seed)
    parts = []
    for a, b in [("2020-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")]:
        y = x[(x.date >= a) & (x.date <= b)]
        parts.append(y.iloc[rng.choice(len(y), min(n_per, len(y)), replace=False)])
    s = pd.concat(parts)
    fac = (traded_price(P) / P["rawc"])
    rows = []
    for _, r in s.iterrows():
        sym = r.ticker.replace("-", ".")
        try:
            off, first = A._official(sym, r.date.strftime("%Y-%m-%d"), "open")
        except RuntimeError:
            off, first = np.nan, np.nan
        f = fac.at[r.date, r.ticker]
        o_y = (O_S.at[r.date, r.ticker]) * f            # Yahoo open in traded units
        c_y = P["rawc"].at[r.date, r.ticker] * f
        rows.append(dict(date=r.date, ticker=r.ticker, o_yahoo=o_y, o_official=off, first_trade=first, close=c_y,
                         r_yahoo=c_y / o_y - 1, r_official=c_y / off - 1 if off == off else np.nan, cost=r.cost))
    out = pd.DataFrame(rows)
    out["diff_bps"] = 1e4 * (out.o_official / out.o_yahoo - 1)
    print("official open found:", out.o_official.notna().sum(), "of", len(out))
    print("official / Yahoo open - 1 (bps):", out.diff_bps.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_dict())
    if os.environ.get("S62_CHECK_OUT"):
        out.to_csv(os.environ["S62_CHECK_OUT"], index=False)
    bad = out.diff_bps.abs() > 2000
    print("pairs with |official/Yahoo - 1| > 20% (split/symbol mismatch, excluded below):", int(bad.sum()))
    for lab, m in [("2020-23", out.date < "2024-01-01"), ("2024-26", out.date >= "2024-01-01")]:
        z = out[m & out.o_official.notna() & ~bad]
        print(lab, "n", len(z), "short net bps, Yahoo open:", round(1e4 * (-z.r_yahoo - z.cost).mean(), 1),
              " official open:", round(1e4 * (-z.r_official - z.cost).mean(), 1),
              " median diff bps:", round(z.diff_bps.median(), 1))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "checkopen":
    check_open()
