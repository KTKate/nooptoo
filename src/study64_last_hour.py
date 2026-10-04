"""Study 64: the last hour.

(a) Does the 09:30 -> 15:00 (or 15:30) return predict the 15:00 -> close / 15:30 -> close return (continuation into
    the close, e.g. institutional rebalancing or leveraged-ETF hedging), for stocks and for SPY/QQQ/IWM/DIA?
(b) Do 15:30-15:45 volume surges (vs the stock's usual share of daily volume at that time) predict the 15:45 -> close
    move and the next overnight (close -> next open) return?
(c) Overlap with the overnight blend (study 33: 2/3 ensemble rank + 1/3 jump-minus-drop rank, top 10 bought at the
    close and sold at the next open): can a last-30-minute signal choose better among the blend's top 20?

Data
  Stocks (a, b): the 500 most traded stocks each day (lagged 20-day median dollar volume, traded price > $5).
    2020-23: 15-minute SIP bars 14:45 .. 15:45 (data/local/m15s64).  price(15:00) = close of the 14:45 bar, entry
      15:00 = open of the 15:00 bar; price(15:30) = close of the 15:15 bar, entry = open of the 15:30 bar;
      price(15:45) = close of the 15:30 bar, entry = open of the 15:45 bar; v(15:30-15:45) = volume of the 15:30 bar.
    2024-26: 5-minute SIP bars 14:55, 15:00, 15:25 (data/local/m5s64) + 15:30-15:55 (data/local/m5snap); same
      definitions on 5-minute bars (price(15:00) = close of the 14:55 bar, entry = open of the 15:00 bar, ...).
    Fetched by src/s6364_fetch.py. Exit = official close (Yahoo rawc, split-only units); next open = official open.
    Day return = price(T) / official open - 1 (also from the dividend-adjusted previous close, and minus SPY's).
  ETFs: 1-minute SIP bars (data/local/m1), price(T) = close of the bar ending at T, entry one minute later.
  Volume surge = v(15:30-15:45) / (usual share x 20-day median daily volume), usual share = median over the previous
    20 days of v(15:30-15:45) / daily volume (both lagged one day; known at 15:45).
  Blend (c): results/study23_pred.parquet + results/study33_pred.parquet, 2024-01 .. 2026-09; 15:30-15:55 bars for
    the picks from m5snap + m5snapx (about 4,600 tickers).
Costs per side: continuous-trading entries at 15:00/15:30/15:45 = s6162_common.cont_cost_bps (15:00 and 15:30 are
  interpolated between the 12:00 and 15:45 calibration points) ; closing or opening auction = exec_cost_bps(P,
  "auction") + 2.5 bp. Intraday/overnight shorts: no borrow fee charged (top-500 names, locate assumed).
Data checks: last trade before 16:00 within 3% of the official close; signal and entry prices inside the Yahoo day's
  low-high range (+-1%); ticker-months with median disagreement > 0.5% dropped.
Free plan: price signals (a) can be computed from IEX real-time trades/quotes; the SIP volume of 15:30-15:45 (b) is
  only available after 16:00 on the free plan (15-minute delay), so (b) needs IEX volume as a proxy (IEX is ~2-3% of
  consolidated volume; not tested here) or the $99/month SIP plan.

    python src/study64_last_hour.py      # writes results/study64_last_hour.csv
"""
import os
import sys

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
import bt
from core import load_panel, stock_cols, ann_stats, exec_cost_bps, RES
from s6162_common import cont_cost_bps, auction_cost_bps, with_traded_price

PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-10-31"),
       ("val", "2024-01-01", "2025-06-30"), ("oos", "2025-07-01", "2026-10-31")]
rows = []
warnings.filterwarnings("ignore")
FIELDS = ["p1500", "e1500", "p1530", "e1530", "p1545", "e1545", "v3045", "last"]


def add_rows(part, name, r_net, r_gross=None, extra=None, active=None):
    for per, a, b in PER:
        x = r_net.loc[a:b]
        if len(x) < 5:
            continue
        st = ann_stats(x)
        act = (active.loc[a:b] if active is not None else (x != 0))
        row = dict(part=part, variant=name, period=per, n_days=st["n"], active_days=int(act.sum()),
                   sharpe=st["sharpe"], tstat=st["tstat"], ann_ret=st["ann_ret"], maxdd=st["maxdd"],
                   net_bps_active=1e4 * x[act].mean() if act.any() else np.nan)
        if r_gross is not None:
            g = r_gross.loc[a:b]
            row["sharpe_gross"] = ann_stats(g)["sharpe"]
            row["gross_bps_active"] = 1e4 * g[act].mean() if act.any() else np.nan
            row["tstat_gross"] = ann_stats(g)["tstat"]
        row.update(extra or {})
        rows.append(row)


def nw_t(y, x, lags=5):
    m = pd.concat([y, x], axis=1).dropna()
    if len(m) < 30:
        return np.nan, np.nan, len(m)
    yv, xv = m.iloc[:, 0].values, m.iloc[:, 1].values
    X = np.column_stack([np.ones(len(xv)), xv])
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ b
    XtX = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        G = u[L:].T @ u[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    V = XtX @ S @ XtX
    return b[1], b[1] / np.sqrt(V[1, 1]), len(m)


def _snap_month(fn, keep):
    """15:30-15:55 5-minute bars of one month -> e1530, p1545, e1545, v3045, last (index date, ticker)."""
    d = pd.read_parquet(fn, columns=["ts", "ticker", "o", "c", "v"])
    if keep is not None:
        d = d[d.ticker.isin(keep)]
    hm = d.ts.dt.hour * 100 + d.ts.dt.minute
    d = d[hm >= 1530]
    hm = hm[hm >= 1530]
    d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
    ix = ["date", "ticker"]
    x = pd.DataFrame({"e1530": d[hm == 1530].set_index(ix).o})
    x = x.join(pd.DataFrame({"p1545": d[hm == 1540].set_index(ix).c}), how="outer")
    x = x.join(pd.DataFrame({"e1545": d[hm == 1545].set_index(ix).o}), how="outer")
    x = x.join(pd.DataFrame({"v3045": d[hm < 1545].groupby(ix).v.sum(),
                             "last": d.groupby(ix).c.last()}), how="outer")
    return x


def load_stock_bars(keep24):
    parts = []
    d0 = os.path.join(A.LOCAL, "m15s64")
    for fn in sorted(os.listdir(d0)):
        if not fn.endswith(".parquet"):
            continue
        d = pd.read_parquet(os.path.join(d0, fn), columns=["ts", "ticker", "o", "c", "v"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
        g = lambda h: d[hm == h].set_index(["date", "ticker"])
        b1445, b1500, b1515, b1530, b1545 = g(1445), g(1500), g(1515), g(1530), g(1545)
        x = pd.DataFrame({"p1500": b1445.c}).join(pd.DataFrame({"e1500": b1500.o}), how="outer")
        x = x.join(pd.DataFrame({"p1530": b1515.c}), how="outer")
        x = x.join(pd.DataFrame({"e1530": b1530.o, "p1545": b1530.c, "v3045": b1530.v}), how="outer")
        x = x.join(pd.DataFrame({"e1545": b1545.o, "last": b1545.c}), how="outer")
        parts.append(x)
    d0 = os.path.join(A.LOCAL, "m5s64")
    for fn in sorted(os.listdir(d0)):
        if not fn.endswith(".parquet"):
            continue
        d = pd.read_parquet(os.path.join(d0, fn), columns=["ts", "ticker", "o", "c"])
        hm = d.ts.dt.hour * 100 + d.ts.dt.minute
        d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
        g = lambda h: d[hm == h].set_index(["date", "ticker"])
        x = pd.DataFrame({"p1500": g(1455).c}).join(pd.DataFrame({"e1500": g(1500).o}), how="outer")
        x = x.join(pd.DataFrame({"p1530": g(1525).c}), how="outer")
        sn = os.path.join(A.LOCAL, "m5snap", fn)
        if os.path.exists(sn):
            x = x.join(_snap_month(sn, keep24), how="outer")
        parts.append(x)
    x = pd.concat(parts)
    x = x[~x.index.duplicated(keep="last")].reset_index()
    x["ticker"] = x.ticker.str.replace(".", "-", regex=False)
    print("stock bars", len(x), x.date.min().date(), x.date.max().date(), flush=True)
    return {k: x.pivot(index="date", columns="ticker", values=k) for k in FIELDS}


def load_snap_all():
    """m5snap + m5snapx 15:30-16:00 fields for every ticker (2024+), for the blend overlap."""
    parts = []
    for ds in ["m5snap", "m5snapx"]:
        d0 = os.path.join(A.LOCAL, ds)
        for fn in sorted(os.listdir(d0)):
            if fn.endswith(".parquet") and fn >= "2023-12":
                parts.append(_snap_month(os.path.join(d0, fn), None))
    x = pd.concat(parts)
    x = x[~x.index.duplicated(keep="first")].reset_index()
    x["ticker"] = x.ticker.str.replace(".", "-", regex=False)
    return {k: x.pivot(index="date", columns="ticker", values=k) for k in ["e1530", "p1545", "v3045", "last"]}


# ================================================================== ETFs
def part_etf(P, PT, F, C):
    O_S = P["o"] / F
    PC_DIV = P["rawc"].shift(1) * F.shift(1) / F
    for t in ["SPY", "QQQ", "IWM", "DIA"]:
        d = A.read("m1", tickers=[t], columns=["ts", "ticker", "c"])
        d = d[d.ts >= "2020-01-01"]
        hm = d.ts.dt.strftime("%H:%M")
        d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
        g = lambda h: d[hm == h].set_index("date").c
        f = pd.DataFrame({"p1500": g("14:59"), "e1500": g("15:00"), "p1530": g("15:29"), "e1530": g("15:30")})
        days = P["c"].index[P["c"].index >= "2020-01-02"]
        f = f.reindex(days).dropna()
        o, pc, close = O_S[t].reindex(f.index), PC_DIV[t].reindex(f.index), P["rawc"][t].reindex(f.index)
        night = (P["o"][t].shift(-1) / P["c"][t] - 1).reindex(f.index)
        cout = C["auc"][t].reindex(f.index) / 1e4
        add_rows("etf_bench", f"{t}|buy_hold", P["c"][t].pct_change().reindex(f.index))
        for T in ["1500", "1530"]:
            y = close / f["e" + T] - 1
            cin = C[T][t].reindex(f.index) / 1e4
            add_rows("etf_bench", f"{t}|always_long_{T}_close", y - cin - cout, y)
            for sn, s in [("day", f["p" + T] / o - 1), ("pc", f["p" + T] / pc - 1)]:
                for per, a, b in PER:
                    sl, tt, n = nw_t(y.loc[a:b], s.loc[a:b])
                    rows.append(dict(part="etf_reg", variant=f"{t}|{sn}{T}->close", period=per, n_days=n,
                                     slope=sl, tstat=tt))
                    sl, tt, n = nw_t(night.loc[a:b], s.loc[a:b])
                    rows.append(dict(part="etf_reg", variant=f"{t}|{sn}{T}->night", period=per, n_days=n,
                                     slope=sl, tstat=tt))
                thr = s.abs().rolling(250, min_periods=120).quantile(0.7).shift(1)
                for rn, pos in [("ls", np.sign(s)), ("lo", (s > 0).astype(float)),
                                ("big_ls", np.sign(s) * (s.abs() > thr))]:
                    pos = pos.fillna(0)
                    gross = pos * y
                    net = gross - (pos != 0) * (cin + cout)
                    add_rows("etf_rule", f"{t}|{sn}{T}->close|{rn}", net, gross, active=(pos != 0))


# ================================================================== stocks
def run_side(Wsel, y, cside, side):
    r = bt.run(Wsel, y, cside, side=side)
    return r


def topk_fixed(S, E, k=10, largest=True):
    s = S.where(E)
    rk = s.rank(axis=1, ascending=not largest, method="first")
    return (rk <= k).astype(float) / k


def part_stocks(P, PT, F, C, snap_all):
    O_S = P["o"] / F
    PC_DIV = P["rawc"].shift(1) * F.shift(1) / F
    sc = stock_cols(P)
    adv0 = P["dv"][sc].rolling(20, min_periods=10).median().shift(1)
    rk0 = adv0.where(PT["rawc"][sc].shift(1) > 5).rank(axis=1, ascending=False)
    keep = set((rk0.loc["2024-01-01":] <= 520).any().pipe(lambda s: s[s].index))
    keep = {t.replace("-", ".") for t in keep}
    W = load_stock_bars(keep)
    cols = sorted(set(sc) & set(W["last"].columns))
    days = P["c"].index[(P["c"].index >= "2020-01-02") & (P["c"].index <= W["last"].index.max())]
    W = {k: v.reindex(index=days, columns=cols) for k, v in W.items()}
    close = P["rawc"][cols].reindex(days)
    o_s, pc = O_S[cols].reindex(days), PC_DIV[cols].reindex(days)
    lo_s, hi_s = (P["l"] / F)[cols].reindex(days), (P["h"] / F)[cols].reindex(days)
    inside = lambda q: (q >= 0.99 * lo_s) & (q <= 1.01 * hi_s)
    dif = (W["last"] / close - 1).abs()
    tm = dif.groupby(dif.index.to_period("M")).transform("median")
    ok = (dif < 0.03) & (tm < 0.005)
    for k in ["p1500", "e1500", "p1530", "e1530", "p1545", "e1545"]:
        ok &= inside(W[k])
    print("stock-days with last price", int(W["last"].notna().sum().sum()), "kept", int(ok.sum().sum()), flush=True)
    adv = adv0[cols].reindex(days)
    rk = adv.where(PT["rawc"][cols].shift(1).reindex(days) > 5).rank(axis=1, ascending=False)
    U = (rk <= 500) & ok
    print("universe per day", U.sum(1).describe().round(0).to_dict(), flush=True)
    for k in ["p1500", "e1500", "p1530", "e1530", "p1545", "e1545"]:
        print(k, "missing share in U", round(float((W[k].isna() & U).sum().sum() / U.sum().sum()), 4))

    night = (P["o"].shift(-1) / P["c"] - 1)[cols].reindex(days)
    spy_o = O_S["SPY"].reindex(days)
    spy = A.read("m1", tickers=["SPY"], columns=["ts", "c"])
    spy["date"] = spy.ts.dt.normalize().astype("datetime64[ns]")
    shm = spy.ts.dt.strftime("%H:%M")
    spy_p = {T: spy[shm == h].set_index("date").c.reindex(days) for T, h in [("1500", "14:59"), ("1530", "15:29")]}
    cin = {T: C[T][cols].reindex(days) for T in ["1500", "1530", "1545"]}
    cout = C["auc"][cols].reindex(days)
    cout_next = C["auc"][cols].shift(-1).reindex(days)

    # volume surge
    V = P["v"][cols].reindex(days)
    share = (W["v3045"] / V).where(ok & (V > 0))
    usual = share.rolling(20, min_periods=10).median().shift(1)
    vmed = P["v"][cols].rolling(20, min_periods=10).median().shift(1).reindex(days)
    surge = W["v3045"] / (usual * vmed)
    print("usual share 15:30-15:45 median", float(usual.where(U).stack().median()),
          "surge quantiles", surge.where(U).stack().quantile([.1, .5, .9, .99]).round(2).to_dict(), flush=True)

    add_rows("bench", "SPY|buy_hold", P["c"]["SPY"].pct_change().reindex(days))
    tg = {"1500": close / W["e1500"] - 1, "1530": close / W["e1530"] - 1, "1545": close / W["e1545"] - 1}
    for T, y in tg.items():
        E = U & y.notna()
        Wu = E.astype(float).div(E.sum(1).replace(0, np.nan), axis=0).fillna(0)
        r = bt.run(Wu, y, (cin[T] + cout) / 2)
        add_rows("bench", f"universe_ew_{T}_close", r.net, r.gross)
    Eu = U & night.notna()
    r = bt.run(Eu.astype(float).div(Eu.sum(1).replace(0, np.nan), axis=0).fillna(0), night, (cout + cout_next) / 2)
    add_rows("bench", "universe_ew_night", r.net, r.gross)

    def ic_rows(tag, s, y, E):
        ic = s.where(E).rank(axis=1).corrwith(y.where(E).rank(axis=1), axis=1)
        for per, a, b in PER:
            x = ic.loc[a:b].dropna()
            if len(x) > 30:
                rows.append(dict(part="ic", variant=tag, period=per, n_days=len(x), ic=x.mean(),
                                 tstat=x.mean() / x.std() * np.sqrt(len(x))))

    def book(tag, s, y, E, cside, part="rule_a"):
        """Long top 10 / long bottom 10 / ls momentum (long top, short bottom) / ls reversal, 1/10 per name."""
        E = E & s.notna() & y.notna()
        Wt, Wb = topk_fixed(s, E, 10, True), topk_fixed(s, E, 10, False)
        rt, rb = bt.run(Wt, y, cside), bt.run(Wb, y, cside)
        act = rt.n > 0
        add_rows(part, tag + "|long_top", rt.net, rt.gross, active=act)
        add_rows(part, tag + "|long_bottom", rb.net, rb.gross, active=act)
        add_rows(part, tag + "|ls_mom", 0.5 * (rt.gross - rb.gross) - 0.5 * (rt.cost + rb.cost),
                 0.5 * (rt.gross - rb.gross), active=act)
        add_rows(part, tag + "|ls_rev", 0.5 * (rb.gross - rt.gross) - 0.5 * (rt.cost + rb.cost),
                 0.5 * (rb.gross - rt.gross), active=act)

    # ---------------- (a) day return -> last hour / half hour / overnight
    for T in ["1500", "1530"]:
        p = W["p" + T]
        sig = {"day": p / o_s - 1, "pc": p / pc - 1}
        sig["day_rel"] = sig["day"].sub(spy_p[T] / spy_o - 1, axis=0)
        y = tg[T]
        y_night = (1 + y) * (1 + night) - 1
        for sn, s in sig.items():
            ic_rows(f"a|{sn}{T}->close", s, y, U)
            ic_rows(f"a|{sn}{T}->night", s, night, U)
            book(f"a|{sn}{T}->close", s, y, U, (cin[T] + cout) / 2)
            book(f"a|{sn}{T}->close+night", s, y_night, U, (cin[T] + cout_next) / 2)
    print("part a done", flush=True)

    # ---------------- (b) volume surge 15:30-15:45
    r3045 = W["p1545"] / W["p1530"] - 1
    y45 = tg["1545"]
    y45n = (1 + y45) * (1 + night) - 1
    Eb = U & surge.notna() & r3045.notna() & (r3045 != 0)
    dirs = np.sign(r3045)
    ss = dirs * np.log(surge.clip(lower=1e-3))
    for nm, s in [("surge_signed", ss), ("r3045", r3045), ("surge_unsigned", surge)]:
        ic_rows(f"b|{nm}->close", s, y45, Eb)
        ic_rows(f"b|{nm}->night", s, night, Eb)
    # pooled decile diagnostic: direction-adjusted later return by surge decile
    dec = surge.where(Eb).rank(axis=1, pct=True)
    for lo_, hi_ in [(0, .5), (.5, .8), (.8, .9), (.9, .97), (.97, 1.01)]:
        m = Eb & (dec > lo_) & (dec <= hi_)
        for tn, y in [("close", y45), ("night", night)]:
            z = (dirs * y).where(m).mean(1)
            add_rows("b_decile", f"surge_pct_{lo_}-{hi_}|dir_x_{tn}", z.fillna(0), z.fillna(0), active=z.notna())
    for thr in [1.5, 2.0, 3.0]:
        up = Eb & (surge >= thr) & (r3045 > 0)
        dn = Eb & (surge >= thr) & (r3045 < 0)
        for tn, y, cs in [("close", y45, (cin["1545"] + cout) / 2), ("night", night, (cout + cout_next) / 2),
                          ("close+night", y45n, (cin["1545"] + cout_next) / 2)]:
            Wu_ = topk_fixed(surge, up, 10, True)
            Wd_ = topk_fixed(surge, dn, 10, True)
            ru, rd = bt.run(Wu_, y, cs), bt.run(Wd_, y, cs)
            tag = f"b|surge>={thr}|{tn}"
            add_rows("rule_b", tag + "|long_up_surges", ru.net, ru.gross, active=ru.n > 0,
                     extra=dict(avg_names=float(ru.n[ru.n > 0].mean())))
            add_rows("rule_b", tag + "|short_down_surges", -rd.gross - rd.cost, -rd.gross, active=rd.n > 0,
                     extra=dict(avg_names=float(rd.n[rd.n > 0].mean())))
            add_rows("rule_b", tag + "|long_down_surges", rd.net, rd.gross, active=rd.n > 0)
            add_rows("rule_b", tag + "|short_up_surges", -ru.gross - ru.cost, -ru.gross, active=ru.n > 0)
    book("b|r3045->close", r3045, y45, U, (cin["1545"] + cout) / 2, "rule_b")
    book("b|r3045->night_auction", r3045, night, U, (cout + cout_next) / 2, "rule_b")
    book("b|surge_signed->close", ss, y45, Eb, (cin["1545"] + cout) / 2, "rule_b")
    book("b|surge_signed->night_auction", ss, night, Eb, (cout + cout_next) / 2, "rule_b")
    print("part b done", flush=True)


# ================================================================== blend overlap
def part_blend(P, F, snap):
    O_S = P["o"] / F
    cols = stock_cols(P)
    days = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-2]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    bd = S.index
    close = P["rawc"][cols].reindex(bd)
    night = (P["o"].shift(-1) / P["c"] - 1)[cols].reindex(bd)
    cauc = exec_cost_bps(P, "auction")[cols] + 2.5
    cside = ((cauc + cauc.shift(-1)) / 2).reindex(bd)
    X = {k: v.reindex(index=bd, columns=cols) for k, v in snap.items()}
    lo_s, hi_s = (P["l"] / F)[cols].reindex(bd), (P["h"] / F)[cols].reindex(bd)
    good = ((X["last"] / close - 1).abs() < 0.03) & (X["p1545"] >= 0.99 * lo_s) & (X["p1545"] <= 1.01 * hi_s) & \
           (X["e1530"] >= 0.99 * lo_s) & (X["e1530"] <= 1.01 * hi_s)
    o_s = O_S[cols].reindex(bd)
    day1545 = (X["p1545"] / o_s - 1).where(good)
    r3045 = (X["p1545"] / X["e1530"] - 1).where(good)            # 15:30 first trade -> 15:45 last trade
    # surge on the snapshot set: history from the same source (only days present)
    V = P["v"][cols].reindex(bd)
    vfull = snap["v3045"].reindex(columns=cols)
    Vf = P["v"][cols].reindex(vfull.index)
    share = (vfull / Vf).where(Vf > 0)
    usual = share.rolling(20, min_periods=10).median().shift(1).reindex(bd)
    vmed = P["v"][cols].rolling(20, min_periods=10).median().shift(1).reindex(bd)
    surge = (X["v3045"] / (usual * vmed)).where(good)
    del V
    top20 = S.rank(axis=1, ascending=False, method="first") <= 20
    top10 = S.rank(axis=1, ascending=False, method="first") <= 10
    cov = {k: float((v.notna() & top20).sum().sum() / top20.sum().sum()) for k, v in
           [("day1545", day1545), ("r3045", r3045), ("surge", surge)]}
    print("blend top-20 coverage of last-30-minute signals", cov, flush=True)
    base = bt.run(top10.astype(float) / 10, night, cside)
    add_rows("blend", "baseline_top10", base.net, base.gross, extra=dict(coverage_top20=np.nan))
    # IC within the top 20
    sigs = {"day1545": day1545, "r3045": r3045, "surge_signed": np.sign(r3045) * np.log(surge.clip(lower=1e-3)),
            "surge": surge}
    for nm, s in sigs.items():
        E = top20 & s.notna() & night.notna()
        ic = s.where(E).rank(axis=1).corrwith(night.where(E).rank(axis=1), axis=1)
        # top-10-in-top-20 spread: picks with the signal above vs below the median of the top 20
        hi = E & (s.where(E).rank(axis=1, pct=True) > 0.5)
        lo = E & (s.where(E).rank(axis=1, pct=True) <= 0.5)
        spread = night.where(hi).mean(1) - night.where(lo).mean(1)
        for per, a, b in PER[1:]:
            x = ic.loc[a:b].dropna()
            z = spread.loc[a:b].dropna()
            if len(x) > 30:
                rows.append(dict(part="blend_ic", variant=f"top20|{nm}->night", period=per, n_days=len(x), ic=x.mean(),
                                 tstat=x.mean() / x.std() * np.sqrt(len(x)),
                                 net_bps_active=1e4 * z.mean(), sharpe=z.mean() / z.std() * np.sqrt(len(z)),
                                 coverage_top20=cov.get(nm, cov["surge"] if "surge" in nm else np.nan)))
        # rerank the top 20 with the signal (missing signal = neutral 0.5), both signs
        sr = s.where(top20).rank(axis=1, pct=True).fillna(0.5)
        base_r = S.where(top20).rank(axis=1, pct=True)
        for sign in [1, -1]:
            for lam in [0.25, 0.5, 1.0]:
                sc = (base_r + lam * (sr if sign > 0 else 1 - sr)).where(top20)
                Wn = (sc.rank(axis=1, ascending=False, method="first") <= 10).astype(float) / 10
                r = bt.run(Wn, night, cside)
                overlap = float(((Wn > 0) & top10).sum(1).mean())
                add_rows("blend", f"rerank_top20|{'+' if sign > 0 else '-'}{nm}|lam{lam}", r.net, r.gross,
                         extra=dict(avg_overlap_with_top10=overlap))
        # drop rule: skip baseline picks in the worst signal quintile of the top 20 (cash instead)
        for sign in [1, -1]:
            q = s.where(top20).rank(axis=1, pct=True)
            bad = (q <= 0.2) if sign > 0 else (q > 0.8)
            Wn = (top10 & ~bad.fillna(False)).astype(float) / 10
            r = bt.run(Wn, night, cside)
            add_rows("blend", f"skip_top10_if_{'low' if sign > 0 else 'high'}_{nm}", r.net, r.gross)
    print("blend done", flush=True)


def main():
    P = load_panel()
    PT = with_traded_price(P)
    F = P["c"] / P["rawc"]
    C = {"1500": cont_cost_bps(PT, "15:00"), "1530": cont_cost_bps(PT, "15:30"), "1545": cont_cost_bps(PT, "15:45"),
         "auc": auction_cost_bps(PT)}
    part_etf(P, PT, F, C)
    print("etf done", flush=True)
    part_stocks(P, PT, F, C, None)
    snap = load_snap_all()
    part_blend(P, F, snap)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "study64_last_hour.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 800)
    pv = lambda x, v: x.pivot_table(index="variant", columns="period", values=v)
    for part, vals in [("etf_reg", ["slope", "tstat"]), ("etf_bench", ["sharpe"]), ("etf_rule", ["sharpe", "gross_bps_active"]),
                       ("bench", ["sharpe", "gross_bps_active", "net_bps_active"]), ("ic", ["ic", "tstat"]),
                       ("rule_a", ["gross_bps_active", "net_bps_active", "sharpe"]),
                       ("b_decile", ["gross_bps_active", "tstat_gross"]),
                       ("rule_b", ["gross_bps_active", "net_bps_active", "sharpe"])]:
        x = df[df.part == part]
        if not len(x):
            continue
        t = pv(x, vals)
        c_ = [(v, p) for v in vals for p in ["2020-23", "2024-26"] if (v, p) in t]
        print("\n==", part)
        print(t[c_].round(3 if part in ("etf_reg", "ic") else 2).to_string())
    for part, vals in [("blend", ["sharpe", "net_bps_active"]), ("blend_ic", ["ic", "tstat", "net_bps_active"])]:
        x = df[df.part == part]
        t = pv(x, vals)
        c_ = [(v, p) for v in vals for p in ["2024-26", "val", "oos"] if (v, p) in t]
        print("\n==", part)
        print(t[c_].round(3).to_string())


if __name__ == "__main__":
    main()
