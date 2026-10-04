"""Study 63: intraday lead-lag within industries.

Question: does the first-hour move (official open -> 10:30) of an industry's leaders predict the 10:30 -> close
return of the industry's smaller members that have not yet moved (laggards)?

Data
  Stocks: the 1000 most traded stocks each day (lagged 20-day median dollar volume, traded price > $5), 5-minute SIP
    bars at 10:25 and 10:30 (data/local/m5s63, fetched by src/s6364_fetch.py), 2020-01 .. 2026-10.
    price at 10:30 (signal) = close of the 10:25 bar (last trade before 10:30); entry = open of the 10:30 bar (first
    trade at/after 10:30), so signal and entry prices are different trades. Exit = official close (Yahoo rawc).
    First-hour move r1 = price(10:30) / official open - 1 (open from Yahoo, split-only units); variant r1g from the
    previous close (dividend-adjusted, gap included).
  Industry labels: data/store/sectors.parquet (current Nasdaq-style sector + industry; a hindsight label: today's
    classification applied to 2020); Blank Checks and unlabeled names are dropped.
  Leaders (three definitions):
    etf    the sector/industry ETF (1-minute SIP bars, data/local/m1): Technology XLK (Semiconductors SMH),
           Finance XLF (banks KRE), Energy XLE, Health Care XLV (biotech industries XBI), Industrials XLI,
           Consumer Discretionary XLY, Consumer Staples XLP, Utilities XLU, Basic Materials XLB, Real Estate XLRE,
           Telecommunications XLC. ETF price at 10:30 = close of the 10:29 bar.
    adv3   the 3 most traded members of the industry that day (equal-weight r1); the rest are the members.
    mcap3  the 3 largest members by point-in-time market cap (SEC filings, data/local/fundamentals_panels.pkl,
           lagged one day; 20-day median dollar volume where market cap is missing, i.e. most of 2020H1).
  Laggards: members (not leaders) with sign(L) * r1_i < k * |L| (k = 0.25 or 0.5; own move in the leader's direction
    smaller than k times the leader's), in industries with |L| >= thr (0.5%, 1%, 2%); for etf leaders every
    industry member of the sector is a member. Industries need >= 3 members besides the leaders.
Portfolios (each name 1/10 of the side's capital, fewer candidates -> cash):
  long   top 10 laggards by catch-up gap (L - r1_i) in rising industries (L >= thr)
  short  top 10 laggards by gap (r1_i - L) in falling industries (L <= -thr)
  ls     half capital long, half short
  placebo: the same rules with SPY's first-hour move as the "leader" for every stock (market lead-lag only; study 61
  found that alone has no edge).
Diagnostics (gross, no selection): pooled Fama-MacBeth slope of the 10:30-close return on L among all members, with
  and without the member's own r1 as a control; mean 10:30-close return of all laggards in rising minus falling
  industries; and the same spread minus the leader ETF's 10:30-close return (industry-specific catch-up).
Costs per side: entry in continuous trading at 10:30 = s6162_common.cont_cost_bps(P, "10:30") (quote-model half-
  spread interpolated between the 09:35 and 12:00 calibration points + 2 bp slippage + 0.3 bp fees + 2.5 bp); exit in
  the closing auction = exec_cost_bps(P, "auction") + 2.5 bp. Intraday shorts pay no borrow fee (top-1000 names; a
  locate is assumed available, today's easy-to-borrow list is not knowable historically).
Data checks: the 10:30 signal and entry prices must lie inside the Yahoo day's low-high range (+-1%) and within 2% of
  each other; ticker-months with a median entry/signal gap above 0.5% are dropped.
Free plan: all inputs are prices at 10:30 (no volume), available in real time from the IEX feed (IEX quotes and
  trades for liquid names track the SIP closely); runnable on the free plan.

    python src/study63_leadlag.py        # writes results/study63_leadlag.csv
"""
import os
import sys

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A
import bt
from core import load_panel, stock_cols, ann_stats, RES, DATA
from s6162_common import cont_cost_bps, auction_cost_bps, with_traded_price

PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-10-31"),
       ("val", "2024-01-01", "2025-06-30"), ("oos", "2025-07-01", "2026-10-31")]
SECTOR_ETF = {"Technology": "XLK", "Finance": "XLF", "Energy": "XLE", "Health Care": "XLV", "Industrials": "XLI",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Utilities": "XLU",
              "Basic Materials": "XLB", "Real Estate": "XLRE", "Telecommunications": "XLC"}
rows = []
warnings.filterwarnings("ignore")


def industry_etf(sector, industry):
    if industry == "Semiconductors":
        return "SMH"
    if industry.startswith("Biotechnology") and sector == "Health Care":
        return "XBI"
    if industry in ("Major Banks", "Commercial Banks", "Banks", "Savings Institutions"):
        return "KRE"
    return SECTOR_ETF.get(sector)


def add_rows(part, name, r_net, r_gross=None, extra=None, active=None):
    for per, a, b in PER:
        x = r_net.loc[a:b]
        if not len(x):
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
        row.update(extra or {})
        rows.append(row)


def load_bars():
    d = A.read("m5s63", columns=["ts", "ticker", "o", "c", "v"])
    hm = d.ts.dt.hour * 100 + d.ts.dt.minute
    d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
    d["ticker"] = d.ticker.str.replace(".", "-", regex=False)
    a = d[hm == 1025].set_index(["date", "ticker"])
    b = d[hm == 1030].set_index(["date", "ticker"])
    x = pd.DataFrame({"p1030": a.c}).join(pd.DataFrame({"e1030": b.o}), how="outer")
    return {k: x[k].unstack() for k in x.columns}


def etf_first_hour(P, F, days):
    out_p, out_e = {}, {}
    tick = sorted(set(SECTOR_ETF.values()) | {"SMH", "XBI", "KRE", "SPY"})
    d = A.read("m1", tickers=tick, columns=["ts", "ticker", "o", "c"])
    hm = d.ts.dt.hour * 100 + d.ts.dt.minute
    d = d[(hm == 1029) | (hm == 1030)]
    d["date"] = d.ts.dt.normalize().astype("datetime64[ns]")
    p = d[d.ts.dt.minute == 29].pivot(index="date", columns="ticker", values="c").reindex(days)
    e = d[d.ts.dt.minute == 30].pivot(index="date", columns="ticker", values="c").reindex(days)  # one minute later
    o = (P["o"] / F)[tick].reindex(days)
    close = P["rawc"][tick].reindex(days)
    return p / o - 1, close / e - 1, close, e


def fama_macbeth(y, xs, mask):
    """Per-day OLS of y on the columns in xs (dict of wide frames) over mask; mean slope and t."""
    names = list(xs)
    Y = y.where(mask)
    out = []
    for d in Y.index:
        yy = Y.loc[d]
        m = yy.notna()
        for k in names:
            m &= xs[k].loc[d].notna()
        if m.sum() < 30:
            continue
        X = np.column_stack([np.ones(m.sum())] + [xs[k].loc[d][m].values for k in names])
        b = np.linalg.lstsq(X, yy[m].values, rcond=None)[0]
        out.append(pd.Series(b[1:], index=names, name=d))
    return pd.DataFrame(out)


def main():
    P = load_panel()
    PT = with_traded_price(P)
    F = P["c"] / P["rawc"]
    O_S = P["o"] / F
    PC_DIV = P["rawc"].shift(1) * F.shift(1) / F
    W = load_bars()
    cols = sorted(set(stock_cols(P)) & set(W["p1030"].columns))
    days = P["c"].index[(P["c"].index >= "2020-01-02") & (P["c"].index <= W["p1030"].index.max())]
    W = {k: v.reindex(index=days, columns=cols) for k, v in W.items()}
    close = P["rawc"][cols].reindex(days)
    o_s = O_S[cols].reindex(days)
    pc = PC_DIV[cols].reindex(days)
    lo_s, hi_s = (P["l"] / F)[cols].reindex(days), (P["h"] / F)[cols].reindex(days)
    inside = lambda q: (q >= 0.99 * lo_s) & (q <= 1.01 * hi_s)
    gap = (W["e1030"] / W["p1030"] - 1).abs()
    tm = gap.groupby(gap.index.to_period("M")).transform("median")
    ok = inside(W["p1030"]) & inside(W["e1030"]) & (gap < 0.02) & (tm < 0.005) & close.notna() & o_s.notna()
    print("stock-days with bars", int(W["p1030"].notna().sum().sum()), "kept", int(ok.sum().sum()), flush=True)

    adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1).reindex(days)
    px = PT["rawc"][cols].shift(1).reindex(days)
    rk = adv.where(px > 5).rank(axis=1, ascending=False)
    U = (rk <= 1000) & ok
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).set_index("ticker")
    sec = sec[(sec.sector.fillna("").str.strip() != "") & (sec.industry != "Blank Checks")]
    sec = sec.reindex(cols)
    ind = (sec.sector + "|" + sec.industry)
    U &= pd.DataFrame(np.broadcast_to(ind.notna().values, U.shape), index=days, columns=cols)
    print("universe per day", U.sum(1).describe().round(0).to_dict(), flush=True)

    r1 = (W["p1030"] / o_s - 1).where(U)
    r1g = (W["p1030"] / pc - 1).where(U)
    y = (close / W["e1030"] - 1).where(U)
    print("r1 describe", r1.stack().describe().round(4).to_dict(), "y describe", y.stack().describe().round(4).to_dict())
    cin = cont_cost_bps(PT, "10:30")[cols].reindex(days)
    cout = auction_cost_bps(PT)[cols].reindex(days)
    cside = (cin + cout) / 2
    print("avg round-trip cost bp (universe)", float((2 * cside).where(U).stack().mean()), flush=True)

    # leader returns per stock-day (the leader's move broadcast to every member of its industry)
    etf_r1, etf_y, _, _ = etf_first_hour(P, F, days)
    etf_of = pd.Series({t: industry_etf(sec.sector[t], sec.industry[t]) if isinstance(ind[t], str) else None
                        for t in cols})
    F_ = pd.read_pickle(os.path.join(DATA, "local", "fundamentals_panels.pkl"))
    mcap = F_["market_cap"].reindex(index=days, columns=cols).shift(1)
    del F_
    ind_codes = ind.astype("category").cat.codes.values           # -1 for missing

    def leaders_by(size):
        """L (leader first-hour move, equal-weight top 3 by size within industry) and member mask."""
        Ls = {"r1": pd.DataFrame(np.nan, index=days, columns=cols), "r1g": pd.DataFrame(np.nan, index=days, columns=cols)}
        member = pd.DataFrame(False, index=days, columns=cols)
        sz = size.where(U)
        R = {"r1": r1.values, "r1g": r1g.values}
        S = sz.values
        Uv = U.values
        outL = {k: np.full(S.shape, np.nan) for k in R}
        outM = np.zeros(S.shape, bool)
        for code in np.unique(ind_codes[ind_codes >= 0]):
            ci = np.where(ind_codes == code)[0]
            if len(ci) < 6:
                continue
            s = S[:, ci]
            u = Uv[:, ci] & np.isfinite(s)
            s = np.where(u, s, -np.inf)
            order = np.argsort(-s, axis=1)
            rank = np.empty_like(order)
            np.put_along_axis(rank, order, np.arange(len(ci))[None, :].repeat(len(days), 0), axis=1)
            lead = (rank < 3) & u
            nmem = (u & ~lead).sum(1)
            good = (lead.sum(1) == 3) & (nmem >= 3)
            for k in R:
                rr = R[k][:, ci]
                lv = np.nanmean(np.where(lead, rr, np.nan), axis=1)
                lv = np.where(good, lv, np.nan)
                outL[k][:, ci] = lv[:, None]
            outM[:, ci] = u & ~lead & good[:, None]
        for k in R:
            Ls[k] = pd.DataFrame(outL[k], index=days, columns=cols)
        member = pd.DataFrame(outM, index=days, columns=cols)
        return Ls, member

    leaders = {}
    leaders["adv3"] = leaders_by(adv)
    leaders["mcap3"] = leaders_by(mcap.where(mcap > 0).fillna(adv / 1e3))   # mcap first; ADV-scaled fallback
    # etf leaders
    Le = {}
    for k, er in [("r1", etf_r1), ("r1g", None)]:
        if er is None:
            # ETF move from the previous close: previous close dividend-adjusted
            etk = sorted(set(etf_of.dropna()))
            p1029 = (etf_r1[etk] + 1) * (O_S[etk].reindex(days))
            er = p1029 / PC_DIV[etk].reindex(days) - 1
        M = pd.DataFrame(np.nan, index=days, columns=cols)
        for e, g in etf_of.dropna().groupby(etf_of.dropna()):
            M[list(g.index)] = np.broadcast_to(er[e].values[:, None], (len(days), len(g)))
        Le[k] = M
    # members per ETF group: need >= 3 universe members
    grp = etf_of.reindex(cols)
    cnt = U.T.groupby(grp).transform("sum").T
    leaders["etf"] = (Le, U & grp.notna().values[None, :] & (cnt >= 3))
    # placebo: SPY first-hour move for everyone
    spy = {"r1": pd.DataFrame(np.broadcast_to(etf_r1["SPY"].values[:, None], U.shape), index=days, columns=cols)}
    p1029 = (etf_r1["SPY"] + 1) * O_S["SPY"].reindex(days)
    spy["r1g"] = pd.DataFrame(np.broadcast_to((p1029 / PC_DIV["SPY"].reindex(days) - 1).values[:, None], U.shape),
                              index=days, columns=cols)
    leaders["spy_placebo"] = (spy, U.copy())

    own = {"r1": r1, "r1g": r1g}
    spy_y = etf_y["SPY"]
    add_rows("bench", "SPY|buy_hold_close_to_close", P["c"]["SPY"].pct_change().reindex(days))
    add_rows("bench", "SPY|10:30_to_close_always_long", spy_y)
    Wu = U.astype(float).div(U.sum(1).replace(0, np.nan), axis=0).fillna(0)
    ru = bt.run(Wu, y, cside)
    add_rows("bench", "universe_ew_10:30_to_close", ru.net, ru.gross)

    # ETF move into each member's own sector ETF 10:30->close for the hedge diagnostic
    etf_y_m = pd.DataFrame(np.nan, index=days, columns=cols)
    for e, g in etf_of.dropna().groupby(etf_of.dropna()):
        etf_y_m[list(g.index)] = np.broadcast_to(etf_y[e].values[:, None], (len(days), len(g)))

    for lname, (Ls, member) in leaders.items():
        for mv in ["r1", "r1g"]:
            L, ri = Ls[mv], own[mv]
            mem = member & L.notna() & ri.notna() & y.notna()
            # diagnostics
            if lname == "spy_placebo":              # L is the same for every stock: no cross-sectional slope
                fm1 = fm2 = pd.DataFrame()
            else:
                fm1 = fama_macbeth(y, {"L": L}, mem)
                fm2 = fama_macbeth(y, {"L": L, "own": ri}, mem)
            for per, a, b in PER:
                for nm, fm in [("FM y~L", fm1), ("FM y~L+own", fm2)]:
                    x = fm.loc[a:b]
                    if len(x) < 30:
                        continue
                    for k in x.columns:
                        rows.append(dict(part="diag_fm", variant=f"{lname}|{mv}|{nm}|{k}", period=per, n_days=len(x),
                                         slope=x[k].mean(), tstat=x[k].mean() / x[k].std() * np.sqrt(len(x))))
            for k_ in [0.25, 0.5]:
                lag = mem & (np.sign(L) * ri < k_ * L.abs())
                for thr in [0.005, 0.01, 0.02]:
                    up = lag & (L >= thr)
                    dn = lag & (L <= -thr)
                    tag = f"{lname}|{mv}|k{k_}|thr{thr}"
                    # all-laggards spread (gross, equal weight inside each side)
                    mu_up = y.where(up).mean(1)
                    mu_dn = y.where(dn).mean(1)
                    sp = (mu_up.fillna(0) * mu_up.notna() - mu_dn.fillna(0) * mu_dn.notna())
                    act = mu_up.notna() | mu_dn.notna()
                    add_rows("diag_spread", tag + "|all_laggards_up_minus_down", sp.where(act, 0), None,
                             dict(avg_up=float(up.sum(1)[act].mean()), avg_dn=float(dn.sum(1)[act].mean())), act)
                    if lname != "spy_placebo":
                        yh = y - etf_y_m
                        hu, hd = yh.where(up).mean(1), yh.where(dn).mean(1)
                        sph = hu.fillna(0) * hu.notna() - hd.fillna(0) * hd.notna()
                        add_rows("diag_spread", tag + "|all_laggards_up_minus_down_etf_hedged", sph.where(act, 0), None,
                                 None, act)
                    gapv = L - ri
                    Wl = (gapv.where(up).rank(axis=1, ascending=False, method="first") <= 10).astype(float) / 10
                    Ws = (gapv.where(dn).rank(axis=1, ascending=True, method="first") <= 10).astype(float) / 10
                    rl = bt.run(Wl, y, cside)
                    rs = bt.run(Ws, y, cside, side=-1)
                    al, as_ = rl.n > 0, rs.n > 0
                    ex = dict(avg_names_long=float(rl.n[al].mean()) if al.any() else 0,
                              avg_names_short=float(rs.n[as_].mean()) if as_.any() else 0)
                    add_rows("rule", tag + "|long", rl.net, rl.gross, ex, al)
                    add_rows("rule", tag + "|short", rs.net, rs.gross, ex, as_)
                    add_rows("rule", tag + "|ls", 0.5 * (rl.net + rs.net), 0.5 * (rl.gross + rs.gross), ex, al | as_)
        print("done", lname, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "study63_leadlag.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 600)
    pv = lambda x, v: x.pivot_table(index="variant", columns="period", values=v)
    fm = df[df.part == "diag_fm"]
    print(pv(fm, ["slope", "tstat"])[[("slope", "2020-23"), ("tstat", "2020-23"), ("slope", "2024-26"),
                                      ("tstat", "2024-26")]].round(4).to_string())
    for part, vals in [("bench", ["sharpe", "net_bps_active"]), ("diag_spread", ["gross_bps_active", "tstat"]),
                       ("rule", ["gross_bps_active", "net_bps_active", "sharpe"])]:
        x = df[df.part == part].copy()
        if part == "diag_spread":
            x["gross_bps_active"] = x["net_bps_active"]
        t = pv(x, vals)
        c_ = [(v, p) for v in vals for p in ["2020-23", "2024-26"] if (v, p) in t]
        print("\n==", part)
        print(t[c_].round(2).to_string())


if __name__ == "__main__":
    main()
