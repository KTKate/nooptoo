"""Studies 101, 104 and 107: account-level questions for running the overnight blend (study 33) in a $10,000 account.

Question:
101  Does a beta hedge help: short 0.25 / 0.5 / 0.75 / 1.0 x SPY or QQQ notional overnight (sell in the closing auction,
     buy back in the opening auction) against the long blend, or a hedge sized by the picks' trailing beta? Two
     earlier checks disagreed (study 95: 50% short SPY 2.91 / 2.92 vs 2.80 / 2.91; a quick check on 2026-10-07: 2.70
     vs 2.85 over 2024-26). Why, and what is the answer at official cross prices?
104  What do whole shares do to a $10,000 account: idle cash, names skipped because one share costs more than the
     per-name budget (equity / 10), and the effect on Sharpe and return vs fractional ideal weights? Variants: (a) floor
     shares, skip names that do not fit; (b) skip and take the next-ranked name; (c) fractional shares (Alpaca
     supports fractional shares only for orders with time in force 'day', not for opening / closing auction orders,
     so (c) is an ideal; (c2) prices it with continuous-market costs instead). Also the day short at 2.5% of equity
     per name (whole shares; Alpaca has no fractional short sales).
107  Yearly dollar results of a $10,000 account in 2024, 2025 and 2026 (Jan-Sep) after short-term capital gains tax
     (30% and 40% combined, on the year's net trading gain, losses offset within the year only), the data plan fee
     (Alpaca Algo Trader Plus $99/month vs the free plan) and borrow fees for the day short; break-even account size
     at which the $99/month fee is under 10% of expected gross profit.

Data: daily panel (core.load_panel); blend scores as study 69 lines 22-46 (results/study23_pred.parquet,
results/study33_pred.parquet); official crosses of the picks from study 94 (data/local/auctions94.parquet, seeded from
auctions88.parquet) with study 94's correction method (S94.cross_ratios: night return at crosses = panel return x
ratio_open(t+1) / ratio_close(t), panel price where no usable cross); SPY and QQQ opening ('O') and closing ('6')
crosses from the listing exchange (SPY NYSE Arca 'P', QQQ Nasdaq 'Q') fetched here from Alpaca SIP trades into
data/local/etf_cross101.parquet (Alpaca's daily bar open is not the cross: 0 of 30 sampled days matched; the daily
close matched to within 1-2 bp). ETF costs as study 95: median quoted half-spread of the year (15:44 quote to sell
short, 09:31 quote to buy back) + 1 bp per side, about 2.2-2.5 bp round trip. Stock costs: exec_cost_bps(P, "auction")
+ 2.5 bp per side (100 bp where the cost panel is missing, as bt.run). Industries: data/store/sectors.parquet (live
rule paper_overnight.cap_by_industry, at most 3 names per industry). Easy-to-borrow and fractionable flags: Alpaca's
current asset list (data/local/alpaca_assets_active.parquet; today's list, which flatters the past).

Design:
101  Blend = uncapped top 10 (study 69 base). Hedged net = blend net - h x (ETF night return) - h x ETF round-trip
     cost, h = hedge notional as a fraction of equity, held as extra notional on top of the full long book (overlay);
     a carved-out version (long 1/(1+h), short h/(1+h), gross 1x) is the overlay divided by (1+h), so its Sharpe is
     the same. Reconciliation at h = 0.5 SPY: (a) study 95's formula net + w x (r - cost) with w = -0.5, which credits
     the cost to the short; (b) cost charged; (c) the quick check's 2 x 2 bp x h; (d) no hedge cost; (e) cross prices;
     (f) blend at crosses with SPY at panel prices. Trailing beta: per stock, beta of its night return on SPY's night
     return over the last 60 nights known at t (panel prices), averaged over the picks; hedge = k x beta (k = 0.5, 1),
     clipped to 0..2 x equity; for QQQ divided by the trailing 250-night QQQ / SPY beta.
104  Event simulation per night from 2024-01 to 2026-09: equity E (compounding from $10,000, or fixed $10,000 each
     night), budget E / 10 per name, shares = floor(budget / closing cross price of t) (panel close on the traded scale
     where no cross); P&L = dollars x (cross-corrected night return - 2 x cost). Live book (3-per-industry cap) and
     the uncapped book. Day short: the held names that are easy to borrow, shorted at t+1's opening cross and covered
     at t+1's closing cross, floor(0.025 x E / opening price) shares, cost auction + 2.5 bp per side at t+1.
     (c2) fractional at continuous-market costs: exec_cost_bps 'close' (15:45 spread + 2 bp slippage + fees) to buy,
     'open' (09:35 spread + slippage + fees) to sell.
107  Variant (b) of study 104 on the live book. Two setups: a fresh $10,000 at the start of each calendar year (fee
     and tax settled at year end), and one account from 2024-01 with the fee taken monthly and tax taken from equity
     at each year end. Tax = rate x max(year's trading gain, 0); the data fee is not deducted (investment expenses are
     not deductible without trader tax status; a column shows the deductible case for the fresh-year setup). Borrow:
     Alpaca charges $0 locate and borrow fees on easy-to-borrow shares for Trading API users and charges hard-to-borrow
     fees even on intraday shorts, so 0 is the base case; 0.25% and 0.5% a year on the shorted notional per day held
     (/360) are shown as sensitivity. Break-even: account size X with $1,188 < 10% x X x (mean daily net x 252) on the
     fixed-$10k simulation, also with half and a quarter of the backtest return.

Output: results/study101_hedge.csv (part = hedge / reconcile 0.5x SPY / beta / etf_data; Sharpe, mean net bp, ann,
max drawdown, worst night, nights <= -2%, by pricing and period), results/study104_mechanics.csv (per book, mode,
sizing, day short, period: Sharpe, return, idle cash, names held, skips, most skipped tickers, share of legs at panel
prices), results/study107_after_tax.csv (per strategy, borrow rate, setup, year, plan, tax rate: dollars).
Periods: 2024-01..2025-06, 2025-07..2026-09, 2024-01..2026-09 by entry date.

    python src/study101_107_account.py fetch   # SPY / QQQ official crosses (incremental)
    python src/study101_107_account.py         # study
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A   # noqa: E402

A.RL = A.RateLimiter(int(os.environ.get("ALPACA_PER_MIN", "40")))
ETF_FN = os.path.join(A.LOCAL, "etf_cross101.parquet")
ETF_VEN = {"SPY": "P", "QQQ": "Q"}          # listing exchange SIP codes: SPY NYSE Arca, QQQ Nasdaq
EARLY = {"2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24", "2026-07-02"}


# ------------------------------------------------------------------ fetch: SPY / QQQ official crosses
def _etf_day(day):
    """Opening ('O') and closing ('6') cross of SPY and QQQ on day from the listing exchange (SIP trades)."""
    out = []
    for kind, cond, wins in [("open", "O", [("09:30:00", "09:30:02"), ("09:30:02", "09:31:00"), ("09:31:00", "09:45:00")]),
                             ("close", "6", ([("12:59:59", "13:05:00")] if day in EARLY else [])
                              + [("15:59:59", "16:00:10"), ("16:00:10", "16:15:00")])]:
        res = {t: np.nan for t in ETF_VEN}
        for wa, wb in wins:
            left = [t for t in ETF_VEN if np.isnan(res[t])]
            if not left:
                break
            p = dict(symbols=",".join(left), start=A_utc(day, wa), end=A_utc(day, wb), limit=10000, feed="sip")
            while True:
                j = A.get("trades", p)
                for t, trs in (j.get("trades") or {}).items():
                    if t in res and np.isnan(res[t]):
                        for z in trs:
                            if cond in z.get("c", []) and z.get("x") == ETF_VEN[t]:
                                res[t] = z["p"]
                                break
                if not j.get("next_page_token") or all(not np.isnan(res[t]) for t in left):
                    break
                p["page_token"] = j["next_page_token"]
        out += [dict(ticker=t, date=pd.Timestamp(day), kind=kind, px=v) for t, v in res.items()]
    return out


def A_utc(day, hms):
    return pd.Timestamp(f"{day} {hms}").tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def fetch_etf():
    from core import load_panel
    days = load_panel()["c"].index
    days = days[(days >= "2023-12-01") & (days <= "2026-10-07")]
    have = pd.read_parquet(ETF_FN) if os.path.exists(ETF_FN) else pd.DataFrame(columns=["ticker", "date", "kind", "px"])
    done = set(pd.to_datetime(have.date)) if len(have) else set()
    todo = [d for d in days if d not in done]
    print("days to fetch", len(todo), flush=True)
    rows = []
    for i, d in enumerate(todo):
        try:
            rows += _etf_day(d.strftime("%Y-%m-%d"))
        except RuntimeError as e:
            print("err", d, e, flush=True)
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if len(have) else pd.DataFrame(rows)
            out["date"] = pd.to_datetime(out.date)
            out.to_parquet(ETF_FN, index=False)
            print(i + 1, "/", len(todo), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    fetch_etf()
    sys.exit()


# ------------------------------------------------------------------ study
import study94_cross_rescore as S94   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, traded_close, RES, DATA   # noqa: E402
import bt   # noqa: E402

PER = [("2024-25H1", "2024-01", "2025-06"), ("2025H2-26", "2025-07", "2026-09"), ("2024-26", "2024-01", "2026-09")]
START, END = "2024-01", "2026-09"


def metrics(x):
    x = x.dropna()
    st = ann_stats(x)
    return dict(n=len(x), sharpe=st["sharpe"], mean_bp=1e4 * x.mean(), ann=st["ann_ret"], maxdd=st["maxdd"],
                worst_night=x.min(), nights_le_m2pct=int((x <= -0.02).sum()))


def setup():
    """Panel, blend scores, picks (uncapped, live-capped, study 68 day short), cross-corrected returns."""
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    ens = ens.loc[ens.index < days[-1]]
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    del pred
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr, ok
    W = bt.select_topk(S, S.notna(), 10)
    tc = traded_close(P)[cols]
    Q = S94.cross_ratios(dict(P=P, cols=cols, days=days, tc=tc))
    ro = Q["ro"].reindex(index=days, columns=cols)
    rc = Q["rc"].reindex(index=days, columns=cols)
    o, c = P["o"][cols], P["c"][cols]
    Rn_p = (o.shift(-1) / c - 1).reindex_like(S)
    Rn_x = (1 + Rn_p) * ro.fillna(1.0).shift(-1).reindex_like(S) / rc.fillna(1.0).reindex_like(S) - 1
    miss_n = (ro.shift(-1).isna() | rc.isna()).reindex_like(S)
    Rd_p = (c.shift(-1) / o.shift(-1) - 1).reindex_like(S)                        # day session of t+1
    Rd_x = (1 + Rd_p) * rc.fillna(1.0).shift(-1).reindex_like(S) / ro.fillna(1.0).shift(-1).reindex_like(S) - 1
    C69 = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex_like(S).fillna(100.0)     # missing cost: 100 bp as bt.run
    C68 = (exec_cost_bps(P, "auction")[cols].shift(-1) + 2.5).reindex_like(S).fillna(100.0)
    # traded-scale prices for share counts: closing cross of t (else panel raw close), opening cross of t+1 (else
    # panel open on the traded scale)
    PXC = Q["XC"].reindex(index=days, columns=cols).where(rc.notna()).fillna(Q["PRC"].reindex(index=days, columns=cols))
    PXC = PXC.fillna(tc)
    PXO = Q["XO"].reindex(index=days, columns=cols).where(ro.notna()).fillna(Q["PRO"].reindex(index=days, columns=cols))
    a = pd.read_parquet(f"{DATA}/local/alpaca_assets_active.parquet")
    etb = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)])
    frac = set(a.symbol[a.fractionable.astype(bool)])
    ind = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index(
        "ticker").industry
    return dict(P=P, cols=cols, days=days, S=S, W=W, Rn_p=Rn_p, Rn_x=Rn_x, miss_n=miss_n, Rd_p=Rd_p, Rd_x=Rd_x,
                C69=C69, C68=C68, PXC=PXC, PXO=PXO, etb=etb, frac=frac, ind=ind, ro=ro, rc=rc)


# ------------------------------------------------------------------ 101 beta hedge
def etf_returns(P):
    """Night returns (close t -> open t+1) of SPY / QQQ: panel (Yahoo adjusted; open = first trade) and official
    crosses (ratio method of study 94 on the raw scale; SPY / QQQ had no splits in 2023-26). Costs per study 95."""
    E = ["SPY", "QQQ"]
    days = P["c"].index
    o, c, rawc = (P[k][E].astype("float64") for k in ["o", "c", "rawc"])
    r_p = o.shift(-1) / c - 1
    x = pd.read_parquet(ETF_FN)
    X = {k: g.pivot(index="date", columns="ticker", values="px").reindex(index=days, columns=E) for k, g in x.groupby("kind")}
    pro = o / c * rawc
    ro_e = (X["open"] / pro).where(lambda z: (z > 0.95) & (z < 1.05))
    rc_e = (X["close"] / rawc).where(lambda z: (z > 0.95) & (z < 1.05))
    r_x = (1 + r_p) * ro_e.fillna(1.0).shift(-1) / rc_e.fillna(1.0) - 1
    miss = (ro_e.shift(-1).isna() | rc_e.isna())
    # quick-check version: unadjusted panel open / close (r_spy = next open / close - 1 from the panel) is r_p
    q = pd.read_parquet(f"{DATA}/local/study95_quotes.parquet")
    q["year"] = pd.to_datetime(q.date).dt.year
    hs = q[q.ticker.isin(E)].groupby(["ticker", "hm", "year"]).half_spread_bps.median()
    yrs = pd.Series(days.year, index=days)
    cbuy = pd.DataFrame({t: yrs.map(hs.xs((t, "09:31"))) for t in E}) + 1.0      # buy back at t+1's open
    csell = pd.DataFrame({t: yrs.map(hs.xs((t, "15:44"))) for t in E}) + 1.0     # sell short at t's close
    cost_rt = (csell + cbuy.shift(-1).fillna(cbuy)) / 1e4
    return r_p, r_x, miss, cost_rt, X


def trailing_beta(K, r_spy, win=60):
    """Beta of each stock's night return on SPY's night return over the last win nights known at t (nights up to
    t-1, panel prices). Returns the pick-weighted beta of the uncapped blend per entry day."""
    cols = K["cols"]
    y = (K["P"]["o"][cols].shift(-1) / K["P"]["c"][cols] - 1).loc["2023-06":].astype("float64")
    y = y.shift(1)
    x = r_spy.reindex(y.index).shift(1)
    mx = x.rolling(win, min_periods=40).mean()
    vx = x.rolling(win, min_periods=40).var()
    mxy = y.mul(x, axis=0).rolling(win, min_periods=40).mean()
    my = y.rolling(win, min_periods=40).mean()
    cov = mxy.sub(my.mul(mx, axis=0)) * (win / (win - 1))
    beta = cov.div(vx, axis=0).clip(-3, 5)
    W = K["W"].reindex(beta.index).fillna(0)
    hb = (W * beta.fillna(1.0)).sum(axis=1)
    return hb.reindex(K["S"].index)


def study101(K):
    P = K["P"]
    rows = []
    r_p, r_x, miss, cost_rt, X = etf_returns(P)
    base = {"panel": bt.run(K["W"], K["Rn_p"], K["C69"]).net, "cross": bt.run(K["W"], K["Rn_x"], K["C69"]).net}
    for pr in base:
        base[pr] = base[pr].loc[:END]

    def add(name, pricing, etf, h, net, **kw):
        for lab, a, b in PER:
            rows.append(dict(part="hedge", variant=name, pricing=pricing, etf=etf, h=h, period=lab,
                             **metrics(net.loc[a:b]), **kw))
    for pr in base:
        add("blend alone", pr, "-", 0.0, base[pr])
    # data notes
    for t in ["SPY", "QQQ"]:
        for lab, a, b in PER:
            m = miss[t].loc[a:b].reindex(base["cross"].loc[a:b].index)
            rows.append(dict(part="etf_data", etf=t, period=lab, n=len(m), share_no_cross=float(m.mean()),
                             night_panel_bp=1e4 * r_p[t].loc[a:b].mean(), night_cross_bp=1e4 * r_x[t].loc[a:b].mean(),
                             panel_minus_cross_bp=1e4 * (r_p[t] - r_x[t]).loc[a:b].mean(),
                             cost_rt_bp=1e4 * cost_rt[t].loc[a:b].mean(),
                             corr_blend_cross=base["cross"].loc[a:b].corr(r_x[t].reindex(base["cross"].index).loc[a:b])))
    # (1) reconcile study 95 and the 2026-10-07 quick check (panel prices, 0.5 x SPY)
    idx = base["panel"].index
    rp, rx, cr = (z["SPY"].reindex(idx) for z in (r_p, r_x, cost_rt))
    h = 0.5
    recon = [("a study 95 formula: net + w x (r - cost), w = -0.5 (cost sign flipped: the short earns the cost)",
              "panel", base["panel"] - h * (rp - cr)),
             ("b same, cost charged: net - 0.5 r - 0.5 cost (study 95 quoted cost, ~2.3 bp round trip)", "panel",
              base["panel"] - h * rp - h * cr),
             ("c quick check: net - 0.5 r - 2 x 2 bp x 0.5 (4 bp round trip)", "panel", base["panel"] - h * rp - 2 * 2e-4 * h),
             ("d quick check, no hedge cost", "panel", base["panel"] - h * rp),
             ("e correct cost, cross prices (blend and SPY)", "cross", base["cross"] - h * rx - h * cr),
             ("f correct cost, blend at crosses, SPY at panel open/close", "cross", base["cross"] - h * rp - h * cr),
             ("g carved out of the book (long 1/1.5, short 0.5/1.5), cross prices", "cross",
              (base["cross"] - h * rx - h * cr) / (1 + h))]
    for nm, pr, net in recon:
        for lab, a, b in PER:
            rows.append(dict(part="reconcile 0.5x SPY", variant=nm, pricing=pr, etf="SPY", h=h, period=lab,
                             **metrics(net.loc[a:b])))
    # (2) grid at cross prices (and panel), correct cost, extra notional (overlay)
    for t in ["SPY", "QQQ"]:
        for hh in [0.25, 0.5, 0.75, 1.0]:
            for pr, rr in [("cross", r_x), ("panel", r_p)]:
                net = base[pr] - hh * rr[t].reindex(idx) - hh * cost_rt[t].reindex(idx)
                add(f"short {hh} x {t} (extra notional)", pr, t, hh, net)
    # (3) trailing-beta sized hedge
    hb = trailing_beta(K, r_p["SPY"])
    for lab, a, b in PER:
        x = hb.loc[a:b]
        rows.append(dict(part="beta", variant="pick-weighted trailing 60-night beta to SPY", period=lab,
                         mean_bp=np.nan, beta_mean=x.mean(), beta_p10=x.quantile(0.1), beta_p90=x.quantile(0.9)))
        y, z = base["cross"].loc[a:b], r_x["SPY"].reindex(idx).loc[a:b]
        rows.append(dict(part="beta", variant="realized beta of blend net on SPY night (cross prices, OLS)", period=lab,
                         beta_mean=np.cov(y, z)[0, 1] / z.var()))
    for t in ["SPY", "QQQ"]:
        for k in [0.5, 1.0]:
            hh = (k * hb.reindex(idx)).clip(0, 2).fillna(0)
            if t == "QQQ":      # QQQ hedge sized by SPY beta scaled by the QQQ / SPY beta (trailing 250 nights)
                bq = (r_p["QQQ"].rolling(250).cov(r_p["SPY"]) / r_p["SPY"].rolling(250).var()).shift(1).reindex(idx)
                hh = (hh / bq).clip(0, 2)
            for pr, rr in [("cross", r_x), ("panel", r_p)]:
                net = base[pr] - hh * rr[t].reindex(idx) - hh * cost_rt[t].reindex(idx)
                add(f"short {k} x trailing beta x {t}", pr, t, float(hh.loc[START:END].mean()), net)
    return rows, base


# ------------------------------------------------------------------ 104 / 107 account simulation
def ranked_lists(K, capped=True, depth=40):
    """Per entry day: tickers in rank order (live 3-per-industry cap applied), up to depth names."""
    S, ind = K["S"].loc[START:END], K["ind"]
    out = {}
    for d, row in S.iterrows():
        r = row.dropna().sort_values(ascending=False, kind="stable")
        lst, cnt = [], {}
        for t in r.index:
            if capped:
                g = ind.get(t)
                g = g if isinstance(g, str) and g else t
                if cnt.get(g, 0) >= 3:
                    continue
                cnt[g] = cnt.get(g, 0) + 1
            lst.append(t)
            if len(lst) == depth:
                break
        out[d] = lst
    return out


def simulate(K, lists, mode, short=0.0, E0=10_000.0, compound=True, start=START, end=END, fee_month=0.0,
             borrow=0.0, tax=None, n=10):
    """Live book with a cash account of E0. mode: 'floor' (whole shares, skip names that do not fit), 'replace'
    (whole shares, skip and take the next-ranked name), 'frac' (fractional shares, auction cost), 'frac_cont'
    (fractional shares, continuous-market cost: 15:45 spread + slippage to buy, 09:35 spread + slippage to sell).
    short: day short of the held names that are easy to borrow, short x equity per name (0.025 = quarter size),
    whole shares only (Alpaca has no fractional shorts). borrow: annual fee on shorted notional (/360 per day held).
    fee_month: platform fee taken on the first trading day of each month. tax: rate taken from equity at the last
    trading day of each calendar year on that year's positive trading gain (gross of fees, no carryforward).
    Returns daily frame (entry day t) and per-night skip details."""
    Rn, Rd, C69, C68 = K["Rn_x"], K["Rd_x"], K["C69"], K["C68"]
    PXC, PXO, etb, miss_n = K["PXC"], K["PXO"], K["etb"], K["miss_n"]
    if mode == "frac_cont":
        cb = exec_cost_bps(K["P"], "close")[K["cols"]].reindex_like(K["S"]).fillna(100.0)
        cs = exec_cost_bps(K["P"], "open")[K["cols"]].shift(-1).reindex_like(K["S"]).fillna(100.0)
    E = E0
    out = []
    ys_gain, cur_year, cur_month = 0.0, None, None
    ds = [d for d in lists if pd.Timestamp(start) <= d <= pd.Timestamp(end) + pd.offsets.MonthEnd(0)]
    for i, d in enumerate(ds):
        fee = tx = 0.0
        if cur_month != (d.year, d.month):
            cur_month = (d.year, d.month)
            fee = fee_month
        if cur_year != d.year:
            cur_year, ys_gain = d.year, 0.0
        Eb = E if compound else E0
        B = Eb / n
        lst = lists[d]
        held, skipped, fb = [], [], 0
        for t in lst:
            if len(held) == n:
                break
            if t in held:
                continue
            px = PXC.at[d, t]
            r = Rn.at[d, t]
            if not np.isfinite(px) or not np.isfinite(r):
                continue
            if mode in ("floor", "replace"):
                sh = np.floor(B / px)
                if sh < 1:
                    skipped.append(t)
                    if mode == "floor":
                        held.append(None)      # slot stays in cash
                    continue
                dollars = sh * px
            else:
                dollars = B
            held.append((t, dollars, r))
            fb += bool(miss_n.at[d, t])
        held = [h for h in held if h is not None]
        inv = sum(h[1] for h in held)
        pnl = 0.0
        for t, dl, r in held:
            if mode == "frac_cont":
                cst = (cb.at[d, t] + cs.at[d, t]) / 1e4
            else:
                cst = 2 * C69.at[d, t] / 1e4
            pnl += dl * (r - cst)
        # day short at t+1 (easy-to-borrow names among the held ones)
        spnl, snot, sskip, sn = 0.0, 0.0, 0, 0
        if short > 0:
            Es = (E + pnl) if compound else E0
            for t, _, _ in held:
                if t.replace("-", ".") not in etb:
                    continue
                px = PXO.at[d, t] if t in PXO.columns else np.nan
                if not np.isfinite(px):
                    pxc = PXC.at[d, t]
                    px = pxc * (1 + Rn.at[d, t])
                rd = Rd.at[d, t]
                if not np.isfinite(rd) or not np.isfinite(px):
                    continue
                sh = np.floor(short * Es / px)
                if sh < 1:
                    sskip += 1
                    continue
                dl = sh * px
                sn += 1
                snot += dl
                spnl += dl * (-rd - 2 * C68.at[d, t] / 1e4) - dl * borrow / 360
        gain = pnl + spnl
        ys_gain += gain
        last_of_year = (i == len(ds) - 1) or (ds[i + 1].year != d.year)
        if tax is not None and last_of_year:
            tx = tax * max(ys_gain, 0.0)
        ret = gain / Eb
        E = E + gain - fee - tx if compound else E0
        out.append(dict(date=d, equity_start=Eb, ret=ret, net_ret=(gain - fee - tx) / Eb, pnl_long=pnl, pnl_short=spnl,
                        fee=fee, tax=tx, invested=inv, idle=1 - inv / Eb, names=len(held), skipped=len(skipped),
                        skipped_names=",".join(skipped), fallback=fb, short_names=sn, short_skip=sskip,
                        short_notional=snot, equity_end=E))
    return pd.DataFrame(out).set_index("date")


def study104(K):
    rows = []
    lists = {"capped (live)": ranked_lists(K, True), "uncapped": ranked_lists(K, False)}
    sims = {}
    for book, L in lists.items():
        for mode in ["floor", "replace", "frac", "frac_cont"]:
            for comp in [True, False]:
                for sh in [0.0, 0.025]:
                    if book == "uncapped" and (sh > 0 or mode == "frac_cont"):
                        continue
                    x = simulate(K, L, mode, short=sh, compound=comp)
                    sims[(book, mode, comp, sh)] = x
                    for lab, a, b in PER:
                        y = x.loc[a:b]
                        sk = y.skipped_names.str.split(",").explode()
                        sk = sk[sk != ""].value_counts()
                        rows.append(dict(book=book, mode=mode, sizing="compounding from $10k" if comp else "fixed $10k",
                                         day_short=sh, period=lab, **metrics(y.ret),
                                         final_equity=y.equity_end.iloc[-1] if comp else np.nan,
                                         idle_cash_pct=100 * y.idle.mean(), names_held=y.names.mean(),
                                         nights_with_skip_pct=100 * (y.skipped > 0).mean(), skips_per_night=y.skipped.mean(),
                                         top_skipped=";".join(f"{k}:{v}" for k, v in sk.head(6).items()),
                                         fallback_panel_px_pct=100 * y.fallback.sum() / max(y.names.sum(), 1),
                                         short_names=y.short_names.mean() if sh else np.nan,
                                         short_skipped=y.short_skip.mean() if sh else np.nan,
                                         short_notional_pct=100 * (y.short_notional / y.equity_start).mean() if sh else np.nan))
    # reference: bt.run ideal fractional book (study 94 live row) at cross prices
    Wc = pd.DataFrame(0.0, index=K["S"].index, columns=K["cols"])
    for d, l in lists["capped (live)"].items():
        Wc.loc[d, l[:10]] = 0.1
    ref = bt.run(Wc, K["Rn_x"], K["C69"]).net.loc[START:END]
    for lab, a, b in PER:
        rows.append(dict(book="capped (live)", mode="reference: bt.run equal weight, cross", sizing="weights",
                         day_short=0.0, period=lab, **metrics(ref.loc[a:b])))
    # share of picks Alpaca lists as fractionable (today's list)
    fr = np.mean([t.replace("-", ".") in K["frac"] for l in lists["capped (live)"].values() for t in l[:10]])
    rows.append(dict(book="capped (live)", mode="note: share of picks fractionable on Alpaca (today's list)",
                     period="2024-26", mean_bp=np.nan, frac_share=fr))
    return rows, lists


def study107(K, lists):
    rows = []
    L = lists["capped (live)"]
    years = [("2024", "2024-01", "2024-12"), ("2025", "2025-01", "2025-12"), ("2026 YTD (Jan-Sep)", "2026-01", "2026-09")]
    plans = [("free plan", 0.0), ("Algo Trader Plus $99/mo", 99.0)]
    for strat, sh in [("blend only", 0.0), ("blend + quarter-size day short", 0.025)]:
        for borrow in ([0.0] if sh == 0 else [0.0, 0.0025, 0.005]):
            # (i) a fresh $10k at the start of each year; fee and tax settled at year end
            for y, a, b in years:
                x = simulate(K, L, "replace", short=sh, start=a, end=b, borrow=borrow)
                months = x.index.to_period("M").nunique()
                gross = (x.pnl_long + x.pnl_short).sum()
                for plan, fee_m in plans:
                    fees = fee_m * months
                    for tr in [0.30, 0.40]:
                        tax = tr * max(gross, 0)                      # fees not deductible (base case)
                        net = gross - fees - tax
                        rows.append(dict(strategy=strat, borrow_rate=borrow, setup="fresh $10k each year", year=y,
                                         plan=plan, tax_rate=tr, equity_start=10_000.0, trading_gain=gross,
                                         long_gain=x.pnl_long.sum(), short_gain=x.pnl_short.sum(),
                                         borrow_cost=(x.short_notional * borrow / 360).sum(), platform_fees=fees,
                                         tax=tax, net_after_tax=net, net_return_pct=100 * net / 10_000,
                                         gross_return_pct=100 * gross / 10_000,
                                         net_if_fees_deductible=gross - fees - tr * max(gross - fees, 0)))
            # (ii) one account from 2024-01: fee taken monthly and tax at each year end from the equity
            for plan, fee_m in plans:
                for tr in [0.30, 0.40]:
                    x = simulate(K, L, "replace", short=sh, borrow=borrow, fee_month=fee_m, tax=tr)
                    for y, a, b in years:
                        z = x.loc[a:b]
                        gross = (z.pnl_long + z.pnl_short).sum()
                        e0 = z.equity_start.iloc[0]
                        net = gross - z.fee.sum() - z.tax.sum()
                        rows.append(dict(strategy=strat, borrow_rate=borrow, setup="one account from 2024-01", year=y,
                                         plan=plan, tax_rate=tr, equity_start=e0, equity_end=z.equity_end.iloc[-1],
                                         trading_gain=gross, long_gain=z.pnl_long.sum(), short_gain=z.pnl_short.sum(),
                                         borrow_cost=(z.short_notional * borrow / 360).sum(), platform_fees=z.fee.sum(),
                                         tax=z.tax.sum(), net_after_tax=net, net_return_pct=100 * net / e0,
                                         gross_return_pct=100 * gross / e0))
    # break-even account size: $1,188 a year < 10% of expected gross profit -> size > 11,880 / annual simple return
    fx = simulate(K, L, "replace", compound=False)
    for lab, a, b in PER:
        r = fx.ret.loc[a:b].mean() * 252
        for hc, f in [("backtest", 1.0), ("half of backtest", 0.5), ("quarter of backtest", 0.25)]:
            rows.append(dict(strategy="blend only", setup="break-even ($99/mo < 10% of gross profit)", year=lab,
                             plan=hc, gross_return_pct=100 * r * f, breakeven_account=99 * 12 / 0.10 / (r * f)))
    return pd.DataFrame(rows)




def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    K = setup()
    r101, base = study101(K)
    d101 = pd.DataFrame(r101)
    d101.to_csv(f"{RES}/study101_hedge.csv", index=False)
    print(d101.round(4).to_string())
    r104, lists = study104(K)
    d104 = pd.DataFrame(r104)
    d104.to_csv(f"{RES}/study104_mechanics.csv", index=False)
    print(d104.round(3).to_string())
    d107 = study107(K, lists)
    d107.to_csv(f"{RES}/study107_after_tax.csv", index=False)
    print(d107.round(1).to_string())


if __name__ == "__main__":
    main()
