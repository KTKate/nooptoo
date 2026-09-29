"""Study 20: statistical arbitrage with auction-only execution (research, no orders).

Tests
  1 resid : residual (stock minus group EW) reversal, closing-auction entry/exit, hold 1/3/5 days
  2 pairs : within-industry most-correlated pairs (prior two years), spread z-score, auction execution
  3 night : residual close-to-close move -> next overnight residual (close auction in, open auction out)
Costs: per side exec_cost_bps(P, "auction") + 2.5 bp. Shorts need price > $10; borrow 0.3 bp/night if
20d median dollar volume > $50M else 2 bp/night (calendar nights). Long-short books are 0.5 long / 0.5 short.
Signals "at close" use the closing price that the auction itself sets (not strictly tradable: MOC orders are due
before 15:50); "1545" variants use the 15:45 price (2024+ only) and are the realistic ones.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, deflated_sharpe, RES, DATA

PERIODS = [("dev", "2020-01-01", "2023-12-31"), ("val", "2024-01-01", "2025-06-30"),
           ("hold", "2025-07-01", "2026-09-25")]
EXTRA = 2.5
N_PICK = 20
EXTRA_TRIALS = 700


# ------------------------------------------------------------------ data
def setup():
    P = load_panel()
    cols = stock_cols(P)
    c = P["c"][cols].astype("float64")
    o = P["o"][cols].astype("float64")
    rawc = P["rawc"][cols].astype("float64")
    tc = traded_close(P)[cols]
    px = tc.where(tc.notna(), rawc)                       # traded close where known (Dec 2023+), else Yahoo raw
    adv = P["dv"][cols].rolling(20, min_periods=10).median()
    px_l, adv_l = px.shift(1), adv.shift(1)               # universe uses info known before day t
    cost = (exec_cost_bps(P, "auction")[cols].astype("float64") + EXTRA) / 1e4
    r = (c / c.shift(1) - 1).clip(-0.9, 2.0)
    on = (o / c.shift(1) - 1).clip(-0.9, 2.0)            # overnight close(t-1) -> open(t)
    dates = c.index
    nights = pd.Series(dates, index=dates).diff().dt.days.shift(-1).fillna(1).values  # nights t -> t+1
    borrow = np.where(adv_l.values > 5e7, 0.3e-4, 2e-4)
    D = dict(P=P, cols=cols, c=c, o=o, px=px, px_l=px_l, adv_l=adv_l, cost=cost.fillna(50e-4), r=r, on=on,
             dates=dates, nights=nights, borrow=borrow)
    # groups
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).set_index("ticker")
    ind = sec["industry"].reindex(cols)
    codes = pd.Series(pd.factorize(ind)[0], index=cols).where(ind.notna(), -1).astype(int).values
    D["L_industry"] = np.tile(codes, (len(dates), 1))
    g = pd.read_parquet(os.path.join(RES, "study14_groups.parquet")).reset_index()
    g.columns = ["year", "kind", "ticker", "group"]
    g = g[g.kind == "comove"]
    Lc = np.full((len(dates), len(cols)), -1, dtype=int)
    yrs = dates.year.values
    for y, gy in g.groupby("year"):
        lab = gy.set_index("ticker")["group"].reindex(cols).fillna(-1).astype(int).values
        Lc[yrs == y] = lab
    D["L_comove"] = Lc
    # 15:45 price, adjusted scale (2024+)
    from study8_exec_retest import price_1545
    p15 = price_1545("none").reindex(index=dates, columns=cols).astype("float64")
    p15 = p15 * (c / P["rawc"][cols].astype("float64"))
    bad = (p15 / c - 1).abs() > 0.5
    D["r15"] = (p15.mask(bad) / c.shift(1) - 1).clip(-0.9, 2.0)
    return D


def universe(D, min_adv):
    return (D["px_l"] > 5) & (D["adv_l"] > min_adv) & D["c"].notna() & D["c"].shift(1).notna()


def group_demean(x, L, U, min_n=5):
    """x - equal-weight mean of x over universe members of the same group (row-wise). NaN if group < min_n."""
    X = x.values if hasattr(x, "values") else x
    out = np.full(X.shape, np.nan)
    Um = U.values & np.isfinite(X) & (L >= 0)
    ng = L.max() + 2
    for t in range(X.shape[0]):
        m = Um[t]
        if m.sum() < 20:
            continue
        lab = L[t, m]
        s = np.bincount(lab, weights=X[t, m], minlength=ng)
        n = np.bincount(lab, minlength=ng)
        mu = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)
        out[t, m] = X[t, m] - mu[lab]
    return out


def pick(S, n, largest=False, min_valid=60):
    """Boolean T x N selection of the n smallest (largest) finite values per row."""
    A = -S if largest else S
    A = np.where(np.isfinite(A), A, np.inf)
    sel = np.zeros(S.shape, bool)
    valid = np.isfinite(S).sum(1)
    idx = np.argpartition(A, n, axis=1)[:, :n]
    rows = np.arange(S.shape[0])[:, None]
    sel[rows, idx] = True
    sel[valid < min_valid] = False
    return sel


def book_stats(ret, turn, dates, name, meta):
    ret = pd.Series(ret, index=dates)
    turn = pd.Series(turn, index=dates)
    rows = []
    for per, a, b in PERIODS:
        rr = ret.loc[a:b].dropna()
        rr = rr[rr.ne(0).cummax()] if len(rr) else rr     # start at first traded day
        s = ann_stats(rr)
        rows.append(dict(variant=name, period=per, **meta, sharpe=s["sharpe"], ann_ret=s["ann_ret"],
                         maxdd=s["maxdd"], ann_vol=s["ann_vol"], days=s["n"],
                         turnover=turn.loc[rr.index].mean() if len(rr) else np.nan))
    return rows


# ------------------------------------------------------------------ test 1
def run_resid(D):
    rows, series = [], {}
    fwd = D["r"].shift(-1).fillna(0).values          # close t -> close t+1
    cost, nights, borrow = D["cost"].values, D["nights"], D["borrow"]
    short_ok = (D["px_l"] > 10).values
    for uname, madv in [("adv5M", 5e6), ("adv50M", 5e7)]:
        U = universe(D, madv)
        for gname in ["comove", "industry"]:
            L = D["L_" + gname]
            res = group_demean(D["r"], L, U)
            res15 = group_demean(D["r15"], L, U)
            vol = pd.DataFrame(res).rolling(60, min_periods=30).std().shift(1).values
            for k in [1, 3, 5]:
                rk = pd.DataFrame(res).rolling(k, min_periods=k).sum().values
                prev = pd.DataFrame(res).shift(1).rolling(k - 1, min_periods=k - 1).sum().values if k > 1 else 0.0
                rk15 = res15 + prev
                for timing, base in [("close", rk), ("1545", rk15)]:
                    for sig in ["raw", "z"]:
                        S = base if sig == "raw" else base / (vol * np.sqrt(k))
                        S = np.where(U.values, S, np.nan)
                        lo = pick(S, N_PICK)
                        Ss = np.where(short_ok, S, np.nan)
                        sh = pick(Ss, N_PICK, largest=True)
                        for side in ["LO", "LS"]:
                            if side == "LO":
                                W = lo / N_PICK
                            else:
                                W = 0.5 * lo / N_PICK - 0.5 * sh / N_PICK
                            for h in [1, 3, 5]:
                                H = W.copy() if h == 1 else sum(np.roll(W, j, axis=0) * (np.arange(len(W)) >= j)[:, None]
                                                                for j in range(h)) / h
                                gross = (H * fwd).sum(1)
                                dH = np.abs(np.diff(np.vstack([np.zeros((1, H.shape[1])), H]), axis=0))
                                tc = (dH * cost).sum(1)
                                bc = (np.clip(-H, 0, None) * borrow).sum(1) * nights
                                net = gross - tc - bc
                                # returns are realised the day after the position is set
                                net = np.concatenate([[0.0], net[:-1]])
                                turn = np.concatenate([[0.0], dH.sum(1)[:-1]])
                                name = f"resid_{gname}_{uname}_k{k}_{timing}_{sig}_{side}_h{h}"
                                meta = dict(test="resid", group=gname, univ=uname, k=k, timing=timing, sig=sig,
                                            side=side, hold=h, gross_sharpe=np.nan)
                                gs = pd.Series(np.concatenate([[0.0], gross[:-1]]), index=D["dates"])
                                st = book_stats(net, turn, D["dates"], name, meta)
                                for rrow in st:
                                    a, b = [p[1:] for p in PERIODS if p[0] == rrow["period"]][0]
                                    g = gs.loc[a:b]
                                    g = g[g.ne(0).cummax()]
                                    rrow["gross_sharpe"] = ann_stats(g)["sharpe"]
                                rows += st
                                series[name] = pd.Series(net, index=D["dates"])
            print("resid", uname, gname, "done", flush=True)
    return pd.DataFrame(rows), series


# ------------------------------------------------------------------ test 3
def run_night(D):
    rows, series = [], {}
    on_next = D["on"].shift(-1)                               # close t -> open t+1
    cost, nights, borrow = D["cost"].values, D["nights"], D["borrow"]
    short_ok = (D["px_l"] > 10).values
    ics = []
    for uname, madv in [("adv5M", 5e6), ("adv50M", 5e7)]:
        U = universe(D, madv)
        for gname in ["comove", "industry"]:
            L = D["L_" + gname]
            res = group_demean(D["r"], L, U)
            res15 = group_demean(D["r15"], L, U)
            onres = group_demean(on_next, L, U)               # next overnight residual (target)
            vol = pd.DataFrame(res).rolling(60, min_periods=30).std().shift(1).values
            on_fill = on_next.fillna(0).values
            for k in [1, 3]:
                rk = pd.DataFrame(res).rolling(k, min_periods=k).sum().values
                prev = pd.DataFrame(res).shift(1).rolling(k - 1, min_periods=k - 1).sum().values if k > 1 else 0.0
                for timing, base in [("close", rk), ("1545", res15 + prev)]:
                    # rank IC of residual move vs next overnight residual
                    a = pd.DataFrame(np.where(U.values, base, np.nan), index=D["dates"])
                    ic = a.rank(axis=1).corrwith(pd.DataFrame(onres, index=D["dates"]).rank(axis=1), axis=1)
                    for per, s0, s1 in PERIODS:
                        x = ic.loc[s0:s1].dropna()
                        ics.append(dict(univ=uname, group=gname, k=k, timing=timing, period=per,
                                        ic_mean=x.mean(), ic_t=x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 5 else np.nan))
                    for sig in ["raw", "z"]:
                        S = base if sig == "raw" else base / (vol * np.sqrt(k))
                        S = np.where(U.values, S, np.nan)
                        lo = pick(S, N_PICK)
                        sh = pick(np.where(short_ok, S, np.nan), N_PICK, largest=True)
                        for side in ["LO", "LS"]:
                            W = lo / N_PICK if side == "LO" else 0.5 * lo / N_PICK - 0.5 * sh / N_PICK
                            gross = (W * on_fill).sum(1)
                            # opening-auction exit cost approximated by the same auction cost as the entry
                            tc = (np.abs(W) * cost).sum(1) * 2
                            bc = (np.clip(-W, 0, None) * borrow).sum(1) * nights
                            net = np.concatenate([[0.0], (gross - tc - bc)[:-1]])
                            turn = np.concatenate([[0.0], (2 * np.abs(W).sum(1))[:-1]])
                            gs = pd.Series(np.concatenate([[0.0], gross[:-1]]), index=D["dates"])
                            name = f"night_{gname}_{uname}_k{k}_{timing}_{sig}_{side}"
                            meta = dict(test="night", group=gname, univ=uname, k=k, timing=timing, sig=sig, side=side,
                                        hold=0, gross_sharpe=np.nan)
                            st = book_stats(net, turn, D["dates"], name, meta)
                            for rrow in st:
                                a0, b0 = [p[1:] for p in PERIODS if p[0] == rrow["period"]][0]
                                g = gs.loc[a0:b0]
                                g = g[g.ne(0).cummax()]
                                rrow["gross_sharpe"] = ann_stats(g)["sharpe"]
                            rows += st
                            series[name] = pd.Series(net, index=D["dates"])
            print("night", uname, gname, "done", flush=True)
    pd.DataFrame(ics).to_csv(os.path.join(RES, "study20_night_ic.csv"), index=False)
    return pd.DataFrame(rows), series


# ------------------------------------------------------------------ test 2
def form_pairs(D, year, n_pairs=200):
    dates = D["dates"]
    a, b = pd.Timestamp(f"{year - 2}-01-01"), pd.Timestamp(f"{year - 1}-12-31")
    win = (dates >= a) & (dates <= b)
    end = np.where(win)[0][-1]
    adv_end = D["adv_l"].iloc[end + 1] if end + 1 < len(dates) else D["adv_l"].iloc[end]
    px_end = D["px"].iloc[end]
    lc = np.log(D["c"].loc[win])
    ok = (adv_end > 5e7) & (px_end > 10) & (lc.notna().mean() > 0.95)
    names = ok[ok].index
    lr = lc[names].diff().iloc[1:]
    ind = pd.Series(D["L_industry"][end], index=D["cols"])[names]
    cand = []
    for g, members in ind.groupby(ind):
        if g < 0 or len(members) < 2:
            continue
        m = members.index
        cm = lr[m].corr(min_periods=100).values
        iu = np.triu_indices(len(m), 1)
        for i, j in zip(*iu):
            if np.isfinite(cm[i, j]):
                cand.append((cm[i, j], m[i], m[j]))
    cand.sort(reverse=True)
    out = []
    for rho, x, y in cand[:n_pairs]:
        la, lb = lc[x], lc[y]
        mk = la.notna() & lb.notna()
        beta = np.polyfit(lb[mk], la[mk], 1)[0]
        sp = la - beta * lb
        sv = (lr[x] - beta * lr[y]).std()
        out.append(dict(year=year, a=x, b=y, rho=rho, beta=beta, sv=sv))
    return out


def run_pairs(D):
    dates = D["dates"]
    lc = np.log(D["c"])
    lc15 = np.log(D["c"].shift(1) * (1 + D["r15"]))
    ci = {t: i for i, t in enumerate(D["cols"])}
    rv = D["r"].values
    cost, nights, borrow = D["cost"].values, D["nights"], D["borrow"]
    pairs = []
    for y in range(2020, 2027):
        pairs += form_pairs(D, y)
    pdf = pd.DataFrame(pairs)
    pdf.to_csv(os.path.join(RES, "study20_pairs_list.csv"), index=False)
    print("pairs formed", len(pdf), flush=True)
    rows, series = [], {}
    for hedge in ["beta", "dollar"]:
        for timing in ["close", "1545"]:
            # z-score series per (year, pair): spread today vs 60d mean/std up to yesterday
            Z = {}
            for p in pairs:
                bt = p["beta"] if hedge == "beta" else 1.0
                sp = lc[p["a"]] - bt * lc[p["b"]]
                mu = sp.rolling(60, min_periods=40).mean().shift(1)
                sd = sp.rolling(60, min_periods=40).std().shift(1)
                cur = sp if timing == "close" else lc15[p["a"]] - bt * lc15[p["b"]]
                Z[(p["year"], p["a"], p["b"])] = ((cur - mu) / sd).values
            H = np.zeros((len(dates), len(D["cols"])))           # holdings set at close t
            open_pos = {}                                        # key -> dict(sign, days, wa, wb)
            yrs = dates.year.values
            for t in range(len(dates)):
                Ht = np.zeros(len(D["cols"]))
                # manage exits
                for key in list(open_pos):
                    ps = open_pos[key]
                    z = Z[key][t]
                    ps["days"] += 1
                    if (np.isfinite(z) and abs(z) < 0.5) or ps["days"] >= 10 or not np.isfinite(D["c"].values[t, ci[key[1]]] * D["c"].values[t, ci[key[2]]]):
                        del open_pos[key]
                # entries
                if len(open_pos) < 10:
                    cands = []
                    for p in pairs:
                        if p["year"] != yrs[t]:
                            continue
                        key = (p["year"], p["a"], p["b"])
                        if key in open_pos or any(k[1] in key[1:] or k[2] in key[1:] for k in open_pos):
                            continue
                        z = Z[key][t]
                        if np.isfinite(z) and abs(z) > 2:
                            cands.append((abs(z), key, p))
                    cands.sort(key=lambda x: -x[0])
                    for az, key, p in cands[:10 - len(open_pos)]:
                        z = Z[key][t]
                        sign = -1 if z > 0 else 1                 # z>0: spread rich -> short A, long B
                        bt = p["beta"] if hedge == "beta" else 1.0
                        # equal risk: gross notional scaled to spread vol, target 0.1 of capital at 1.5% daily
                        size = 0.1 * np.clip(0.015 / max(p["sv"], 1e-4), 0.33, 2.0)
                        wa, wb = size / (1 + abs(bt)), size * abs(bt) / (1 + abs(bt))
                        short_px = D["px"].values[t, ci[p["b"] if sign > 0 else p["a"]]]
                        if not (short_px > 10):
                            continue
                        open_pos[key] = dict(sign=sign, days=0, wa=wa, wb=wb)
                for key, ps in open_pos.items():
                    Ht[ci[key[1]]] += ps["sign"] * ps["wa"]
                    Ht[ci[key[2]]] -= ps["sign"] * ps["wb"]
                H[t] = Ht
            fwd = np.nan_to_num(np.roll(rv, -1, axis=0))
            fwd[-1] = 0
            gross = (H * fwd).sum(1)
            dH = np.abs(np.diff(np.vstack([np.zeros((1, H.shape[1])), H]), axis=0))
            tc = (dH * cost).sum(1)
            bc = (np.clip(-H, 0, None) * borrow).sum(1) * nights
            net = np.concatenate([[0.0], (gross - tc - bc)[:-1]])
            turn = np.concatenate([[0.0], dH.sum(1)[:-1]])
            gs = pd.Series(np.concatenate([[0.0], gross[:-1]]), index=dates)
            name = f"pairs_{hedge}_{timing}"
            meta = dict(test="pairs", group="industry", univ="adv50M", k=0, timing=timing, sig="z", side="LS",
                        hold=10, gross_sharpe=np.nan)
            st = book_stats(net, turn, dates, name, meta)
            for rrow in st:
                a0, b0 = [p[1:] for p in PERIODS if p[0] == rrow["period"]][0]
                g = gs.loc[a0:b0]
                g = g[g.ne(0).cummax()]
                rrow["gross_sharpe"] = ann_stats(g)["sharpe"]
                rrow["avg_gross"] = np.abs(H[(dates >= a0) & (dates <= b0)]).sum(1).mean()
            rows += st
            series[name] = pd.Series(net, index=dates)
            print("pairs", hedge, timing, "done", flush=True)
    return pd.DataFrame(rows), series


def dsr_best(df, series, test):
    """Best variant chosen on dev+val (2020-01..2025-06) Sharpe; DSR on that window, trials = variants + 700."""
    n_var = df.variant.nunique()
    best, bsr = None, -np.inf
    for v in df.variant.unique():
        r = series[v].loc["2020-01-01":"2025-06-30"]
        r = r[r.ne(0).cummax()]
        if len(r) < 120:
            continue
        s = r.mean() / r.std()
        if s > bsr:
            best, bsr, rb = v, s, r
    from scipy.stats import skew, kurtosis
    d = deflated_sharpe(bsr, n_var + EXTRA_TRIALS, len(rb), skew(rb), kurtosis(rb, fisher=False))
    return dict(test=test, variants=n_var, best=best, best_sr_devval=bsr * np.sqrt(252), days=len(rb), dsr=d)


if __name__ == "__main__":
    which = sys.argv[1:] or ["resid", "night", "pairs"]
    D = setup()
    print("setup done", flush=True)
    summ = []
    for w in which:
        df, ser = {"resid": run_resid, "night": run_night, "pairs": run_pairs}[w](D)
        df.to_csv(os.path.join(RES, f"study20_{w}.csv"), index=False)
        summ.append(dsr_best(df, ser, w))
        print(pd.DataFrame(summ).to_string(), flush=True)
    f = os.path.join(RES, "study20_dsr.csv")
    old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    new = pd.DataFrame(summ)
    if len(old):
        new = pd.concat([old[~old.test.isin(new.test)], new])
    new.to_csv(f, index=False)
