"""Study 206: hedge the live overnight blend's chip-sector exposure with a short SMH position.

Question: the live book (top 10 of the capped blend, closing-auction buy, opening-auction sell) tracks SMH closely on
bad nights (entries 11/3, 11/4, 11/5 2026: book -2.7 / -5.4 / +6.8% vs SMH overnight -1.4 / -4.4 / +5.7%). Studies 101
(SPY/QQQ beta hedge) and 90/92/172 (tech hardware halving and caps) were rejected; none hedged with SMH. Does shorting
SMH overnight (closing auction to opening auction) raise the Sharpe or cut the drawdown of the book?

Data: blend = (2 rank(ensemble) + rank(P(jump) - P(drop))) / 3 over names scored by both, top 10 with at most 3 per
industry (same rule as paper_overnight.cap_by_industry, data/store/sectors.parquet), equal weight. Ensemble scores:
results/study23_pred.parquet (2024-01-02..2026-09-25, the last saved prediction). P(jump), P(drop): study 33's
results/study33_pred.parquet is not in this checkout (git-ignored, not on Hugging Face), so it is rebuilt here
('rescore'): study 33's quarterly models data/models/jump_{jump,drop}_<q>.txt (Hugging Face models/) scored on study
23's 15:45 frame (15:40 SIP bar close from data/local/m5snap, SPY/IWM 15:44 from data/local/m1, SNAP_MODE none) plus
earn_tonight and news_today (news_features.build(save=False) from data/local/news_scored); written to
data/local/study206_jump_pred.parquet. The earnings store is today's (it now fills days_to_e for Aug-Sep 2026, which
study 158 found missing at the time), so late-2026 scores can differ slightly from study 33's.
Prices: daily panel (core.load_panel, Yahoo): close t (official close) to open t+1. The panel open is the first trade,
about 5-11 bp above the official opening cross for the picks (study 94), so the book's level is a little optimistic;
the hedge leg (SMH, a deep ETF) differs by far less. No official-cross cache is in this checkout (only 5,072 rows of
data/local/auctions.parquet), so the backtest is at panel prices for both legs. Cost per side per leg
core.exec_cost_bps(P, 'auction') + 2.5 = 1.3 + 0.1 x modelled 15:45 half-spread + 2.5 bp (book and SMH).
Tech hardware: study 90/135 definition (TECH_HW industries incl. Semiconductors, plus 'Electrical Products' and
'Industrial Machinery/Components' inside Technology; sectors.parquet is one current snapshot for all dates).
Live: logs/paper/virtual_days.csv (book net at the official auction prints) and logs/paper/virtual.csv names (read
only); SMH overnight from the panel, and from Yahoo (yfinance, in memory only) for nights after the panel ends.

Design (pre-registered in the task, 2026-11-09): hedged net = book net - h x SMH overnight - h x 2 x SMH cost.
  a       h = beta of book net on SMH overnight, OLS over the 60 previous nights (t-60..t-1), at least 40, clipped 0..3
  b       h = share of the night's picks that are tech hardware
  a_half  0.5 x a;   b_half  0.5 x b
Choose the variant with the highest net Sharpe on 2024-01..2025-06 only, freeze it to results/study206_frozen.json,
then score 2025-07..2026-09-25 once. Pass: hedged Sharpe above unhedged with a paired circular block bootstrap
(10-night blocks, 5,000 draws, seed 206) t > 2 on the confirmation period, or a max drawdown cut of at least one third
with Sharpe not lower. Live nights from 2026-09-29 reported with the frozen rule (h from the backtest + live series).

    python src/study206_smh_hedge.py rescore     # rebuild P(jump), P(drop) (about 1 hour, ~1.5 GB)
    python src/study206_smh_hedge.py discover    # choice period, writes the frozen choice
    python src/study206_smh_hedge.py confirm     # confirmation and live nights (run once)

Output: results/study206_smh_hedge.csv (table = check / choice / confirm / confirm_boot / live_nights / live),
        results/study206_frozen.json, data/local/study206_jump_pred.parquet (git-ignored).
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, exec_cost_bps, ann_stats, RES, DATA   # noqa: E402
import bt   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC = os.path.join(DATA, "local")
PAPER = os.path.join(ROOT, "logs", "paper")
JPRED = os.path.join(LOC, "study206_jump_pred.parquet")
FROZEN = os.path.join(RES, "study206_frozen.json")
OUT = os.path.join(RES, "study206_smh_hedge.csv")
CHOICE = ("2024-01", "2025-06")
CONF = ("2025-07", "2026-09")
VARIANTS = ["a", "b", "a_half", "b_half"]
TECH_HW = ["Semiconductors", "Electronic Components", "Computer peripheral equipment", "Computer Manufacturing",
           "Computer Communications Equipment", "Retail: Computer Software & Peripheral Equipment"]
TECH_HW_TECHSECTOR = ["Electrical Products", "Industrial Machinery/Components"]
BETA_N, BETA_MIN, BLOCK, NBOOT, SEED = 60, 40, 10, 5000, 206


# ------------------------------------------------------------------ step 1: rebuild the jump scores
def rescore():
    import lightgbm as lgb
    import alpaca_data as A
    import ml_features as M
    import news_features as NF
    import store
    P, cols, days = M.P, M.cols, M.days
    earn = M.earnings_features()
    parts = []
    for f in sorted(os.listdir(os.path.join(A.LOCAL, "m5snap"))):
        if f.endswith(".parquet") and f[:7] >= "2023-12":
            x = pd.read_parquet(os.path.join(A.LOCAL, "m5snap", f), columns=["ts", "ticker", "c"])
            parts.append(x[(x.ts.dt.hour == 15) & (x.ts.dt.minute == 40)])
    s = pd.concat(parts, ignore_index=True)
    s["date"] = s.ts.dt.normalize()
    p1545 = s.pivot(index="date", columns="ticker", values="c").reindex(columns=cols)
    del s, parts
    m1 = A.read("m1", start="2023-12", tickers=["SPY", "IWM"])
    m1 = m1[m1.ts.dt.strftime("%H:%M") == "15:44"]
    m1["date"] = m1.ts.dt.normalize()
    etf1545 = m1.pivot(index="date", columns="ticker", values="c")
    o, h, l, c, rawc, v, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "v", "dv"])
    adjf = c / rawc
    live_cols = [t for t in cols if P["dv"][t].loc["2023-06":].max() > 3e6]
    test_days = [d for d in days if pd.Timestamp("2024-01-02") <= d <= pd.Timestamp("2026-09-25")
                 and d in p1545.index and d in etf1545.index]
    # study 33 extra inputs
    E = store.read("earnings")
    di = days.searchsorted(pd.to_datetime(E.date))
    EARN = pd.DataFrame(0.0, index=days, columns=cols)
    ci = {t: i for i, t in enumerate(cols)}
    for d, t in zip(di, E.symbol):
        if t in ci and 0 < d < len(days):
            EARN.iat[d, ci[t]] = 1.0
            EARN.iat[d - 1, ci[t]] = 1.0
    NEWS = NF.build(save=False)["n_news"].reindex(index=days, columns=cols)

    def frame_1545(d):   # study23_ensemble_timing.frame_1545, verbatim logic
        i = days.get_loc(d)
        sl = slice(max(0, i - 260), i + 1)
        oo, hh, ll, cc, rr, vv, dd = (x[live_cols].iloc[sl].copy() for x in (o, h, l, c, rawc, v, dv))
        px = p1545.loc[d].reindex(live_cols)
        ok = px.notna()
        cc.iloc[-1] = np.where(ok, px * adjf.loc[d, live_cols], np.nan)
        rr.iloc[-1] = np.where(ok, px, np.nan)
        hh.iloc[-1] = np.maximum(hh.iloc[-1], cc.iloc[-1])
        ll.iloc[-1] = np.minimum(ll.iloc[-1], cc.iloc[-1])
        vv.iloc[-1] = vv.iloc[-1] * 0.85
        dd.iloc[-1] = dd.iloc[-1] * 0.85
        spy = P["c"]["SPY"].iloc[sl].copy()
        spy.iloc[-1] = etf1545.at[d, "SPY"] * (P["c"]["SPY"].loc[d] / P["rawc"]["SPY"].loc[d])
        iwm = P["c"]["IWM"].iloc[sl].copy()
        iwm.iloc[-1] = etf1545.at[d, "IWM"] * (P["c"]["IWM"].loc[d] / P["rawc"]["IWM"].loc[d])
        F, mkt, adv20 = M.build(oo, hh, ll, cc, rr, vv, dd, spy, P["c"]["^VIX"].iloc[sl], P["c"]["^VIX3M"].iloc[sl],
                                iwm, earn=earn)
        last = {k: f.iloc[[-1]] for k, f in F.items()}
        univ = ((rr > 5) & (adv20 > 5e6)).iloc[[-1]] & ok.values
        return M.features_frame(last, mkt.iloc[[-1]], univ)

    models = {}
    rows = []
    for n, d in enumerate(test_days):
        q = str(pd.Period(d, freq="Q"))
        if q not in models:
            models[q] = {k: lgb.Booster(model_file=f"{DATA}/models/jump_{k}_{q}.txt") for k in ["jump", "drop"]}
        X = frame_1545(d)
        r = days.get_indexer(X.index.get_level_values(0))
        cc_ = pd.Index(cols).get_indexer(X.index.get_level_values(1))
        for k, panel in [("earn_tonight", EARN), ("news_today", NEWS)]:
            X[k] = panel.values[r, cc_].astype("float32")
        X = X[models[q]["jump"].feature_name()]
        rows.append(pd.DataFrame({"p_jump": models[q]["jump"].predict(X),
                                  "p_drop": models[q]["drop"].predict(X)}, index=X.index))
        if n % 50 == 0:
            print("scored", d.date(), n, "/", len(test_days), flush=True)
    pd.concat(rows).to_parquet(JPRED)
    print("wrote", JPRED)


# ------------------------------------------------------------------ book and hedge series
def cap_by_industry(sorted_names, ind, n=10, cap=3):
    out, cnt = [], {}
    for t in sorted_names:
        g = ind.get(t)
        g = g if isinstance(g, str) and g else t
        if cnt.get(g, 0) >= cap:
            continue
        cnt[g] = cnt.get(g, 0) + 1
        out.append(t)
        if len(out) == n:
            break
    return out


def sectors():
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    tech = sec.industry.isin(TECH_HW) | (sec.industry.isin(TECH_HW_TECHSECTOR) & (sec.sector == "Technology"))
    return sec, tech


def build():
    P = load_panel()
    cols = stock_cols(P)
    sec, tech = sectors()
    ind = sec.industry.to_dict()
    ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
    pred = pd.read_parquet(JPRED)
    pj = pred.p_jump.unstack().reindex(index=ens.index, columns=cols)
    pdr = pred.p_drop.unstack().reindex(index=ens.index, columns=cols)
    ok = ens.notna() & pj.notna()
    S = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
    del ens, pj, pdr, pred
    W = pd.DataFrame(0.0, index=S.index, columns=cols)
    techshare = pd.Series(np.nan, index=S.index)
    for d, row in S.iterrows():
        picks = cap_by_industry(row.dropna().sort_values(ascending=False, kind="stable").index, ind)
        if picks:
            W.loc[d, picks] = 1.0 / len(picks)
            techshare[d] = tech.reindex(picks).fillna(False).mean()
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1).reindex(index=S.index)
    C = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=S.index)
    b = bt.run(W, R, C)
    b = b[b.n > 0]
    smh_on = P["o"]["SMH"].shift(-1) / P["c"]["SMH"] - 1
    smh_cost = (exec_cost_bps(P, "auction")["SMH"] + 2.5) / 1e4
    df = pd.DataFrame({"book": b.net, "book_gross": b.gross, "n_missing": b.n_missing,
                       "smh": smh_on.reindex(b.index), "smh_cost": smh_cost.reindex(b.index),
                       "tech_share": techshare.reindex(b.index)}).dropna(subset=["book", "smh"])
    return df, P


def rolling_beta(book, smh):
    """beta of book on smh over the BETA_N previous nights (excludes night t)."""
    x, y = smh.shift(1), book.shift(1)
    cov = (x * y).rolling(BETA_N, min_periods=BETA_MIN).mean() - x.rolling(BETA_N, min_periods=BETA_MIN).mean() * \
        y.rolling(BETA_N, min_periods=BETA_MIN).mean()
    var = x.rolling(BETA_N, min_periods=BETA_MIN).var(ddof=0)
    return (cov / var).clip(0, 3)


def hedged(df):
    beta = rolling_beta(df.book, df.smh)
    hs = {"a": beta, "b": df.tech_share, "a_half": 0.5 * beta, "b_half": 0.5 * df.tech_share}
    out = {"unhedged": df.book}
    H = {}
    for k, h in hs.items():
        h = h.fillna(0.0)
        H[k] = h
        out[k] = df.book - h * df.smh - h * 2 * df.smh_cost
    return pd.DataFrame(out), pd.DataFrame(H)


def stats_row(r, base=None):
    st = ann_stats(r)
    row = dict(nights=st["n"], mean_bp=1e4 * r.mean(), sharpe=st["sharpe"], ann_vol=st["ann_vol"], maxdd=st["maxdd"],
               total=st["total"], hit=st["hit"])
    if base is not None:
        d = (r - base).dropna()
        row["diff_bp"] = 1e4 * d.mean()
        row["diff_t"] = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan
    return row


def boot_sharpe_diff(h, u):
    """paired circular block bootstrap of Sharpe(h) - Sharpe(u)."""
    rng = np.random.default_rng(SEED)
    hv, uv = h.values, u.values
    n = len(hv)
    nb = int(np.ceil(n / BLOCK))
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)   # noqa: E731
    obs = sr(hv) - sr(uv)
    ds = np.empty(NBOOT)
    for i in range(NBOOT):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
        ds[i] = sr(hv[idx]) - sr(uv[idx])
    return obs, ds.std(), obs / ds.std(), (ds <= 0).mean()


def checks(df, Hd, Hh):
    rows = []
    for lab, (a, b) in [("choice", CHOICE), ("confirm", CONF), ("2024-26", (CHOICE[0], CONF[1]))]:
        x = df.loc[a:b]
        rows.append(dict(table="check", period=lab, variant="unhedged",
                         **stats_row(x.book), corr_smh=x.book.corr(x.smh), smh_sharpe=ann_stats(x.smh)["sharpe"],
                         smh_mean_bp=1e4 * x.smh.mean(), tech_share=x.tech_share.mean(),
                         beta_mean=Hh.a.loc[a:b].mean(), n_missing=x.n_missing.sum()))
    return rows


def discover():
    df, _ = build()
    Hd, Hh = hedged(df)
    rows = checks(df, Hd, Hh)
    a, b = CHOICE
    u = Hd.unhedged.loc[a:b]
    for k in ["unhedged"] + VARIANTS:
        rows.append(dict(table="choice", period="2024-01..2025-06", variant=k, **stats_row(Hd[k].loc[a:b], u),
                         h_mean=Hh[k].loc[a:b].mean() if k in Hh else 0.0))
    g = pd.DataFrame(rows)
    print(g.round(3).to_string(index=False))
    ch = g[(g.table == "choice") & g.variant.isin(VARIANTS)].sort_values("sharpe", ascending=False).iloc[0]
    fz = dict(variant=ch.variant, choice_sharpe=float(ch.sharpe), choice_unhedged_sharpe=float(
        g[(g.table == "choice") & (g.variant == "unhedged")].sharpe.iloc[0]), choice_maxdd=float(ch.maxdd),
        frozen_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
        rule="highest net Sharpe on 2024-01..2025-06 among a, b, a_half, b_half",
        pass_rule=("confirmation 2025-07..2026-09-25: Sharpe(hedged) > Sharpe(unhedged) with paired block-bootstrap "
                   "t > 2 (10-night blocks), or max drawdown cut by at least 1/3 with Sharpe not lower"),
        beta=dict(nights=BETA_N, min=BETA_MIN, clip=[0, 3]))
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(OUT, index=False)
    print(json.dumps(fz, indent=1))


def smh_live(P, days_needed):
    on = (P["o"]["SMH"].shift(-1) / P["c"]["SMH"] - 1).dropna()
    src = pd.Series("panel", index=on.index)
    miss = [d for d in days_needed if d not in on.index]
    if miss:
        import yfinance as yf
        y = yf.download("SMH", start=str(min(miss).date()), end=str((max(miss) + pd.Timedelta(days=7)).date()),
                        auto_adjust=False, progress=False)
        y.columns = y.columns.droplevel(1) if isinstance(y.columns, pd.MultiIndex) else y.columns
        yo = (y.Open.shift(-1) / y.Close - 1).dropna()
        yo.index = pd.to_datetime(yo.index).tz_localize(None)
        for d in miss:
            if d in yo.index:
                on[d] = yo[d]
                src[d] = "yahoo"
    return on.sort_index(), src


def confirm():
    fz = json.load(open(FROZEN))
    k = fz["variant"]
    df, P = build()
    Hd, Hh = hedged(df)
    rows = pd.read_csv(OUT).to_dict("records")
    rows = [r for r in rows if r["table"] in ("check", "choice")]
    a, b = CONF
    u = Hd.unhedged.loc[a:b]
    for v in ["unhedged"] + VARIANTS:
        rows.append(dict(table="confirm", period="2025-07..2026-09-25", variant=v, frozen=(v == k),
                         **stats_row(Hd[v].loc[a:b], u), h_mean=Hh[v].loc[a:b].mean() if v in Hh else 0.0))
    for v in VARIANTS:
        obs, sd, t, p = boot_sharpe_diff(Hd[v].loc[a:b], u)
        dd_h, dd_u = ann_stats(Hd[v].loc[a:b])["maxdd"], ann_stats(u)["maxdd"]
        rows.append(dict(table="confirm_boot", period="2025-07..2026-09-25", variant=v, frozen=(v == k),
                         sharpe_diff=obs, boot_sd=sd, boot_t=t, boot_p_le0=p, maxdd=dd_h, maxdd_unhedged=dd_u,
                         dd_cut=1 - dd_h / dd_u,
                         passed=bool(t > 2 or (1 - dd_h / dd_u >= 1 / 3 and obs >= 0))))
    # by half-year inside the confirmation period (descriptive)
    for lab, (x0, x1) in [("2025H2", ("2025-07", "2025-12")), ("2026H1", ("2026-01", "2026-06")),
                          ("2026Q3", ("2026-07", "2026-09"))]:
        for v in ["unhedged", k]:
            rows.append(dict(table="confirm_sub", period=lab, variant=v, frozen=(v == k),
                             **stats_row(Hd[v].loc[x0:x1], Hd.unhedged.loc[x0:x1])))
    # live nights
    vd = pd.read_csv(os.path.join(PAPER, "virtual_days.csv"), parse_dates=["day"]).set_index("day")
    vn = pd.read_csv(os.path.join(PAPER, "virtual.csv"), parse_dates=["day"])
    _, tech = sectors()
    smh, src = smh_live(P, list(vd.index))
    smh_cost = (exec_cost_bps(P, "auction")["SMH"] + 2.5) / 1e4
    cost_last = float(smh_cost.dropna().iloc[-1])
    live = pd.DataFrame({"book": vd.net})
    live["smh"] = smh.reindex(live.index)
    live["smh_src"] = src.reindex(live.index)
    live["smh_cost"] = smh_cost.reindex(live.index).fillna(cost_last)
    live["tech_share"] = vn.groupby("day").symbol.apply(lambda s: tech.reindex(s).fillna(False).mean()).reindex(
        live.index)
    live["entry_strategy"] = [json.load(open(os.path.join(PAPER, f"entry_{d.date()}.json"))).get("strategy")
                              if os.path.exists(os.path.join(PAPER, f"entry_{d.date()}.json")) else None
                              for d in live.index]
    hist = pd.concat([df[["book", "smh"]], live[["book", "smh"]]]).sort_index()
    hist = hist[~hist.index.duplicated(keep="last")].dropna()
    beta = rolling_beta(hist.book, hist.smh)
    live["beta"] = beta.reindex(live.index)
    hmap = {"a": live.beta, "b": live.tech_share, "a_half": 0.5 * live.beta, "b_half": 0.5 * live.tech_share}
    lv = live.dropna(subset=["smh"])
    for v, h in hmap.items():
        live[f"h_{v}"] = h
        live[f"net_{v}"] = live.book - h * live.smh - h * 2 * live.smh_cost
    for d, r in live.iterrows():
        rows.append(dict(table="live_nights", period=str(d.date()), variant=k, book=r.book, smh=r.smh,
                         smh_src=r.smh_src, tech_share=r.tech_share, beta=r.beta, h=r[f"h_{k}"],
                         hedged=r[f"net_{k}"], entry_strategy=r.entry_strategy))
    lv = live.dropna(subset=["smh"])
    for v in ["unhedged"] + VARIANTS:
        r = lv.book if v == "unhedged" else lv[f"net_{v}"]
        rows.append(dict(table="live", period=f"{lv.index[0].date()}..{lv.index[-1].date()}", variant=v,
                         frozen=(v == k), **stats_row(r, lv.book), corr_smh=lv.book.corr(lv.smh)))
    g = pd.DataFrame(rows)
    g.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    for t in ["check", "choice", "confirm", "confirm_boot", "confirm_sub", "live"]:
        print(g[g.table == t].dropna(axis=1, how="all").round(4).to_string(index=False))
    print(g[g.table == "live_nights"].dropna(axis=1, how="all").round(4).to_string(index=False))


if __name__ == "__main__":
    {"rescore": rescore, "discover": discover, "confirm": confirm}[sys.argv[1]]()
