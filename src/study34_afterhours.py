"""Study 34: do the overnight picks' after-hours moves hold until the next open?

Picks: the study-33 blend top 10 (2 * ensemble rank + jump-minus-drop rank, 15:45 features), 2024-01 .. 2026-09.
For each pick, SIP 5-minute bars from 16:00 on the entry day to 09:30 on the next trading day (extended hours;
one request per day, cached in data/local/afterhours.parquet). Price at a checkpoint T = close of the last bar that
ends at or before T. ret_T = price_T / raw close - 1; the hold return is the official close-to-open return.
Question 1: E[open return - ret_T | ret_T bucket], i.e. whether after-hours moves continue or fade by the open.
Question 2: exit rules: sell at T (limit order in the extended session) when ret_T > x, else at the open; charged an
extra after-hours cost of 25 or 50 bp (half-spread plus impact outside the auction) on the names sold at T.
Output: results/study34_afterhours.csv
"""
import os
import numpy as np
import pandas as pd
import alpaca_data as A
from core import load_panel, stock_cols, ann_stats, exec_cost_bps, traded_close, RES, DATA
import bt

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
pred = pd.read_parquet(f"{RES}/study33_pred.parquet")
ens = pd.read_parquet(f"{RES}/study23_pred.parquet")["ensemble"].unstack().reindex(columns=cols)
ens = ens.loc[ens.index < days[-1]]
pj = pred["p_jump"].unstack().reindex(index=ens.index, columns=cols)
pdr = pred["p_drop"].unstack().reindex(index=ens.index, columns=cols)
ok = ens.notna() & pj.notna()
score = (2 * ens.where(ok).rank(axis=1, pct=True) + (pj - pdr).where(ok).rank(axis=1, pct=True)) / 3
W = bt.select_topk(score, score.notna(), 10)
picks = W.stack()
picks = picks[picks > 0].index
FN = f"{DATA}/local/afterhours.parquet"
CK = ["16:30", "17:00", "18:00", "20:00", "08:00", "09:00", "09:25"]


def fetch_day(d, tick):
    nxt = days[days.get_loc(d) + 1]
    a = pd.Timestamp(f"{d.date()} 16:00").tz_localize("America/New_York").tz_convert("UTC")
    b = pd.Timestamp(f"{nxt.date()} 09:30").tz_localize("America/New_York").tz_convert("UTC")
    p = dict(symbols=",".join(tick), timeframe="5Min", start=a.strftime("%Y-%m-%dT%H:%M:%SZ"),
             end=b.strftime("%Y-%m-%dT%H:%M:%SZ"), limit=10000, adjustment="raw", feed="sip", sort="asc")
    rows = []
    while True:
        j = A.get("bars", p)
        for s, bl in (j.get("bars") or {}).items():
            for x in bl:
                rows.append((d, s, x["t"], x["c"], x["v"]))
        if not j.get("next_page_token"):
            break
        p["page_token"] = j["next_page_token"]
    return rows


if __name__ == "__main__":
    if os.path.exists(FN):
        bars = pd.read_parquet(FN)
    else:
        from concurrent.futures import ThreadPoolExecutor
        by_day = pd.Series(picks.get_level_values(1), index=picks.get_level_values(0)).groupby(level=0).apply(list)
        with ThreadPoolExecutor(4) as ex:
            out = list(ex.map(lambda kv: fetch_day(kv[0], kv[1]), by_day.items()))
        bars = pd.DataFrame([r for o in out for r in o], columns=["day", "ticker", "t", "c", "v"])
        bars["ts"] = pd.to_datetime(bars.t, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        bars = bars.drop(columns="t")
        bars.to_parquet(FN)
    # bar start + 5 minutes = bar end; price at T = last bar ending at or before T
    bars["end"] = bars.ts + pd.Timedelta(minutes=5)
    rawc = traded_close(P)[cols]   # the traded close (Yahoo raw closes are adjusted for later splits)
    R = (P["o"][cols].shift(-1) / P["c"][cols] - 1)
    cost = exec_cost_bps(P, "auction")[cols] + 2.5
    rec = pd.DataFrame({"day": picks.get_level_values(0), "ticker": picks.get_level_values(1)})
    rec["c0"] = [rawc.at[d, t] for d, t in zip(rec.day, rec.ticker)]
    rec["hold"] = [R.at[d, t] for d, t in zip(rec.day, rec.ticker)]
    rec["cost"] = [2 * cost.at[d, t] / 1e4 for d, t in zip(rec.day, rec.ticker)]
    g = bars.sort_values("end").groupby(["day", "ticker"])
    for T in CK:
        rows = []
        for (d, t), x in g:
            nxt = days[days.get_loc(d) + 1]
            dd = d if T >= "16:00" else nxt
            cut = pd.Timestamp(f"{dd.date()} {T}")
            y = x[x.end <= cut]
            y = y[y.end > pd.Timestamp(f"{d.date()} 16:00")]
            rows.append((d, t, y.c.iloc[-1] if len(y) else np.nan, y.v.sum()))
        px = pd.DataFrame(rows, columns=["day", "ticker", "p", "v"]).set_index(["day", "ticker"])
        rec = rec.join(px.rename(columns={"p": f"p{T}", "v": f"v{T}"}), on=["day", "ticker"])
        rec[f"r{T}"] = rec[f"p{T}"] / rec.c0 - 1
    rec = rec.dropna(subset=["hold"])
    # a split between the close and the extended-hours print shows as a huge move that is absent from the
    # (split-adjusted) close-to-open return; drop those events
    bad = np.zeros(len(rec), bool)
    for T in CK:
        bad |= (np.abs(np.log(rec[f"p{T}"] / rec.c0)) - np.abs(np.log1p(rec.hold)) > 0.4).fillna(False).values
    print("dropped split-like events:", int(bad.sum()))
    rec = rec[~bad]
    rec.to_parquet(f"{RES}/study34_rec.parquet")
    out = []
    # Q1: continuation vs fade by bucket of the move so far
    for T in CK:
        r = rec[f"r{T}"]
        for lo, hi in [(-1, -0.05), (-0.05, -0.02), (-0.02, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 9)]:
            m = (r > lo) & (r <= hi)
            if m.sum() >= 20:
                out.append(dict(kind="bucket", T=T, lo=lo, hi=hi, n=int(m.sum()), mean_ret_T=r[m].mean(),
                                mean_hold=rec.hold[m].mean(), mean_open_minus_T=(rec.hold - r)[m].mean(),
                                t=((rec.hold - r)[m].mean() / ((rec.hold - r)[m].std() / np.sqrt(m.sum())))))
    # Q2: sell at T when ret_T > x (else hold to the open)
    base = rec.groupby("day").apply(lambda z: (z.hold - z.cost).mean())
    for T in CK:
        for x in [0.02, 0.05, 0.1]:
            for extra in [0.0025, 0.005]:
                sell = (rec[f"r{T}"] > x) & (rec[f"v{T}"] > 0)
                ret = np.where(sell, rec[f"r{T}"] - rec.cost / 2 - extra, rec.hold - rec.cost)
                s = pd.Series(ret, index=rec.index).groupby(rec.day).mean()
                d = s - base
                out.append(dict(kind="rule", T=T, lo=x, extra=extra, n=int(sell.sum()),
                                sharpe=ann_stats(s)["sharpe"], base_sharpe=ann_stats(base)["sharpe"],
                                bp_per_day=1e4 * d.mean(), t=d.mean() / (d.std() / np.sqrt(len(d))),
                                val_bp=1e4 * d.loc["2024-01":"2025-06"].mean(), oos_bp=1e4 * d.loc["2025-07":].mean()))
    df = pd.DataFrame(out)
    df.to_csv(f"{RES}/study34_afterhours.csv", index=False)
    pd.set_option("display.width", 200)
    print("picks", len(rec), "with an after-hours price at 20:00:", int(rec["p20:00"].notna().sum()))
    print(df[df.kind == "bucket"].drop(columns=["kind", "extra", "sharpe", "base_sharpe", "bp_per_day", "val_bp", "oos_bp"], errors="ignore").round(4).to_string())
    print(df[df.kind == "rule"].drop(columns=["kind", "hi", "mean_ret_T", "mean_hold", "mean_open_minus_T"], errors="ignore").round(3).to_string())
