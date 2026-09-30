"""Study 34b: the study-34 exit rule priced at the real extended-hours bid.

Study 34 found that picks already up 2-10% at 09:00 (pre-market) or 20:00 (after hours) give back part of it by the
opening auction, so selling them at T instead of the open could add 2-9 bp a day, depending on the extended-hours
cost, which study 34 assumed (25 or 50 bp). Here: the last SIP NBBO quote before T for every such event (plus 20:00
and 08:00), and the rule re-priced as a limit sell at the bid (fees 2.5 bp). Events with a split between the close
and T (|log(price_T / close)| > 0.5 and a split-sized mismatch) are dropped.
Output: results/study34b_quotes.csv (per event) and printed summary.
"""
import numpy as np
import pandas as pd
import alpaca_data as A
from concurrent.futures import ThreadPoolExecutor
import study34_afterhours as S

P, days = S.P, S.days


def quote_at(t, ts):
    """Last NBBO quote in the 2 minutes before ts (ET, naive): (bid, ask)."""
    a = (ts - pd.Timedelta(minutes=2)).tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    b = ts.tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        j = A.get("quotes", dict(symbols=t, start=a, end=b, limit=10000, feed="sip"))
    except RuntimeError:
        return np.nan, np.nan
    q = (j.get("quotes") or {}).get(t, [])
    q = [x for x in q if x.get("bp", 0) > 0 and x.get("ap", 0) > 0]
    return (q[-1]["bp"], q[-1]["ap"]) if q else (np.nan, np.nan)


if __name__ == "__main__":
    rec = pd.read_parquet(f"{S.RES}/study34_rec.parquet")
    out = []
    for T in ["20:00", "08:00", "09:00", "09:25"]:
        ev = rec[(rec[f"r{T}"] > 0.02) & (rec[f"v{T}"] > 0) & (rec[f"r{T}"] < 0.5)].copy()
        ev["ts"] = [pd.Timestamp(f"{(d if T >= '16:00' else days[days.get_loc(d) + 1]).date()} {T}") for d in ev.day]
        with ThreadPoolExecutor(6) as ex:
            q = list(ex.map(lambda r: quote_at(r[0], r[1]), zip(ev.ticker, ev.ts)))
        ev["bid"], ev["ask"] = [x[0] for x in q], [x[1] for x in q]
        ev["T"] = T
        out.append(ev)
    ev = pd.concat(out)
    ev["half_spread_bp"] = 1e4 * (ev.ask - ev.bid) / (ev.ask + ev.bid)
    ev["bid_vs_last_bp"] = 1e4 * (ev.bid / ev.apply(lambda r: r[f"p{r['T']}"], axis=1) - 1)
    ev["r_bid"] = ev.bid / ev.c0 - 1
    ev.to_parquet(f"{S.RES}/study34b_quotes.parquet")
    base = rec.groupby("day").apply(lambda z: (z.hold - z.cost).mean())
    rows = []
    for T in ["20:00", "08:00", "09:00", "09:25"]:
        for x in [0.02, 0.03, 0.05]:
            e = ev[(ev["T"] == T) & (ev[f"r{T}"] > x) & ev.bid.notna()]
            gain = (e.r_bid - 2.5e-4 - e.cost / 2) - (e.hold - e.cost)          # per event vs holding to the open
            per_day = gain.groupby(e.day).sum() / 10
            d = per_day.reindex(base.index).fillna(0)
            rows.append(dict(T=T, x=x, events=len(e), quoted=int(e.bid.notna().sum()),
                             median_half_spread_bp=e.half_spread_bp.median(), mean_bid_vs_last_bp=e.bid_vs_last_bp.mean(),
                             gain_per_event_bp=1e4 * gain.mean(), t_event=gain.mean() / (gain.std() / np.sqrt(len(gain))),
                             bp_per_day=1e4 * d.mean(), val_bp=1e4 * d.loc["2024-01":"2025-06"].mean(),
                             oos_bp=1e4 * d.loc["2025-07":].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(f"{S.RES}/study34b_quotes.csv", index=False)
    print(df.round(2).to_string())
