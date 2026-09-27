"""Study 11: behavior around quarterly earnings announcements, before, at and after, split into overnight and
intraday segments, and trades that fit the executable pattern (15:45 signal, closing-auction buy, opening-auction
or later closing-auction sell).

Events: Nasdaq earnings calendar (data/store/earnings), 2020-01 .. 2026-09, stocks with traded price > $2 and
20d median dollar volume > $1M. Announcement timing: the calendar rarely has it, so it is taken from the first
Benzinga earnings headline for the symbol (data/local/news_scored): before 09:30 on the report date = before the
open (BMO), 16:00 on the report date to 09:30 next day = after the close (AMC). Events without a matching
headline are left out of timing-specific results. Reaction day R = report date (BMO) or the next day (AMC); the
first price reaction is the overnight gap into R.
Part 1, event-time profile: mean market-adjusted overnight and intraday returns for k = -10 .. +10 trading days
around R, by surprise sign, by timing, by size tier.
Part 2, trades (daily portfolios, up to 10 names equal weight, cash when there are no events, auction costs):
  hold_gap        buy close R-1, sell open R (holds through the announcement)
  pre_run         buy close R-6, sell close R-1
  post_up_night   at 15:45 of day R, stocks whose reaction (close R-1 -> 15:45 R) is in the top third and whose
                  earnings headline sentiment is positive: buy close R, sell open R+1
  post_down_night same for the bottom third with negative sentiment (long the loser: reversal)
  post_up_hold5   as post_up_night but sell at the close of R+5 (post-earnings drift)
  post_rev_hold5  bottom third, hold 5 days (reversal)
Output: results/study11_profile.csv, results/study11_trades.csv
"""
import os
import numpy as np
import pandas as pd
import store
from core import load_panel, stock_cols, traded_close, exec_cost_bps, ann_stats, RES, DATA
from study8_exec_retest import price_1545

P = load_panel()
cols = stock_cols(P)
days = P["c"].index
o, c = P["o"][cols], P["c"][cols]
night = o / c.shift(1) - 1                     # overnight into day t
intra = c / o - 1
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
pxT = traded_close(P)[cols].shift(1)
px = pxT.fillna(P["rawc"][cols].shift(1))
elig = (px > 2) & (adv > 1e6)
mkt_n, mkt_i = night.where(elig).mean(1), intra.where(elig).mean(1)
tier = pd.DataFrame(np.select([adv > 5e7, adv > 5e6], ["L", "M"], "S"), index=adv.index, columns=adv.columns)

E = store.read("earnings")
E["date"] = pd.to_datetime(E.date)
E = E[E.symbol.isin(cols) & (E.date >= "2020-01-01") & (E.date <= days[-12])].drop_duplicates(["symbol", "date"])
E["di"] = days.searchsorted(E.date)
E = E[E.di < len(days) - 11]

# timing and headline sentiment from the news archive
src = os.path.join(DATA, "local", "news_scored")
N = pd.concat([pd.read_parquet(os.path.join(src, f)) for f in sorted(os.listdir(src)) if f.endswith(".parquet")])
N = N[N.headline.str.contains(r"(?i)(\bEPS\b|earnings|results|revenue|sales)", regex=True)]
N["sym"] = N.symbols.str.split("|")
N = N.explode("sym")
N = N[N.sym.isin(set(E.symbol))]
N["t"] = pd.to_datetime(N.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
news_start = N.t.min()
N = N.sort_values("t")
# first earnings headline at or after 16:00 of the day before the report date (merge_asof per symbol)
E = E.sort_values("date")
E["wstart"] = E.date - pd.Timedelta(hours=8)
N2 = N[["sym", "t", "sent"]].rename(columns={"sym": "symbol"}).sort_values("t")
M = pd.merge_asof(E.sort_values("wstart"), N2, left_on="wstart", right_on="t", by="symbol", direction="forward")
wend = M.date + pd.Timedelta(days=1, hours=9, minutes=30) + pd.to_timedelta(np.where(M.date.dt.dayofweek == 4, 2, 0), "D")
hit = M.t.notna() & (M.t < wend)
M["timing"] = np.where(~hit, "unknown", np.where(M.t < M.date + pd.Timedelta(hours=9, minutes=30), "BMO",
                       np.where(M.t >= M.date + pd.Timedelta(hours=16), "AMC", "during")))
M["sent"] = M.sent.where(hit)
E = M.drop(columns=["wstart", "t"])
E = E[E.timing.isin(["BMO", "AMC"])].copy()
E["R"] = E.di + (E.timing == "AMC").astype(int)
E = E[E.R < len(days) - 11]
print("events with timing:", len(E), E.timing.value_counts().to_dict(), "news from", news_start.date(), flush=True)

ci = {t: i for i, t in enumerate(cols)}
Nv, Iv = night.values, intra.values
mn, mi = mkt_n.values, mkt_i.values
E["j"] = E.symbol.map(ci)
E["surp_sign"] = np.sign(E.surprise.fillna(0))
E["tier"] = [tier.values[R - 1, j] for R, j in zip(E.R, E.j)]
E = E[[bool(elig.values[R - 1, j]) for R, j in zip(E.R, E.j)]]
prof = []
for k in range(-10, 11):
    i = E.R.values + k
    en = Nv[i, E.j.values] - mn[i]
    ei = Iv[i, E.j.values] - mi[i]
    for grp, m in [("all", np.ones(len(E), bool)), ("surprise_pos", E.surp_sign.values > 0),
                   ("surprise_neg", E.surp_sign.values < 0), ("BMO", (E.timing == "BMO").values),
                   ("AMC", (E.timing == "AMC").values), ("L", (E.tier == "L").values), ("M", (E.tier == "M").values),
                   ("S", (E.tier == "S").values)]:
        prof.append(dict(k=k, group=grp, n=int(m.sum()), night_bps=1e4 * np.nanmean(en[m]),
                         intra_bps=1e4 * np.nanmean(ei[m]), night_abs_bps=1e4 * np.nanmean(np.abs(en[m]))))
prof = pd.DataFrame(prof)
prof.to_csv(f"{RES}/study11_profile.csv", index=False)
print(prof[prof.group == "all"].round(1).to_string(index=False))
print(prof[prof.group.isin(["surprise_pos", "surprise_neg"])].pivot_table(
    index="k", columns="group", values=["night_bps", "intra_bps"]).round(1).to_string())

# ---------------- trades
cost = exec_cost_bps(P, "auction")[cols] + 2.5                 # per side, bps
Cv = cost.values
p1545 = price_1545("none").reindex(index=days, columns=cols)
raw = P["rawc"][cols]
react = (p1545 / raw.shift(1) - 1).values                      # close R-1 -> 15:45 R (raw scale)
cv = c.values


def book(entries, name):
    """entries: list of (entry_day_index, exit_day_index, j, exit_kind 'open'|'close'). Daily portfolio: up to 10
    names, equal weight 1/10 each (cash for unused slots), marked close to close; an open exit realizes the
    overnight return on the exit day."""
    pnl = np.zeros(len(days))
    live = np.zeros(len(days))
    for a, b, j, kind in entries:
        if a + 1 >= len(days) or b >= len(days):
            continue
        w = 0.1
        if kind == "open":                                        # hold close a -> open b (b = a + 1)
            r = o.values[b, j] / cv[a, j] - 1
            pnl[b] += w * (r - 2 * Cv[a, j] / 1e4)
            live[b] += 1
        else:
            for t in range(a + 1, b + 1):
                pnl[t] += w * (cv[t, j] / cv[t - 1, j] - 1)
                live[t] += 1
            pnl[a + 1] -= w * 2 * Cv[a, j] / 1e4
    s = pd.Series(pnl, index=days)
    over = pd.Series(live, index=days)
    s = s.where(over <= 10, s * 10 / over.replace(0, np.nan))  # scale down days with more than 10 positions
    return s.fillna(0.0)


R_ = E.R.values
J = E.j.values
S_ = E.sent.values
rq = react[R_, J]
q_hi = pd.Series(rq).rank(pct=True).values
trades = {
    "hold_gap": [(R - 1, R, j, "open") for R, j in zip(R_, J)],
    "pre_run": [(R - 6, R - 1, j, "close") for R, j in zip(R_, J)],
    "post_up_night": [(R, R + 1, j, "open") for R, j, q, s in zip(R_, J, q_hi, S_) if q > 2 / 3 and s > 0.3],
    "post_down_night": [(R, R + 1, j, "open") for R, j, q, s in zip(R_, J, q_hi, S_) if q < 1 / 3 and s < -0.3],
    "post_up_hold5": [(R, R + 5, j, "close") for R, j, q, s in zip(R_, J, q_hi, S_) if q > 2 / 3 and s > 0.3],
    "post_rev_hold5": [(R, R + 5, j, "close") for R, j, q in zip(R_, J, q_hi) if q < 1 / 3],
    "post_any_night": [(R, R + 1, j, "open") for R, j in zip(R_, J)],
}
out = []
spy = (P["c"]["SPY"] / P["c"]["SPY"].shift(1) - 1)
for name, ent in trades.items():
    s = book([e for e in ent if not np.isnan(react[e[0], e[2]]) or name in ("hold_gap", "pre_run")], name)
    for per, a, b in [("2020-23", "2020-01", "2023-12"), ("val", "2024-01", "2025-06"), ("oos", "2025-07", "2026-09")]:
        x = s.loc[a:b]
        st = ann_stats(x)
        out.append(dict(trade=name, period=per, n_events=len(ent), sharpe=st["sharpe"], ann_ret=st["ann_ret"],
                        maxdd=st["maxdd"], corr_spy=float(x.corr(spy.reindex(x.index)))))
    print(name, "done", flush=True)
df = pd.DataFrame(out)
df.to_csv(f"{RES}/study11_trades.csv", index=False)
print(df.round(3).to_string())
