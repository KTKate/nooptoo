"""Study 1: battery of simple intraday / overnight / intraweek cross-sectional rules
on the full US common-stock universe with daily OHLC bars.

Every rule is long-only top-k (k=10, the size a $10k account can hold) plus a
decile long-short spread (gross) as a measure of raw signal strength.
Liquidity tiers are evaluated separately because cost and survivorship differ.
Output: results/study1_battery.csv
"""
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, exec_cost_bps, split_stats, RES
import bt

P = load_panel()
cols = stock_cols(P)
o, h, l, c, rawc, dv = (P[k][cols] for k in ["o", "h", "l", "c", "rawc", "dv"])
# per-side costs by execution: signals that use the open print must trade after it (continuous market,
# opening spread) and exit in the closing auction; close->open trades enter in the closing auction and
# exit in the first minutes (the Yahoo open is the first trade, not the opening cross); close->close
# trades use the closing auction on both sides. bt.run charges 2 x the per-side frame, so pass the mean.
c_auc = exec_cost_bps(P, "auction")[cols]
c_open = exec_cost_bps(P, "open")[cols]
COST = {"open": (c_open + c_auc) / 2, "close_night": (c_auc + c_open.shift(-1)) / 2, "close_cc": c_auc}

adv = dv.rolling(20, min_periods=10).median().shift(1)   # known before open t
px = rawc.shift(1)
tiers = {
    "L": (px > 5) & (adv > 5e7),
    "M": (px > 5) & (adv > 5e6) & (adv <= 5e7),
    "S": (px > 2) & (adv > 1e6) & (adv <= 5e6),
}

ret1 = c / c.shift(1) - 1
gap = o / c.shift(1) - 1
oc = c / o - 1
co_next = o.shift(-1) / c - 1          # overnight return after close t (indexed by t)
cc_next = c.shift(-1) / c - 1
night = o / c.shift(1) - 1
on_mom20 = np.log1p(night).rolling(20, min_periods=15).mean()   # avg overnight return through open t
id_mom20 = np.log1p(oc).rolling(20, min_periods=15).mean()
ret5 = c / c.shift(5) - 1
vol20 = np.log1p(ret1).rolling(20, min_periods=15).std()
rng = (h / l - 1)

# strategies: (name, signal, holding return, eligibility shift, roundtrip, hold days, cost frame)
# decision at OPEN of t: use info <= c_{t-1} and o_t. holding: o_t -> c_t
# decision at CLOSE of t: info <= c_t. holding: c_t -> o_{t+1} or c_t -> c_{t+1}
S = []
S.append(("open:gap_down_fade", -gap, oc, "open"))  # 4th field: cost key
S.append(("open:gap_up_go", gap, oc, "open"))
S.append(("open:gap_down_fade_volnorm", -gap / vol20.shift(1), oc, "open"))
S.append(("open:prev_loser", -ret1.shift(1), oc, "open"))
S.append(("open:prev_winner", ret1.shift(1), oc, "open"))
S.append(("open:intraday_mom20", id_mom20.shift(1), oc, "open"))
S.append(("close:overnight_mom20", on_mom20, co_next, "close_night"))
S.append(("close:intraday_loser_overnight", -oc, co_next, "close_night"))
S.append(("close:day_loser_overnight", -ret1, co_next, "close_night"))
S.append(("close:day_winner_overnight", ret1, co_next, "close_night"))
S.append(("close:lowvol_overnight", -vol20, co_next, "close_night"))
S.append(("close:rev1_cc", -ret1, cc_next, "close_cc"))
S.append(("close:rev5_cc", -ret5, cc_next, "close_cc"))
S.append(("close:mom5_cc", ret5, cc_next, "close_cc"))
S.append(("close:bigrange_loser_cc", -ret1 * rng, cc_next, "close_cc"))

rows = []
for tname, Emask in tiers.items():
    for name, sig, R, when in S:
        # eligibility uses liquidity known before the open of t (valid for both open and close decisions)
        E = Emask & sig.notna()
        W = bt.select_topk(sig, E, 10)
        cs = COST[when]
        r = bt.run(W, R, cs, roundtrip=True)
        # decile long-short gross (signal quality)
        Wl = bt.select_quantile(sig, E, 0.1, True)
        Ws = bt.select_quantile(sig, E, 0.1, False)
        ls = bt.run(Wl, R, cs)["gross"] - bt.run(Ws, R, cs)["gross"]
        st = split_stats(r["net"])
        stg = split_stats(r["gross"])
        stls = split_stats(ls)
        for per in ["dev", "val", "oos"]:
            rows.append(dict(tier=tname, strat=name, period=per,
                             net_sharpe=st.loc[per, "sharpe"], net_annret=st.loc[per, "ann_ret"],
                             gross_sharpe=stg.loc[per, "sharpe"],
                             ls_gross_sharpe=stls.loc[per, "sharpe"], maxdd=st.loc[per, "maxdd"]))
        # per-trade averages by period
        for per, (a, b) in {"dev": ("2020", "2023"), "val": ("2024", "2025-06"), "oos": ("2025-07", "2026-09")}.items():
            sub = r.loc[a:b]
            m = [x for x in rows if x["tier"] == tname and x["strat"] == name and x["period"] == per][0]
            m["gross_bps_day"] = 1e4 * sub["gross"].mean()
            m["cost_bps_day"] = 1e4 * sub["cost"].mean()
            m["ls_bps_day"] = 1e4 * ls.loc[a:b].mean()
        print(tname, name, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/study1_battery.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(df.pivot_table(index=["tier", "strat"], columns="period",
                     values=["net_sharpe", "gross_bps_day", "cost_bps_day", "ls_gross_sharpe"]).round(2))
