"""Analyst price targets and company announcements from the Alpaca/Benzinga headline archive, known at 15:45 ET.

Headlines like "Stifel Maintains Buy on MongoDB, Lowers Price Target to $385" (274k with a target, 2019-06..).
Each article is assigned to the first trading day whose 15:45 ET cutoff is at or after its timestamp (as in
news_features.py); only articles tagged with exactly one symbol are used for targets.
Panels (date x ticker, float32):
  pt_n          price-target headlines in today's window
  pt_up, pt_dn  of which "Raises" / "Lowers" (today's window)
  pt_net20      raises - lowers over the last 20 windows (including today)
  pt_gap        median of each firm's latest target in the last 120 days / previous close - 1 (firm = text before
                the action verb; targets converted to split-adjusted units with the traded/adjusted close factor of
                their day, so a split does not create a gap; targets outside 0.25-4x the previous close are dropped); NaN
                with no target
  pt_firms      number of firms with a target in the last 120 days
  pt_gap_chg20  pt_gap today - pt_gap 20 days ago (target revisions net of price moves)
  ev_buyback    repurchase / buyback headline (today's window); ev_buyback20 = last 20 windows
  ev_guid_up, ev_guid_dn, ev_guid_aff   guidance raised / lowered / affirmed (today), *_20 = last 20 windows
  ev_exec       CEO / CFO / President departure or appointment headline (today); ev_exec20 = last 20 windows
Output: data/local/analyst_panels.pkl
"""
import os
import re
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, traded_close, DATA
import news_features as NF

FN = os.path.join(DATA, "local", "analyst_panels.pkl")
PT = re.compile(r"^(?P<firm>.+?) (?:Maintains|Upgrades|Downgrades|Initiates Coverage On|Initiates|Reiterates|Assumes|"
                r"Resumes|Reinstates|Reaffirms)\b.*?(?P<act>Raises|Lowers|Maintains|Announces|Sets|Cuts|Boosts)? ?"
                r"Price Target to \$(?P<px>[0-9][0-9,]*\.?[0-9]*)")
PAT = {
    "ev_buyback": r"(?i)(repurchase|buyback|buy back)",
    "ev_guid_up": r"(?i)\b(raises|boosts|increases|lifts)\b.*\b(guidance|outlook|forecast)\b",
    "ev_guid_dn": r"(?i)\b(lowers|cuts|reduces|slashes|withdraws)\b.*\b(guidance|outlook|forecast)\b",
    "ev_guid_aff": r"(?i)\b(affirms|reaffirms|reiterates|maintains)\b.*\b(guidance|outlook)\b",
    "ev_exec": r"(?i)\b(CEO|CFO|Chief Executive|Chief Financial|President)\b.*\b(steps? down|resign|depart|leav|"
               r"succeed|appoint|names|named|retire|terminat|ousted|replac)",
}


def build(save=True):
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    d = NF.load_news()
    d = d[d.sym.isin(cols)]
    nsym = d.groupby("id").sym.transform("size")
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    cut = pd.DatetimeIndex(days) + pd.Timedelta(hours=15, minutes=45)
    i = cut.searchsorted(ts.values, side="left")
    d = d.assign(di=i, nsym=nsym.values)[i < len(days)]
    d["day"] = days[d.di.values]
    h = d.headline.fillna("")
    F = {}
    # announcements (articles with 1-3 symbols, like news_features)
    a = d[d.nsym <= 3]
    ha = a.headline.fillna("")
    for k, p in PAT.items():
        m = ha.str.contains(p, regex=True)
        x = a[m].groupby(["day", "sym"]).size().unstack().reindex(index=days, columns=cols).fillna(0).astype("float32")
        F[k] = x
        F[k + "20"] = x.rolling(20, min_periods=1).sum().astype("float32")
    # price targets (single-symbol articles)
    t = d[(d.nsym == 1) & h.str.contains("Price Target to $", regex=False)].copy()
    mt = t.headline.str.extract(PT)
    t = t.assign(firm=mt.firm.str.strip(), act=mt.act.fillna(""), px=pd.to_numeric(mt.px.str.replace(",", ""), errors="coerce"))
    t = t.dropna(subset=["firm", "px"])
    t = t[t.px > 0]
    print("price-target headlines parsed:", len(t))
    g = t.groupby(["day", "sym"])
    F["pt_n"] = g.size().unstack().reindex(index=days, columns=cols).fillna(0).astype("float32")
    up = t.act.isin(["Raises", "Boosts"])
    dn = t.act.isin(["Lowers", "Cuts"])
    F["pt_up"] = t[up].groupby(["day", "sym"]).size().unstack().reindex(index=days, columns=cols).fillna(0).astype("float32")
    F["pt_dn"] = t[dn].groupby(["day", "sym"]).size().unstack().reindex(index=days, columns=cols).fillna(0).astype("float32")
    F["pt_net20"] = (F["pt_up"] - F["pt_dn"]).rolling(20, min_periods=1).sum().astype("float32")
    # latest target per firm and symbol, carried 120 trading days; mean across firms
    # targets are in the share price of their day; convert to today's split-adjusted units with the factor
    # f = traded close / adjusted close of that day (changes only at splits, so carried forward over gaps)
    fac = (traded_close(P)[cols] / P["c"][cols]).ffill()
    t["f"] = fac.stack().reindex(pd.MultiIndex.from_arrays([t.day, t.sym])).values
    t = t.dropna(subset=["f"])
    t["px"] = t.px / t.f
    # drop targets far from the price of their day (stale pre-split targets published on a split day, unit errors)
    cprev = P["c"][cols].shift(1).stack().reindex(pd.MultiIndex.from_arrays([t.day, t.sym])).values
    t = t[(t.px / cprev > 0.25) & (t.px / cprev < 4)]
    last = t.sort_values("ts").groupby(["day", "sym", "firm"]).px.last().reset_index()
    pv = last.pivot_table(index="day", columns=["sym", "firm"], values="px", aggfunc="last")
    pv = pv.reindex(days).ffill(limit=120)
    mean_t = pv.T.groupby(level=0).median().T.reindex(columns=cols)
    nfirm = pv.notna().T.groupby(level=0).sum().T.reindex(columns=cols)
    px = P["c"][cols].shift(1)
    gap = (mean_t / px - 1).where(nfirm > 0).clip(-0.9, 5)
    F["pt_gap"] = gap.astype("float32")
    F["pt_firms"] = nfirm.fillna(0).astype("float32")
    F["pt_gap_chg20"] = (gap - gap.shift(20)).astype("float32")
    if save:
        pd.to_pickle(F, FN)
    return F


def load():
    return pd.read_pickle(FN) if os.path.exists(FN) else build()


if __name__ == "__main__":
    F = build()
    for k, v in F.items():
        print(k, float(np.nanmean(v.values)), int(np.isfinite(v.values).sum()))
