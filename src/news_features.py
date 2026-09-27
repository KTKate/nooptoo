"""Daily factor panels (date x ticker) from news, short-sale volume and sectors, known at 15:45 ET of day t.

News window for day t: (15:45 ET of the previous trading day, 15:45 ET of day t]. Only articles tagged with
1-3 symbols count (data/local/news_scored, src/score_news.py). Panels:
  n_news        articles in the window
  n_news5       articles in the 5 prior windows (excluding today)
  sent          mean headline sentiment (P(pos) - P(neg)) in the window, NaN without news
  sent5         mean sentiment over the 5 prior windows
  ev_analyst    analyst-ratings articles (upgrades, downgrades, price targets)
  ev_up / ev_down  analyst headlines containing upgrade / downgrade
  ev_fda        headline mentions FDA
  ev_offering   share offering / registered direct / ATM (dilution)
  ev_mna        merger, acquisition, buyout, "to acquire", "to be acquired"
  ev_gov        government / regulation words: tariff, antitrust, FTC, DOJ, SEC, sanction, export control,
                regulator, probe, subpoena, lawsuit, investigation
  ev_earn       earnings category or EPS headline
Short volume (FINRA Reg SHO, day t-1, published after that close):
  short_ratio   short volume / total volume on t-1; short_z = (ratio - 60d mean) / 60d std
Sectors (Nasdaq screener, current classification): sector and industry labels per ticker.
Output: data/local/factor_panels.pkl (dict of DataFrames on the load_panel grid)
"""
import os
import numpy as np
import pandas as pd
from core import load_panel, stock_cols, DATA

LOCAL = os.path.join(DATA, "local")
PAT = {
    "ev_fda": r"\bFDA\b",
    "ev_offering": r"(?i)(public offering|registered direct|at-the-market|prices? .*offering|proposed offering|shelf)",
    "ev_mna": r"(?i)(merger|to acquire|acquisition of|to be acquired|buyout|takeover|definitive agreement)",
    "ev_gov": r"(?i)(tariff|antitrust|\bFTC\b|\bDOJ\b|\bSEC\b|sanction|export control|regulator|probe|subpoena|"
              r"lawsuit|investigation|ban\b|executive order|White House|Congress|Senate)",
    "ev_earn": r"(?i)(\bEPS\b|earnings|quarterly results|\bQ[1-4] 20\d\d\b)",
    "ev_up": r"(?i)upgrade",
    "ev_down": r"(?i)downgrade",
}


def load_news():
    src = os.path.join(LOCAL, "news_scored")
    parts = [pd.read_parquet(os.path.join(src, f)) for f in sorted(os.listdir(src)) if f.endswith(".parquet")]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True).drop_duplicates("id")
    d["sym"] = d.symbols.str.split("|")
    d = d.explode("sym")
    return d


def build(save=True):
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    d = load_news()
    d = d[d.sym.isin(cols)]
    ts = pd.to_datetime(d.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    # assign each article to the first trading day whose 15:45 cutoff is at or after the timestamp
    cut = pd.DatetimeIndex(days) + pd.Timedelta(hours=15, minutes=45)
    i = cut.searchsorted(ts.values, side="left")
    d = d.assign(di=i)[i < len(days)]
    d["day"] = days[d.di.values]
    for k, p in PAT.items():
        d[k] = d.headline.fillna("").str.contains(p, regex=True).astype("float32")
    d["ev_analyst"] = d.category.fillna("").str.contains("analyst-ratings").astype("float32")
    d["ev_earn"] = np.maximum(d.ev_earn, d.category.fillna("").str.contains("earnings").astype("float32"))
    g = d.groupby(["day", "sym"])
    F = {"n_news": g.size().unstack(), "sent": g.sent.mean().unstack()}
    for k in list(PAT) + ["ev_analyst"]:
        F[k] = g[k].sum().unstack()
    for k in F:
        F[k] = F[k].reindex(index=days, columns=cols).astype("float32")
    for k in ["n_news"] + list(PAT) + ["ev_analyst"]:
        F[k] = F[k].fillna(0.0)
    F["n_news5"] = F["n_news"].shift(1).rolling(5, min_periods=1).sum()
    ssum = (F["sent"].fillna(0) * F["n_news"]).shift(1).rolling(5, min_periods=1).sum()
    F["sent5"] = (ssum / F["n_news5"].replace(0, np.nan)).astype("float32")
    first = d.day.min()
    for k in F:                                          # before the archive starts the panels are unknown
        F[k].loc[:first - pd.Timedelta(days=1)] = np.nan
    # short-sale volume (t-1)
    sv_dir = os.path.join(LOCAL, "shortvol")
    if os.path.isdir(sv_dir):
        s = pd.concat([pd.read_parquet(os.path.join(sv_dir, f)) for f in sorted(os.listdir(sv_dir))])
        s = s[s.ticker.isin(cols)]
        sv = s.pivot_table(index="date", columns="ticker", values="short_vol", aggfunc="sum")
        tv = s.pivot_table(index="date", columns="ticker", values="total_vol", aggfunc="sum")
        r = (sv / tv.replace(0, np.nan)).reindex(index=days, columns=cols)
        F["short_ratio"] = r.shift(1).astype("float32")
        m, sd = r.rolling(60, min_periods=20).mean(), r.rolling(60, min_periods=20).std()
        F["short_z"] = ((r - m) / sd).shift(1).clip(-5, 5).astype("float32")
    sec = pd.read_parquet(os.path.join(DATA, "store", "sectors.parquet")).drop_duplicates("ticker").set_index("ticker")
    F["_sector"] = sec.sector.reindex(cols).replace("", np.nan)
    F["_industry"] = sec.industry.reindex(cols).replace("", np.nan)
    if save:
        pd.to_pickle(F, os.path.join(LOCAL, "factor_panels.pkl"))
    return F


def load():
    return pd.read_pickle(os.path.join(LOCAL, "factor_panels.pkl"))


if __name__ == "__main__":
    F = build()
    for k, v in F.items():
        if isinstance(v, pd.DataFrame):
            print(k, float(v.loc["2024":].notna().mean().mean()), float(v.loc["2024":].mean().mean()))
