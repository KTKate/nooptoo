"""Study 74: the day session BEFORE an earnings announcement (no overnight hold).

Question (owner): for companies reporting after the close on day t (AMC) or before the open on day t+1 (BMO), is
there a tradable pattern in the day session of t: buy or short at the opening auction (or 09:35 / 10:00) and exit at
the closing auction (or 15:30 / 15:45), i.e. before the report? Split by sector / industry, news sentiment, analyst
revisions, the previous report's outcome, pre-earnings run-up, size, volatility and the market's day.

Events: Nasdaq earnings calendar (store 'earnings'), report dates 2020-01 .. 2026-09-25 that are trading days, stocks
in the daily panel, one row per (symbol, date), dropping symbols with another calendar row within 10 trading days.
Timing: 'time' is 'time-not-supplied' for ~99% of rows. Companies publish their timing in advance, so it was knowable
on the day, but our history lacks it. It is RECOVERED (not predicted) per event:
  gap rule (primary, as asked): g_pre = open d / close d-1 - 1, g_post = open d+1 / close d - 1 (d = calendar date).
      |g_post| >= 2 |g_pre| and |g_post| >= 1%  -> reaction after session d  (AMC on d)  -> trade session t = d
      |g_pre| >= 2 |g_post| and |g_pre| >= 1%   -> reaction before session d (BMO on d) -> trade session t = d-1
      else ambiguous -> excluded. Uses later prices only to recover a fact that was public beforehand.
  headline rule (robustness): first Benzinga EPS-result headline for the symbol (data/local/news_scored, 1 symbol,
      'EPS' + Beats/Misses/Inline/vs Est., not guidance) between 16:00 of the previous trading day and 09:30 of the
      next trading day: before 09:30 of d -> BMO, at/after 16:00 of d -> AMC, during the session -> excluded.
      This selects nothing on reaction size (the gap rule keeps only events with a clear reaction).
  calendar rule: the ~0.8% rows with time-pre-market / time-after-hours.
In every case the traded session t is the session immediately before the reaction overnight; "BMO" events are the
mirror case (report before the open of t+1).
Universe on t (known before the open): traded price of t-1 > $5 (s6162_common.traded_price), 20-day median dollar
volume through t-1 > $20M, open and close of t present, no split dated t or t-1.
Returns: day session r = close auction / opening auction - 1 (adjusted panel; ratio within a day). Benchmarks: equal
weight mean of the same liquid universe on t, and SPY's open-to-close. Excess = r - universe mean.
t-stats: cluster-robust by date (many events share a date).
Features (all known before the open of t, i.e. data through t-1):
  sector / industry (current Nasdaq classification: mild look-ahead), sent5 & n_news5 (5 news windows to 15:45 of
  t-1), sent_prev (window ending 15:45 t-1), analyst pt_net20, pt_gap, pt_gap_chg20, ev_guid_up20 / ev_guid_dn20 at
  t-1, previous report (calendar row 30-120 trading days earlier): EPS surprise sign (eps vs epsForecast) and size
  (surprise %), its two-day reaction close(d_prev-1) -> close(d_prev+1) (timing free), run-up ret5 / ret20 to the
  close of t-1, size (20-day median dollar volume), 20-day volatility, the stock's own opening gap on t and SPY's
  opening gap (known at the open), and SPY's open-to-close on t (contemporaneous: descriptive, not tradable).
  Buckets: terciles with breakpoints fixed on 2020-23 (applied unchanged to 2024-26), plus 'none' buckets.
Intraday entry/exit variants (5-minute SIP bars, split-adjusted; price at hh:mm = open of the bar starting hh:mm):
  2024-26: data/local/m5snap (09:30-10:00 and 15:30-16:00, ~3,100 tickers) + m5snapx (15:30-16:00 only, 1,125 small
  caps) + m5full (ORB study pairs, fill-in). 2020-23: data/local/m30s61 (30-minute bars 09:30, 10:00, 15:30 for the
  500 most traded stocks of each day): only the 10:00 and 15:30 points exist. Opening/closing-auction legs combined
  with bar prices: open -> X = (1 + r) * p_X / last trade before 16:00 - 1; X -> close = last trade / p_X - 1.
  Bars are checked against the daily panel (first-trade-to-last-trade vs open-to-close within 5%).
Costs per side: auction trades core.exec_cost_bps(P,'auction') + 2.5 bp (s6162_common.auction_cost_bps on traded
prices); continuous trades s6162_common.cont_cost_bps(hh:mm) (quoted half-spread at that time + 2.3 bp + 2.5 bp).
Rules: at most 3, chosen mechanically on 2020-23 (largest |cluster t| of the excess return among split cells with at
least 300 events, direction = sign), then run on 2024-26 as books: each day equal weight across that day's
qualifying events, at most 10% per name (cash otherwise), net of costs; shorts only in names on Alpaca's current
easy-to-borrow list (data/local/alpaca_assets_active.parquet: today's list, flatters the past).
Survivorship: the panel holds only tickers alive in 2026 (one-day holds: small effect).
Output: results/study74_earnings_day.csv (sections: timing, coverage, splits, variants, rules, robustness)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "2")
import store
import alpaca_data as A
from core import load_panel, stock_cols, ann_stats, RES, DATA
from s6162_common import with_traded_price, auction_cost_bps, cont_cost_bps
import news_features as NF
import analyst_features as AF

END = pd.Timestamp("2026-09-25")
PER = [("2020-23", "2020-01-01", "2023-12-31"), ("2024-26", "2024-01-01", "2026-09-30")]
OUT = os.path.join(RES, "study74_earnings_day.csv")
ROWS = []


def emit(section, **kw):
    ROWS.append(dict(section=section, **kw))


def cl_t(x, g):
    """Mean and date-clustered t-stat."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(g)[ok]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    m = x.mean()
    e = pd.Series(x - m).groupby(g).sum().values
    G = len(e)
    se = np.sqrt((e ** 2).sum() * G / max(G - 1, 1)) / n
    return m, m / se if se > 0 else np.nan, n


# ------------------------------------------------------------------ panels
P = load_panel()
PT = with_traded_price(P)
cols = stock_cols(P)
alld = P["c"].index
o, c = P["o"][cols], P["c"][cols]
TP = PT["rawc"][cols]
adv = P["dv"][cols].rolling(20, min_periods=10).median().shift(1)
ret1 = c / c.shift(1) - 1
vol20 = np.log(c / c.shift(1)).rolling(20, min_periods=10).std().shift(1)
R = c / o - 1
spl = pd.read_csv(os.path.join(DATA, "local", "events", "splits_yf.csv"), parse_dates=["date"])
spl = spl[spl.ticker.isin(cols)]
splitday = pd.DataFrame(False, index=alld, columns=cols)
for t, d in zip(spl.ticker, spl.date):
    i = alld.searchsorted(d)
    for k in (i, i + 1):
        if k < len(alld):
            splitday.iloc[k, cols.index(t)] = True
univ = (TP.shift(1) > 5) & (adv > 20e6) & o.notna() & c.notna() & c.shift(1).notna() & ~splitday
mkt = R.where(univ & (R.abs() < 0.5)).mean(1)
spy_oc = P["c"]["SPY"] / P["o"]["SPY"] - 1
spy_gap = P["o"]["SPY"] / P["c"]["SPY"].shift(1) - 1
COST_A = auction_cost_bps(PT)[cols]
a = pd.read_parquet(os.path.join(DATA, "local", "alpaca_assets_active.parquet"))
ETB = set(a.symbol[a.easy_to_borrow.astype(bool) & a.shortable.astype(bool)].str.replace(".", "-", regex=False))
ci = {t: i for i, t in enumerate(cols)}

# ------------------------------------------------------------------ events and timing
E = store.read("earnings")
E["date"] = pd.to_datetime(E.date).dt.normalize()
E = E[E.symbol.isin(ci)].drop_duplicates(["symbol", "date"])
Eall = E.copy()                                     # for previous-report features (any date)
E = E[(E.date >= "2020-01-01") & (E.date <= END) & E.date.isin(alld)].copy()
E["di"] = alld.get_indexer(E.date)
E = E.sort_values(["symbol", "di"])
gap_prev = E.groupby("symbol").di.diff()
gap_next = -E.groupby("symbol").di.diff(-1)
E = E[~((gap_prev < 10) | (gap_next < 10))].copy()
E = E[(E.di >= 21) & (E.di < len(alld) - 1)]
E["j"] = E.symbol.map(ci)
Ov, Cv = o.values, c.values
di, j = E.di.values, E.j.values
E["g_pre"] = Ov[di, j] / Cv[di - 1, j] - 1
E["g_post"] = Ov[di + 1, j] / Cv[di, j] - 1
ap, aq = E.g_pre.abs(), E.g_post.abs()
E["t_gap"] = np.select([(aq >= 2 * ap) & (aq >= 0.01), (ap >= 2 * aq) & (ap >= 0.01)], ["AMC", "BMO"], "amb")
E.loc[E.g_pre.isna() | E.g_post.isna(), "t_gap"] = "amb"
E["t_cal"] = E.time.map({"time-after-hours": "AMC", "time-pre-market": "BMO"}).fillna("unk")

# headline timing
N = NF.load_news()
N = N[N.symbols.str.count(r"\|") == 0]
h = N.headline.fillna("")
isres = (h.str.contains(r"\bEPS\b") & h.str.contains(r"(?i)(beats?|miss(es)?|in-?line|in line|vs\.? .{0,20}est|"
                                                     r"estimate|up from|down from)")
         & ~h.str.contains(r"(?i)(\bsees\b|guidance|outlook|raises|lowers|affirms|reaffirms|forecast)"))
N = N[isres & N.sym.isin(set(E.symbol))]
N["t"] = pd.to_datetime(N.ts, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
N = N[["sym", "t"]].rename(columns={"sym": "symbol"}).sort_values("t")
E["wstart"] = alld[E.di - 1] + pd.Timedelta(hours=16)
E["wend"] = alld[E.di + 1] + pd.Timedelta(hours=9, minutes=30)
M = pd.merge_asof(E.sort_values("wstart"), N, left_on="wstart", right_on="t", by="symbol", direction="forward")
hit = M.t.notna() & (M.t < M.wend)
M["t_news"] = np.where(~hit, "unk", np.where(M.t < M.date + pd.Timedelta(hours=9, minutes=30), "BMO",
                       np.where(M.t >= M.date + pd.Timedelta(hours=16), "AMC", "during")))
M.loc[M.date < N.t.min() + pd.Timedelta(days=30), "t_news"] = "unk"
E = M.drop(columns=["wstart", "wend", "t"]).sort_values(["symbol", "di"]).reset_index(drop=True)

# timing agreement table
for per, a0, b0 in PER + [("all", "2020-01-01", "2026-12-31")]:
    x = E[(E.date >= a0) & (E.date <= b0)]
    ct = pd.crosstab(x.t_gap, x.t_news)
    for r_ in ct.index:
        for c_ in ct.columns:
            emit("timing", period=per, split="gap_vs_news", bucket=f"{r_}|{c_}", n=int(ct.loc[r_, c_]))
    ct = pd.crosstab(x.t_gap, x.t_cal)
    for r_ in ct.index:
        for c_ in ct.columns:
            emit("timing", period=per, split="gap_vs_cal", bucket=f"{r_}|{c_}", n=int(ct.loc[r_, c_]))
both = E[E.t_gap.isin(["AMC", "BMO"]) & E.t_news.isin(["AMC", "BMO"])]
print("gap vs headline agreement:", round((both.t_gap == both.t_news).mean(), 3), "n", len(both), flush=True)
bc = E[E.t_gap.isin(["AMC", "BMO"]) & E.t_cal.isin(["AMC", "BMO"])]
print("gap vs calendar agreement:", round((bc.t_gap == bc.t_cal).mean(), 3), "n", len(bc), flush=True)


def session_frame(timing_col):
    """One row per event with a known timing: traded session ti, return, benchmark, features."""
    x = E[E[timing_col].isin(["AMC", "BMO"])].copy()
    x["timing"] = x[timing_col]
    x["ti"] = x.di - (x.timing == "BMO").astype(int)
    ti, jj = x.ti.values, x.j.values
    x["tdate"] = alld[ti]
    x["u"] = univ.values[ti, jj]
    x = x[x.u].copy()
    ti, jj = x.ti.values, x.j.values
    x["r"] = R.values[ti, jj]
    x["mkt"] = mkt.values[ti]
    x["spy"] = spy_oc.values[ti]
    x["ex"] = x.r - x.mkt
    x["exspy"] = x.r - x.spy
    x["cost_a"] = COST_A.values[ti, jj]
    x["gap_t"] = Ov[ti, jj] / Cv[ti - 1, jj] - 1
    x["spy_gap"] = spy_gap.values[ti]
    x["ret5"] = Cv[ti - 1, jj] / Cv[ti - 6, jj] - 1
    x["ret20"] = Cv[ti - 1, jj] / Cv[ti - 21, jj] - 1
    x["adv"] = adv.values[ti, jj]
    x["vol20"] = vol20.values[ti, jj]
    x["etb"] = x.symbol.isin(ETB)
    return x[x.r.notna()].copy()


def add_features(x):
    F = NF.load()
    Aa = AF.load()
    ti, jj = x.ti.values, x.j.values
    fcols = F["sent"].columns
    fj = fcols.get_indexer(x.symbol)

    def take(df, lag):
        v = df.reindex(index=alld, columns=fcols).values
        out = v[ti - lag, np.maximum(fj, 0)]
        return np.where(fj >= 0, out, np.nan)
    x["sent5"] = take(F["sent5"], 0)              # windows t-5..t-1 (to 15:45 of t-1)
    x["n_news5"] = take(F["n_news5"], 0)
    x["sent_prev"] = take(F["sent"], 1)           # window ending 15:45 t-1
    for k in ["pt_net20", "pt_gap", "pt_gap_chg20", "ev_guid_up20", "ev_guid_dn20", "pt_firms"]:
        v = Aa[k].reindex(index=alld, columns=fcols).values
        x[k] = np.where(fj >= 0, v[ti - 1, np.maximum(fj, 0)], np.nan)
    x["sector"] = F["_sector"].reindex(x.symbol).values
    x["industry"] = F["_industry"].reindex(x.symbol).values
    # previous report: latest calendar row for the symbol 30..120 trading days before d
    pe = Eall[Eall.date.isin(alld)].copy()
    pe["pdi"] = alld.get_indexer(pe.date)
    pe = pe[(pe.pdi >= 1) & (pe.pdi < len(alld) - 1)]
    pe["pj"] = pe.symbol.map(ci)
    pe["p_react"] = Cv[pe.pdi + 1, pe.pj] / Cv[pe.pdi - 1, pe.pj] - 1
    pe["p_surp"] = pe.surprise
    pe["p_beat"] = np.sign(pe.eps - pe.epsForecast)
    pe = pe[["symbol", "pdi", "p_react", "p_surp", "p_beat"]].sort_values("pdi")
    x = x.sort_values("di")
    x["key"] = x.di - 30
    m = pd.merge_asof(x, pe, left_on="key", right_on="pdi", by="symbol", direction="backward")
    stale = (m.di - m.pdi) > 120
    for k in ["p_react", "p_surp", "p_beat"]:
        m.loc[stale, k] = np.nan
    return m.drop(columns=["key"])


# ------------------------------------------------------------------ intraday bars for variants
def bar_points(pairs):
    """pairs: DataFrame(symbol, tdate). Returns per pair: first (09:30 bar open), p0935, p1000, p1530, p1545, last."""
    want = pairs.assign(tk=pairs.symbol.str.replace("-", ".", regex=False))
    out = []
    for ds, a0, b0 in [("m30s61", "2020-01", "2023-12"), ("m5snap", "2024-01", "2026-09"),
                       ("m5snapx", "2024-01", "2026-09"), ("m5full", "2024-01", "2026-09")]:
        for m_ in pd.period_range(a0, b0, freq="M").astype(str):
            fn = os.path.join(A.LOCAL, ds, f"{m_}.parquet")
            w = want[want.tdate.dt.strftime("%Y-%m") == m_]
            if not os.path.exists(fn) or w.empty:
                continue
            d = pd.read_parquet(fn, columns=["ts", "ticker", "o", "c"], filters=[("ticker", "in", sorted(set(w.tk)))])
            d["tdate"] = d.ts.dt.normalize()
            d = d.merge(w[["tk", "tdate"]].rename(columns={"tk": "ticker"}), on=["ticker", "tdate"])
            if d.empty:
                continue
            hm = d.ts.dt.strftime("%H:%M")
            g = {}
            for lab, t_ in [("first", "09:30"), ("p0935", "09:35"), ("p1000", "10:00"), ("p1530", "15:30"),
                            ("p1545", "15:45")]:
                g[lab] = d[hm == t_].set_index(["ticker", "tdate"]).o
            g["last"] = d[hm >= "15:30"].sort_values("ts").groupby(["ticker", "tdate"]).c.last()
            z = pd.DataFrame(g)
            z["src"] = ds
            out.append(z)
    z = pd.concat(out)
    # keep the first source per pair that has each field (m5snap before m5snapx / m5full)
    z = z.reset_index().groupby(["ticker", "tdate"]).first()
    z.index = z.index.set_levels(z.index.levels[0].str.replace(".", "-", regex=False), level=0)
    return z


VARIANTS = [  # name, entry, exit
    ("open-close", "open", "close"), ("open-1530", "open", "p1530"), ("open-1545", "open", "p1545"),
    ("0935-close", "p0935", "close"), ("0935-1545", "p0935", "p1545"), ("1000-close", "p1000", "close"),
    ("1000-1530", "p1000", "p1530")]


def variant_returns(x, z):
    """Adds r_<variant> and cost_<variant> (round trip, decimal)."""
    k = pd.MultiIndex.from_arrays([x.symbol, x.tdate])
    b = z.reindex(k)
    b.index = x.index
    ok = (b["first"] > 0) & (b["last"] > 0)
    cons = (np.log(b["last"] / b["first"]) - np.log1p(x.r)).abs() < 0.05      # bars agree with the daily panel
    x["bars_ok"] = ok & cons
    x["bars_src"] = b.src
    px = {"p0935": b.p0935, "p1000": b.p1000, "p1530": b.p1530, "p1545": b.p1545}
    ti, jj = x.ti.values, x.j.values
    cc = {hm: cont_cost_bps(PT, hm)[cols].values[ti, jj] for hm in ["09:35", "10:00", "15:30", "15:45"]}
    for name, en, ex in VARIANTS:
        if en == "open" and ex == "close":
            r = x.r.values
        elif en == "open":
            r = (1 + x.r) * px[ex] / b["last"] - 1
        elif ex == "close":
            r = b["last"] / px[en] - 1
        else:
            r = px[ex] / px[en] - 1
        r = np.where(x.bars_ok | ((en == "open") & (ex == "close")), r, np.nan)
        ce = x.cost_a.values if en == "open" else cc[en[1:3] + ":" + en[3:]]
        cx = x.cost_a.values if ex == "close" else cc[ex[1:3] + ":" + ex[3:]]
        x["r_" + name] = r
        x["c_" + name] = (ce + cx) / 1e4
        # benchmark for the same window: universe mean open-close scaled is not available intraday; use the
        # events' own same-day mean (excess vs other events of the day) only for open-close; report raw and vs SPY
    return x


# ------------------------------------------------------------------ splits
def buckets(x):
    """Returns dict split -> Series of bucket labels (breakpoints from 2020-23)."""
    dev = x[x.tdate <= "2023-12-31"]
    B = {"all": pd.Series("all", index=x.index), "timing": x.timing}

    def terc(col, nanlab="none", extra=None):
        ref = dev[col].dropna()
        if len(ref) < 500:                      # not available in 2020-23 (pt_gap needs the traded close, 2023-12+):
            ref = x.loc[(x.tdate >= "2024-01-01") & (x.tdate <= "2024-12-31"), col].dropna()   # 2024 breakpoints
        q = ref.quantile([1 / 3, 2 / 3]).values
        lab = np.select([x[col] <= q[0], x[col] <= q[1], x[col] > q[1]], ["T1", "T2", "T3"], nanlab)
        return pd.Series(lab, index=x.index)
    B["sector"] = x.sector.fillna("none")
    ind_n = dev.industry.value_counts()
    keep = set(ind_n[ind_n >= 150].index)
    B["industry"] = x.industry.where(x.industry.isin(keep), "other").fillna("none")
    B["sent5"] = terc("sent5")
    B["sent_prev"] = terc("sent_prev")
    B["n_news5"] = pd.Series(np.select([x.n_news5.isna(), x.n_news5 == 0, x.n_news5 <= 3], ["none", "0", "1-3"],
                                       ">3"), index=x.index)
    B["pt_net20"] = pd.Series(np.select([x.pt_net20.isna(), x.pt_net20 < 0, x.pt_net20 == 0], ["none", "<0", "0"],
                                        ">0"), index=x.index)
    B["pt_gap"] = terc("pt_gap")
    B["pt_gap_chg20"] = terc("pt_gap_chg20")
    B["guidance20"] = pd.Series(np.select([x.ev_guid_dn20 > 0, x.ev_guid_up20 > 0], ["down", "up"], "neither"),
                                index=x.index)
    B["prev_beat"] = pd.Series(np.select([x.p_beat > 0, x.p_beat < 0, x.p_beat == 0], ["beat", "miss", "inline"],
                                         "none"), index=x.index)
    B["prev_surp"] = terc("p_surp")
    B["prev_react"] = terc("p_react")
    B["ret5"] = terc("ret5")
    B["ret20"] = terc("ret20")
    B["size_adv"] = terc("adv")
    B["vol20"] = terc("vol20")
    B["gap_t"] = terc("gap_t")
    B["spy_gap"] = terc("spy_gap")
    B["spy_day(not tradable)"] = terc("spy")
    B["weekday"] = x.tdate.dt.day_name().str[:3]
    return B


def split_table(x, B, tag, ycol="ex", only=None):
    cells = []
    for s, lab in B.items():
        if only and s not in only:
            continue
        for bk in sorted(lab.unique()):
            for per, a0, b0 in PER:
                m = (lab == bk) & (x.tdate >= a0) & (x.tdate <= b0)
                y = x[m]
                if len(y) < 30:
                    continue
                mr, tr, n = cl_t(y.r, y.tdate)
                me, te, _ = cl_t(y[ycol], y.tdate)
                ms, ts_, _ = cl_t(y.exspy, y.tdate)
                emit("splits", sample=tag, split=s, bucket=bk, period=per, n=n, days=y.tdate.nunique(),
                     raw_bp=1e4 * mr, t_raw=tr, ex_bp=1e4 * me, t_ex=te, exspy_bp=1e4 * ms, t_exspy=ts_,
                     hit_ex=(y[ycol] > 0).mean(), cost_rt_bp=2e4 * y.cost_a.mean() / 1e4,
                     etb_share=y.etb.mean())
                cells.append(dict(split=s, bucket=bk, period=per, n=n, ex=me, t=te))
    return pd.DataFrame(cells)


# ------------------------------------------------------------------ books
def book(x, mask, side, var="open-close", d0="2020-01-01", d1="2026-09-30", cap=0.10):
    y = x[mask & (x.tdate >= d0) & (x.tdate <= d1)].copy()
    if side < 0:
        y = y[y.etb]
    rc, cc_ = "r_" + var, "c_" + var
    y = y[y[rc].notna()]
    n = y.groupby("tdate").size()
    w = np.minimum(cap, 1.0 / n.reindex(y.tdate).values)
    y["gross"] = side * w * y[rc].values
    y["cost"] = w * y[cc_].values
    y["w"] = w
    g = y.groupby("tdate")
    dd = alld[(alld >= d0) & (alld <= min(pd.Timestamp(d1), END))]
    daily = pd.DataFrame({"gross": g.gross.sum(), "cost": g.cost.sum(), "w": g.w.sum(), "n": g.size()}).reindex(dd)
    daily = daily.fillna(0.0)
    daily["net"] = daily.gross - daily.cost
    # hedged: the same weights in the universe mean (long book minus market, short book plus market)
    hm = y.assign(hm=side * y.w * y.mkt).groupby("tdate").hm.sum().reindex(dd).fillna(0.0)
    daily["net_hedged"] = daily.net - hm
    return daily, y


def book_stats(daily, y, years):
    s = ann_stats(daily.net)
    sh = ann_stats(daily.net_hedged)
    act = daily[daily.n > 0]
    return dict(n_trades=len(y), trades_per_yr=len(y) / years, days_active=len(act), avg_names=act.n.mean(),
                avg_gross_w=act.w.mean(), gross_bp_per_trade=1e4 * (y.gross / y.w).mean(),
                net_bp_per_trade=1e4 * ((y.gross - y.cost) / y.w).mean(), cost_bp_rt=1e4 * (y.cost / y.w).mean(),
                ann_ret=s["ann_ret"], sharpe=s["sharpe"], maxdd=s["maxdd"], t=s["tstat"],
                sharpe_hedged=sh["sharpe"], maxdd_hedged=sh["maxdd"])


def main():
    xg = session_frame("t_gap")
    print("gap-timed events in universe:", len(xg), xg.timing.value_counts().to_dict(), flush=True)
    xg = add_features(xg)
    # data-error screen
    big = xg[xg.r.abs() > 0.3]
    print("events with |open-close| > 30%:", len(big), flush=True)
    print(big[["symbol", "tdate", "r", "g_pre", "g_post"]].head(20).to_string(), flush=True)
    for col in ["r", "ex"]:
        q = xg[col].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(4).to_dict()
        print(col, q, flush=True)
    # official opening print check (2024+ sample)
    au = pd.read_parquet(os.path.join(DATA, "local", "auctions.parquet"))
    au["date"] = pd.to_datetime(au.date)
    chk = xg.merge(au.rename(columns={"ticker": "symbol", "date": "tdate"}), on=["symbol", "tdate"])
    if len(chk):
        rawo = (P["o"] / P["c"] * P["rawc"])[cols]
        yo = rawo.values[chk.ti, chk.j]
        trf = (PT["rawc"] / P["rawc"])[cols].values[chk.ti, chk.j]
        dev_ = (yo * trf / chk.open_off.astype(float) - 1).abs()
        print("Yahoo open vs official open (events in auctions.parquet):", len(chk), "median |diff|",
              round(float(np.nanmedian(dev_)), 5), "share > 1%", round(float(np.nanmean(dev_ > 0.01)), 4), flush=True)
        emit("checks", split="yahoo_vs_official_open", n=len(chk), raw_bp=1e4 * float(np.nanmedian(dev_)),
             bucket="median_abs_diff")
    emit("checks", split="abs_r_gt_30pct", n=len(big))

    # intraday bars
    z = bar_points(xg[["symbol", "tdate"]].drop_duplicates())
    xg = variant_returns(xg, z)
    for per, a0, b0 in PER:
        y = xg[(xg.tdate >= a0) & (xg.tdate <= b0)]
        for name, en, ex in VARIANTS:
            rr = y["r_" + name]
            ok = rr.notna()
            emit("coverage", period=per, split=name, n=int(ok.sum()), share=ok.mean(),
                 tickers=y[ok].symbol.nunique(), bucket=",".join(sorted(y[ok].bars_src.dropna().unique())))
            print(per, name, "coverage", int(ok.sum()), f"{ok.mean():.2f}", "tickers", y[ok].symbol.nunique(), flush=True)

    # ---------------- splits (primary: gap timing, open-close)
    B = buckets(xg)
    cells = split_table(xg, B, "gap")
    # intraday variants: all events and timing only (raw, vs SPY, net long)
    for per, a0, b0 in PER:
        for tm in ["all", "AMC", "BMO"]:
            y = xg[(xg.tdate >= a0) & (xg.tdate <= b0) & ((xg.timing == tm) | (tm == "all"))]
            for name, en, ex in VARIANTS:
                rr = y["r_" + name]
                ok = rr.notna()
                if ok.sum() < 30:
                    continue
                mr, tr, n = cl_t(rr[ok], y.tdate[ok])
                me, te, _ = cl_t((rr - y.spy)[ok], y.tdate[ok])
                # same-sample open-close for comparison
                mo, to_, _ = cl_t(y.r[ok], y.tdate[ok])
                emit("variants", sample="gap", split=name, bucket=tm, period=per, n=n, raw_bp=1e4 * mr, t_raw=tr,
                     exspy_bp=1e4 * me, t_exspy=te, oc_same_sample_bp=1e4 * mo,
                     cost_rt_bp=1e4 * y["c_" + name][ok].mean(), net_long_bp=1e4 * (mr - y["c_" + name][ok].mean()),
                     net_short_bp=1e4 * (-mr - y["c_" + name][ok].mean()))

    # ---------------- robustness: headline timing and calendar timing (main splits)
    xn = add_features(session_frame("t_news"))
    Bn = buckets(xn)
    split_table(xn, Bn, "news")
    xc = session_frame("t_cal")
    xc = add_features(xc)
    if len(xc) > 100:
        split_table(xc, buckets(xc), "calendar", only=["all", "timing"])
    agree = E[(E.t_gap == E.t_news) & E.t_gap.isin(["AMC", "BMO"])]
    xa = add_features(session_frame("t_gap"))
    xa = xa[xa.set_index(["symbol", "di"]).index.isin(agree.set_index(["symbol", "di"]).index)]
    split_table(xa, buckets(xa), "gap&news", only=["all", "timing", "sector", "prev_beat", "sent5", "ret5"])
    # ambiguous events (excluded) for reference: day session before the calendar date d, and on d
    for tag, sh in [("amb_session_d", 0)]:
        x = E[E.t_gap == "amb"].copy()
        x = x[univ.values[x.di, x.j]]
        x["r"] = R.values[x.di, x.j]
        x["ex"] = x.r - mkt.values[x.di]
        x["tdate"] = alld[x.di]
        for per, a0, b0 in PER:
            y = x[(x.tdate >= a0) & (x.tdate <= b0)]
            me, te, n = cl_t(y.ex, y.tdate)
            emit("robustness", sample=tag, split="all", bucket="ambiguous", period=per, n=n, ex_bp=1e4 * me, t_ex=te)

    # ---------------- rules: chosen on 2020-23 (gap timing)
    dev = cells[(cells.period == "2020-23") & (cells.split != "all") & (cells.n >= 300)
                & ~cells.split.str.contains("not tradable") & (cells.bucket != "none")]
    dev = dev.assign(at=dev.t.abs()).sort_values("at", ascending=False)
    n_cells = len(cells[cells.period == "2020-23"])
    print("split cells tested (2020-23):", n_cells, flush=True)
    print(dev.head(15).to_string(), flush=True)
    rules = []
    for _, rw in dev.iterrows():
        if len(rules) == 3:
            break
        if rw.split in [r_[0] for r_ in rules]:
            continue
        rules.append((rw.split, rw.bucket, int(np.sign(rw.ex))))
    for s, bk, side in rules:
        mask = B[s] == bk
        for var in ["open-close", "1000-1530", "0935-1545"]:
            for per, a0, b0 in PER:
                daily, y = book(xg, mask, side, var, a0, b0)
                if len(y) < 20:
                    continue
                yrs = len(daily) / 252
                st = book_stats(daily, y, yrs)
                spy = ann_stats((P["c"]["SPY"].pct_change()).loc[daily.index])
                emit("rules", sample="gap", split=s, bucket=bk, side=side, variant=var, period=per,
                     spy_sharpe=spy["sharpe"], **st)
                print("RULE", s, bk, side, var, per, {k: round(v, 3) if isinstance(v, float) else v
                                                      for k, v in st.items()}, flush=True)
        # same rule on the headline-timed sample (robustness)
        mask_n = Bn[s] == bk if s in Bn else None
        if mask_n is not None:
            xn2 = variant_returns(xn.copy(), z)
            for per, a0, b0 in PER:
                daily, y = book(xn2, mask_n, side, "open-close", a0, b0)
                if len(y) < 20:
                    continue
                st = book_stats(daily, y, len(daily) / 252)
                emit("rules", sample="news", split=s, bucket=bk, side=side, variant="open-close", period=per, **st)
    # all-events long / short books for reference
    for side in (1, -1):
        for per, a0, b0 in PER:
            daily, y = book(xg, pd.Series(True, index=xg.index), side, "open-close", a0, b0)
            st = book_stats(daily, y, len(daily) / 252)
            emit("rules", sample="gap", split="all", bucket="all", side=side, variant="open-close", period=per, **st)
    n_var = n_cells + 3 * len(VARIANTS) + len(rules) * 3
    emit("count", split="variants_tried", n=n_var,
         bucket=f"{n_cells} split cells (2020-23) + {3 * len(VARIANTS)} entry/exit x timing + {len(rules) * 3} rule books")
    print("variants tried:", n_var, flush=True)
    pd.DataFrame(ROWS).to_csv(OUT, index=False)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
