"""Study 204: Wikipedia page-view attention as a 1-5 day stock signal.

Question: does a jump in retail attention, measured by daily English Wikipedia page views of a company's article,
predict the stock's return over the next 1, 3 or 5 trading days (Da, Engelberg and Gao 2011 found Google search jumps
predict higher returns over 1-2 weeks; Barber and Odean 2008 found attention-driven retail buying), by enough to pay
for trading it at the closing auctions?
Data: ticker -> enwiki title from Wikidata SPARQL (P414 NYSE Q13677 / Nasdaq Q82059 statements with a P249 ticker
qualifier, enwiki sitelink), restricted to data/local/hks_universe.csv; ambiguous tickers resolved by statement dates /
rank and a short hand list (MANUAL) chosen by name only. Daily user page views 2016-01-01.. from the Wikimedia REST
API, summed over the current title and its former titles (page moves found in the article's revision comments since
2015, so a rename does not show up as a fake attention jump). Cache: data/local/wikiviews.parquet,
data/local/wiki_titles.csv. Prices: Yahoo daily store (store.read("daily")) for 2016-2019, Alpaca daily bars
(alpaca_data.read("d1"), close = official closing cross) from 2020-01 on.
Universe on day t: the 500 stocks with the highest 20-day median dollar volume up to t-1, prior close above $5, with
a mapped article whose prior-60-day median views are at least MIN_MED.
Signal (known before the close of t; page views are UTC daily counts, so only views through day t-1 are used):
  abn_W = log(mean daily views over days t-W .. t-1) - log(median daily views over the 60 days before that), W in {1,3}.
Trades: at the closing auction of t buy the top 10 (or the bottom 10) by abn_W, hold K in {1,3,5} trading days, sell
at the closing auction of t+K. Overlapping holds: each daily cohort gets 1/K of capital, so the daily book return is
the mean over the last K cohorts of that day's close-to-close return. Returns are excess over the same universe's
equal-weight average (survivorship-biased panel). Directions: top (long top 10), bottom (long bottom 10), and
top-minus-bottom (long top, short bottom, both legs pay costs).
Costs per side (bp): auction 1.3 + 0.1 x half-spread(15:45 model) + 2.5; a cohort pays entry + exit once per K days.
Protocol (pre-registered 2026-11-09): choose W, K and direction on 2016-2021 only by net Sharpe of daily excess
returns, write results/study204_frozen.json, then score 2022-01..2026-10 once. Pass: net Sharpe > 1, t > 2, and a net
gain in at least 4 of the 5 confirmation years (2022..2026). No changes after the confirmation run.
    python src/study204_wiki_attention.py map | fetch | moves | discover | confirm
Output: results/study204_wiki_attention.csv (grid, confirmation and per-year rows), results/study204_frozen.json,
        results/study204_daily.csv (confirmation daily returns)
"""
import json
import os
import re
import sys
import time
import urllib.parse as up
import numpy as np
import pandas as pd
import requests
import alpaca_data as A
from core import RES, half_spread_model

UA = {"User-Agent": "nooptoo-research/1.0 (stock research script)"}
WINS, HOLDS, DIRS = [1, 3], [1, 3, 5], ["top", "bottom", "top_minus_bottom"]
N, MIN_MED = 10, 20
START, END = "2016-01-01", "2026-10-31"
DISC_END, CONF_START = "2021-12-31", "2022-01-01"
FROZEN = os.path.join(RES, "study204_frozen.json")
OUT = os.path.join(RES, "study204_wiki_attention.csv")
TITLES = os.path.join(A.LOCAL, "wiki_titles.csv")
VIEWS = os.path.join(A.LOCAL, "wikiviews.parquet")
MOVES = os.path.join(A.LOCAL, "wiki_moves.csv")

# tickers with several plausible articles after the date / rank filter: the listed company's own article, by name
MANUAL = {"ALK": "Alaska_Air_Group", "AMR": "Alpha_Metallurgical_Resources", "BKNG": "Booking_Holdings",
          "CNP": "CenterPoint_Energy", "COHR": "Coherent_Corp.", "DELL": "Dell_Technologies", "DOW": "Dow_Inc.",
          "EWBC": "East_West_Bancorp", "FHN": "First_Horizon_Corporation", "FTI": "TechnipFMC", "GEN": "Gen_Digital",
          "HPQ": "HP_Inc.", "HTZ": "Hertz_Global_Holdings", "LAC": "Lithium_Americas",
          "NSC": "Norfolk_Southern_Corporation", "SAN": "Santander_Group", "SNOW": "Snowflake_Inc.",
          "SNX": "TD_Synnex", "SU": "Suncor_Energy", "TKO": "TKO_Group_Holdings", "TW": "Tradeweb",
          "VIP": "Veon_(company)", "WBC": "WABCO_Holdings", "WEN": "The_Wendy's_Company",
          "ZION": "Zions_Bancorporation", "AAL": "American_Airlines_Group", "GL": "Globe_Life",
          "AMAT": "Applied_Materials", "C": "Citigroup", "CHRD": "Chord_Energy", "B": "Barnes_Group",
          "LIN": "Linde_plc", "PTON": "Peloton_Interactive", "ROK": "Rockwell_Automation", "S": "SentinelOne",
          "GOOG": "Alphabet_Inc.", "GOOGL": "Alphabet_Inc.", "DNKN": "Dunkin'_Brands", "FOXA": "Fox_Corporation"}


def http_get(url, params=None, tries=40):
    wait = 4.0
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(wait)
            wait = min(wait * 2, 120)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        ra = r.headers.get("retry-after", "")
        time.sleep(float(ra) + 1 if ra.isdigit() else wait)     # 429 or 5xx: honour Retry-After, else back off
        wait = min(wait * 2, 120)
    raise RuntimeError(f"gave up on {url}")


def build_map():
    q = """SELECT ?item ?ticker ?article ?rank ?start ?end ?iend WHERE {
      VALUES ?exch { wd:Q13677 wd:Q82059 } ?item p:P414 ?st . ?st ps:P414 ?exch ; pq:P249 ?ticker ;
      wikibase:rank ?rank . OPTIONAL { ?st pq:P580 ?start } OPTIONAL { ?st pq:P582 ?end }
      OPTIONAL { ?item wdt:P576 ?iend }
      ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }"""
    r = http_get("https://query.wikidata.org/sparql", {"query": q, "format": "json"})
    rows = [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
    m = pd.DataFrame(rows)
    m["title"] = m.article.str.replace("https://en.wikipedia.org/wiki/", "", regex=False).map(up.unquote)
    m["ticker"] = m.ticker.str.upper().str.strip().str.replace(".", "-", regex=False)
    u = set(pd.read_csv(os.path.join(A.LOCAL, "hks_universe.csv")).ticker)
    m = m[m.ticker.isin(u)].copy()
    for c in ["start", "end", "iend"]:
        m[c] = pd.to_datetime(m.get(c), errors="coerce", utc=True).dt.tz_localize(None)
    m = m[~(m.end < START) & ~(m.iend < START)]
    m["dep"] = m["rank"].str.endswith("DeprecatedRank")
    keep = m.groupby(["ticker", "title"]).dep.transform("min") == 0     # keep a title if any statement is not deprecated
    m = m[keep | (m.groupby("ticker").dep.transform("min") == 1)]
    t = m.groupby("ticker").title.agg(lambda s: sorted(set(s)))
    out = []
    for tk, ts in t.items():
        title = MANUAL.get(tk) if len(ts) > 1 else ts[0]
        if title is None:
            print("unresolved", tk, ts)
            continue
        out.append((tk, title))
    out = pd.DataFrame(out, columns=["ticker", "title"])
    out.to_csv(TITLES, index=False)
    print(len(out), "tickers mapped of", len(u))


def fetch_views(titles, done):
    rows = []
    for i, ti in enumerate(titles):
        if ti in done:
            continue
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
               + up.quote(ti.replace(" ", "_"), safe="") + "/daily/20160101/" + time.strftime("%Y%m%d"))
        r = http_get(url)
        items = r.json().get("items", []) if r is not None else []
        rows.append(pd.DataFrame({"title": ti, "date": [x["timestamp"][:8] for x in items],
                                  "views": [x["views"] for x in items]}) if items
                    else pd.DataFrame({"title": [ti], "date": [None], "views": [np.nan]}))
        time.sleep(1.0)
        if len(rows) >= 20 or i == len(titles) - 1:
            yield pd.concat(rows)
            rows = []
    if rows:
        yield pd.concat(rows)


def save_views(new):
    old = pd.read_parquet(VIEWS) if os.path.exists(VIEWS) else None
    v = pd.concat([old, new]) if old is not None else new
    v.to_parquet(VIEWS, index=False)
    return v


def fetch(titles=None):
    titles = titles if titles is not None else list(pd.read_csv(TITLES).title.unique())
    done = set(pd.read_parquet(VIEWS, columns=["title"]).title) if os.path.exists(VIEWS) else set()
    todo = [t for t in titles if t not in done]
    print(len(todo), "titles to fetch", flush=True)
    for chunk in fetch_views(todo, done):
        v = save_views(chunk)
        print(time.strftime("%H:%M:%S"), v.title.nunique(), "titles cached", flush=True)


def moves():
    """Former titles: 'moved page [[Old]] to [[New]]' in revision comments since 2015-06 (follow chains)."""
    titles = list(pd.read_csv(TITLES).title.unique())
    have = pd.read_csv(MOVES) if os.path.exists(MOVES) else pd.DataFrame(columns=["title", "old", "ts", "scanned"])
    pat = re.compile(r"moved page \[\[([^\]]+)\]\] to \[\[([^\]]+)\]\]")
    for i, ti in enumerate(titles):
        if ti in set(have.title):
            continue
        found, cont = [], {}
        for _ in range(40):
            p = {"action": "query", "format": "json", "prop": "revisions", "titles": ti.replace("_", " "),
                 "rvprop": "comment|timestamp", "rvlimit": "500", "rvend": "2015-06-01T00:00:00Z", **cont}
            r = http_get("https://en.wikipedia.org/w/api.php", p)
            d = r.json()
            for pg in d.get("query", {}).get("pages", {}).values():
                for rv in pg.get("revisions", []):
                    mt = pat.search(rv.get("comment", ""))
                    if mt:
                        found.append((ti, mt.group(1).replace(" ", "_"), rv["timestamp"], 1))
            time.sleep(1.0)
            if "continue" not in d:
                break
            cont = d["continue"]
        rows = found or [(ti, None, None, 1)]
        have = pd.concat([have, pd.DataFrame(rows, columns=["title", "old", "ts", "scanned"])])
        if i % 20 == 0:
            have.to_csv(MOVES, index=False)
            print(time.strftime("%H:%M:%S"), i, "scanned", have.old.notna().sum(), "moves", flush=True)
    have.to_csv(MOVES, index=False)
    olds = sorted(set(have.old.dropna()) - set(titles))
    print(len(olds), "former titles; fetching their views", flush=True)
    fetch(olds)


# ------------------------------------------------------------------ panel and signal
def prices():
    """Daily close panel (wide). Alpaca d1pre (2016-01..2019-11, fetched for this study) + d1 (2019-12..).
    Alpaca bars are split-adjusted as of each month's fetch date, so a split after a month was fetched shows up as a
    fake jump; from 2019-06 on, a day whose Alpaca return differs from the Yahoo adjusted return by more than 10 points
    takes the Yahoo return. Returns beyond +-60% left after that are dropped as data errors."""
    u = list(pd.read_csv(os.path.join(A.LOCAL, "hks_universe.csv")).ticker)
    d = pd.concat([A.read("d1pre", columns=["ts", "ticker", "c", "v"]),
                   A.read("d1", start="2019-12", columns=["ts", "ticker", "c", "v"])], ignore_index=True)
    d = d[d.ticker.isin(u)]
    d["date"] = d.ts.dt.normalize()
    d = d.drop_duplicates(["date", "ticker"], keep="last")
    C = d.pivot(index="date", columns="ticker", values="c").astype("float64")
    V = d.pivot(index="date", columns="ticker", values="v").astype("float64")
    C = C[(C.index >= START) & (C.index <= END)]
    V = V.reindex(C.index)
    R = C / C.shift(1) - 1
    from core import load_panel
    P = load_panel()
    Y = P["c"].astype("float64")
    ry = (Y / Y.shift(1) - 1).reindex(index=R.index, columns=R.columns)
    fix = (R - ry).abs() > 0.10
    print("returns replaced by Yahoo (split jumps):", int(fix.sum().sum()), flush=True)
    R = R.mask(fix, ry)
    R = R.mask(R.abs() > 0.6)
    return C, V, R


def views_panel(dates):
    """Wide panel of daily views per ticker (current + former titles summed), on calendar days."""
    t = pd.read_csv(TITLES)
    v = pd.read_parquet(VIEWS).dropna(subset=["date"])
    mv = pd.read_csv(MOVES).dropna(subset=["old"]) if os.path.exists(MOVES) else pd.DataFrame(columns=["title", "old"])
    cur = set(t.title)
    mv = mv[mv.title.isin(cur) & ~mv.old.isin(cur)]        # a former title that is now another mapped article: skip
    alias = pd.concat([pd.DataFrame({"title": sorted(cur), "src": sorted(cur)}),
                       mv[["title", "old"]].rename(columns={"old": "src"})]).drop_duplicates()
    v = v.merge(alias, left_on="title", right_on="src", suffixes=("_src", ""))
    v["date"] = pd.to_datetime(v.date, format="%Y%m%d")
    w = v.groupby(["date", "title"]).views.sum().unstack()
    cal = pd.date_range(START, max(dates.max(), w.index.max()))
    w = w.reindex(cal)
    # the API omits days with zero views (and days before the article existed): zero only once the article has data
    started = w.notna().cumsum() > 0
    w = w.fillna(0).where(started)
    return t, w


def signal(dates):
    t, w = views_panel(dates)
    lw = np.log1p(w)
    out = {}
    for W in WINS:
        recent = np.log1p(w.rolling(W, min_periods=W).mean())
        base = w.shift(W).rolling(60, min_periods=50).median()
        a = (recent - np.log1p(base)).where(base >= MIN_MED)
        a = a.shift(1)                              # value at calendar day t uses views through t-1
        out[W] = a.reindex(dates)
    del lw
    return t, out


def build():
    C, V, R = prices()
    dates = C.index
    t, sig = signal(dates)
    px = C.shift(1)
    adv = (C * V).rolling(20, min_periods=10).median().shift(1)
    vol = np.log(C / C.shift(1)).rolling(20, min_periods=10).std().shift(1)
    rk = adv.where(px > 0).rank(axis=1, ascending=False)
    elig = (rk <= 500) & (px > 5)
    hs = half_spread_model(adv.values, px.values, vol.clip(lower=1e-3).fillna(0.02).values, "15:45")
    side = pd.DataFrame(1.3 + 0.1 * hs + 2.5, index=C.index, columns=C.columns)
    S = {}
    for W, a in sig.items():
        s = pd.DataFrame(np.nan, index=dates, columns=C.columns)
        for title, g in t.groupby("title"):
            tk = [x for x in g.ticker if x in s.columns]
            if not tk or title not in a.columns:
                continue
            for x in tk:
                s[x] = a[title].values
        # one ticker per article per day: the share class with the larger dollar volume
        for title, g in t.groupby("title"):
            tk = [x for x in g.ticker if x in s.columns]
            if len(tk) > 1:
                best = adv[tk].idxmax(axis=1)
                for x in tk:
                    s[x] = s[x].where(best == x)
        S[W] = s.where(elig & C.notna())
    return R, S, side


def cohorts(R, s, side, K, which):
    """Per signal day t: excess return of the picks on each of days t+1..t+K and round-trip cost (bp)."""
    rr = s.rank(axis=1, ascending=(which == "bottom"), method="first")
    pick = rr <= N
    n_ok = s.notna().sum(axis=1)
    pick = pick & (n_ok >= 4 * N).values[:, None]
    univ = s.notna()
    ex = {}
    for h in range(1, K + 1):
        f = R.shift(-h)
        ex[h] = (f.where(pick).mean(axis=1) - f.where(univ).mean(axis=1)) * 1e4
    cost = 2 * side.where(pick).mean(axis=1)
    return pd.DataFrame(ex), cost.where(pick.any(axis=1))


def book(R, s, side, K, d):
    """Daily book (1/K of capital per cohort): gross excess, cost, net in bp, plus per-trade (per cohort) numbers."""
    if d == "top_minus_bottom":
        e1, c1 = cohorts(R, s, side, K, "top")
        e2, c2 = cohorts(R, s, side, K, "bottom")
        ex, cost = e1 - e2, c1 + c2
    else:
        ex, cost = cohorts(R, s, side, K, d)
    ok = cost.notna() & ex.notna().all(axis=1)
    ex, cost = ex.where(ok, 0.0), cost.where(ok, 0.0)
    g = sum(ex[h].shift(h) for h in range(1, K + 1)) / K       # day d return of the cohort formed on d-h
    c = sum(cost.shift(h) for h in range(1, K + 1)) / K / K   # round trip spread over the K days of the hold
    live = sum(ok.shift(h, fill_value=False).astype(int) for h in range(1, K + 1)) > 0
    daily = pd.DataFrame({"gross": g, "cost": c})[live]
    daily["net"] = daily.gross - daily.cost
    trades = pd.DataFrame({"gross": ex.sum(axis=1), "cost": cost})[ok]
    trades["net"] = trades.gross - trades.cost
    return daily, trades


def stats(r):
    r = r.dropna()
    sd = r.std()
    return dict(n=len(r), mean_bp=r.mean(), sharpe=r.mean() / sd * np.sqrt(252) if sd else np.nan,
                t=r.mean() / sd * np.sqrt(len(r)) if sd else np.nan)


def summarize(daily, trades, tag, **kw):
    row = dict(tag=tag, **kw, days=len(daily), trades=len(trades),
               gross_bp_trade=trades.gross.mean(), cost_bp_trade=trades.cost.mean(), net_bp_trade=trades.net.mean())
    for c in ["gross", "net"]:
        st = stats(daily[c])
        row.update({f"{c}_day_bp": st["mean_bp"], f"{c}_sharpe": st["sharpe"], f"{c}_t": st["t"]})
    return row


def discover():
    R, S, side = build()
    rows = []
    for W in WINS:
        for K in HOLDS:
            for d in DIRS:
                dl, tr = book(R, S[W], side, K, d)
                dl, tr = dl[dl.index <= DISC_END], tr[tr.index <= DISC_END]
                rows.append(summarize(dl, tr, "discover_2016_2021", W=W, K=K, dir=d))
                for y in sorted(set(dl.index.year)):
                    rows.append(summarize(dl[dl.index.year == y], tr[tr.index.year == y], f"discover_{y}",
                                          W=W, K=K, dir=d))
    g = pd.DataFrame(rows)
    main = g[g.tag == "discover_2016_2021"]
    pd.set_option("display.width", 250)
    print(main.round(2).to_string(index=False))
    b = main.sort_values("net_sharpe", ascending=False).iloc[0]
    fz = dict(W=int(b.W), K=int(b.K), dir=b.dir, N=N, min_median_views=MIN_MED,
              discovery_net_sharpe=float(b.net_sharpe), discovery_net_t=float(b.net_t),
              frozen_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
              pass_rule="confirmation 2022-01..2026-10: net Sharpe > 1, t > 2, net gain in at least 4 of 2022..2026")
    json.dump(fz, open(FROZEN, "w"), indent=1)
    g.to_csv(OUT, index=False)
    print(json.dumps(fz, indent=1))


def confirm():
    fz = json.load(open(FROZEN))
    R, S, side = build()
    dl, tr = book(R, S[fz["W"]], side, fz["K"], fz["dir"])
    dl, tr = dl[(dl.index >= CONF_START) & (dl.index <= END)], tr[(tr.index >= CONF_START) & (tr.index <= END)]
    kw = dict(W=fz["W"], K=fz["K"], dir=fz["dir"])
    rows = [summarize(dl, tr, "confirm_2022_2026", **kw)]
    for y in sorted(set(dl.index.year)):
        rows.append(summarize(dl[dl.index.year == y], tr[tr.index.year == y], f"confirm_{y}", **kw))
    c = pd.DataFrame(rows)
    old = pd.read_csv(OUT)
    old = old[~old.tag.str.startswith("confirm")]
    pd.concat([old, c]).to_csv(OUT, index=False)
    dl.to_csv(os.path.join(RES, "study204_daily.csv"))
    pd.set_option("display.width", 250)
    print(c.round(2).to_string(index=False))


if __name__ == "__main__":
    {"map": build_map, "fetch": fetch, "moves": moves, "discover": discover, "confirm": confirm}[sys.argv[1]]()
