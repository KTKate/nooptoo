"""Studies 190 and 191: did any part of the blend lose much less in October 2026, and how are the queued leads doing
on the live nights?

190  October by blend part. Question: study 166 found that the jump-only score (P(jump) - P(drop)) holds fewer tech
     hardware names (1.8 vs 2.4 a night in 2026) at a lower 2024-26 Sharpe (2.15 vs 2.45). Study 179 rescored the
     October entry days 2026-09-28..10-14 with the frozen 2026Q4 models. Scored the same way, did the ensemble alone,
     the jump score alone or a w=3 blend lose much less than the live blend on the October nights?
191  Forward-test tracker. Question: on every live night from 2026-09-29, what would each queued lead have returned
     at the official auction prints, against the live book? Leads (fixed before these nights):
       93    halve the book on nights when any of the live picks is tech hardware (study 90 lead);
       100   skip Monday-night entries (study 65 lead);
       134a  blended top 8 instead of top 10 (study 110 lead): the first 8 of the live capped list;
       134b  ensemble top 5 + jump top 5 (study 118 lead b1: weight 1/10 per name per list, a name in both lists
             gets 2/10, no industry cap, as study 118).

Data
  190: study 179 scores (data/local/study179_scores.parquet: ensemble, p_jump, p_drop, jmd, blend per name for frames
  B = backtest 15:45 and C = daily close, 2026Q4 models) for entry days 2026-09-28..10-14, extended here to 10-15 and
  10-16 with the same code: 15:40 SIP 5-minute bars, SPY/IWM 15:44 and news counts fetched into
  data/local/study190_bars.parquet / study190_news.parquet, frames by the study 154 replica
  (src/study150_154_live_check.Ctx) into data/local/study190_frames.parquet, scored with study179.score_set into
  data/local/study190_scores.parquet (core.load_panel ends 2026-10-16). Official crosses: every cached
  data/local/auctions*.parquet plus data/local/auctions190.parquet fetched here (study 94 rule: listing-venue 'O' /
  '6' print in widening windows, second pass the largest 'O' / '6' print of any listing exchange; Alpaca SIP
  trades, 60 requests a minute). Tech hardware: study 90/92 definition (TECH_HW industries, plus 'Electrical
  Products' and 'Industrial Machinery/Components' inside the Technology sector; data/store/sectors.parquet snapshot).
  Live: logs/paper/virtual_days.csv.
  191: logs/paper/entry_<day>.json (submitted entries: capped top 10 'ml_top' with blend scores in rank order, the
  orders actually sent) and logs/paper/members_<day>.json (each ensemble member's top 10, plus 'ensemble_top' and
  'jmd_top': the top 10 of the ensemble and of P(jump) - P(drop) over the whole scored universe, before the
  tradable filter and the industry cap). Only top 10 lists are logged, not the full scored universe; this is
  enough for every lead here (134b needs 5 tradable names from each list). members_2026-09-29.json has member
  lists only (that night ran the ensemble strategy without the jump models), so 134b cannot be scored on 09-29 and
  134a there is the top 8 of the live ensemble list. Prices: logs/paper/virtual.csv (per name official close and
  next official open from alpaca_data._official, cost) for the names the live book traded; other names priced the
  same way as paper_overnight.virtual() (alpaca_data._official close of the entry day and open of the next trading
  day, the daily bar if no cross print; cost 2 x (1.3 + 0.1 x half_spread_model(adv20, price, vol20, '15:45') +
  2.5) bp), cached in data/local/auctions191.parquet (study 94 leg format: ticker, date, kind, px = official
  cross or NaN, plus bar_px = daily bar fallback, entry_day, cost on the close leg, rule 'official', so the other
  studies' glob over auctions*.parquet can read it).

Design 190: per entry day 2026-09-28..10-16 (15 days), frames B (main) and C (check). Parts, all within-day ranks on
  the scored universe: blend = (2 rank(ensemble) + rank(jmd)) / 3 (live), ensemble only, jump only (jmd), w=3 =
  (3 rank(ensemble) + rank(jmd)) / 4. Each: Alpaca-tradable names, top 10 with at most 3 per industry
  (paper_overnight.cap_by_industry), equal weight. Night return at crosses as study 94 / 179 (panel night return x
  open cross / panel open on t+1 / (close cross / panel close on t); legs without a usable cross at the panel
  price); for the entry day 10-16 the panel has no 10-19 row, so that night is the raw cross-to-cross return (open
  cross 10-19 / close cross 10-16 - 1; no dividend adjustment; legs without a cross dropped). Cost
  core.exec_cost_bps(P, 'auction') + 2.5 bp per side. Per night: net, gross, tech hardware count, overlap with the
  blend picks. Summary over the 13 nights with a live entry (09-29, 10-01..10-16) and over all 15 nights: mean net
  bp, compounded, hit rate, mean tech hardware picks, paired t of the nightly difference against the blend.
Design 191: per live night (submitted entry from 2026-09-29 whose exit open has printed): per-name net = next
  official open / official close - 1 - cost. live_book = equal-weight mean over the names actually ordered (this is
  the virtual book, logs/paper/virtual_days.csv); live_10 = the same over all 10 capped picks (differs only when a
  pick got no order, 10-15 LITE). 93 = live_book x 1/2 on nights with any tech hardware name among the live picks.
  100 = 0 on Monday entries. 134a = mean over the first 8 of the capped list. 134b = ensemble top 5 + jmd top 5
  (first 5 Alpaca-tradable names of each logged top 10). Cumulative per lead: nights, mean net bp, compounded, hit
  rate, mean difference vs live_book and paired t, tech hardware names a night. update() appends nights not yet
  priced (rerun weekly: python src/study190_191_live_leads.py track).

Data checks: the B blend over the 11 live nights to 10-14 gives -95.2 bp a night, as study 179 (-95.3). 956 legs for
  190 (610 cached, 346 fetched into auctions190, 1 not found: one C ensemble leg at the panel price). 191: live_book
  equals logs/paper/virtual_days.csv net on every night (max difference 0.0 bp); 74 names outside the virtual book
  priced, all with official prints (no daily bar fallback). Peak memory 1.6 GB (190 report), 0.5 GB (191).

Result 190 (net bp a night at crosses, frame B; C in brackets; 13 live nights 09-29..10-16, live virtual book -58.4,
  compounded -7.8%): blend -57.6 [-46.6] (compounded -7.7% [-6.4%]), ensemble only -43.6 [-42.3] (-5.9% [-5.7%];
  paired t vs blend +0.57 [+0.20]), jump only -81.2 [-84.3] (-10.4% [-10.8%]; t -0.71 [-1.43]), w=3 -60.3 [-46.6]
  (t -0.59 [0.00]). Over all 15 nights 09-28..10-16 (adds 09-28 and 09-30, no live entry): blend -47.2, ensemble
  -38.1, jump -39.3, w=3 -48.4 (jump earned +257 and +209 bp on those two nights). Tech hardware picks a night (13
  live nights): blend 4.2, ensemble 3.8, jump 2.2, w=3 4.2 (live 3.8). Every part lost on 10-06, 10-13 and 10-14
  (-250 to -560 bp), and all gained 438-516 bp on 10-16. Nightly correlation with the live book 0.92-0.97.
Reading 190: no part lost much less. The jump-only score held about half as many tech hardware names but lost more on
  the live nights (it lost -270 bp on 10-09 and -204 bp on 10-12 with no tech hardware pick), so fewer tech hardware picks did not
  protect it. The ensemble alone lost about 14 bp a night less in B and 4 bp in C, well within noise (t 0.2-0.6).
  w=3 is the blend within 3 bp. October is a loss for every way of combining the two scores.
Result 191 (13 live nights 09-29..10-16, exits to 10-19; net bp a night, compounded): live_book -58.4, -7.75%;
  93 halve on tech hardware -26.7, -3.53% (12 of 13 nights halved, only 09-29 had no tech hardware pick; diff vs
  live +31.7, t +0.83); 100 skip Monday -47.9, -6.45% (skipped 10-05 +117 and 10-12 -254; diff +10.5, t +0.47);
  134a top 8 -57.9, -7.75% (diff +0.5, t +0.07); 134b ensemble 5 + jump 5 -67.8, -8.30% over 12 nights 10-01..10-16
  (live on the same nights -8.35%; diff +0.9, t +0.04). live_10 (all 10 capped picks) -58.5: LITE on 10-15 had no
  order (price above the per-name budget).
Reading 191: 13 nights are far too few to judge any lead (a 30 bp a night edge would need about a year). 93 is
  mostly a half-size book in a losing stretch: it loses half as much because tech hardware was in the picks almost
  every night, so it says little about timing. 100 gained from skipping one bad Monday. 134a and 134b track the live
  book within 1 bp a night.
Caveats: 190 uses the study 179 scores (backtest-style frames, not the live 15:45 inputs); the 10-16 night is raw cross
  to cross (no dividend adjustment), and sectors.parquet and the tradable list are current snapshots. 191 134b uses
  the logged top 10 lists, which are on the scored universe before the tradable filter (the first 5 tradable names
  are taken); no 134b on 09-29 (no jump scores logged that night, the live job ran the ensemble strategy). Costs are
  the model cost, not realized fills. The leads are scored on the live picks, so they share the live book's
  October market exposure.

Output: results/study190_october_parts.csv (table = nightly / summary), results/study191_lead_tracker.csv
  (table = nightly / summary / names)
    python src/study190_191_live_leads.py fetchbars   # 190: bars and news for 10-15, 10-16
    python src/study190_191_live_leads.py frames      # 190: B / C frames for 10-15, 10-16
    python src/study190_191_live_leads.py score       # 190: 2026Q4 scores for 10-15, 10-16
    python src/study190_191_live_leads.py fetch       # 190: official crosses of the parts' picks
    python src/study190_191_live_leads.py report      # 190
    python src/study190_191_live_leads.py track       # 191: price new nights and rewrite the tracker (weekly)
"""
import datetime as dt
import glob
import json
import os
import sys
import time

os.environ.setdefault("ALPACA_PER_MIN", "60")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import alpaca_data as A   # noqa: E402
from core import load_panel, stock_cols, exec_cost_bps, RES, DATA   # noqa: E402

A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))
LOC = os.path.join(DATA, "local")
LOG = os.path.join(A.ROOT, "logs", "paper")
BARS190 = os.path.join(LOC, "study190_bars.parquet")
NEWS190 = os.path.join(LOC, "study190_news.parquet")
FRAMES190 = os.path.join(LOC, "study190_frames.parquet")
SCORES190 = os.path.join(LOC, "study190_scores.parquet")
AUC190 = os.path.join(LOC, "auctions190.parquet")
AUC191 = os.path.join(LOC, "auctions191.parquet")
OUT190 = os.path.join(RES, "study190_october_parts.csv")
OUT191 = os.path.join(RES, "study191_lead_tracker.csv")
NEW_DAYS = ("2026-10-15", "2026-10-16")
OCT = ("2026-09-28", "2026-10-16")
PARTS = ["blend", "ensemble", "jump", "w3"]
START191 = "2026-09-29"


def rearm():
    A.RL = A.RateLimiter(int(os.environ["ALPACA_PER_MIN"]))   # other study modules reset it at import


# ================================================================== 190
def fetchbars():
    import study150_154_live_check as S154
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    rearm()
    mine = pd.read_parquet(BARS190) if os.path.exists(BARS190) else pd.DataFrame(columns=["day"])
    mn = pd.read_parquet(NEWS190) if os.path.exists(NEWS190) else pd.DataFrame(columns=["day"])
    parts, nparts = [mine], [mn]
    for ds in NEW_DAYS:
        day = pd.Timestamp(ds)
        assert day in days, f"panel has no {ds}"
        if day not in set(pd.to_datetime(mine.day)):
            _, cand = S154.universe_sets(P, cols, day)
            tick = sorted(set(cand) | {"SPY", "IWM"})
            x = pd.concat([A.bars(tick[i:i + 400], "5Min", S154.utc(day, "15:40"), S154.utc(day, "15:45"))
                           for i in range(0, len(tick), 400)])
            x["day"], x["tf"] = day, "5Min"
            y = A.bars(["SPY", "IWM"], "1Min", S154.utc(day, "15:44"), S154.utc(day, "15:45"))
            y["day"], y["tf"] = day, "1Min"
            parts += [x, y]
            print(ds, "bars", len(x), len(y), flush=True)
        if day not in set(pd.to_datetime(mn.day)):
            prev = days[days < day][-1]
            n = S154.news_counts(S154.utc(prev, "15:45"), S154.utc(day, "15:45"))
            nparts.append(pd.DataFrame({"day": day, "ticker": list(n), "n": list(n.values())}))
            print(ds, "news symbols", len(n), flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(BARS190)
    pd.concat(nparts, ignore_index=True).to_parquet(NEWS190)


def frames():
    """B and C frames for the new days, as study179.frames()."""
    import study150_154_live_check as S154
    ctx = S154.Ctx()
    P = ctx.P
    bars = pd.read_parquet(BARS190)
    bars["day"] = pd.to_datetime(bars.day)
    news = pd.read_parquet(NEWS190)
    news["day"] = pd.to_datetime(news.day)
    out = []
    for ds in NEW_DAYS:
        d = pd.Timestamp(ds)
        _, cand = S154.universe_sets(P, ctx.cols, d)
        n = news[news.day == d]
        nw = pd.Series(n.n.values, index=n.ticker.values, dtype="float64")
        for kind in ["B", "C"]:
            if kind == "B":
                b = bars[bars.day == d]
                hm = b.ts.dt.strftime("%H:%M")
                p5 = b[(b.tf == "5Min") & (hm == "15:40")].groupby("ticker").c.last()
                m1 = b[(b.tf == "1Min") & (hm == "15:44")].groupby("ticker").c.last()
                X = ctx.frame(d, "B", cand, intr=pd.DataFrame({"p1545": p5.reindex(cand)}), m1=m1)
            else:
                X = ctx.frame(d, "C", cand)
            t = list(X.index.get_level_values(1))
            X = X.astype("float32")
            X["earn_tonight"] = ctx.earn_tonight_ext(t).loc[d].reindex(t).fillna(0.0).values.astype("float32")
            X["news_today"] = nw.reindex(t).fillna(0.0).values.astype("float32")
            X["kind"] = kind
            out.append(X)
            print(ds, kind, len(X), flush=True)
    F = pd.concat(out)
    F.index = F.index.set_names(["date", "ticker"])
    F.to_parquet(FRAMES190)


def score():
    import study179_180_model_check as S179
    F = pd.read_parquet(FRAMES190)
    rows = []
    for kind in ["B", "C"]:
        X = F[F.kind == kind].drop(columns="kind")
        s = S179.score_set(X, "2026Q4")
        s["kind"], s["q"] = kind, "2026Q4"
        rows.append(s)
    S = pd.concat(rows)
    S.to_parquet(SCORES190)
    print("scored", S.groupby(["kind", S.index.get_level_values(0)]).size().to_dict(), flush=True)


def all_scores():
    s = pd.read_parquet(os.path.join(LOC, "study179_scores.parquet"))
    s = s[(s.q.astype(str) == "2026Q4") & s.kind.astype(str).isin(["B", "C"])]
    d = s.index.get_level_values(0)
    s = s[(d >= OCT[0]) & (d <= OCT[1])]
    if os.path.exists(SCORES190):
        n = pd.read_parquet(SCORES190)
        s = s[~s.index.get_level_values(0).isin(n.index.get_level_values(0))]
        s = pd.concat([s, n])
    s["kind"] = s.kind.astype(str)
    return s[["ensemble", "jmd", "blend", "kind"]]


def part_picks():
    """(kind, part) -> {day: [tickers in rank order]}: tradable, top 10, at most 3 per industry."""
    from paper_overnight import cap_by_industry, tradable
    S = all_scores()
    ok = set(tradable(sorted(set(S.index.get_level_values(1)))))
    out = {}
    for kind in ["B", "C"]:
        for day, g in S[S.kind == kind].groupby(level=0):
            g = g.droplevel(0)
            d = g.index.get_level_values(0)
            re, rj = g.ensemble.rank(pct=True), g.jmd.rank(pct=True)
            sc = dict(blend=g.blend, ensemble=g.ensemble, jump=g.jmd, w3=(3 * re + rj) / 4)
            for part, v in sc.items():
                v = v[v.index.isin(ok)].sort_values(ascending=False, kind="stable")
                out.setdefault((kind, part), {})[day] = list(cap_by_industry(v, 10, 3).index)
    return out


def next_days(days):
    import study150_154_live_check as S154
    g = S154.extended_grid(days)
    return pd.Series(g[1:], index=g[:-1])


def fetch():
    import study94_cross_rescore as S94
    import study166_167_blend_parts as S166
    from concurrent.futures import ThreadPoolExecutor
    P = load_panel()
    nxt = next_days(P["c"].index)
    del P
    rearm()
    pk = part_picks()
    need = []
    for picks in pk.values():
        for d, t in picks.items():
            need += [dict(ticker=x, date=d, kind="close") for x in t] + [dict(ticker=x, date=nxt[d], kind="open") for x in t]
    need = pd.DataFrame(need)
    need["ticker"] = need.ticker.str.replace("-", ".", regex=False)
    need = need[need.date <= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)].drop_duplicates()
    ca = S166.cached_crosses()
    ok = ca[ca.px.notna() | (ca.rule == "largest")][["ticker", "date", "kind"]]
    m = need.merge(ok, how="left", indicator=True)
    todo = m[m._merge == "left_only"].drop(columns="_merge")
    print("crosses needed", len(need), "cached", len(need) - len(todo), "to fetch", len(todo), flush=True)
    have = pd.read_parquet(AUC190) if os.path.exists(AUC190) else pd.DataFrame(
        columns=["ticker", "date", "kind", "first_px", "px", "sz", "ts", "x"])
    have["date"] = pd.to_datetime(have.date)
    ven = S94.listing_venue()
    jobs = [(d.strftime("%Y-%m-%d"), k, sorted(g.ticker)) for (d, k), g in todo.groupby(["date", "kind"])]
    print("day-kind jobs", len(jobs), flush=True)
    S94.AUC_FN = AUC190
    rearm()
    rows, t0 = [], time.time()

    def one(job):
        try:
            return S94._day_kind(job[0], job[1], job[2], ven)
        except RuntimeError as e:
            print("err", job[0], job[1], e, flush=True)
            return []
    for i in range(0, len(jobs), 24):
        with ThreadPoolExecutor(2) as ex:
            for r in ex.map(one, jobs[i:i + 24]):
                rows += r
        S94._save(pd.concat([have, pd.DataFrame(rows)], ignore_index=True))
        print("jobs", min(i + 24, len(jobs)), "/", len(jobs), "rows", len(rows), f"{time.time() - t0:.0f}s", flush=True)
    if rows:
        S94.refetch_missing()


def night_returns(P, cols, idx):
    """Per name and entry day: net return at crosses (study 94 / 166 method; the last night raw cross to cross)."""
    import study166_167_blend_parts as S166
    days = P["c"].index
    in_panel = idx[idx < days[-1]]
    Rx, miss, C = S166.returns(P, cols, days, in_panel)
    C_all = (exec_cost_bps(P, "auction")[cols] + 2.5).reindex(index=idx)
    last = idx[idx >= days[-1]]
    nxt = next_days(days)
    if len(last):
        X = S166.cached_crosses().dropna(subset=["px"])
        X["ticker"] = X.ticker.str.replace(".", "-", regex=False)
        for d in last:
            c0 = X[(X.date == d) & (X.kind == "close")].set_index("ticker").px.reindex(cols)
            o1 = X[(X.date == nxt[d]) & (X.kind == "open")].set_index("ticker").px.reindex(cols)
            sane = (c0 / P["rawc"][cols].loc[d] - 1).abs() < 0.2
            r = (o1 / c0 - 1).where(sane)
            Rx.loc[d] = r
            miss.loc[d] = r.isna()
    return Rx.reindex(idx), miss.reindex(idx), C_all


def report():
    import study179_180_model_check as S179
    P = load_panel()
    cols = stock_cols(P)
    days = P["c"].index
    _, ind, hw = S179.sectors()
    hwc = hw.reindex(cols).fillna(False).astype(bool)
    pk = part_picks()
    idx = days[(days >= OCT[0]) & (days <= OCT[1])]
    Rx, miss, C = night_returns(P, cols, idx)
    vd = pd.read_csv(os.path.join(LOG, "virtual_days.csv"), parse_dates=["day"]).set_index("day")
    live = S179.live_entries()
    rows = []
    nets = {}
    for (kind, part), picks in pk.items():
        for d in idx:
            t = [x for x in picks.get(d, []) if x in Rx.columns]
            r = Rx.loc[d, t]
            if d == days[-1]:
                t = list(r.dropna().index)                 # raw cross night: drop legs without a cross
                r = r.dropna()
            else:
                r = r.fillna(0.0)
            net = float(r.mean() - 2 * C.loc[d, t].mean() / 1e4) if t else np.nan
            nets[(kind, part, d)] = net
            ds = str(d.date())
            rows.append(dict(table="nightly", frame=kind, part=part, day=ds, live_night=ds in live, net_bp=1e4 * net,
                             gross_bp=1e4 * float(r.mean()) if t else np.nan, names=len(t),
                             legs_at_panel_or_dropped=int(miss.loc[d, picks.get(d, [])].sum()) if d in miss.index else 0,
                             techhw_picks=int(hwc.reindex(picks.get(d, [])).fillna(False).sum()),
                             overlap_blend=len(set(picks.get(d, [])) & set(pk[(kind, "blend")].get(d, []))),
                             overlap_live=len(set(picks.get(d, [])) & {x.replace(".", "-") for x in live[ds]["ml_top"]})
                             if ds in live else np.nan,
                             live_book_bp=1e4 * vd.net.get(d, np.nan), picks=" ".join(picks.get(d, []))))
    nt = pd.DataFrame(rows)
    for d in idx:                                          # live virtual book rows for reference
        ds = str(d.date())
        lr = live.get(ds)
        nt = pd.concat([nt, pd.DataFrame([dict(table="nightly", frame="live", part="live_virtual_book", day=ds,
                                               live_night=lr is not None, net_bp=1e4 * vd.net.get(d, np.nan),
                                               gross_bp=1e4 * vd.gross.get(d, np.nan),
                                               techhw_picks=int(hwc.reindex([x.replace(".", "-") for x in lr["ml_top"]]).fillna(False).sum()) if lr else np.nan,
                                               picks=" ".join(lr["ml_top"]) if lr else "")])], ignore_index=True)
    srows = []
    live_days = [str(d.date()) for d in idx if str(d.date()) in live]
    for (frame, part), g in nt.groupby(["frame", "part"]):
        x = g.set_index("day").net_bp / 1e4
        base = nt[(nt.frame == frame) & (nt.part == "blend")].set_index("day").net_bp / 1e4 if frame != "live" else None
        for lab, sel in [("13 live nights", x.index.isin(live_days)), ("15 nights 09-28..10-16", np.ones(len(x), bool))]:
            xx = x[sel].dropna()
            gg = g.set_index("day").loc[xx.index]
            st = S179.stats(xx)
            row = dict(table="summary", frame=frame, part=part, item=lab, **st, gross_bp=gg.gross_bp.mean(),
                       techhw_picks=gg.techhw_picks.mean(), worst_bp=1e4 * xx.min(), best_bp=1e4 * xx.max())
            if base is not None and part != "blend":
                b = base.reindex(xx.index)
                row.update(diff_vs_blend_bp=1e4 * (xx - b).mean(), t_vs_blend=S179.paired_t(xx, b),
                           overlap_blend=gg.overlap_blend.mean())
            if frame != "live":
                row["overlap_live"] = gg.overlap_live.mean()
            if frame != "live":
                lv = vd.net.reindex(pd.to_datetime(xx.index))
                row["corr_with_live"] = float(np.corrcoef(xx.values[lv.notna().values], lv.dropna().values)[0, 1]) if lv.notna().sum() > 3 else np.nan
            srows.append(row)
    df = pd.concat([nt, pd.DataFrame(srows)], ignore_index=True)
    df.to_csv(OUT190, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for fr in ["B", "C"]:
        print(f"frame {fr}: nightly net bp")
        q = nt[nt.frame.isin([fr, "live"])].pivot_table(index="day", columns="part", values="net_bp")
        print(q.round(0).to_string())
        print(nt[nt.frame == fr].pivot_table(index="day", columns="part", values="techhw_picks").to_string())
    sm = pd.DataFrame(srows)
    print(sm[["frame", "part", "item", "nights", "net_bp", "compounded", "hit", "techhw_picks", "worst_bp",
              "diff_vs_blend_bp", "t_vs_blend", "overlap_blend", "overlap_live", "corr_with_live"]].round(3).to_string(index=False))
    print("legs at panel price / dropped:", nt.groupby(["frame", "part"]).legs_at_panel_or_dropped.sum().to_dict())


# ================================================================== 191
def live_nights():
    """Submitted live entries from START191, oldest first: dict day -> (entry record, members record or {})."""
    out = {}
    for f in sorted(glob.glob(os.path.join(LOG, "entry_*.json"))):
        if "replay" in f:
            continue
        r = json.load(open(f))
        if not r.get("submitted") or r["day"] < START191:
            continue
        mf = os.path.join(LOG, f"members_{r['day']}.json")
        out[r["day"]] = (r, json.load(open(mf)) if os.path.exists(mf) else {})
    return out


_NEXT = None


def next_trading_day(day):
    """Next trading day from the panel days plus the extended business-day grid (study 154)."""
    global _NEXT
    if _NEXT is None:
        _NEXT = next_days(load_panel()["c"].index)
    return _NEXT[pd.Timestamp(day)]


def exit_printed(next_day):
    now = pd.Timestamp.now(tz="America/New_York")
    return now >= pd.Timestamp(f"{next_day.date()} 09:50").tz_localize("America/New_York")


def lead_names(r, m):
    """Name -> weight for each book on one night (Alpaca symbols)."""
    from paper_overnight import tradable
    top = list(r["ml_top"])
    orders = [o["symbol"] for o in r.get("orders", [])]
    books = {"live_book": {t: 1 / len(orders) for t in orders}, "live_10": {t: 1 / len(top) for t in top},
             "134a_top8": {t: 1 / len(top[:8]) for t in top[:8]}}
    if m.get("ensemble_top") and m.get("jmd_top"):
        e, j = list(m["ensemble_top"]), list(m["jmd_top"])
        ok = set(tradable(e + j))
        w = {}
        for lst in ([t for t in e if t in ok][:5], [t for t in j if t in ok][:5]):
            for t in lst:
                w[t] = w.get(t, 0.0) + 0.1
        books["134b_ens5_jump5"] = w
    return books


COLS191 = ["ticker", "date", "kind", "px", "first_px", "sz", "ts", "x", "rule", "bar_px", "entry_day", "cost"]


def load191():
    """Cache in the study 94 leg format (ticker, date, kind, px = official cross or NaN), so the other studies'
    glob over data/local/auctions*.parquet reads it; bar_px = daily bar fallback, cost on the close leg."""
    if not os.path.exists(AUC191):
        return pd.DataFrame(columns=COLS191)
    c = pd.read_parquet(AUC191)
    if "symbol" in c.columns:                       # first version: one row per entry day and symbol
        rows = []
        for r in c.itertuples():
            for kind, d, px in [("close", r.day, r.close), ("open", r.next_day, r.next_open)]:
                rows.append(dict(ticker=r.symbol, date=pd.Timestamp(d), kind=kind,
                                 px=px if r.src == "official" else np.nan, rule="official",
                                 bar_px=np.nan if r.src == "official" else px, entry_day=r.day,
                                 cost=r.cost if kind == "close" else np.nan))
        c = pd.DataFrame(rows, columns=COLS191)
        save191(c)
    c["date"] = pd.to_datetime(c.date)
    return c


def save191(c):
    c = c.reindex(columns=COLS191).copy()
    for k in ["ticker", "ts", "x", "rule", "entry_day"]:
        c[k] = c[k].astype("string")
    for k in ["px", "first_px", "sz", "bar_px", "cost"]:
        c[k] = c[k].astype("float64")
    c["date"] = pd.to_datetime(c.date)
    c.to_parquet(AUC191, index=False)


def price_names(day, names):
    """Official close of day and open of the next trading day per symbol, as paper_overnight.virtual(); cached."""
    from core import half_spread_model
    cache = load191()
    v = pd.read_csv(os.path.join(LOG, "virtual.csv"))
    v = v[v.day == day].set_index("symbol")
    out, todo = {}, []
    for t in names:
        if t in v.index:
            x = v.loc[t]
            out[t] = dict(close=x.close, next_open=x.next_open, cost=x.cost, src="virtual")
            continue
        c = cache[(cache.entry_day == day) & (cache.ticker == t)].set_index("kind")
        if {"close", "open"} <= set(c.index):
            cl, op = c.loc["close"], c.loc["open"]
            out[t] = dict(close=cl.px if cl.px == cl.px else cl.bar_px, next_open=op.px if op.px == op.px else op.bar_px,
                          cost=cl.cost, src="official" if (cl.px == cl.px and op.px == op.px) else "daily_bar")
        else:
            todo.append(t)
    if todo:
        rearm()
        d0 = pd.Timestamp(day)
        nd = next_trading_day(d0)
        d = A.bars(todo, "1Day", (d0 - pd.Timedelta(days=40)).strftime("%Y-%m-%dT00:00:00Z"),
                   (nd + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"), adjustment="raw")
        d["date"] = d.ts.dt.tz_convert(None).dt.normalize() if d.ts.dt.tz is not None else d.ts.dt.normalize()
        new = []
        for t in todo:
            x = d[d.ticker == t].set_index("date").sort_index()
            try:
                c0 = A._official(t, day, "close")[0]
                o1 = A._official(t, str(nd.date()), "open")[0]
            except RuntimeError:
                c0 = o1 = np.nan
            bc = float(x.loc[d0, "c"]) if (c0 != c0 and d0 in x.index) else np.nan
            bo = float(x.loc[nd, "o"]) if (o1 != o1 and nd in x.index) else np.nan
            pc, po = (c0 if c0 == c0 else bc), (o1 if o1 == o1 else bo)
            h = x.loc[x.index < d0].tail(20)
            if len(h) < 5 or pc != pc:
                cost = np.nan
            else:
                adv = float((h.c * h.v).median())
                vol = float(np.log(h.c / h.c.shift(1)).std())
                cost = 2 * (1.3 + 0.1 * float(half_spread_model(adv, pc, max(vol, 1e-3), "15:45")) + 2.5) / 1e4
            new += [dict(ticker=t, date=d0, kind="close", px=c0, bar_px=bc, rule="official", entry_day=day, cost=cost),
                    dict(ticker=t, date=nd, kind="open", px=o1, bar_px=bo, rule="official", entry_day=day)]
            out[t] = dict(close=pc, next_open=po, cost=cost, src="official" if (c0 == c0 and o1 == o1) else "daily_bar")
        save191(pd.concat([cache, pd.DataFrame(new)], ignore_index=True))
        print(day, "priced", len(todo), "new names", flush=True)
    return out


def update():
    """Price every live night not yet in the cache (append), then rebuild the tracker table from the cache."""
    import study150_154_live_check as S154
    _, _, hw = S154.sectors()
    rows, names = [], []
    for day, (r, m) in live_nights().items():
        nd = next_trading_day(pd.Timestamp(day))
        if not exit_printed(nd):
            print(day, "exit open", nd.date(), "not printed yet; skipped", flush=True)
            continue
        books = lead_names(r, m)
        px = price_names(day, sorted({t for b in books.values() for t in b}))
        net = {t: (p["next_open"] / p["close"] - 1 - p["cost"]) for t, p in px.items()
               if p["close"] == p["close"] and p["next_open"] == p["next_open"] and p["cost"] == p["cost"]}
        is_hw = lambda t: bool(hw.get(t.replace(".", "-"), False))
        any_hw = any(is_hw(t) for t in r["ml_top"])
        monday = pd.Timestamp(day).weekday() == 0
        res = {}
        for b, w in books.items():
            ok = {t: x for t, x in w.items() if t in net}
            s = sum(ok.values())
            res[b] = sum(x * net[t] for t, x in ok.items()) / s if s > 0 else np.nan
            for t, x in w.items():
                p = px.get(t, {})
                names.append(dict(table="names", day=day, book=b, symbol=t, weight=x, techhw=is_hw(t),
                                  close=p.get("close"), next_open=p.get("next_open"), cost=p.get("cost"),
                                  src=p.get("src"), net_bp=1e4 * net.get(t, np.nan)))
            rows.append(dict(table="nightly", day=day, weekday=pd.Timestamp(day).day_name(), book=b,
                             net_bp=1e4 * res[b], names=len(ok), names_unpriced=len(w) - len(ok),
                             techhw_names=sum(is_hw(t) for t in w), live_strategy=r.get("strategy")))
        res["93_halve_techhw"] = res["live_book"] * (0.5 if any_hw else 1.0)
        res["100_skip_monday"] = 0.0 if monday else res["live_book"]
        for b in ["93_halve_techhw", "100_skip_monday"]:
            rows.append(dict(table="nightly", day=day, weekday=pd.Timestamp(day).day_name(), book=b,
                             net_bp=1e4 * res[b], names=len(books["live_book"]), live_strategy=r.get("strategy"),
                             techhw_names=sum(is_hw(t) for t in books["live_book"]),
                             exposure=0.5 if (b.startswith("93") and any_hw) else (0.0 if (b.startswith("100") and monday) else 1.0)))
        if "134b_ens5_jump5" not in books:
            rows.append(dict(table="nightly", day=day, weekday=pd.Timestamp(day).day_name(), book="134b_ens5_jump5",
                             net_bp=np.nan, live_strategy=r.get("strategy"),
                             note="members log has no ensemble_top / jmd_top (ensemble-only night)"))
    nt = pd.DataFrame(rows)
    nt["exposure"] = nt["exposure"].fillna(1.0) if "exposure" in nt else 1.0
    lv = nt[nt.book == "live_book"].set_index("day").net_bp
    srows = []
    for b, g in nt.groupby("book", sort=False):
        x = g.set_index("day").net_bp.dropna() / 1e4
        if not len(x):
            continue
        base = lv.reindex(x.index) / 1e4
        d = x - base
        srows.append(dict(table="summary", book=b, first_night=x.index.min(), last_night=x.index.max(), nights=len(x),
                          net_bp=1e4 * x.mean(), compounded=float((1 + x).prod() - 1), hit=float((x > 0).mean()),
                          worst_bp=1e4 * x.min(), best_bp=1e4 * x.max(),
                          diff_vs_live_bp=1e4 * d.mean(), t_vs_live=d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 4 and d.std() > 0 else np.nan,
                          live_compounded_same_nights=float((1 + base).prod() - 1),
                          techhw_names=g.techhw_names.mean() if "techhw_names" in g else np.nan,
                          mean_exposure=g.exposure.mean()))
    df = pd.concat([nt, pd.DataFrame(srows), pd.DataFrame(names)], ignore_index=True)
    df.to_csv(OUT191, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(nt.pivot_table(index="day", columns="book", values="net_bp", sort=False).round(0).to_string())
    print(pd.DataFrame(srows).drop(columns="table").round(4).to_string(index=False))
    # check: live_book vs the virtual book
    vd = pd.read_csv(os.path.join(LOG, "virtual_days.csv")).set_index("day").net
    c = (lv / 1e4 - vd.reindex(lv.index)).abs().max()
    print("max |live_book - virtual_days.net| (bp):", round(1e4 * c, 3) if c == c else "n/a")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "track"
    {"fetchbars": fetchbars, "frames": frames, "score": score, "fetch": fetch, "report": report,
     "track": update}[what]()
    import resource
    print("peak memory GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
