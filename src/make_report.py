"""Build the final HTML report (reports/overnight_report.html) from the result files.

    python src/make_report.py

Reads results/final_summary.csv, final_series.parquet, final_summary_base.csv (if present), study3_robust.csv,
study9_smallcap_robust.csv, study9_trade_check.csv and the per-study CSVs. All numbers in the text come from
these files; the verdict wording is fixed here.
"""
import html
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "reports", "overnight_report.html")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fs = pd.read_csv(f"{RES}/final_summary.csv")
series = pd.read_parquet(f"{RES}/final_series.parquet")
fsb = pd.read_csv(f"{RES}/final_summary_base.csv") if os.path.exists(f"{RES}/final_summary_base.csv") else None
r3 = pd.read_csv(f"{RES}/study3_robust.csv")
r9 = pd.read_csv(f"{RES}/study9_smallcap_robust.csv")
tc = pd.read_csv(f"{RES}/study9_trade_check.csv")


def g(df, strategy, period, col="sharpe"):
    x = df[(df.strategy == strategy) & (df.period == period)][col]
    return float(x.iloc[0]) if len(x) else np.nan


def f2(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}"


def pct(x, d=0):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


def cls(x, lo=0.0):
    return "neg" if x < lo else ""


# ------------------------------------------------------------------ chart: growth of $10,000 (log scale)
def chart():
    cols = [("spy_buy_hold", "SPY buy and hold", "var(--c-spy)"),
            ("ml_overnight_k10", "ML overnight ranker", "var(--c-ml)"),
            ("s_intraday_loser", "Small-cap intraday loser", "var(--c-sc)"),
            ("combo", "Combo (5 + 5 names)", "var(--c-combo)")]
    eq = (1 + series[[c for c, _, _ in cols]].fillna(0)).cumprod() * 10000
    W, H, L, R, T, B = 860, 340, 64, 150, 16, 34
    lo, hi = np.log(eq.min().min() * 0.95), np.log(eq.max().max() * 1.05)
    xs = lambda i: L + (W - L - R) * i / (len(eq) - 1)
    ys = lambda v: T + (H - T - B) * (1 - (np.log(v) - lo) / (hi - lo))
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Growth of $10,000, log scale">']
    for v in [5000, 10000, 20000, 40000, 80000]:
        if lo <= np.log(v) <= hi:
            y = ys(v)
            parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{L - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">${v // 1000}k</text>')
    for yr in sorted(set(eq.index.year)):
        i = eq.index.searchsorted(pd.Timestamp(f"{yr}-01-01"))
        if 0 < i < len(eq) or i == 0:
            x = xs(i)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B}" class="grid"/>')
            parts.append(f'<text x="{x + 4:.1f}" y="{H - B + 18}" class="axis">{yr}</text>')
    i0 = eq.index.searchsorted(pd.Timestamp("2025-07-01"))
    parts.append(f'<rect x="{xs(i0):.1f}" y="{T}" width="{xs(len(eq) - 1) - xs(i0):.1f}" height="{H - T - B}" class="hold"/>')
    parts.append(f'<text x="{xs(i0) + 6:.1f}" y="{T + 14}" class="axis">holdout</text>')
    ends = []
    for c, name, col in cols:
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(eq[c].values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{2.2 if c == "ml_overnight_k10" else 1.6}"/>')
        ends.append([ys(eq[c].iloc[-1]), name, col, eq[c].iloc[-1]])
    ends.sort()
    for j in range(1, len(ends)):                       # keep end labels apart
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 30)
    for y, name, col, v in ends:
        parts.append(f'<circle cx="{xs(len(eq) - 1):.1f}" cy="{ys(v):.1f}" r="3" fill="{col}"/>')
        parts.append(f'<text x="{W - R + 8}" y="{y + 4:.1f}" class="lab" fill="{col}">{html.escape(name)}</text>')
        parts.append(f'<text x="{W - R + 8}" y="{y + 18:.1f}" class="axis">${v:,.0f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------ candidate table
cand = [("ml_overnight_k10", "ML overnight ranker (top 10)", "paper", "Paper trade"),
        ("combo", "Combo: 5 ML + 5 small-cap", "no", "Reject"),
        ("s_intraday_loser", "Small-cap intraday loser", "no", "Reject"),
        ("s_day_loser", "Small-cap day loser", "no", "Reject"),
        ("m_day_winner", "Mid-cap day winner", "no", "Reject"),
        ("spy_buy_hold", "SPY buy and hold (benchmark)", "bench", "Benchmark")]
rows = []
for k, name, kind, verdict in cand:
    before = "" if fsb is None or k == "spy_buy_hold" else f2(g(fsb, k, "2024-26"))
    rows.append(f"""<tr><td>{name}</td><td><span class="chip {kind}">{verdict}</span></td>
<td class="num {cls(g(fs, k, 'val'))}">{f2(g(fs, k, 'val'))}</td><td class="num {cls(g(fs, k, 'oos'))}">{f2(g(fs, k, 'oos'))}</td>
<td class="num strong">{f2(g(fs, k, '2024-26'))}</td><td class="num dim">{before}</td>
<td class="num">{pct(g(fs, k, '2024-26', 'ann_ret'))}</td><td class="num">{pct(g(fs, k, '2024-26', 'ann_vol'))}</td>
<td class="num neg">{pct(g(fs, k, '2024-26', 'maxdd'))}</td>
<td class="num">{'' if pd.isna(g(fs, k, '2024-26', 'gross_bps')) else f"{g(fs, k, '2024-26', 'gross_bps'):.0f} / {g(fs, k, '2024-26', 'cost_bps'):.0f}"}</td>
<td class="num">{'' if pd.isna(g(fs, k, '2024-26', 'ci_lo')) else f"{g(fs, k, '2024-26', 'ci_lo'):.2f} to {g(fs, k, '2024-26', 'ci_hi'):.2f}"}</td>
<td class="num">{f2(g(fs, k, '2024-26', 'dsr_250')) if k != 'spy_buy_hold' else ''}</td></tr>""")
cand_table = "\n".join(rows)

# ------------------------------------------------------------------ ML robustness
c3 = r3[(r3.test == "cost") & (r3.period == "2024-26")].pivot_table(index="auction_frac", columns="extra_bps", values="sharpe")
cost_rows = "\n".join(
    f"<tr><th>{int(fr * 100)}%</th>" + "".join(f'<td class="num {cls(v)}">{v:.2f}</td>' for v in c3.loc[fr].values) + "</tr>"
    for fr in c3.index)
k3 = r3[r3.test == "k"].pivot_table(index="k", columns="period", values="sharpe")
k_rows = "\n".join(f'<tr><th>{int(k)}</th><td class="num">{k3.loc[k, "val"]:.2f}</td><td class="num">{k3.loc[k, "oos"]:.2f}</td>'
                   f'<td class="num">{k3.loc[k, "2024-26"]:.2f}</td></tr>' for k in k3.index)
hy = r3[r3.test == "half_year"][["period", "sharpe", "gross_bps"]]
hy_rows = "\n".join(f'<tr><th>{p}</th><td class="num {cls(s)}">{s:.2f}</td><td class="num {cls(gb)}">{gb:.0f}</td></tr>'
                    for p, s, gb in hy.itertuples(index=False))
hed = r3[r3.test == "hedged_vs_spy_night"].iloc[0]
dsr_oos = float(r3[(r3.test == "deflated_sharpe_prob") & (r3.period == "oos")].value.iloc[0])

# ------------------------------------------------------------------ small-cap robustness
x9 = r9[r9.period == "2024-26"]
filt = r9[r9.test == "filter"].pivot_table(index=["rule", "mode"], columns="period", values="sharpe")
base9 = r9[(r9.test == "base")].pivot_table(index="rule", columns="period", values="sharpe")
uni9 = x9[x9.test == "universe"].set_index("rule").sharpe
label = {"base": "Whole-ticker filter (earlier default, hindsight)", "month": "Ticker-month filter (hindsight within month)",
         "strict": "Strict whole-ticker filter (hindsight)", "alpaca_only": "Alpaca-only signal, m5snap universe",
         "alpaca_only_full_universe": "Alpaca-only signal, full universe"}
f_rows = [f'<tr><th>No filter, full universe (main)</th>' + "".join(
    f'<td class="num">{base9.loc[r, p]:.2f}</td>' for r in ["s_intraday_loser", "s_day_loser"] for p in ["val", "oos", "2024-26"]) + "</tr>"]
for m in ["base", "month", "strict", "alpaca_only", "alpaca_only_full_universe"]:
    cells = ""
    for r in ["s_intraday_loser", "s_day_loser"]:
        for p in ["val", "oos", "2024-26"]:
            v = filt.loc[(r, m), p] if (r, m) in filt.index else np.nan
            cells += f'<td class="num">{f2(v)}</td>'
    f_rows.append(f"<tr><th>{label[m]}</th>{cells}</tr>")
filter_rows = "\n".join(f_rows)
c9 = x9[x9.test == "cost"].pivot_table(index=["rule", "auction_frac"], columns="extra_bps", values="sharpe")
tails = x9[x9.test.isin(["tails_drop_top1pct", "tails_winsor_1_99"])].pivot_table(index="rule", columns="test", values="sharpe")
tcx = tc.dropna(subset=["r_off"]).copy()
tcx["d"] = 1e4 * (tcx.r_off - tcx.r_yahoo)
tcx = tcx[tcx.d.abs() < 2000]
tc_top = tcx[tcx.bucket == "top2pct"]

# ------------------------------------------------------------------ all studies
S = pd.read_csv(f"{RES}/study1_battery.csv")
s8 = pd.read_csv(f"{RES}/study8_exec_retest.csv")
studies = [
    ("0", "Calendar and ETF effects (overnight-only SPY, turn of month, weekday)", "Reject",
     "Gross overnight SPY returns are real, but a daily round trip costs more than the edge; net overnight-only SPY "
     "Sharpe 0.22 (val) and 0.86 (holdout) versus 1.15 and 1.61 for buy and hold."),
    ("1", "Daily-bar battery: 15 cross-sectional rules x 3 liquidity tiers", "Leads only",
     "All rules that enter at the open lose after the opening spread (net Sharpe -2 to -14). Overnight rules "
     "(buy at the close, sell at the open) were the only leads, and study 8 re-tested them with executable timing."),
    ("2", "Earnings: post-earnings drift, earnings-announcement returns, pre-earnings run-up", "Reject",
     "PEAD and announcement-return rules are flat or negative after 2024. The pre-earnings run-up (Sharpe 1.47 val, "
     "1.15 holdout) is no better than SPY buy and hold with similar market exposure."),
    ("3", "LightGBM cross-sectional models (overnight, next day, 1 and 5 days)", "Overnight only",
     "Next-day intraday and close-to-close models lose to costs. The 5-day model (Sharpe 1.1-1.3 holdout) is "
     "benchmark-like. The overnight model is the one survivor (below)."),
    ("4", "Intraday on 60-minute bars (ETF momentum, stocks in play, first-hour and last-half-hour rules)", "Reject",
     "Every variant is negative after costs in both halves of 2024-09..2026-09."),
    ("5", "Noise-area intraday momentum on SPY/QQQ/IWM/DIA (1-minute bars)", "Reject",
     "Gross Sharpe 1.5 on SPY in 2019-23, then negative after publication (2024-26 gross -0.05)."),
    ("6", "Opening-range breakout on stocks in play (5- and 1-minute bars)", "Reject",
     "-60 to -90 bp per trade net; even with optimistic fills -30 to -40 bp."),
    ("7", "ETF last-half-hour momentum", "Reject", "Median net Sharpe -1.1 (val) and -1.1 (holdout) across 162 variants."),
    ("8", "Executable re-test of the study-1 leads (15:45 signal, auction or 09:35 exits)", "Leads only",
     "Only market-on-close entry with market-on-open exit survives; any exit in the continuous market after the "
     "open loses (opening spread 35-190 bp). Gap fades at 09:35/10:00 entries lose."),
    ("9", "Robustness of the small-cap overnight rules (this session)", "Reject",
     "Two hindsight effects inflated them (details below). Without them the Sharpe is about 1.0, below SPY."),
    ("10", "News (Benzinga via Alpaca, 2019-2026), headline sentiment, event types (analyst, FDA, offering, M&A, "
     "earnings, government and regulation words), FINRA short-sale volume, sector momentum", "Reject",
     "Small caps that fall without news rebound overnight (+10-14 bp vs the average small cap, t 3-8), but a traded "
     "version is no better than SPY after costs. Added to the ML ranker, the factors are mixed (Sharpe 0.97/2.20/2.63 "
     "vs 0.79/2.36/2.42 for 2022-23 / validation / holdout); news and event features get almost no weight."),
    ("11", "Earnings: 10 days before to 10 days after each report, overnight and intraday, timing from news "
     "timestamps (82,801 reports)", "Reject",
     "Reaction night +11 bp beyond the market on average (beats +139 bp, misses -282 bp); misses keep falling for "
     "two nights. Holding through the report, pre-report run-ups and post-reaction trades are market-like or negative."),
    ("12", "Insider open-market purchases and sales (SEC Form 4, 2019-2026Q1)", "Reject",
     "Cluster buys: Sharpe about 0 in 2020-23, 0.5-1.0 in 2024-25, 1.9-2.3 in 2025-26 (5-20 day holds). Not stable."),
    ("13", "One model per stock and per sector vs one pooled model (300 most traded stocks)", "Reject",
     "The pooled model is better in every period (Sharpe 1.26/1.49 vs 0.28-0.88); per-stock models have too little data."),
    ("14", "Cohorts of similar stocks (co-movement and behavior groups, rebuilt yearly) with a model per group", "Candidate",
     "Per-behavior-group models: Sharpe 0.52/3.00/2.66 vs 0.79/2.36/2.42 for the pooled model (close-of-day features). "
     "The gain is lower volatility at the same mean return; the difference is not statistically detectable. Next candidate."),
    ("15", "Insider purchases in depth: role, size, cluster, opportunistic vs routine insiders, 10b5-1 plans, "
     "holds of 1-60 days, market-hedged (400 variants)", "Candidate",
     "5-20 day holds work only since 2024 (hedged Sharpe -0.5 in 2020-23). Mid caps with 2+ buyers held 60 days: "
     "hedged Sharpe 1.55/1.42/1.33, about 6% a year because 15% of capital is used; deflated Sharpe 0.86. Shadow test."),
    ("16", "International: US-listed ADRs and country ETFs, Europe and Asia lead-lag, ADR cross-section", "Candidate",
     "Lead-lag effects are absent (ADR prices already reflect the US afternoon). Buying the 5 ADRs with the largest "
     "loss to 15:45: Sharpe 0.52 (validation) and 1.74 (holdout), deflated Sharpe 0.08. Shadow test only."),
    ("17", "Corporate events: S&P 500 additions and deletions, Russell reconstitution, ex-dividend, splits, options "
     "expiration (25 variants)", "Reject",
     "S&P additions jump +339 bp the first night, before an auction order is possible; the later run-up is +20 bp. "
     "Forward splits: the night before averages +184 bp but the median is 26-38 bp and liquid names give 1.02/1.45/0.22. "
     "Reverse splits fall 20% around the event (short only)."),
    ("18", "Predicting the earnings reaction before the report (64,391 reports, 28 inputs)", "Reject",
     "Top 10 predicted: Sharpe 1.22/1.66/0.40; accuracy (rank correlation) falls from 0.049 to 0.014 and the sign is "
     "right 50-51% of the time."),
    ("19", "Portfolio construction of the ML strategy: long-short, SPY hedge, weighting, keeping picks through the day, "
     "regime filters, minimum score (21 variants)", "Reject",
     "No variant beats equal-weight top 10 in both periods. Hedging costs about 10 bp a night; holding through the day "
     "loses because the picks give back their overnight gains during the day."),
    ("20", "Statistical arbitrage at auctions: industry/cluster residual reversal, pairs, overnight residuals (356 variants)",
     "Reject", "Best residual-reversal variant 1.11/1.09 (2024-26 with 15:45 prices) but about 0 in 2020-23; deflated 0.03."),
    ("21", "Crypto at Alpaca (0.15-0.25% fees): time of day, weekends, trend, cross-sectional momentum, lead-lag with SPY",
     "Reject", "Daily trading costs about 0.5% per round trip. Best rule (ETH 30-day trend) deflated Sharpe 0.17."),
    ("22", "Options premium selling (put spreads, iron condors, cash-secured puts), 2024-02..2026-09", "Reject",
     "Best: SPY 10-delta weekly put spread, Sharpe 0.38 then 3.26, worst month -86% of capital at risk."),
    ("23", "Five-model ensemble (pooled, per-group and group-feature models) with 15:45 features", "Adopted",
     "Sharpe 2.30 / 2.21 (validation / holdout) vs 1.78 / 1.78, max drawdown -18% vs -24%, market-hedged 1.72 vs 1.13, "
     "deflated 0.66 vs 0.36; +7 bp a day (t 1.4). Paper account traded it on 2026-09-29; replaced by study 33."),
    ("32", "Retroactive look at big overnight jumps; classifiers for jumps (> +5%) and drops (< -5%), close prices",
     "Leads only",
     "Of 7,250 jumps above +10% since 2020, 65% had news after 15:45 and 43% were earnings nights, so most are not "
     "predictable at 15:45. Still, the top 10 by P(jump) held 15-24% jumps vs a 1% base rate, and ranking by "
     "P(jump) - P(drop) beat the rank model in 2022-23, validation and holdout (1.02 / 2.43 / 2.90 vs 0.79 / 2.36 / 2.39)."),
    ("33", "Study-32 classifiers at 15:45, blended with the ensemble (2/3 ensemble rank + 1/3 jump-minus-drop rank)",
     "Adopted",
     "Sharpe 2.80 / 2.91 (validation / holdout) vs 2.30 / 2.15; better in 4 of 6 half-years; 2.27 vs 1.60 without the 20 "
     "best trades; 2.59 vs 2.33 without earnings nights; hedged 2.43 vs 1.69; at twice the cost 2.22 vs 1.52; deflated "
     "0.91; paired bootstrap against the ensemble p = 0.01. Paper account trades it from 2026-09-30."),
    ("34", "Selling picks early in the after-hours or pre-market session when they are already up 2-10%", "Reject",
     "Moves of +2-5% by 09:00 give back about 0.6% by the open (t -12.6), but the extended-hours bid is 45-60 bp "
     "below the last trade (median half-spread about 20 bp). Sold at the real bid, every variant loses 11-43 bp per "
     "event against holding to the opening auction."),
    ("35", "Buying after-hours drops (2-10% below the close at 17:00-20:00), selling in the next opening auction", "Reject",
     "Measured on last trades the drops rebound 1.4-2.9% by the open (t above 20), but the last trade is not a price "
     "anyone could buy at: in a sample of 395 events the ask was 0.4% above to 4.4% below the close (median half-spread "
     "0.3-1.1%), and bought at the ask every group lost 1.1-1.8% on average by the open."),
    ("24", "Model settings and target tuning, chosen on 2022-23 only", "Reject",
     "Chosen settings: 2.56 / 1.89 vs default 2.36 / 2.42; single changes (market-adjusted target, 600 rounds) mixed."),
    ("26", "Training history from 2011 instead of 2020 (backfilled daily data)", "Reject",
     "0.34 / 2.79 / 3.84 vs 0.52 / 2.72 / 2.47 (2022-23 / val / holdout); worse in 2022-23 and with 20 names."),
    ("31", "Random-seed noise of the model", "Caveat",
     "Five seeds of the same model: holdout Sharpe 1.66 to 2.78, validation 2.45 to 2.91 (top 10). Differences below "
     "about 0.5 between variants are within run-to-run noise; this includes the ensemble's lead over the single model."),
    ("25", "Premarket gaps and volume, opening-auction to closing-auction day trades (41 variants)", "Reject",
     "Gap rules fail; a short-side model works only through stocks that cannot be borrowed (see 29)."),
    ("27", "ETF rotation, volatility management, leveraged ETFs with trend filter, SPY/QQQ mean reversion", "Candidate",
     "Only mean reversion (buy after 3 down days) has a higher Sharpe than SPY (1.40/1.01/1.96) with lower return; "
     "shadow-tracked."),
    ("28", "Intraday features (15:30-15:45 and first 30 minutes) stacked on the ranker", "Candidate",
     "Holdout 2.77 vs 1.78 but needs real-time full-market data at 15:45 (Alpaca Algo Trader Plus, $99 a month)."),
    ("29", "Shorting predicted day losers (opening-auction short, closing-auction cover)", "Reject",
     "1.39/2.75/4.09 on all stocks, 0.07/0.84/1.67 on easy-to-borrow stocks: the profit is in names that cannot be shorted."),
    ("30", "Neural-network ensemble member", "Reject", "No gain in top-10 books; worse in 2024-25 (t -2.2)."),
    ("-", "Macro calendar (FOMC, CPI, jobs report nights)", "Reject",
     "The ML strategy's jobs-report nights average -17 to -32 bp, but only 15-18 nights per period (t below 0.6)."),
]
chipcls = {"Reject": "no", "Leads only": "lead", "Overnight only": "paper", "Candidate": "lead", "Adopted": "paper", "Caveat": "lead",
           "Running": "bench"}
study_rows = "\n".join(f'<tr><td class="num">{n}</td><td>{html.escape(t)}</td><td><span class="chip {chipcls[v]}">{v}</span></td>'
                       f'<td>{html.escape(d)}</td></tr>' for n, t, v, d in studies)

ml_sh, spy_sh = g(fs, "ml_overnight_k10", "2024-26"), g(fs, "spy_buy_hold", "2024-26")
ml_oos, spy_oos = g(fs, "ml_overnight_k10", "oos"), g(fs, "spy_buy_hold", "oos")
eq_end = ((1 + series.fillna(0)).cumprod() * 10000).iloc[-1]
first, last = series.index[0].strftime("%Y-%m-%d"), series.index[-1].strftime("%Y-%m-%d")

page = f"""<title>Overnight Strategy Findings</title>
<meta name="description" content="Stock strategy research 2024-2026 for a $10k computer-driven account">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
:root {{
  --bg: #f4f5f3; --panel: #ffffff; --ink: #1b2126; --muted: #5a6570; --rule: #d6dbd6; --accent: #0e6770;
  --pos: #2d7a47; --neg: #a5402c; --lead: #8a6a12; --hold: rgba(14,103,112,0.07);
  --c-spy: #6b7682; --c-ml: #0e6770; --c-sc: #b0572f; --c-combo: #7a5aa6;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --cond: "IBM Plex Sans Condensed", "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #111517; --panel: #181d20; --ink: #e1e5e2; --muted: #98a3ab; --rule: #2b3337; --accent: #5cb9c0;
    --pos: #72c48d; --neg: #e38a73; --lead: #d6b457; --hold: rgba(92,185,192,0.08);
    --c-spy: #9aa5b0; --c-ml: #5cb9c0; --c-sc: #e0885e; --c-combo: #b39ae0;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #111517; --panel: #181d20; --ink: #e1e5e2; --muted: #98a3ab; --rule: #2b3337; --accent: #5cb9c0;
  --pos: #72c48d; --neg: #e38a73; --lead: #d6b457; --hold: rgba(92,185,192,0.08);
  --c-spy: #9aa5b0; --c-ml: #5cb9c0; --c-sc: #e0885e; --c-combo: #b39ae0;
}}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--sans); }}
main {{ max-width: 920px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 64px; display: grid; gap: 36px; }}
h1, h2, h3 {{ font-family: var(--cond); text-wrap: balance; line-height: 1.2; margin: 0; }}
h1 {{ font-size: 34px; font-weight: 600; letter-spacing: -0.01em; }}
h2 {{ font-size: 23px; font-weight: 600; padding-top: 6px; border-top: 2px solid var(--ink); }}
h3 {{ font-size: 17px; font-weight: 600; }}
section {{ display: grid; gap: 14px; }}
p, li {{ max-width: 68ch; margin: 0; }}
ul, ol {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; }}
.meta {{ font: 13px var(--mono); color: var(--muted); }}
.lede {{ font-size: 17px; }}
.eyebrow {{ font: 500 12px var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
thead th {{ font: 500 11.5px var(--mono); text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
  border-bottom: 1.5px solid var(--ink); white-space: nowrap; }}
tbody th {{ font-weight: 500; white-space: nowrap; }}
.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
.strong {{ font-weight: 600; }}
.dim {{ color: var(--muted); }}
.neg {{ color: var(--neg); }}
.chip {{ display: inline-block; font: 500 11.5px var(--mono); padding: 2px 8px; border-radius: 3px; white-space: nowrap;
  border: 1px solid currentColor; }}
.chip.no {{ color: var(--neg); }} .chip.paper {{ color: var(--pos); }} .chip.bench {{ color: var(--muted); }}
.chip.lead {{ color: var(--lead); }}
.panel {{ background: var(--panel); border: 1px solid var(--rule); padding: 18px 20px; display: grid; gap: 10px; align-content: start; }}
.finding {{ border-left: 3px solid var(--neg); padding-left: 14px; display: grid; gap: 6px; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; }}
svg {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .axis {{ fill: var(--muted); font: 11px var(--mono); }}
svg .lab {{ font: 500 12px var(--sans); }}
svg .hold {{ fill: var(--hold); }}
code, pre {{ font-family: var(--mono); font-size: 12.5px; }}
pre {{ background: var(--panel); border: 1px solid var(--rule); padding: 14px 16px; overflow-x: auto; margin: 0; line-height: 1.55; }}
.note {{ font-size: 13px; color: var(--muted); }}
@media (max-width: 520px) {{ h1 {{ font-size: 27px; }} main {{ padding-inline: 16px; }} }}
</style>
<main>
<header style="display:grid;gap:10px">
  <div class="eyebrow">Stock strategy research, net of costs, paper trading only</div>
  <h1>Overnight Strategy Findings</h1>
  <div class="meta">Data {first} to {last} · validation 2024-01..2025-06 · holdout 2025-07..2026-09 · $10,000 account</div>
  <p class="lede">One strategy remains a paper-trading candidate: a LightGBM model that ranks liquid stocks at 15:45,
  buys the top 10 in the closing auction and sells them in the next opening auction. It returned a Sharpe ratio of
  {ml_sh:.2f} over 2024-26 against {spy_sh:.2f} for SPY buy and hold, but the result is not statistically
  significant after the roughly 250 variants tried in this project, it lost money in 2026H2 so far, and half of its
  return is leveraged exposure to the market's overnight drift. Every intraday rule lost money after costs. The
  small-cap overnight rules that looked strongest in the previous session were inflated by two hindsight errors,
  found and fixed in this session. News, headline sentiment, earnings timing, insider trades, short-sale volume,
  sector effects, per-stock models, cohorts of similar stocks, insider purchases, international ADRs, corporate events
  and earnings prediction were tested next (studies 10-18); none clearly improves on the ML ranker. Two candidates
  (small-cap no-news losers and the ADR loser rule) are recorded daily in shadow mode (picks logged, no orders)
  from 2026-09-29; the behavior-cohort models and 60-day insider cluster buys still need live data feeds before
  they can be shadowed.</p>
</header>

<section>
  <h2>Candidates against buy and hold</h2>
  <p>Daily net returns, signal at 15:45, market-on-close buy, market-on-open sell, auction costs plus 2.5 bp per
  side for the measured gap between the Yahoo open and the official opening cross. Sharpe ratios are annualized.
  "Previous" is the same row with the hindsight filter that was used before this session.</p>
  <div class="scroll"><table>
    <thead><tr><th>Strategy</th><th>Verdict</th><th class="num">Val</th><th class="num">Holdout</th><th class="num">2024-26</th>
    <th class="num">Previous</th><th class="num">Ann. return</th><th class="num">Ann. vol</th><th class="num">Max DD</th>
    <th class="num">Gross / cost bp</th><th class="num">95% CI Sharpe</th><th class="num">DSR</th></tr></thead>
    <tbody>{cand_table}</tbody>
  </table></div>
  <p class="note">Gross / cost: mean per day in basis points of equity. 95% CI: 10-day block bootstrap. DSR: deflated
  Sharpe ratio, the probability that the true Sharpe is above zero after 250 trials (0.95 or more would be significant).
  Correlation of the ML ranker with SPY daily returns: {series.corr().loc['ml_overnight_k10', 'spy_buy_hold']:.2f}.</p>
</section>

<section>
  <h3>Growth of $10,000, log scale, net of costs</h3>
  <div class="scroll" style="min-width:0">{chart()}</div>
  <p class="note">No compounding limits, taxes or position caps are applied. The strategies compound daily at full
  equity; a live account also pays short-term capital-gains tax on every gain.</p>
</section>

<section>
  <h2>Corrections made in this session</h2>
  <div class="finding">
    <h3>1. The consistency filter removed real losses with hindsight</h3>
    <p>The backtests combine Alpaca 5-minute prices (for the 15:45 signal) with Yahoo daily prices. The filter that
    kept the two sources consistent dropped every ticker whose prices disagreed by more than 0.5% in any month of
    2024-26 (483 tickers). Many of those are distressed small caps whose later reverse splits were adjusted
    differently by the two sources, so the filter removed future losers using information from after the trade.
    Official auction prints confirm the removed losses are real: for the 40 worst removed trades the mean overnight
    return is the same at the official crosses as in the Yahoo data (about -31%). A signal built only from Alpaca
    prices, with returns only from Yahoo, needs no filter; for the small-cap day-loser rule it gives a 2024-26 Sharpe
    of {f2(filt.loc[('s_day_loser', 'alpaca_only_full_universe'), '2024-26'])}.</p>
  </div>
  <div class="finding">
    <h3>2. The price filter and the small-cap universe used future information</h3>
    <p>Yahoo's "unadjusted" close is adjusted for later splits: a stock that did a 1-for-50 reverse split in 2025
    shows 50 times its traded price for 2024. The $2 and $5 price floors now use the close as it traded (Alpaca raw
    daily bars). The 5-minute data also covered only stocks that reached $5M daily volume at some point up to
    2026-09, which keeps small caps that later became active; the missing 1,125 small caps were fetched. Without the
    filter, the earlier universe gives Sharpe {uni9.get('s_intraday_loser', np.nan):.2f} (intraday loser) and
    {uni9.get('s_day_loser', np.nan):.2f} (day loser); the full universe gives {base9.loc['s_intraday_loser', '2024-26']:.2f}
    and {base9.loc['s_day_loser', '2024-26']:.2f}.</p>
  </div>
  <p>The ML ranker is not affected by either error: without the filter its 2024-26 Sharpe is
  {ml_sh:.2f} (previously {f2(g(fsb, 'ml_overnight_k10', '2024-26')) if fsb is not None else 'n/a'}), and only 2.4% of its picks traded below $5.</p>
</section>

<section>
  <h2>ML overnight ranker</h2>
  <p>LightGBM regression on 53 price, volume, volatility, market and earnings-calendar features, trained each quarter
  on all data up to the quarter start minus a 10-day embargo, target = cross-sectional rank of the close-to-next-open
  return. Universe: price above $5 and 20-day median dollar volume above $5M. Features for day t use the 15:45 price,
  the day's range up to then and 85% of the day's volume, so the orders can be sent before the 15:50 market-on-close
  cutoff.</p>
  <div class="grid2">
    <div class="panel"><h3>Cost sensitivity, 2024-26 Sharpe</h3>
      <p class="note">Rows: share of the quoted 15:45 half-spread paid in each auction. Columns: extra bp per side.
      Base case 10% and 2.5 bp.</p>
      <div class="scroll"><table><thead><tr><th></th>{''.join(f'<th class="num">+{c:g} bp</th>' for c in c3.columns)}</tr></thead>
      <tbody>{cost_rows}</tbody></table></div></div>
    <div class="panel"><h3>Number of names held</h3>
      <div class="scroll"><table><thead><tr><th>k</th><th class="num">Val</th><th class="num">Holdout</th><th class="num">2024-26</th></tr></thead>
      <tbody>{k_rows}</tbody></table></div></div>
    <div class="panel"><h3>Half-years</h3>
      <div class="scroll"><table><thead><tr><th>Period</th><th class="num">Sharpe</th><th class="num">Gross bp/day</th></tr></thead>
      <tbody>{hy_rows}</tbody></table></div>
      <p class="note">2026H2 covers July to September 2026 only.</p></div>
    <div class="panel"><h3>What the return consists of</h3>
      <ul>
        <li>Beta to SPY's overnight return: {hed.beta:.2f}. After hedging it, alpha is {hed.alpha_bps:.1f} bp per night and
        the Sharpe is {hed.sharpe:.2f}.</li>
        <li>Deflated Sharpe probability: {g(fs, 'ml_overnight_k10', '2024-26', 'dsr_250'):.2f} for 2024-26 and {dsr_oos:.2f}
        for the holdout alone. Neither is significant.</li>
        <li>Break-even: the strategy stays positive until the auctions cost about half the quoted half-spread per side.</li>
        <li>In 2022-23 the same model design had close to zero alpha (study 3, development period).</li>
      </ul></div>
  </div>
</section>

<section>
  <h2>Small-cap overnight rules</h2>
  <p>Buy the 10 small caps ($1-5M daily dollar volume, traded price above $2) with the largest loss from the open
  (or from the previous close) to 15:45, sell at the next open. These looked like the best strategies before this
  session. Sharpe by consistency-filter treatment:</p>
  <div class="scroll"><table>
    <thead><tr><th>Treatment</th><th class="num">Intraday loser val</th><th class="num">Holdout</th><th class="num">2024-26</th>
    <th class="num">Day loser val</th><th class="num">Holdout</th><th class="num">2024-26</th></tr></thead>
    <tbody>{filter_rows}</tbody></table></div>
  <ul>
    <li>The large overnight winners are real: re-priced at the official closing and opening crosses, the top 2% of
    trades lose {-tc_top.d.mean():.0f} bp on average against the Yahoo prices, and all trades together lose 2-5 bp,
    within the 5 bp already charged.</li>
    <li>Clipping both tails of the trade returns at the 1st and 99th percentile gives Sharpe
    {tails.loc['s_intraday_loser', 'tails_winsor_1_99']:.2f} and {tails.loc['s_day_loser', 'tails_winsor_1_99']:.2f};
    removing only the best 1% of trades gives {tails.loc['s_intraday_loser', 'tails_drop_top1pct']:.2f} and
    {tails.loc['s_day_loser', 'tails_drop_top1pct']:.2f}.</li>
    <li>Paying 50% of the quoted half-spread in each auction gives Sharpe
    {c9.loc[('s_intraday_loser', 0.5), 2.5]:.2f} and {c9.loc[('s_day_loser', 0.5), 2.5]:.2f}.</li>
    <li>The edge sits in stocks trading at $2-5; with a $5 floor the Sharpe falls below 1.</li>
    <li>Only 19 delisted names have Alpaca data for 2024-26, so survivorship bias in small caps is not measured
    well. The Yahoo universe holds only stocks listed in 2026-09.</li>
  </ul>
</section>

<section>
  <h2>All studies</h2>
  <div class="scroll"><table>
    <thead><tr><th class="num">#</th><th>Study</th><th>Verdict</th><th>Result</th></tr></thead>
    <tbody>{study_rows}</tbody></table></div>
</section>

<section>
  <h2>Data not used</h2>
  <p>Options data (implied volatility, put-to-call volume, skew) is a documented predictor, but Alpaca's free
  options history starts in 2024-02 and comes per contract (hundreds per stock), so a daily history for 1,500
  stocks is not practical here; paid sources (OptionMetrics, ORATS, Cboe DataShop) sell it ready-made. Futures are
  not tradable at Alpaca and the index information is already in the SPY, IWM and VIX inputs. Analyst estimate
  revisions, order-flow and low-latency news feeds are paid products.</p>
</section>

<section>
  <h2>Practical constraints for a $10,000 account</h2>
  <div class="grid2">
    <div class="panel"><h3>Pattern day trader rule</h3><p>The SEC approved FINRA's removal of the pattern day
    trader rule and its $25,000 minimum on 2026-04-14, effective 2026-06-04 (brokers have until 2027-10-20 to
    implement it). Alpaca switched to its intraday margin framework on 2026-06-04. Intraday strategies are therefore
    open to a $10,000 account; they were rejected here because they lose money after costs, not because of the rule.
    The overnight strategy holds no position within one day in any case.</p></div>
    <div class="panel"><h3>Cash or margin</h3><p>In a cash account, Tuesday morning's sale settles
    on Wednesday (T+1). Buying Tuesday afternoon with those proceeds and selling Wednesday morning can count as a
    good-faith violation, depending on when the broker credits settlement. A margin account used without leverage
    avoids this. The alternative in a cash account is to trade each half of the
    capital on alternate nights, which halves exposure. About 250 round trips a year.</p></div>
    <div class="panel"><h3>Borrow</h3><p>The ML ranker is long only. No short positions, so no borrow fees or
    locate risk.</p></div>
    <div class="panel"><h3>Auction fills</h3><p>Market-on-close orders must reach Alpaca before 15:50; market-on-open
    orders between 19:00 and 09:28. $1,000 per name is below 0.05% of a typical pick's daily volume. The risk is a
    worse auction price than the model assumes, and the break-even is about 50% of the quoted half-spread.</p></div>
    <div class="panel"><h3>Gaps and concentration</h3><p>Ten overnight positions carry earnings, news and halt risk.
    Single trades lost up to 75% overnight in the small-cap sample; ML picks are more liquid but the 1% worst trades
    lose more than 10%.</p></div>
    <div class="panel"><h3>Taxes and data</h3><p>All gains are short-term. The free Alpaca data plan gives SIP bars
    with a 15-minute delay and IEX trades in real time, so the live 15:45 features are approximations of the backtest
    inputs (a replay of 2026-09-25 matched 7 of the backtest's 10 picks).</p></div>
  </div>
</section>

<section>
  <h2>Paper-trading plan</h2>
  <ol>
    <li>2026-09-28: one dry run of the scheduled job (orders written to <code>logs/paper/</code>, nothing sent).</li>
    <li>From 2026-09-29: the scheduled job sends market-on-close buys and market-on-open sells to the Alpaca paper
    account every trading day (owner approved). Holidays and early-close days are skipped. Each run commits its log,
    and <code>reconcile</code> compares every fill with the official auction print.</li>
    <li>Stop or re-evaluate if the paper account is more than 15% below its high, if the mean fill is more than 5 bp
    per side worse than the auction print, or if the Sharpe over 60 nights is below 0.</li>
    <li>Retrain at the start of each quarter (next: 2026Q4).</li>
    <li>Real money is a separate decision after the paper period, compared against SPY buy and hold.</li>
  </ol>
  <pre>Weekdays, America/New_York (scheduled cloud runs)
15:05  bash src/paper_job.sh entry     # setup, data update, waits until 15:46, market-on-close buys
09:05  bash src/paper_job.sh exit      # market-on-open sells of all positions
Manual: python src/paper_overnight.py reconcile     # fills vs official auction prints, equity history
Quarterly: rm data/ml_frame.parquet; Q_START=2026Q4 Q_END=2026Q4 python src/study3_ml.py night
Replay a past day: python src/paper_overnight.py entry --day=2026-09-25</pre>
</section>

<section>
  <h2>Reproduction</h2>
  <pre>pip install -r requirements.txt huggingface_hub
python src/hf_sync.py pull                          # minute data, auction cache, models (HF_TOKEN)
python src/fetch_rawdaily.py                        # traded (unadjusted) daily closes
python src/fetch_snap_smallcap.py                   # 15:30-16:00 bars for small caps outside m5snap
SNAP_MODE=none python src/study3_timing.py          # ML predictions from 15:45 features
python src/study3_robust.py                         # ML robustness
python src/study9_smallcap_robust.py                # small-cap robustness
python src/study9_trade_check.py                    # official auction re-pricing
python src/final_series.py                          # candidate series and summary (no filter)
SNAP_MODE=base python src/final_series.py           # the previous (hindsight) version, for comparison
python src/make_report.py                           # this page</pre>
  <p class="note">Studies 0-8: <code>python src/study0_calendar_etf.py</code> through <code>study8_exec_retest.py</code>;
  outputs in <code>results/</code>. Branch <code>claude/youthful-edison-ca09h8</code>.</p>
</section>
</main>
"""
open(OUT, "w").write(page)
print("wrote", OUT, len(page))
