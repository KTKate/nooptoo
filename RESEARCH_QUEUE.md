# Research queue

Scope (owner, 2026-10-03): intraday and intraweek strategies only. Holding periods from minutes up to 5 trading days.
Monthly or longer holds are out of scope.

Target: about 5 studies per trading day. Each study gets a script `src/studyNN_*.py` with a docstring (question, data,
design, output), a CSV in `results/`, and a row in `src/make_report.py` with a verdict. Rules that apply to every study:
point-in-time data only, auction costs + 2.5 bp per side (quoted spreads for non-auction trades), walk-forward or a
held-out period, paired comparison against the current strategy where one exists, and a search for data errors before
believing a large result. Live constraint: the free Alpaca plan has real-time IEX data and 15-minute-delayed SIP, so a
signal needed before 15:30 must work with IEX or with data older than 15 minutes.
The daily panel has survivorship bias (only tickers alive in 2026): multi-day studies report excess returns over the
same universe's average (src/horizon_lib.py).

## Running
- 77: earnings-event models with small leaves; interaction analysis of the overnight models (agent)

## Done
- 75 Candidate (weak): insider buys after after-close filings +14-20 bp next session (t 3-5, both periods), thin after
  costs; Congress, 13D, sells, earnings link rejected
- 74 Reject: no pre-report day-session pattern survives out of sample; leads: target raises before report, Fridays
- 73 Reject: fixed top 10 beats variable counts in the selection period; jump threshold (~9 names) a lead
- 70 Reject: shorting the biggest overnight gainers in the day session is weak (Sharpe ~0.6, about zero in 2020-23)
- 44 Reject: no 1-5 day reversal after drops; no-news drops keep falling
- 67 Candidate: blend + half-size day-session long model, Sharpe 2.56 / 3.09 vs 2.41 / 2.91; margin needed
- 69 Reject: second-night re-entry lowers Sharpe (2.35 vs 2.85)
- 59 Reject: guidance-cut and earnings-reaction filters do not improve the blend
- 65 Leads only: Monday-night entries weakest in both periods but still positive; skipping: Sharpe 2.97 vs 2.85
- 61 Reject: opening-30-minute momentum gone in 2024-26; cross-section far below costs
- 62 Reject: gap fills/continuations lose after costs; premarket gap-up >10% short at the open is the only
  consistent cell (borrow-limited; part of study 70)
- 53 Reject: holding longer loses in each day session; picks rise at night and fall in the day for several days
- 68 Candidate: day-session short of the picks (easy to borrow) +20-26 bp/day; with the overnight blend at quarter
  size Sharpe 3.04 vs 2.85; shadow-tracked (shadow.py day_short)
- 39 Reject: weekly ML weak (t up to 1.3); fundamentals/analyst inputs made it worse
- 60 Candidate: day-session long model 4-21 bp/day net, Sharpe 0.4-1.3 (2024-26); test combined with the overnight blend
- 40 (20-day ML) stopped: out of scope
- 41 Reject: no announcement drift; at 1-5 days nothing above costs; guidance cuts trail later (avoid filter)
- 42 Reject: no post-earnings drift; worst earnings reactions keep falling for days (avoid filter)
- 43 out of scope (monthly): low EV/sales top 20 strong in 2020-23, weak in 2024-26

## Next (intraday: open to close, or within the session)
- (done) 60: day-session long model (buy at the opening auction, sell at the closing auction) with all inputs: price,
  news since the previous close, analyst targets, fundamentals, premarket move; long side only (study 29 found the
  short side works only in hard-to-borrow names)
- 63: intraday lead-lag: sector ETF or industry leader moves in the first hour predicting laggards by the close
- 64: last-hour effects: 15:00-16:00 return vs the day's move, imbalance-like proxies from 15:30-15:45 volume

## Next (intraweek: 1-5 trading days)
- 76: cohort models by earnings behavior and news reaction (clusters on past reactions), as extra ensemble members
- 72: short isolated no-news drops below -10% for 2-5 days (study 44 lead), with borrow limits, fresh cut-offs
- 46: pre-earnings run-up: buy 1-5 days before reports, sell before the report
- 78: live Form 4 feed (EDGAR XML from 2026-04) for a shadow test of insider buys > 5% of ADV; insider features
  (recent buy, size) as inputs to the day-session and jump models
- 48: S&P 500 additions: candidates and announced additions, 1-5 days around announcement and effective date
- 49: short squeeze setups (high short volume ratio + news + price breakout), 1-5 day holds
- 50: weekly patterns: Monday open to Friday close, turn of month, pre-holiday, options expiration week
- 52: cross-asset moves (rates, dollar, oil, gold, bitcoin) predicting sector ETFs over 1-5 days
- 57: pairs within industries, 1-5 day reversion of the spread
- 71: full day cycle on one account: overnight blend long + day-session long model + quarter-size day short of the
  picks (studies 33, 60, 68), with margin and settlement rules
- 66: weekly ML (study 39) combined with the overnight blend: capital split, combined Sharpe and drawdown
