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

## Running (batch 2026-10-04)

## Done
- 79 Reject: after-hours spreads (~3%) erase the insider-filing jump; tight-spread subset tiny
- 46 Reject: pre-earnings drift only in 2024-26 (+20 bp/5 days, +35 with target raises); recheck in 2027
- 63 Reject: industry lead-lag 1-3 bp gross vs ~26 bp costs
- 64 Reject: last-hour continuation, volume surges and last-30-minute re-ranking add nothing after costs
- 80 Candidate: earnings-night model holds at 15:45 (long-only liquid Sharpe 1.24 vs 1.37 with close inputs);
  adds ~0.1 Sharpe to the blend at 25% on report nights; shorts lose
- 77 Candidate: earnings-event model for the reaction overnight, Sharpe 0.9-2.2 in all periods with close inputs;
  falls to 0.2-0.5 with previous-day inputs, so it needs the 15:45 rebuild (study 80); pre-report session rejected
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
- 82: check whether the Yahoo spin-off price-scale errors found in study 64 (about 5% of large-stock days in 2020)
  affect the overnight model's training data and backtests
- (done) 60: day-session long model (buy at the opening auction, sell at the closing auction) with all inputs: price,
  news since the previous close, analyst targets, fundamentals, premarket move; long side only (study 29 found the
  short side works only in hard-to-borrow names)

## Next (intraweek: 1-5 trading days)
- 81: live shadow of the earnings-night model (study 80): train on all events, score tonight's reporters at 15:45
  in shadow.py, long-only top quintile, scored at official prints
- 76: cohort models by earnings behavior and news reaction (clusters on past reactions), as extra ensemble members
- 72: short isolated no-news drops below -10% for 2-5 days (study 44 lead), with borrow limits, fresh cut-offs
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
