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

## Running (batch 2026-10-18)
- 181: input drift: feature distributions July-October 2026 vs 2024-26 (population stability), for the inputs the
  models rely on most
- 182: did every model family weaken since July? Each ensemble member's and the jump score's own top 10 by month at
  cross prices
- 183: simple overnight rules (high-beta, day losers, volatility decile) by month: did all overnight approaches fail
  since July, or only the model?
- 81: deferred (live earnings-model pipeline is a large build for ~0.1 Sharpe)

## Done
- 179 Caveat: backtest-style picks with frozen 2026Q4 models lose as much as live in October (-95 vs -102 bp)
- 180 Caveat: every model vintage is weak since July; the Q3 refit is not the cause
- 173 Reject: correlation-aware selection lowers the mean in both periods
- 175 Reject: pause rules lower the 2024-26 mean by 1.5-11 bp a night
- 174 Caveat: edge weak since July 2026 (-3 bp a night); recent 72 nights worse than any 2024-26 window (t -2.05)
- 176 Caveat: tech hardware overnight -75 bp a night in October (3rd percentile) after a strong 2026Q3
- 178 Caveat: about half the live gap is a market-wide high-beta overnight reversal; the rest is a model shortfall
  visible since July (t about -2)
- 149 Reject: monthly refits 2.33 vs quarterly 2.38 over 2024-26 at crosses (t -0.4); keep quarterly
- 166 Reject: the blend beats ensemble-only (1.88) and jump-only (2.15); no member clearly harmful
- 167 Reject: live 2:1 weight best in the choice period; keep it
- 172 Caveat: live picks 35% tech hardware (losses concentrated there); no halving or cap rule helps in both periods;
  pick correlation rising (0.21 to 0.38)
- 168 Reject: no earnings-season effect; October losses were off season
- 171 Reject: same-day news and tone filters flip sign between periods
- 159 Reject: at official crosses small-cap no-news losers 0.11 / 0.73 and live-style ADR losers 0.46 / 1.33;
  live shadow at crosses +6 bp (ADR) and -37 bp (small caps) over 10 nights
- 158 Caveat: days_to_e missing live for ~100% of names and for most Aug-Sep 2026 training rows; fixing it changes
  1.3 picks a night, +2.6 bp (t 0.75); does not explain the weak 2026Q3; fix live and rebuild before next refit
- 162 Caveat: the live losing run is about a 1-in-10 outcome (10th-14th percentile); streaks do not reliably
  predict more losses
- 164 Caveat: 12 of 13 entries and all exits ran; the 9/30 miss had known, fixed causes
- 151 Reject: rolling 2- and 3-year training windows lose to the expanding window in both periods (-0.73 / -0.45
  Sharpe at crosses over 2024-26)
- 154 Caveat: live job runs the backtest models correctly; October losses are shared by the backtest picks; bug:
  live days_to_e missing (earnings calendar grid ends the day before), changes ~1.5 picks a night; fix pending
  owner approval
- 150 Caveat: tech hardware share rise is half stock traits (more volatile, higher beta), half model refits
- 145 Reject: premarket sells at the bid lose to the cross; lead: resting limit at the last premarket trade, then
  the auction if unfilled (+20 to +30 bp in 2025H2-26 only; fills unverified)
- 139 Reject: earnings-night picks are the best picks in 2024-25H1 and weak on the holdout (late NYSE crosses);
  excluding or halving them loses in the choice period
- 142 Caveat: missing crosses negligible (2 of 6,850); live exit needs a fallback sell for cancelled market-on-open
  orders; capacity fine at $10-25k, strained at $100k
- 135 Caveat: only SPY below its 50-day average is weaker in both periods (not significant); tech hardware share
  rising (7% to 30%)
- 137 Caveat: no measurable edge decay 2024-26 (trend t -0.5); 2026Q3 the only negative quarter
- 136 Caveat: 1/3/6/12-month loss chance 27/14/6/1% at backtest returns, 39/32/25/16% at half; 50% chance of a
  -20% drawdown within a year; paper so far at the 28th percentile
- 121 Reject: overnight shorts of the blend's bottom lose on the holdout (-15 bp a night); cross pricing helps ~1 bp
- 128 Leads only: overnight persistence real (IC 0.03-0.05) but weak after costs; blend has night20; tilts fail
  on the holdout
- 133 Reject: large-cap intraday losers work only in 2025H2-26 and lower the blend's Sharpe
- 118 Leads only: ensemble top 5 + jump top 5 gives 2.57 vs 2.38 at crosses (t 0.9 / 0.4); full side-by-side lists
  are worse
- 127 Reject: score thresholds and book scaling move money out of names that still earn
- 108 Reject: 5-seed averaging +0.5 bp (t 0.2) vs single seeds (sd 0.23 Sharpe); lead: hold the 5 single-seed
  books at a fifth each (2.34)
- 113 Reject: the free-plan job already uses a ~15:45 IEX price; paid data adds nothing (-2.2 bp, t -1.2); log
  IEX vs SIP prices for the picks to confirm
- 110 Leads only: top 8 won the choice period, +3 bp (t 0.9) on the holdout; keep 10
- 112 Reject: price and volume floors lose more return than the open bias they remove
- 116 Leads only: after 7 nights ADR losers +71 bp (t 1.3), small-cap no-news -4 bp, day short +143 bp per name
  (4 sessions); recheck after 40 nights (about 2026-11-24)
- 99 Reject: gap-corrected target and weekend flag within seed noise (seed alone moves Sharpe by 0.28); combined
  is worst
- 101 Reject: every SPY/QQQ hedge lowers Sharpe (2.32 to 1.75 vs 2.38 at crosses)
- 104 Caveat: whole shares cost ~nothing (skip and replace); fractional needs continuous orders (negative); check
  the broker's pattern-day-trader status before the day short in a $10k account
- 107 Caveat: after 30% tax, free plan: $9.1k / $14.8k / $3.4k a year on a fresh $10k (backtest returns); $99 plan
  pays off above ~$12.5k-25k
- 106 Caveat: ensemble-only scoring took 92 s (orders 15:47:50); blend scoring time not logged (job waits to 15:55);
  log it before going live
- 95 Reject: ETF overnight legs positive in 2024+ only and lower the blend's Sharpe; filters fail (its hedge gain
  was a cost sign error, see 101)
- 97 Reject: weekend-only model and Friday skip rules fail on 2025H2-26; weekend flag +2 to +6 bp (lead for retrain)
- 94 Caveat: at official crosses the blend is 2.48 / 2.27 / 2.38 (was 2.80 / 2.91 / 2.85); day legs still add
  (+0.12 short, +0.16 long, +0.29 both); bias only at the open and mostly in the picks; variant rankings unchanged
- 96 Leads only: quarterly expiration nights strong in 2025H2-26 (+286 bp vs other nights, 5 nights); closing
  price pressure reverses overnight but is known only after 16:00
- 98 Reject: closing-auction share -7 bp decile spread, below costs; no blend gain
- 88 Caveat: panel open = first trade, 3.5 / 12 bp above the official cross for the picks, so overnight backtests
  are too high by that much (blend roughly 2.6 / 2.2); no trading use of the post-open fall survives
- 78 Reject: insider inputs add nothing reliable to the day model, blend or pooled model; purchases disclosed
  before the open +7 then +20 bp in the day (unstable); EDGAR Atom feed ~40 s latency
- 90 Reject: nothing known at 15:45 predicts the 2%+ losing nights in both periods (98 rules, logistic AUC 0.56);
  lead: halve when any pick is tech hardware (found after seeing the holdout)
- 91 Reject: volatility-scaled weights lose on 2024-25H1; book scaling +0.08 on holdout at 84% size (t 1.1)
- 92 Reject: sector, tech-hardware and correlation caps do not hold on the holdout; live industry cap is neutral
- 89 Done as a tool: src/week_dayshort.py scores the paper day short at fills and at auction prints each week
- 83 Candidate: day-session long model loses only ~2 bp/day with SIP data cut at 09:10 (free plan); cycle Sharpe
  2.82 / 3.21 vs 2.80 / 3.31; IEX trades add nothing reliable; needs opening-auction entry (live account)
- 85 Caveat: day short works only entered in the opening auction; picks fall 18 bp by 09:31 and 33 bp by 09:35, so
  the paper timing (09:31 entry) loses 45-52 bp a name; covering at 15:55 is fine on price
- 86 Reject: stops on the day short cost more mean than they save; only a loose 8% name stop is neutral
- 87 Reject: selling longs after the open loses 15-35 bp a name before costs; keep the opening auction
- 48 Reject: S&P additions' gain is the untradable announcement-night jump; candidate lists do not predict
- 49 Reject: short-squeeze setups fail out of sample; short volume adds nothing to the overnight rise
- 52 Reject: cross-asset lead-lag for sector ETFs below costs; no rule beats SPY
- 57 Reject: fundamental-similarity pairs gross 4-16 bp vs ~17 bp costs
- 71 Candidate: blend + half-size day long + quarter-size day short, Sharpe 2.80 / 3.31 vs 2.42 / 2.91; margin and
  premarket data needed
- 76 Reject: reaction cohorts help 2022-23 only; per-cohort models unstable; ensemble gain not significant
- 72 Reject: pre-registered drop-short rule loses in 2024-26; effect only in hard-to-borrow names (needs borrow data)
- 82 Caveat: Yahoo spin-off scale errors cancel in returns; no result changes; cent-rounded penny prices make the
  blend backtest slightly conservative (2.85 -> 2.93 corrected)
- 50 Reject: calendar windows do not beat SPY after costs; pre-holiday day +18-21 bp but only ~9 days a year
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
- (done) 60: day-session long model (buy at the opening auction, sell at the closing auction) with all inputs: price,
  news since the previous close, analyst targets, fundamentals, premarket move; long side only (study 29 found the
  short side works only in hard-to-borrow names)

- 93: fresh test of "halve the book when any pick is tech hardware" (study 90 lead) on nights after 2026-10-08,
  scored from the paper picks at auction prints; decide in 2027 Q1


- 100: skip Monday-night entries as a pre-registered forward test (study 65 lead; at crosses 2.55 vs 2.38)

- 109: confirm whether Alpaca still enforces the pattern-day-trader rule for accounts under $25k (owner check, not a
  study; the day short depends on it)

- 117: log the IEX last trade and the SIP 15:45 price for each night's picks in the paper job (needs owner approval;
  study 113 assumed they match)

- 134: forward test (paper picks, auction prints) of the concentration leads: ensemble top 5 + jump top 5 (118) and
  blended top 8 (110), alongside 93 and 100; decide in 2027 Q1

- 143: live-account exit design: market-on-open sells plus a 09:45 check that sells any position still open
  (Alpaca cancels unfilled market-on-open orders); needed before going live, not for paper

- 155: paper test of the premarket resting limit sell (study 145 lead): measure real fill rates vs the cross; a
  paper-job change, needs owner approval

- 156: fix the live earnings-calendar gap (src/ml_features.py earnings_features grid ends the previous day, so
  reports dated today or later are dropped) and check the same gap in the 2026Q3 training rows (days_to_e missing
  for 84-99% of Aug-Sep 2026 rows); a paper-job change, needs owner approval
- 157: log the live feature rows for the picks each night (study 154 could not rebuild exact inputs)


- 81: live shadow of the earnings-night model (study 80): train on all events, score tonight's reporters at 15:45
  in shadow.py, long-only top quintile, scored at official prints
