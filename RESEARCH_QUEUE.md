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

## Running (batch 2026-10-10)
- 99: retrain the ensemble and jump models with targets at official opening crosses and the weekend flag
- 101: beta hedge at cross prices (short 0.25-0.75x SPY or QQQ overnight against the blend)
- 104: $10k account mechanics: whole-share rounding, names priced above the per-name budget, minimum sizes
- 106: live-account timing: entry job runtime vs the 15:50 market-on-close order cutoff (from the paper logs)
- 107: after-tax and after-fee return for a $10k account (short-term gains, data plan, borrow fees)
- 81: deferred (live earnings-model pipeline is a large build for ~0.1 Sharpe)

## Done
- 95 Reject: ETF overnight legs positive in 2024+ only and lower the blend's Sharpe; filters fail; short-index
  hedge a small lead
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

## Next (intraweek: 1-5 trading days)
- 81: live shadow of the earnings-night model (study 80): train on all events, score tonight's reporters at 15:45
  in shadow.py, long-only top quintile, scored at official prints
