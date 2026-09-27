# Handoff (session 3 -> next)

Branch `claude/youthful-edison-ca09h8`. Paper trading only; never place orders (`--submit`) without the owner's go-ahead.
Report: `reports/overnight_report.html` (built by `python src/make_report.py`), published as a private Artifact.

## Data
- `data/store/` (git): Yahoo daily/earnings/60m. `data/local/` (git-ignored) and `data/models/` are on the private
  Hugging Face dataset `Incarnadin/nooptoo-market-data`: `python src/hf_sync.py pull` (needs `HF_TOKEN`) restores both
  (models are stored under `models/` in the repo and copied into `data/models`).
- New in session 3: `data/local/d1raw.parquet` (Alpaca unadjusted daily closes, `src/fetch_rawdaily.py`),
  `data/local/m5snapx/` (15:30-16:00 5-minute bars for 1,125 small caps missing from m5snap,
  `src/fetch_snap_smallcap.py`).
- Alpaca free plan: SIP requests must end more than 15 minutes in the past.
- Never use `pkill -f`/`pgrep -f` or `ps | grep <pattern> | xargs kill` with a pattern that appears in your own command
  line; it kills the shell (happened again this session).

## Corrections made in session 3 (both were look-ahead)
1. `study8_exec_retest.snap_panels("base")` dropped 483 whole tickers whose Alpaca and Yahoo prices disagreed in any
   month of 2024-26. Many were distressed small caps whose later reverse splits were adjusted differently, so the filter
   removed future losers. Verified with official auction prints (`study9_trade_check`, and the 40 worst removed trades
   re-priced: same -31% mean). Default is now `SNAP_MODE=none` in `final_series.py`, `study3_robust.py`, study 9.
   `snap_panels(mode)` supports base / month / strict / none.
2. Yahoo's "raw" close is adjusted for later splits (price filters used prices that did not exist on the day).
   `core.traded_close(P)` gives the traded close; small-cap and M tiers use it. The m5snap universe itself was chosen
   with hindsight (ADV > $5M on any day up to 2026-09); `study8_exec_retest.price_1545()` adds the missing small caps.

## Verdicts (2024-01..2025-06 val / 2025-07..2026-09 holdout, net, no hindsight filter; results/final_summary.csv)
| strategy | val | holdout | 2024-26 | previous (base filter) |
|---|---|---|---|---|
| ml_overnight_k10 | 1.78 | 1.78 | 1.78 | 1.72 |
| s_intraday_loser | 0.82 | 1.25 | 1.03 | 2.21 |
| s_day_loser | 1.03 | 0.80 | 0.92 | 2.32 |
| m_day_winner | 0.46 | 1.29 | 0.90 | 1.36 |
| combo 5 ML + 5 small-cap | 1.19 | 0.87 | 1.02 | 2.64 |
| SPY buy and hold | 1.15 | 1.58 | 1.28 | |
Only the ML overnight ranker remains (paper trade). Caveats (results/study3_robust.csv): DSR 0.54 (not significant),
beta 1.9 to SPY overnight return (hedged Sharpe 1.13), half-years 2026H1 +0.25 and 2026H2 -1.16, break-even at ~50%
of the quoted half-spread per auction side. All intraday studies (0, 4-8) and the earnings rules (2) are rejected.

## Models
`data/models/night_2024Q1..2026Q3.txt` regenerated and pushed to Hugging Face. For 2026Q4 (from 2026-10-01):
`python src/update_data.py daily; rm data/ml_frame.parquet; Q_START=2026Q4 Q_END=2026Q4 python src/study3_ml.py night`
(Q_END with no test data now trains and saves the model). Training needs ~6 GB RAM: do not run it next to study 9.

## Paper runner
`src/paper_overnight.py` default `--strategy=ml` (also smallcap / combo for comparison). Dry run by default; replay with
`--day=YYYY-MM-DD` (uses the SIP 15:40-15:45 bar). Replay of 2026-09-25: 7 of 10 picks match the backtest.
Schedule (weekdays ET): 15:45 entry, 09:15 exit, 17:30 `update_data.py daily`. Plan and stop rules in the report.

## Open items
- Owner decision: approve paper submission (`--submit`) after a 2-week dry run; a scheduler is needed to run the
  15:45 / 09:15 jobs (this container is ephemeral).
- Cash vs margin: T+1 settlement makes nightly round trips in a cash account a possible good-faith violation.
- Survivorship for small caps is poorly measured (Alpaca has only 19 delisted names with 2024-26 data).
