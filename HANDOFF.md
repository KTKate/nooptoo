# Handoff (session 2 -> session 3)

Branch `claude/youthful-edison-ca09h8`. Paper trading only; never place orders without the owner's go-ahead.

## Data
- `data/store/` (git): Yahoo daily/earnings/60m. `data/local/` (git-ignored minute data) is on the private
  Hugging Face dataset `Incarnadin/nooptoo-market-data`: run `python src/hf_sync.py pull` first (needs `HF_TOKEN`).
  Folders: `m1` = 1-minute ETF bars 2019-06..2026-09, `m5snap` = 5-minute bars 09:30-10:00 and 15:30-16:00 for
  3,564 stocks 2024-01..2026-09, `m5full`/`m1full` = full-session bars for ORB candidates. One file per month.
- Alpaca keys: `ALPACA_PAPER_KEY_ID` / `ALPACA_PAPER_SECRET_KEY` (paper account, $10k, margin 4x).
- Rate limit is shared across processes via `data/local/.ratelimit` (alpaca_data.RateLimiter).
- Never use `pkill -f`/`pgrep -f` with a pattern that also appears in your own shell command (it kills the shell).

## Fixes made in session 2
- Spread estimator: look-ahead (shift 2) and upward bias (average then clip). Replaced by a quote-calibrated
  model (`core.half_spread_model`, 3,076 Alpaca NBBO samples, `results/spread_model.json`), inputs clipped,
  capped 300 bp. Execution-specific costs: `core.exec_cost_bps(P, "auction"|"open"|"mid"|"close")`.
- Yahoo open = opening cross in the median (auction check, `validate_overnight.py`); cross 2-7 bp lower on
  average for overnight picks -> +2.5 bp per side added. Yahoo close = official closing cross.
- Mixing Alpaca intraday with Yahoo daily needs the consistency filter in `study8_exec_retest.snap_panels`
  (spin-off/stock-dividend adjustments differ; the "large-cap gap-up fade Sharpe 8" was this artifact).

## Verdicts (2024-01..2025-06 val / 2025-07..2026-09 holdout, net of costs)
Rejected: calendar/ETF effects (study0), noise-area SPY/QQQ momentum (study5: gross edge 2019-23, gone
after publication), last-half-hour momentum (study7), ORB stocks in play (study6: -60..-80 bp/trade;
optimistic fills still -30..-40), gap fades at 9:35/10:00 entries (study8), PEAD/EAR (study2),
any overnight rule exited in the continuous market after the open (opening spread 35-190 bp).
Weak/benchmark-like: pre-earnings run-up (study2, Sharpe 1.5/1.0), ML cc5 (1.8/1.1-1.3).

Survivors (signal at 15:45, market-on-close buy, market-on-open sell; `results/final_summary.csv`):
| strategy | val | holdout | 2024-26 | 95% CI | DSR(250 trials) |
|---|---|---|---|---|---|
| ml_overnight_k10 (LightGBM, 15:45 features) | 1.65 | 1.80 | 1.72 | 0.63-2.93 | 0.50 |
| s_intraday_loser (small caps $1-5M ADV) | 2.09 | 2.39 | 2.23 | 1.01-3.37 | 0.82 |
| s_day_loser | 2.24 | 2.59 | 2.39 | 1.23-3.55 | 0.89 |
| m_day_winner | 1.56 | 1.47 | 1.50 | 0.38-2.72 | 0.36 |
| combo 50/50 ml + s_intraday_loser | 2.41 | 2.68 | 2.53 | 1.36-3.76 | 0.91 |
| SPY buy and hold | 1.15 | 1.58 | 1.28 | | |
Correlation with SPY ~0. Caveats: ML alpha ~0 in 2022-23 (regime-dependent); break-even round-trip cost
~40 bp (ML) so auction fill quality is the main risk; small-cap auctions are thin; survivorship check
(`survivorship.py`) shows <2 bp/day effect for these 1-day rules (lower bound, only 628 delisted names).
Close->open holds are not day trades (no PDT issue). Snap filter drops 483 tickers (15% of stock-days):
re-check the small-cap rules with a looser filter (e.g. only drop tickers with a persistent >0.5% level
shift, not monthly medians) to be sure the survivors are not selected by the filter.

## Remaining
1. Robustness for the small-cap rules like `study3_robust.py` does for ML (cost sensitivity incl. full
   half-spread in auctions, k neighborhood, half-years, looser/stricter snap filter).
2. Report: `reports/` HTML (load artifact-design skill first), comparisons vs buy-and-hold, verdict per
   strategy, practical constraints ($10k, PDT, cash vs margin, borrow, survivorship), reproduction commands;
   publish as an Artifact and give the owner the link.
3. Extend `src/paper_overnight.py` (dry-run default, tested for 2026-09-25) to the combo strategy and describe
   a paper-trading schedule; do not submit orders without approval.
