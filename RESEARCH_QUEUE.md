# Research queue

Target: about 5 studies per trading day. Each study gets a script `src/studyNN_*.py` with a docstring (question, data,
design, output), a CSV in `results/`, and a row in `src/make_report.py` with a verdict. Rules that apply to every study:
point-in-time data only, auction costs + 2.5 bp per side, walk-forward or a held-out period, paired comparison
against the current strategy where one exists, and a search for data errors before believing a large result.
The daily panel has survivorship bias (only tickers alive in 2026): multi-day studies report excess returns over the
same universe's average (src/horizon_lib.py).

## Running
- 39: 5-day ML with price / fundamentals / analyst inputs (weekly rebalance)
- 40: 20-day ML, same inputs (monthly rebalance)
- 41: drift after announcements (guidance, buybacks, executive changes, target changes), 1-60 days
- 42: post-earnings drift by EPS surprise and earnings-day reaction, 20 and 60 days
- 43: monthly factor portfolios (value, quality, growth, analyst, momentum, low volatility)

## Next
- 44: multi-day reversal after large non-news drops (3-10 day holds, liquid stocks)
- 45: sector and industry rotation with ETFs: momentum, aggregated fundamentals and analyst revisions
- 46: pre-earnings positioning: buy 5 days before reports with rising targets / high target gap
- 47: insider cluster buying (study 15 candidate) combined with value and quality filters, 60-day holds
- 48: S&P 500 addition candidates: eligible by market cap and profitability but not yet in the index
- 49: FINRA bi-monthly short interest (% of float, days to cover) and squeeze setups, multi-week
- 50: seasonality: turn of month, pre-holiday, options expiration week, by weekday, for stocks and ETFs
- 51: lead-lag: large-cap moves predicting smaller same-industry stocks over the next 1-5 days
- 52: cross-asset signals (rates, dollar, oil, gold, bitcoin) for sector ETFs over 1-20 days
- 53: capital split between the overnight blend and the best multi-day strategy (combined Sharpe, drawdown)
- 54: dividend-related: ex-dividend run-up and drop, special dividends
- 55: volatility regime switching for the overnight strategy (VIX term structure, realized vol)
- 56: post-IPO and lockup-expiry effects (lockup dates from IPO date + 180 days)
- 57: pairs within industries chosen by fundamental similarity, multi-day reversion
- 58: earnings-call timing: report time of day and day of week vs reaction drift
