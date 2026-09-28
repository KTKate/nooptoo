"""15:30-16:00 5-minute Alpaca bars for the ADR and country-ETF universe of study 16 (data/local/intl/universe.csv),
2024-01 .. latest month, so the ADR intraday-loser rule can be tested with the 15:45 price instead of the close.
Output: data/local/m5intl/<month>.parquet (alpaca_data.fetch_windows format)."""
import pandas as pd
import alpaca_data as A

u = pd.read_csv(f"{A.LOCAL}/intl/universe.csv")
col = "ticker" if "ticker" in u.columns else u.columns[0]
tick = sorted(set(u[col].dropna().astype(str)))
print("tickers", len(tick), flush=True)
A.fetch_windows("m5intl", start="2024-01", end=pd.Timestamp.today().strftime("%Y-%m"), windows=(("15:30", "16:00"),),
                tickers=tick, chunk=400, workers=4)
