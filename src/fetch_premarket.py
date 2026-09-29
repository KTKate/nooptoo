"""Premarket 5-minute SIP bars 07:00-09:30 ET for the m5snap universe -> data/local/m5pre (study 25).
bars() keeps only 07:00-16:00, so 04:00-07:00 is not stored. Incremental per day (done_m5pre.parquet)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpaca_data as A

if __name__ == "__main__":
    tick = sorted(A.read("m5snap", columns=["ticker"]).ticker.unique())
    A.fetch_windows("m5pre", start="2024-01", windows=(("07:00", "09:30"),), tickers=tick, chunk=400, workers=4)
