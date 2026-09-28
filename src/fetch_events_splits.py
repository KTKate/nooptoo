"""yfinance Ticker.splits for every store stock whose 20d median dollar volume exceeded $1M on some day since 2020.
Output: data/local/events/splits_yf.csv (ticker, date, ratio). Resumable; sequential (1 core, light memory)."""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_panel, stock_cols, DATA
import yfinance as yf
OUT = os.path.join(DATA, "local", "events", "splits_yf.csv")
DONE = os.path.join(DATA, "local", "events", "splits_yf_done.txt")
P = load_panel()
cols = stock_cols(P)
adv = P["dv"][cols].loc["2020":].rolling(20, min_periods=10).median().max()
tick = sorted(adv[adv > 1e6].index)
del P
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
print(len(tick), "tickers,", len(done), "done", flush=True)
for i, t in enumerate(tick):
    if t in done:
        continue
    for attempt in range(3):
        try:
            s = yf.Ticker(t).splits
            break
        except Exception as e:
            print(t, "err", e, flush=True)
            time.sleep(10)
            s = None
    if s is not None and len(s):
        df = pd.DataFrame({"ticker": t, "date": pd.to_datetime(s.index).tz_localize(None).normalize(), "ratio": s.values})
        df = df[df.date >= "2019-06-01"]
        if len(df):
            df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    with open(DONE, "a") as f:
        f.write(t + "\n")
    if i % 200 == 0:
        print(i, t, flush=True)
    time.sleep(0.2)
print("finished", flush=True)
