"""Headline sentiment for the Alpaca/Benzinga news archive (data/local/news -> data/local/news_scored).

Model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis (Financial PhraseBank fine-tune,
labels negative/neutral/positive), CPU, about 110 headlines per second. Only articles tagged with 1-3 symbols are
scored (market wraps and lists mention many tickers and say little about each). sent = P(positive) - P(negative).
Processes every month file that has no scored file yet; with --follow it keeps waiting for new month files
until fetch_news.py has finished.
"""
import os
import sys
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "local", "news")
DST = os.path.join(ROOT, "data", "local", "news_scored")
os.makedirs(DST, exist_ok=True)
NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
torch.set_num_threads(3)
tok = AutoTokenizer.from_pretrained(NAME)
mdl = AutoModelForSequenceClassification.from_pretrained(NAME).eval()


def score(texts, bs=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i + bs], padding=True, truncation=True, max_length=48, return_tensors="pt")
            out.append(torch.softmax(mdl(**x).logits, -1))
    p = torch.cat(out).numpy() if out else None
    return p


def month(fn):
    d = pd.read_parquet(os.path.join(SRC, fn))
    if not len(d):
        pd.DataFrame().to_parquet(os.path.join(DST, fn))
        return
    n = d.symbols.fillna("").str.count(r"\|") + (d.symbols.fillna("") != "")
    d = d[(n >= 1) & (n <= 3)].copy()
    p = score(d.headline.fillna("").str.slice(0, 300).tolist())
    d["p_neg"], d["p_pos"] = p[:, 0], p[:, 2]
    d["sent"] = d.p_pos - d.p_neg
    d[["id", "ts", "symbols", "category", "headline", "p_neg", "p_pos", "sent"]].to_parquet(
        os.path.join(DST, fn), compression="zstd", index=False)
    print("scored", fn, len(d), flush=True)


if __name__ == "__main__":
    follow = "--follow" in sys.argv
    while True:
        cur = pd.Timestamp.today().strftime("%Y-%m")
        todo = sorted(f for f in os.listdir(SRC) if f.endswith(".parquet")
                      and (not os.path.exists(os.path.join(DST, f)) or f.startswith(cur)))
        todo = [f for f in todo if not (f.startswith(cur) and os.path.exists(os.path.join(DST, f)))]
        for f in todo[::-1]:
            month(f)
        fetching = os.system("ps -eo args | grep -q '[f]etch_news.py'") == 0
        if not follow or not fetching:
            if not todo:
                break
            if not follow:
                break
        time.sleep(60)
