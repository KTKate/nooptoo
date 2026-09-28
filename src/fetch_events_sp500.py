"""Fetch S&P 500 index changes from Wikipedia. The 'Selected changes' table was moved on 2026-08-11 from
'List of S&P 500 companies' to 'Historical components of the S&P 500'; fall back to the last old revision."""
import os, io, requests, pandas as pd
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "local", "events")
os.makedirs(OUT, exist_ok=True)
H = {"User-Agent": "nooptoo-research/1.0 (python-requests)"}
urls = ["https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
        "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1368675864"]
for k, url in enumerate(urls):
    r = requests.get(url, headers=H, timeout=60)
    r.raise_for_status()
    open(os.path.join(OUT, f"sp500_wiki_{k}.html"), "w").write(r.text)
    tabs = pd.read_html(io.StringIO(r.text))
    for i, t in enumerate(tabs):
        print(k, i, t.shape, [str(c)[:40] for c in t.columns][:8])
    cand = [t for t in tabs if t.shape[0] > 100 and "Reason" in str(t.columns)]
    if cand:
        ch = cand[0]
        ch.columns = ["_".join(dict.fromkeys(str(x) for x in c)) if isinstance(c, tuple) else str(c) for c in ch.columns]
        print(ch.head(), ch.columns.tolist(), ch.shape)
        ch.to_csv(os.path.join(OUT, f"sp500_changes_raw_{k}.csv"), index=False)

# ---- announcement dates from the citations (S&P press release date or yyyymmdd in the announcement URL)
import re, json, html as _html
t = open(os.path.join(OUT, "sp500_wiki_0.html")).read()
refdate = {}
for m in re.finditer(r'<li about="#cite_note-[^"]*" id="cite_note-[^"]*" data-mw-footnote-number="(\d+)">(.*?)</li>', t, re.S):
    n, body = int(m.group(1)), _html.unescape(m.group(2))
    ds = []
    for d in re.findall(r'"date":\{"wt":"([^"]+)"\}', body):
        x = pd.to_datetime(d, errors="coerce")
        if pd.notna(x):
            ds.append(x)
    for d in re.findall(r"announcements/(20\d{6})-", body):
        ds.append(pd.to_datetime(d, format="%Y%m%d"))
    refdate[n] = min(ds) if ds else pd.NaT
ch = pd.read_csv(os.path.join(OUT, "sp500_changes_raw_0.csv"))
ch["eff"] = pd.to_datetime(ch["Effective Date"], errors="coerce")
def ann(refs, eff):
    ds = [refdate.get(int(x)) for x in re.findall(r"\[(\d+)\]", str(refs))]
    ds = [d for d in ds if d is not None and pd.notna(d) and eff - pd.Timedelta(days=45) <= d <= eff]
    return min(ds) if ds else pd.NaT
ch["ann"] = [ann(r, e) for r, e in zip(ch.Refs, ch.eff)]
rows = []
for _, r in ch.iterrows():
    for side, col in [("add", "Added_Ticker"), ("del", "Removed_Ticker")]:
        if isinstance(r[col], str) and r[col].strip():
            rows.append(dict(ticker=r[col].strip().replace(".", "-"), side=side, eff=r.eff, ann=r.ann,
                             reason=r.Reason, mcap_change="capitalization" in str(r.Reason).lower()))
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(OUT, "sp500_events.csv"), index=False)
print(ev.tail(), len(ev), ev.ann.notna().mean(), ev.eff.min())
print(ev[ev.eff >= "2020-01-01"].groupby("side").agg(n=("ticker", "size"), has_ann=("ann", lambda x: x.notna().mean())))
