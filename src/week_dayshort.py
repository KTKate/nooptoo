"""Weekly tally of the paper day short (study 68): what the account earned with and without it.

For every submitted logs/paper/dayshort_<day>.json:
  paper     the actual paper fills: short sells after 09:30 that day, and the cover buy of the same symbol and
            quantity that afternoon (the first such buy; cover_shorts runs before the next night's buys).
  auction   the same shares scored from the official opening print to the official closing print, minus the
            virtual book's auction cost per side (what a live account with opening/closing auction orders would get).
Borrow fees are not charged (easy-to-borrow names; about 0.1 bp a day).
Overnight results come from logs/paper/virtual_days.csv (auction prints) and the paper equity history.
    python src/week_dayshort.py [start_day]
Output: logs/paper/dayshort_days.csv (per day) and a printed summary.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import requests
import alpaca_data as A
import paper_overnight as PO
from core import half_spread_model


def fills(day):
    a = pd.Timestamp(f"{day} 09:00").tz_localize("America/New_York").tz_convert("UTC")
    b = a + pd.Timedelta(hours=8)
    j = requests.get(f"{PO.TRADE}/orders", headers=PO.H, timeout=30,
                     params=dict(status="closed", after=a.isoformat(), until=b.isoformat(), limit=500,
                                 direction="asc")).json()
    return [o for o in j if o.get("filled_at") and float(o.get("filled_qty") or 0) > 0]


def auction_cost(t, day, px):
    d = A.bars([t], "1Day", (pd.Timestamp(day) - pd.Timedelta(days=40)).strftime("%Y-%m-%dT00:00:00Z"),
               f"{day}T00:00:00Z", adjustment="raw")
    h = d.tail(20)
    adv = float((h.c * h.v).median())
    vol = float(np.log(h.c / h.c.shift(1)).std())
    return (1.3 + 0.1 * float(half_spread_model(adv, px, max(vol, 1e-3), "15:45")) + 2.5) / 1e4


def main(start="2026-09-29"):
    rows = []
    for f in sorted(x for x in os.listdir(PO.LOG) if x.startswith("dayshort_") and x.endswith(".json")):
        r = json.load(open(os.path.join(PO.LOG, f)))
        day = r["day"]
        if day < start or not r.get("submitted") or not r.get("orders"):
            continue
        fl = fills(day)
        for od in r["orders"]:
            t, q = od["symbol"], int(od["qty"])
            sells = [o for o in fl if o["symbol"] == t and o["side"] == "sell" and o["submitted_at"] >= f"{day}T13:31"
                     and int(float(o["filled_qty"])) == q]
            covers = [o for o in fl if o["symbol"] == t and o["side"] == "buy" and int(float(o["filled_qty"])) == q]
            ps = float(sells[0]["filled_avg_price"]) if sells else np.nan
            pc = float(covers[0]["filled_avg_price"]) if covers else np.nan
            try:
                o0 = A._official(t, day, "open")[0]
                c0 = A._official(t, day, "close")[0]
            except RuntimeError:                 # today's close not printed yet (free plan: SIP delayed 15 min)
                o0 = c0 = np.nan
            cost = auction_cost(t, day, o0) if o0 == o0 else np.nan
            rows.append(dict(day=day, symbol=t, qty=q, paper_sell=ps, paper_cover=pc, paper_pnl=q * (ps - pc),
                             open=o0, close=c0, auction_pnl=q * (o0 - c0) - cost * q * (o0 + c0)))
    v = pd.DataFrame(rows)
    if not len(v):
        print("no day shorts to score")
        return
    v.to_csv(os.path.join(PO.LOG, "dayshort_names.csv"), index=False)
    v = v[v.paper_cover.notna() & v.close.notna()]      # only finished days
    g = v.groupby("day").agg(names=("symbol", "size"), paper_short=("paper_pnl", "sum"),
                             auction_short=("auction_pnl", "sum"))
    vd = pd.read_csv(os.path.join(PO.LOG, "virtual_days.csv"))
    vd = vd[vd.day >= start]
    # a day short on day d follows the night that started on the previous entry day
    nights = vd.set_index("day").pnl
    out = g.copy()
    out.to_csv(os.path.join(PO.LOG, "dayshort_days.csv"))
    pd.set_option("display.width", 200)
    print(v.round(2).to_string(index=False))
    print(out.round(2).to_string())
    ov = nights.sum()
    print(f"\nAuction prices (live-account version), {start} to now:")
    print(f"  overnight blend only:          {ov:+.2f}")
    print(f"  overnight + day short:         {ov + g.auction_short.sum():+.2f}  (day short {g.auction_short.sum():+.2f})")
    print("Paper fills (what the paper account got):")
    print(f"  day short alone:               {g.paper_short.sum():+.2f}")
    a = PO.account()
    eq = float(a["equity"])
    print(f"  paper equity now {eq:.2f}; without the day short {eq - g.paper_short.sum():.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
