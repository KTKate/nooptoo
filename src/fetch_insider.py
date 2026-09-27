"""SEC Form 3/4/5 insider transaction data sets (structured, quarterly), 2019Q3 .. latest published quarter.

Keeps open-market purchases (code P) and sales (code S) of non-derivative securities with the filing date, the
issuer ticker, the reporting owner's role, shares and price. A filing dated d is treated as usable from the
15:45 decision of the next trading day (filings arrive until 22:00 ET).
SEC requires a contact in the User-Agent (SEC_CONTACT environment variable, set to the owner's research address).
Output: data/local/insider.parquet
"""
import io
import os
import zipfile
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": f"nooptoo research {os.environ.get('SEC_CONTACT', 'incarnadins@gmail.com')}"}
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"


def quarter(q):
    r = requests.get(URL.format(q=q), headers=UA, timeout=300)
    if r.status_code != 200:
        print(q, r.status_code)
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    rd = lambda f, cols: pd.read_csv(z.open(f), sep="\t", usecols=cols, dtype=str, on_bad_lines="skip")
    sub = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"])
    tr = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                   "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    ow = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    ow = ow.drop_duplicates("ACCESSION_NUMBER")
    tr = tr[tr.TRANS_CODE.isin(["P", "S"])]
    d = tr.merge(sub, on="ACCESSION_NUMBER").merge(ow, on="ACCESSION_NUMBER", how="left")
    d = d[d.DOCUMENT_TYPE.isin(["4", "4/A"])]
    d["shares"] = pd.to_numeric(d.TRANS_SHARES, errors="coerce")
    d["price"] = pd.to_numeric(d.TRANS_PRICEPERSHARE, errors="coerce")
    d["value"] = d.shares * d.price
    d["filed"] = pd.to_datetime(d.FILING_DATE, format="%d-%b-%Y", errors="coerce")
    d["ticker"] = d.ISSUERTRADINGSYMBOL.str.upper().str.strip()
    out = d[["filed", "ticker", "TRANS_CODE", "value", "shares", "price", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"]]
    out = out.rename(columns={"TRANS_CODE": "code", "RPTOWNERCIK": "owner", "RPTOWNER_RELATIONSHIP": "role"})
    print(q, len(out), flush=True)
    return out


if __name__ == "__main__":
    qs = [f"{y}q{k}" for y in range(2019, 2027) for k in range(1, 5)]
    qs = [q for q in qs if "2019q3" <= q <= "2026q3"]
    parts = [x for x in (quarter(q) for q in qs) if x is not None]
    d = pd.concat(parts, ignore_index=True).dropna(subset=["filed", "ticker"])
    d.to_parquet(os.path.join(ROOT, "data", "local", "insider.parquet"), index=False)
    print("rows", len(d), d.code.value_counts().to_dict(), d.filed.min(), d.filed.max())
