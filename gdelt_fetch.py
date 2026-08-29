"""Pull GDELT news TONE as a timeseries, to test against price.

GDELT is free and historical, which is the only reason this is testable at
all -- Twitter/X full-archive is Enterprise-only at $42k/month, so a social
strategy cannot be backtested and therefore cannot be validated.

Rate limit is one request per 5 seconds, so we fetch in monthly chunks and
cache to disk.
"""
import json, os, time
import pandas as pd
import requests

os.makedirs("gdelt", exist_ok=True)
BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERIES = {
    "market":   "(stock market OR wall street OR nasdaq OR S&P 500)",
    "fed":      "(federal reserve OR interest rates OR inflation)",
    "risk":     "(recession OR selloff OR market crash OR volatility)",
}

def fetch(q, start, end, retries=3):
    p = dict(query=q, mode="timelinetone", startdatetime=start,
             enddatetime=end, format="json")
    for a in range(retries):
        try:
            r = requests.get(BASE, params=p, timeout=60,
                             headers={"User-Agent": "research/1.0"})
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
        except Exception:
            pass
        time.sleep(20)
    return None

def grab(name, q, months):
    out = []
    for (y, m) in months:
        f = f"gdelt/{name}-{y}-{m:02d}.json"
        if os.path.exists(f):
            out.append(json.load(open(f))); continue
        s = f"{y}{m:02d}01000000"
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        e = f"{ny}{nm:02d}01000000"
        d = fetch(q, s, e)
        time.sleep(20)
        if d and d.get("timeline"):
            json.dump(d, open(f, "w")); out.append(d)
            print(f"  {name} {y}-{m:02d}  {len(d['timeline'][0]['data'])} points")
        else:
            print(f"  {name} {y}-{m:02d}  FAILED")
    return out

def to_series(chunks):
    rows = []
    for d in chunks:
        for tl in d.get("timeline", []):
            for pt in tl["data"]:
                rows.append((pt["date"], pt["value"]))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=["date", "tone"])
    s["date"] = pd.to_datetime(s["date"], utc=True, format="mixed")
    return s.drop_duplicates("date").set_index("date")["tone"].sort_index()

if __name__ == "__main__":
    months = [(y, m) for y in (2024, 2025) for m in range(1, 13)]
    for name, q in QUERIES.items():
        print(f"fetching {name}...")
        ch = grab(name, q, months)
        s = to_series(ch)
        if len(s):
            s.to_pickle(f"gdelt/{name}.pkl")
            print(f"  -> {len(s):,} points  {s.index.min()} -> {s.index.max()}")
