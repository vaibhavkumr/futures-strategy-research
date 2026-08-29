"""AUDIT A LIVE TRADER'S ACTUAL CALLS, FROM HIS OWN STREAMS.

This is the one form of evidence that beats everything else measured today.
Screenshots can be chosen. Backtests can be wrong. But a livestream is
timestamped, public, and contains the losers as well as the winners -- he
cannot delete a trade he already narrated to an audience.

Method:
  1. Stream metadata gives release_timestamp -- the exact wall-clock moment
     the broadcast began.
  2. The transcript gives an offset (seconds into the stream) for every
     sentence.
  3. start + offset = the absolute minute he said it.
  4. Find where he narrates entries and exits.
  5. Compare against real 1-minute price data for that minute.

What this can establish: whether his calls, taken at the moment he made them,
made or lost money. What it cannot: trades he took without narrating, or
sessions he did not stream.
"""
from __future__ import annotations

import json
import os
import re
import time
import datetime as dt

import numpy as np
import pandas as pd

OUT = "streams"
os.makedirs(OUT, exist_ok=True)

# Language that marks an actual trade action, not analysis. Deliberately
# narrow -- "I like this level" is not a trade, "I'm in" is.
ENTRY = re.compile(
    r"\b(i'?m in\b|i just (got|jumped) in|took (the|a) (trade|long|short)|"
    r"i'?m (long|short)\b|entering (here|now|the trade)|just entered|"
    r"i'?m buying|i'?m selling|order (is )?filled|got filled|we'?re in\b)", re.I)
EXIT = re.compile(
    r"\b(i'?m out\b|closed (it|out|the trade)|took profit|taking profit|"
    r"stopped out|hit my stop|got stopped|scaled out|i'?m flat\b|"
    r"cut (it|the trade)|closing (it|this) (out|now))", re.I)
DIR_LONG = re.compile(r"\b(long|buying|buy side|calls)\b", re.I)
DIR_SHORT = re.compile(r"\b(short|selling|sell side|puts)\b", re.I)


def fetch_stream(vid, meta):
    """Transcript with absolute timestamps."""
    cache = f"{OUT}/{vid}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        try:
            api = YouTubeTranscriptApi()
            tr = api.fetch(vid)
            segs = [(s.start, s.text) for s in tr]
        except Exception:
            tr = YouTubeTranscriptApi.get_transcript(vid)
            segs = [(s["start"], s["text"]) for s in tr]
    except Exception as e:
        return None
    rt = meta.get("rt")
    if not rt:
        return None
    rows = [{"t": rt + off, "text": txt} for off, txt in segs]
    json.dump(rows, open(cache, "w"))
    return rows


def find_calls(rows):
    """Locate narrated entries and exits, with a direction where stated."""
    calls = []
    for i, r in enumerate(rows):
        txt = r["text"]
        is_e = bool(ENTRY.search(txt))
        is_x = bool(EXIT.search(txt))
        if not (is_e or is_x):
            continue
        # direction from a small window around the utterance
        ctx = " ".join(x["text"] for x in rows[max(0, i-4):i+3])
        nl, ns = len(DIR_LONG.findall(ctx)), len(DIR_SHORT.findall(ctx))
        d = 1 if nl > ns else (-1 if ns > nl else 0)
        calls.append(dict(ts=r["t"], kind="ENTRY" if is_e else "EXIT",
                          dir=d, text=txt.strip()[:90], ctx=ctx[:200]))
    return calls


def get_meta(n=25):
    """Metadata for the most recent n streams."""
    cache = "riley_stream_meta_all.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    import yt_dlp
    s = json.load(open("riley_streams.json"))
    alive = [x for x in s if (x.get("dur") or 0) > 600][:n]
    opts = {"quiet": True, "skip_download": True, "ignoreerrors": True,
            "no_warnings": True}
    out = []
    for x in alive:
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                i = y.extract_info(f"https://www.youtube.com/watch?v={x['id']}",
                                   download=False)
            if i and i.get("release_timestamp"):
                out.append(dict(id=x["id"], dur=x["dur"],
                                rt=i["release_timestamp"],
                                title=i.get("title", "")[:70]))
                print(f"  {x['id']}  {dt.datetime.utcfromtimestamp(i['release_timestamp'])} UTC",
                      flush=True)
        except Exception:
            pass
        time.sleep(0.4)
    json.dump(out, open(cache, "w"), indent=0)
    return out


if __name__ == "__main__":
    meta = get_meta(25)
    print(f"\n{len(meta)} streams with start times")
    allcalls = []
    got = 0
    for m in meta:
        rows = fetch_stream(m["id"], m)
        if not rows:
            continue
        got += 1
        c = find_calls(rows)
        for x in c:
            x["vid"] = m["id"]
            x["title"] = m["title"]
        allcalls.extend(c)
        time.sleep(1.5)
    print(f"transcripts fetched: {got}/{len(meta)}")
    print(f"narrated trade actions found: {len(allcalls)}")
    if allcalls:
        df = pd.DataFrame(allcalls)
        df["when"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(
            "America/New_York")
        df.to_pickle("stream_calls.pkl")
        print("\nby kind:", df.kind.value_counts().to_dict())
        print("\nfirst 20 calls:")
        for _, r in df.sort_values("when").head(20).iterrows():
            print(f"  {r.when:%m-%d %H:%M}  {r.kind:<6} dir={r['dir']:+d}  {r.text[:64]}")
