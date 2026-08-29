"""Retry the livestream transcript pull until YouTube's IP block clears.

~270 caption requests today triggered the ban. It affects youtube-transcript-api,
the signed timedtext URLs, and the browser route is blocked by policy. Bans
usually clear within hours, so poll patiently rather than hammering.
"""
import json, os, time, datetime as dt
from youtube_transcript_api import YouTubeTranscriptApi

meta = json.load(open("riley_stream_meta_all.json"))
os.makedirs("streams", exist_ok=True)

def try_one(vid):
    try:
        api = YouTubeTranscriptApi()
        return [(s.start, s.text) for s in api.fetch(vid)]
    except Exception as e:
        return type(e).__name__

wait = 600            # 10 min between probes
for attempt in range(1, 40):
    probe = try_one(meta[0]["id"])
    stamp = dt.datetime.now().strftime("%H:%M")
    if isinstance(probe, str):
        print(f"[{stamp}] attempt {attempt}: still blocked ({probe})", flush=True)
        time.sleep(wait)
        continue
    print(f"[{stamp}] UNBLOCKED -- pulling {len(meta)} streams", flush=True)
    got = 0
    for m in meta:
        p = f"streams/{m['id']}.json"
        if os.path.exists(p):
            got += 1; continue
        r = try_one(m["id"])
        if isinstance(r, str):
            print(f"   {m['id']}: {r}", flush=True); time.sleep(30); continue
        json.dump([{"t": m["rt"] + off, "text": txt} for off, txt in r], open(p, "w"))
        got += 1
        print(f"   {m['id']} ok ({len(r)} segs)", flush=True)
        time.sleep(4)
    print(f"DONE {got}/{len(meta)} streams cached", flush=True)
    break
