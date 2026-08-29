"""Grab queued videos + the 25 livestreams the moment YouTube unblocks."""
import json, os, time, datetime as dt
from youtube_transcript_api import YouTubeTranscriptApi
os.makedirs("streams", exist_ok=True)
meta = json.load(open("riley_stream_meta_all.json"))
pend = json.load(open("pending_videos.json"))

def pull(vid):
    try:
        return [(s.start, s.text) for s in YouTubeTranscriptApi().fetch(vid)]
    except Exception as e:
        return type(e).__name__

for attempt in range(1, 200):
    probe = pull(pend[0])
    stamp = dt.datetime.now().strftime("%m-%d %H:%M")
    if isinstance(probe, str):
        print(f"[{stamp}] attempt {attempt}: blocked ({probe})", flush=True)
        time.sleep(900)                       # 15 min
        continue
    print(f"[{stamp}] UNBLOCKED", flush=True)
    for v in pend:
        r = pull(v)
        if not isinstance(r, str):
            open(f"yt_{v}.txt", "w", encoding="utf-8").write(
                " ".join(x for _, x in r))
            print(f"   video {v}: {len(r)} segs", flush=True)
        time.sleep(5)
    for m in meta:
        p = f"streams/{m['id']}.json"
        if os.path.exists(p):
            continue
        r = pull(m["id"])
        if isinstance(r, str):
            time.sleep(30); continue
        json.dump([{"t": m["rt"]+o, "text": t} for o, t in r], open(p, "w"))
        print(f"   stream {m['id']} ok", flush=True)
        time.sleep(5)
    print("ALL DONE", flush=True)
    break
