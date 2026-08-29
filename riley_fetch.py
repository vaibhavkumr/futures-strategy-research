"""Pull transcripts for every Riley Coleman video, same as the TJR corpus."""
import json, os, time
from youtube_transcript_api import YouTubeTranscriptApi

vids=json.load(open("riley_videos.json"))
print(f"{len(vids)} videos to fetch", flush=True)
ok=fail=0
try:
    api=YouTubeTranscriptApi(); NEW=True
except Exception:
    NEW=False
for i,v in enumerate(vids):
    vid=v["id"]
    if not vid: continue
    p=f"riley_all/{i:03d}_{vid}.txt"
    if os.path.exists(p): ok+=1; continue
    try:
        if NEW:
            t=api.fetch(vid); segs=[s.text for s in t]
        else:
            t=YouTubeTranscriptApi.get_transcript(vid); segs=[s["text"] for s in t]
        txt=" ".join(segs)
        with open(p,"w",encoding="utf-8") as f:
            f.write(f"### {v['title']}\n### duration_min={(v['dur'] or 0)//60} views={v['views']}\n\n{txt}")
        ok+=1
    except Exception:
        fail+=1
    if (i+1)%25==0:
        print(f"  {i+1}/{len(vids)}  ok={ok} fail={fail}", flush=True)
    time.sleep(0.25)
print(f"DONE ok={ok} fail={fail}", flush=True)
tot=0
for f in os.listdir("riley_all"):
    tot+=len(open(f"riley_all/{f}",encoding="utf-8").read().split())
print(f"TOTAL WORDS: {tot:,}", flush=True)
