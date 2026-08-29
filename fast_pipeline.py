"""FULL-CORPUS STREAM AUDIT -- optimised.

Three speedups over the first version, benchmarked:
  1. BatchedInferencePipeline    38x -> 118x realtime  (3x)
  2. Trim to the TRADING WINDOW. Streams run 84-320 min but trades happen
     between the open and ~11:30 ET. Cutting the rest saves ~40% on the long
     ones and loses nothing.
  3. Download the NEXT stream on a worker thread while the GPU transcribes
     the current one, so network and GPU overlap instead of alternating.

~150 hours of audio -> roughly an hour of GPU time.

Why the full unselected set matters: the 8 streams I picked by title were 6
wins to 2 losses, which measures my title-picking, not his trading. Only the
complete set gives an unbiased read.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import subprocess
import sys
import threading

# trading window, minutes relative to the stream's start
PRE_OPEN_PAD = 20      # start this many min before the 09:30 open
POST_MIN = 150         # ...through ~2.5h after the open
AUD, TXT = "audio", "streams"
os.makedirs(AUD, exist_ok=True)
os.makedirs(TXT, exist_ok=True)


def window_for(rt):
    """Seconds into the stream covering 09:10-11:40 ET."""
    import datetime as dt
    start = dt.datetime.fromtimestamp(rt, dt.timezone.utc)
    et = start.astimezone(dt.timezone(dt.timedelta(hours=-4)))
    open_min = 9 * 60 + 30
    start_min = et.hour * 60 + et.minute
    ss = max((open_min - PRE_OPEN_PAD - start_min) * 60, 0)
    return ss, POST_MIN * 60


def get_audio(vid, rt):
    out = f"{AUD}/{vid}.wav"
    if os.path.exists(out) and os.path.getsize(out) > 1e6:
        return out
    import yt_dlp
    tmp = f"{AUD}/{vid}.raw"
    opts = {"format": "worstaudio/bestaudio", "outtmpl": tmp + ".%(ext)s",
            "quiet": True, "no_warnings": True, "ignoreerrors": True}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([f"https://www.youtube.com/watch?v={vid}"])
    except Exception:
        return None
    got = glob.glob(tmp + ".*")
    if not got:
        return None
    ss, dur = window_for(rt)
    subprocess.run(["ffmpeg", "-y", "-ss", str(ss), "-t", str(dur),
                    "-i", got[0], "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", out], capture_output=True)
    for g in got:
        try:
            os.remove(g)
        except OSError:
            pass
    return out if os.path.exists(out) else None


def main(meta_file="riley_streams_full.json", limit=200):
    allm = json.load(open(meta_file))
    done = {os.path.basename(p)[:-5] for p in glob.glob(f"{TXT}/*.json")}
    todo = [m for m in allm if m["id"] not in done][:limit]
    print(f"{len(done)} already done, {len(todo)} to process", flush=True)
    if not todo:
        return

    # need release timestamps
    import yt_dlp
    opts = {"quiet": True, "skip_download": True, "ignoreerrors": True,
            "no_warnings": True}

    q = queue.Queue(maxsize=2)

    def producer():
        for m in todo:
            try:
                with yt_dlp.YoutubeDL(opts) as y:
                    i = y.extract_info(
                        f"https://www.youtube.com/watch?v={m['id']}", download=False)
                rt = i.get("release_timestamp") if i else None
                if not rt:
                    continue
                w = get_audio(m["id"], rt)
                if w:
                    q.put((m["id"], rt, w, m.get("title", "")))
            except Exception:
                pass
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    from faster_whisper import WhisperModel, BatchedInferencePipeline
    model = WhisperModel("base.en", device="cuda", compute_type="float16")
    bp = BatchedInferencePipeline(model=model)

    n = 0
    while True:
        item = q.get()
        if item is None:
            break
        vid, rt, wav, title = item
        ss, _ = window_for(rt)
        try:
            segs, _ = bp.transcribe(wav, batch_size=16, vad_filter=True)
            rows = [{"t": rt + ss + s.start, "off": round(ss + s.start, 1),
                     "text": s.text.strip()} for s in segs]
            json.dump(rows, open(f"{TXT}/{vid}.json", "w"))
            n += 1
            print(f"  [{n}] {vid}  {len(rows)} segs  {title[:44]}", flush=True)
        except Exception as e:
            print(f"  {vid} FAILED {str(e)[:50]}", flush=True)
        try:
            os.remove(wav)
        except OSError:
            pass
    print(f"DONE {n} streams", flush=True)


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
