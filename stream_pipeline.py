"""AUDIT LIVE-TRADED CALLS: download audio -> transcribe locally -> timestamp.

YouTube IP-blocked the caption endpoint after ~270 requests, but AUDIO
downloads still work. So: pull audio with yt-dlp, transcribe on the GPU with
faster-whisper, and keep per-segment timestamps.

Because stream metadata carries release_timestamp (the exact wall-clock moment
the broadcast began), transcript offset + start time = the real minute he said
something. That makes every narrated entry and exit checkable against actual
price data.

This is the only test that measures the TRADER rather than the published
method. Everything else I have run tests rules; this tests judgement, which is
what every one of these traders says the edge actually is.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

AUD = "audio"
TXT = "streams"
os.makedirs(AUD, exist_ok=True)
os.makedirs(TXT, exist_ok=True)


def get_audio(vid):
    """Download and downsample to 16k mono wav (whisper's native rate)."""
    out = f"{AUD}/{vid}.wav"
    if os.path.exists(out) and os.path.getsize(out) > 1e6:
        return out
    import yt_dlp
    tmp = f"{AUD}/{vid}.raw"
    opts = {"format": "worstaudio/bestaudio", "outtmpl": tmp + ".%(ext)s",
            "quiet": True, "no_warnings": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([f"https://www.youtube.com/watch?v={vid}"])
    got = glob.glob(tmp + ".*")
    if not got:
        return None
    subprocess.run(["ffmpeg", "-y", "-i", got[0], "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", out],
                   capture_output=True)
    for g in got:
        try:
            os.remove(g)
        except OSError:
            pass
    return out if os.path.exists(out) else None


_MODEL = None


def transcribe(wav, rt):
    """Segments with ABSOLUTE unix timestamps."""
    global _MODEL
    from faster_whisper import WhisperModel
    if _MODEL is None:
        _MODEL = WhisperModel("base.en", device="cuda", compute_type="float16")
    segs, _ = _MODEL.transcribe(wav, beam_size=1, vad_filter=True)
    return [{"t": rt + s.start, "off": round(s.start, 1),
             "text": s.text.strip()} for s in segs]


if __name__ == "__main__":
    n_do = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    src = sys.argv[2] if len(sys.argv) > 2 else "riley_stream_meta_all.json"
    meta = json.load(open(src))[:n_do]
    for i, m in enumerate(meta):
        p = f"{TXT}/{m['id']}.json"
        if os.path.exists(p):
            print(f"[{i+1}/{len(meta)}] {m['id']} cached", flush=True)
            continue
        print(f"[{i+1}/{len(meta)}] {m['id']} downloading...", flush=True)
        wav = get_audio(m["id"])
        if not wav:
            print("    download failed", flush=True)
            continue
        print(f"    transcribing ({os.path.getsize(wav)/1e6:.0f} MB)...", flush=True)
        rows = transcribe(wav, m["rt"])
        json.dump(rows, open(p, "w"))
        words = sum(len(r["text"].split()) for r in rows)
        print(f"    {len(rows)} segments, {words:,} words", flush=True)
        try:
            os.remove(wav)          # audio is large; transcript is what we need
        except OSError:
            pass
    print("DONE", flush=True)
