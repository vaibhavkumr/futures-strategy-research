"""Local transcription -- bypasses YouTube entirely.

faster-whisper on the GPU. Keeps per-segment timestamps, which matters for
live-trading video: knowing WHEN he says "I'm in" lets the call be checked
against real price.
"""
import sys, json
from faster_whisper import WhisperModel

wav = sys.argv[1] if len(sys.argv) > 1 else "scarface.wav"
out = sys.argv[2] if len(sys.argv) > 2 else "scarface"

m = WhisperModel("base.en", device="cuda", compute_type="float16")
segs, info = m.transcribe(wav, beam_size=5, vad_filter=True)
rows = []
for s in segs:
    rows.append({"t": round(s.start, 1), "end": round(s.end, 1),
                 "text": s.text.strip()})
json.dump(rows, open(f"{out}.json", "w"), indent=0)
txt = " ".join(r["text"] for r in rows)
open(f"{out}.txt", "w", encoding="utf-8").write(txt)
print(f"{len(rows)} segments, {len(txt.split()):,} words, "
      f"{info.duration/60:.1f} min audio")
