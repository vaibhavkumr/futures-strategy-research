"""Transcribe the video with timestamps, using the GPU pipeline built earlier."""
import json, sys
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel("small.en", device="cuda", compute_type="float16")
bp = BatchedInferencePipeline(model=model)
segs, info = bp.transcribe("vid/clip.wav", batch_size=16, vad_filter=True)
rows = []
for s in segs:
    rows.append({"t": round(s.start, 1), "e": round(s.end, 1), "text": s.text.strip()})
    if len(rows) % 40 == 0:
        print(f"  ...{len(rows)} segments, {s.start/60:.1f} min", flush=True)
json.dump(rows, open("vid/transcript.json", "w"), indent=0)
print(f"DONE {len(rows)} segments, {rows[-1]['e']/60:.1f} minutes")
