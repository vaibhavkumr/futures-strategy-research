"""Rebuild moonshot state under the corrected accounting.

The USO gap loss was REAL and is kept in full. What is removed is the
artifact from the old mark(), which re-levered the book to 30x every 15
minutes and manufactured volatility drag that no real position would pay.
"""
import json

s = json.load(open("moonshot_state.json"))

s["anchor"] = 9700.0                        # equity after the entry cost
uso_w, uso_draw = 0.0721, -0.0686           # from the POSITION_STOP log line
lev = s["lev"]
s["realised"] = uso_w * uso_draw * lev - uso_w * 10.0 / 1e4 * lev
s["equity"] = s["anchor"] * (1 + s["realised"])
s["peak"] = 10000.0
s["weights"].pop("USO", None)
s.get("entry_px", {}).pop("USO", None)

json.dump(s, open("moonshot_state.json", "w"), indent=1)

print(f"  anchor    ${s['anchor']:,.2f}   (start 10,000 less 3% entry cost)")
print(f"  realised  {s['realised']*100:+.2f}%  = USO gap banked as a closed loss")
print(f"  equity    ${s['equity']:,.2f}  before open positions are marked")
print(f"  positions {len(s['weights'])} held")
