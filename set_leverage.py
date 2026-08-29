"""Change leverage on the running book, correctly.

Two things have to happen together, or the accounting silently breaks:

  1. RE-BASELINE. mark() computes equity as anchor*(1 + realised + lev*move),
     where `move` is measured from entry prices. If leverage changes mid
     holding period, the old move would be re-priced at the new leverage --
     applying 20x to gains that were actually earned at 30x. So current
     equity is banked as the new anchor and entry prices are reset to today's
     prices. The P/L clock restarts; nothing already earned is restated.

  2. CHARGE THE DE-LEVERING TRADE. Going 30x -> 20x means selling 10x equity
     of notional. At 10bp that is a real 1.0% cost and it is charged here
     rather than quietly ignored.
"""
import json
import sys

import yfinance as yf

NEW_LEV = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
COST_BP = 10.0

s = json.load(open("moonshot_state.json"))
old = s["lev"]
eq0 = s["equity"]

# cost of trading the difference in notional
notional_traded = abs(old - NEW_LEV)          # in units of equity
cost = notional_traded * COST_BP / 1e4
s["equity"] = eq0 * (1 - cost)

# refresh entry prices to now, so `move` restarts from zero at the new leverage
tk = list(s["weights"])
px = yf.download(tk, period="5d", progress=False, auto_adjust=True)["Close"]
last = px.ffill().iloc[-1]
s["entry_px"] = {a: float(last[a]) for a in tk if a in last and last[a] == last[a]}
s["last_px"] = dict(s["entry_px"])

s["lev"] = NEW_LEV
s["anchor"] = s["equity"]
s["realised"] = 0.0

json.dump(s, open("moonshot_state.json", "w"), indent=1)

print(f"  leverage   {old:.0f}x  ->  {NEW_LEV:.0f}x")
print(f"  de-lever   sold {notional_traded:.0f}x equity of notional, "
      f"cost {cost*100:.2f}% = ${eq0*cost:,.2f}")
print(f"  equity     ${eq0:,.2f}  ->  ${s['equity']:,.2f}")
print(f"  anchor     ${s['anchor']:,.2f}   realised reset to 0")
print(f"  re-based   {len(s['entry_px'])} entry prices to today's close")
