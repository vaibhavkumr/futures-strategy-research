"""TJR's method built to specification, from the course transcripts.

Read closely rather than keyword-matched this time. The pieces that were
missing or wrong in every earlier version:

  TRADING WINDOW   09:50-10:30 ET ONLY. He marks it and sits on his hands
                   until 09:50 even when a valid setup prints earlier. Our
                   bots traded three killzones spanning 7.5 hours.
  TIMEFRAME BRANCH 1H trend decides the entry chart: aligned with 4H -> 5m,
                   opposed -> 15m. Then one level lower for the trigger.
  THIRD CONFLUENCE after the break of structure, price must push INTO a FVG,
                   order block, breaker block, or equilibrium. Never built.
  SMT DIVERGENCE   NQ sweeps a level while ES does not (or vice versa).
                   A confirmation confluence used constantly. Never built.
  ORDER BLOCK      the 1-3 CONSECUTIVE candles that caused the sweep, before
                   the break of structure. Invalidated by a WICK through it,
                   not just a close.
  BREAKER BLOCK    the opposing leg -- the failed retrace before the sweep.
  EQUILIBRIUM      50% of the current leg, swing high to swing low.
  PARTIALS         exits laddered across successive HTF levels, not one shot.
  STOP             first choice beyond the sweep; fall back to the highs/lows
                   inside the confluence when that ruins the R:R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"

# Funnel counters. TJR gets a setup most days; if our rate is 100x lower the
# fault is in OUR implementation of a step, not in his method. This tells us
# WHICH step starves instead of guessing rule by rule.
FUNNEL = {}


def _tick(k):
    FUNNEL[k] = FUNNEL.get(k, 0) + 1


# ------------------------------------------------------------------ helpers
def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"}).dropna()


def swings(df: pd.DataFrame, lb: int = 1):
    w = 2 * lb + 1
    hi = (df["high"].rolling(w, center=True).max() == df["high"]).to_numpy()
    lo = (df["low"].rolling(w, center=True).min() == df["low"]).to_numpy()
    n = len(df)
    hi[:lb] = hi[n - lb:] = False
    lo[:lb] = lo[n - lb:] = False
    return hi, lo


def trend_series(df: pd.DataFrame, lb: int = 1) -> pd.Series:
    """+1 uptrend / -1 downtrend from higher-highs-and-lows structure.
    Shifted so an unclosed candle is never consulted."""
    sh, sl = swings(df, lb)
    hi = pd.Series(np.where(sh, df["high"], np.nan), index=df.index).ffill()
    lo = pd.Series(np.where(sl, df["low"], np.nan), index=df.index).ffill()
    up = (hi > hi.shift(1)) & (lo > lo.shift(1))
    dn = (hi < hi.shift(1)) & (lo < lo.shift(1))
    t = pd.Series(0.0, index=df.index)
    t[up] = 1
    t[dn] = -1
    return t.replace(0, np.nan).ffill().fillna(0).shift(lb + 1)


def fvgs(h, l, i, direction, look=15):
    """Return (lo, hi) of an unfilled gap near bar i, or None."""
    for k in range(max(i - look, 2), i):
        if direction > 0 and l[k - 2] > h[k]:      # bearish gap -> support once inverted
            return (h[k], l[k - 2])
        if direction < 0 and h[k - 2] < l[k]:
            return (h[k - 2], l[k])
    return None


def inverse_fvg(h, l, c, i, direction, look=15) -> bool:
    """A gap price CLOSED THROUGH -- a failure, not a formation."""
    for k in range(max(i - look, 2), i):
        if direction < 0 and h[k - 2] < l[k] and c[i] < h[k - 2]:
            return True
        if direction > 0 and l[k - 2] > h[k] and c[i] > l[k - 2]:
            return True
    return False


def order_block(o, h, l, c, sweep_i, direction, max_len=3):
    """The 1-3 CONSECUTIVE candles that caused the sweep.
    direction +1 = we are going long, so the OB is the down-leg into the low."""
    want_down = direction > 0
    end = sweep_i
    k = end
    cnt = 0
    while k >= 0 and cnt < max_len:
        is_down = c[k] < o[k]
        if is_down != want_down:
            break
        k -= 1
        cnt += 1
    if cnt == 0:
        return None
    seg = slice(k + 1, end + 1)
    return (float(np.min(l[seg])), float(np.max(h[seg])))


def breaker_block(o, h, l, c, sweep_i, direction, max_len=3):
    """The opposing leg immediately before the order block -- the failed retrace."""
    want_down = direction > 0
    k = sweep_i
    cnt = 0
    while k >= 0 and cnt < max_len and (c[k] < o[k]) == want_down:
        k -= 1
        cnt += 1
    end = k
    cnt2 = 0
    while k >= 0 and cnt2 < max_len and (c[k] < o[k]) != want_down:
        k -= 1
        cnt2 += 1
    if cnt2 == 0 or end < 0:
        return None
    seg = slice(k + 1, end + 1)
    return (float(np.min(l[seg])), float(np.max(h[seg])))


def equilibrium(h, l, lo_i, hi_i):
    """50% of the current leg."""
    if lo_i is None or hi_i is None:
        return None
    a, b = float(l[lo_i]), float(h[hi_i])
    if b <= a:
        return None
    mid = (a + b) / 2
    return (mid - (b - a) * 0.02, mid + (b - a) * 0.02)


# ------------------------------------------------------------- key levels
def key_levels(df5: pd.DataFrame) -> pd.DataFrame:
    """1H/4H swing highs & lows plus Asia and London session extremes.
    Each becomes available only after the period forming it has closed."""
    out = pd.DataFrame(index=df5.index)
    day = df5.index.normalize()
    hh = np.asarray(df5.index.hour)
    mins = hh * 60 + np.asarray(df5.index.minute)

    # sessions: Asia 18:00 -> 03:00, London 03:00 -> 08:30 (indices)
    sess = pd.DatetimeIndex(np.where(hh >= 18, day + pd.Timedelta(days=1), day))
    am = (hh >= 18) | (hh < 3)
    a = df5[am]
    out["asia_h"] = day.map(a.groupby(sess[am])["high"].max())
    out["asia_l"] = day.map(a.groupby(sess[am])["low"].min())
    lm = (mins >= 180) & (mins < 510)
    lo = df5[lm]
    out["lon_h"] = day.map(lo.groupby(lo.index.normalize())["high"].max())
    out["lon_l"] = day.map(lo.groupby(lo.index.normalize())["low"].min())
    for col in ("asia_h", "asia_l"):
        out.loc[mins < 180, col] = np.nan
    for col in ("lon_h", "lon_l"):
        out.loc[mins < 510, col] = np.nan

    for rule, tag in (("1h", "h1"), ("4h", "h4")):
        r = resample(df5, rule)
        sh, sl = swings(r, 1)
        hi = pd.Series(np.where(sh, r["high"], np.nan), index=r.index).ffill().shift(2)
        lw = pd.Series(np.where(sl, r["low"], np.nan), index=r.index).ffill().shift(2)
        out[f"{tag}_h"] = hi.reindex(df5.index, method="ffill")
        out[f"{tag}_l"] = lw.reindex(df5.index, method="ffill")
    return out


def smt_divergence(hA, lA, hB, lB, i, direction, look=12) -> bool:
    """Correlated instrument fails to confirm the sweep.
    Going long: A makes a new low, B does not -> bullish divergence."""
    if i < look:
        return False
    w = slice(i - look, i + 1)
    if direction > 0:
        return bool(lA[i] <= np.min(lA[w]) and lB[i] > np.min(lB[w]))
    return bool(hA[i] >= np.max(hA[w]) and hB[i] < np.max(hB[w]))


# ----------------------------------------------------------------- engine
def generate(df5: pd.DataFrame, df1: pd.DataFrame, df15: pd.DataFrame | None = None,
             corr5: pd.DataFrame | None = None, *,
             win_start=590, win_end=630,        # 09:50 - 10:30 ET, in minutes
             use_smt=True, use_window=True, partials=(0.34, 0.33, 0.33),
             slip_frac=0.05, max_hold=240, session_end=960,
             # --- DISCRETION LAYER: rules he states out loud on the charts ---
             min_risk_atr=0.5,   # "no reason to put your stop so tight"
             min_rr=1.0,         # "only taking risk to rewards 1:1 and higher"
             adr_mult=1.0,       # "price isn't going to move all the way up
                                 #  there and move 2.7% in a day"
             min_confluence=1,   # he says the setup OFFERS three and you pick
                                 # one -- "whatever confluence we want"
             drop_dead_levels=True,   # "already pushed past with no reaction"
             trade_laggard=True):     # "NASDAQ is the lagging index"
    """df5/df1 = 5-minute and 1-minute bars. corr5 = correlated index for SMT."""
    if df15 is None:
        df15 = resample(df5, "15min")
    K = key_levels(df5)
    # average daily range, for "price isn't going that far in a day"
    _d = df5.groupby(df5.index.normalize()).agg(h=("high","max"), l=("low","min"))
    _adr = (_d.h - _d.l).rolling(20).mean().shift(1)
    adr = pd.Series(df5.index.normalize(), index=df5.index).map(_adr).to_numpy()
    # running session extremes, to judge how much of the day's range is used
    _day = df5.index.normalize()
    day_hi = df5.groupby(_day)["high"].cummax().to_numpy()
    day_lo = df5.groupby(_day)["low"].cummin().to_numpy()
    t4 = trend_series(resample(df5, "4h")).reindex(df5.index, method="ffill")
    t1 = trend_series(resample(df5, "1h")).reindex(df5.index, method="ffill")

    frames = {"5m": df5, "15m": df15}
    arr = {k: {x: v[x].to_numpy(float) for x in ("open", "high", "low", "close")}
           for k, v in frames.items()}
    sw = {k: swings(v, 2) for k, v in frames.items()}
    lastsw = {}
    for k, v in frames.items():
        sh, sl = sw[k]
        lastsw[k] = (pd.Series(np.where(sh, v["high"], np.nan)).ffill().shift(2).to_numpy(),
                     pd.Series(np.where(sl, v["low"], np.nan)).ffill().shift(2).to_numpy())

    o1, h1, l1, c1 = (df1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    sh1, sl1 = swings(df1, 1)
    lh1 = pd.Series(np.where(sh1, h1, np.nan)).ffill().shift(2).to_numpy()
    ll1 = pd.Series(np.where(sl1, l1, np.nan)).ffill().shift(2).to_numpy()
    idx1 = df1.index

    o5, h5, l5, c5 = (df5[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    tr = pd.concat([df5["high"] - df5["low"],
                    (df5["high"] - df5["close"].shift()).abs(),
                    (df5["low"] - df5["close"].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy()
    mins5 = np.asarray(df5.index.hour) * 60 + np.asarray(df5.index.minute)

    cH = ("asia_h", "lon_h", "h1_h", "h4_h")
    cL = ("asia_l", "lon_l", "h1_l", "h4_l")
    KV = {k: K[k].to_numpy(float) for k in K.columns}

    hB = lB = None
    if corr5 is not None and use_smt:
        cc = corr5.reindex(df5.index, method="ffill")
        hB, lB = cc["high"].to_numpy(float), cc["low"].to_numpy(float)

    FUNNEL.clear()
    rows = []
    busy_ts = None
    n5 = len(df5)

    for i in range(60, n5 - 1):
        ts = df5.index[i]
        if busy_ts is not None and ts <= busy_ts:
            continue
        if use_window and not (win_start <= mins5[i] <= win_end):
            continue
        _tick("1_in_window")
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        bias = t4[i] if np.isfinite(t4[i]) else 0
        if bias == 0:
            continue
        _tick("2_have_4H_bias")
        sgn = int(np.sign(bias))
        # 1H alignment picks the working timeframe
        aligned = np.isfinite(t1[i]) and np.sign(t1[i]) == sgn
        tf = "5m" if aligned else "15m"

        highs = sorted({KV[k][i] for k in cH if np.isfinite(KV[k][i])})
        lows = sorted({KV[k][i] for k in cL if np.isfinite(KV[k][i])})
        # NOTE: the "already pushed past with no reaction" rule is something he
        # says while marking DRAWS ON LIQUIDITY -- it governs what he AIMS AT,
        # not what he sweeps. Applied here to sweep candidates it forced every
        # entry to be a new high/low of day and cut step 3 from 26% to 2%
        # survival (33 trades in 2.5 years across four markets). It is applied
        # to the target pool instead; sweep candidates are left alone.

        # STEP 3: key level swept, in the direction the bias implies
        if sgn > 0:
            hit = [x for x in lows if l5[i] < x <= l5[i] + 2 * a]
        else:
            hit = [x for x in highs if h5[i] > x >= h5[i] - 2 * a]
        if not hit:
            continue
        _tick("3_key_level_swept")
        sweep_i = i
        sweep_px = l5[i] if sgn > 0 else h5[i]

        # STEP 4: confirmation on the working timeframe
        F = frames[tf]
        oX, hX, lX, cX = (arr[tf][x] for x in ("open", "high", "low", "close"))
        pos = F.index.searchsorted(ts)
        if pos >= len(F) - 3:
            continue
        LH, LL = lastsw[tf]
        conf = None
        for j in range(pos, min(pos + 12, len(F))):
            ref = LH[j] if sgn > 0 else LL[j]
            bos = np.isfinite(ref) and (cX[j] > ref if sgn > 0 else cX[j] < ref)
            ifv = inverse_fvg(hX, lX, cX, j, sgn)
            smt = (hB is not None and smt_divergence(
                (l5 if sgn > 0 else h5), (l5 if sgn > 0 else h5),
                (lB if sgn > 0 else hB), (lB if sgn > 0 else hB), i, sgn))
            if bos or ifv or smt:
                conf = j
                break
        if conf is None:
            continue
        _tick("4_confirmation")

        # STEP 5: third confluence -- price must push INTO one of these
        zones = []
        ob = order_block(oX, hX, lX, cX, min(conf, len(cX) - 1), sgn)
        bb = breaker_block(oX, hX, lX, cX, min(conf, len(cX) - 1), sgn)
        fv = fvgs(hX, lX, min(conf + 1, len(cX) - 1), sgn)
        lo_i = int(np.argmin(lX[max(conf - 20, 0):conf + 1])) + max(conf - 20, 0)
        hi_i = int(np.argmax(hX[max(conf - 20, 0):conf + 1])) + max(conf - 20, 0)
        eq = equilibrium(hX, lX, lo_i, hi_i)
        for z in (ob, bb, fv, eq):
            if z and z[1] > z[0]:
                zones.append(z)
        if not zones:
            continue
        _tick("5_zone_exists")

        touch = None
        n_conf = 0
        for j in range(conf + 1, min(conf + 16, len(F))):
            hits = sum(1 for z in zones if lX[j] <= z[1] and hX[j] >= z[0])
            if hits:
                touch = j
                n_conf = hits
                break
        if touch is None:
            continue
        # he does not take a bare touch -- he wants the zones STACKED
        _tick("6_price_touched_zone")
        if n_conf < min_confluence:
            continue

        # STEP 6: trigger one timeframe lower (5m->1m, 15m->5m)
        t_touch = F.index[touch]
        if tf == "5m":
            k0 = int(idx1.searchsorted(t_touch))
            hh_, ll_, cc_ = h1, l1, c1
            LHt, LLt = lh1, ll1
            tindex, nT = idx1, len(idx1)
            step = 30
        else:
            k0 = int(df5.index.searchsorted(t_touch))
            hh_, ll_, cc_ = h5, l5, c5
            LHt, LLt = lastsw["5m"]
            tindex, nT = df5.index, n5
            step = 12
        e_k = None
        for k in range(k0 + 1, min(k0 + 1 + step, nT)):
            ref = LHt[k] if sgn > 0 else LLt[k]
            bos = np.isfinite(ref) and (cc_[k] > ref if sgn > 0 else cc_[k] < ref)
            if bos or inverse_fvg(hh_, ll_, cc_, k, sgn):
                e_k = k
                break
        if e_k is None:
            continue
        _tick("7_trigger_fired")

        entry = cc_[e_k]
        # stop: beyond the sweep first; else the confluence extreme
        stop = (sweep_px - 0.1 * a) if sgn > 0 else (sweep_px + 0.1 * a)
        risk = abs(entry - stop)
        if risk > 2.5 * a:
            seg = slice(max(k0 - step, 0), e_k + 1)
            stop = (np.min(ll_[seg]) - 0.05 * a) if sgn > 0 else (np.max(hh_[seg]) + 0.05 * a)
            risk = abs(entry - stop)
        # BUG #12: the stop could land on the WRONG SIDE of the entry -- price
        # ran past the sweep between the sweep bar and the trigger, so a short
        # ended up with its stop BELOW entry. risk = abs(entry - stop) hid it,
        # and the exit loop then "stopped out" instantly at a ~+1R PROFIT.
        # 17.6% of trades, 96.7% win rate, t=+69: a pure fake-winner generator.
        # If the stop is not on the losing side, the setup is already invalid.
        if (entry - stop) * sgn <= 0:
            continue
        _tick("8_stop_valid_side")
        if risk < min_risk_atr * a or risk > 3 * a:
            continue
        _tick("9_stop_size_ok")

        # targets: successive HTF levels, taken as partials
        # "price isn't going to move all the way up there and move 2.7% in a
        # day" -- a draw beyond what the day can still travel is not a target.
        day_used = day_hi[i] - day_lo[i]
        room = max(adr[i] - day_used, 0.35 * adr[i]) if np.isfinite(adr[i]) else np.inf
        pool = [x for x in (highs if sgn > 0 else lows)
                if (x - entry) * sgn > 0.4 * risk and abs(x - entry) <= adr_mult * room]
        if drop_dead_levels and i > 0:
            # do not aim at a pool the session has already run through
            pool = [x for x in pool
                    if (x >= day_hi[i - 1] if sgn > 0 else x <= day_lo[i - 1])]
        pool = sorted(pool) if sgn > 0 else sorted(pool, reverse=True)
        if not pool:
            continue
        _tick("10_target_exists")
        # "we are only taking risk to rewards that are 1:1 and higher.
        #  Is this a 1:1? No. So are you going to take this trade? No."
        if (pool[-1] - entry) * sgn < min_rr * risk:
            continue
        _tick("11_rr_ok")
        tgts = pool[:len(partials)]
        while len(tgts) < len(partials):
            tgts.append(tgts[-1])

        slip = slip_frac * a
        filled_R = 0.0
        remaining = 1.0
        done = False
        # BUG #9: this ladder used to be the OUTER `partials` tuple, mutated in
        # place. On a timed-out trade the reset below was skipped by `continue`,
        # so the next trade inherited a half-consumed ladder and could never
        # fully close. Keep it strictly local to the trade.
        ladder = list(partials)
        # BUG #11: max_hold used to be counted in BARS on the trigger timeframe,
        # so the 15m branch (5m trigger) held 240*5min = 20 HOURS -- overnight,
        # collecting bull-market drift TJR never holds for. Count real minutes,
        # and force flat at the session close like a day trader.
        t_entry = tindex[e_k]
        deadline = t_entry + pd.Timedelta(minutes=max_hold)
        k = e_k
        k_last = e_k
        for k in range(e_k + 1, nT):
            tk = tindex[k]
            if tk > deadline or (tk.hour * 60 + tk.minute) >= session_end                     or tk.date() != t_entry.date():
                break
            k_last = k
            if (ll_[k] <= stop if sgn > 0 else hh_[k] >= stop):
                px = stop - sgn * slip
                filled_R += remaining * ((px - entry) * sgn) / risk
                done = True
                busy_ts = tindex[k]
                break
            for ti, tg in enumerate(tgts):
                w = ladder[ti]
                if w <= 0:
                    continue
                if (hh_[k] >= tg if sgn > 0 else ll_[k] <= tg):
                    filled_R += w * ((tg - entry) * sgn) / risk
                    remaining -= w
                    ladder[ti] = 0.0
                    if remaining <= 1e-9:
                        done = True
                        busy_ts = tindex[k]
                    break
            if done:
                break
        # BUG #10: timed-out trades used to be dropped entirely -- survivorship
        # bias, since a trade still open at the horizon is not a non-event.
        # Close the remainder at the last price instead.
        timed_out = not done
        if timed_out:
            k = k_last
            filled_R += remaining * ((cc_[k] - entry) * sgn) / risk
            busy_ts = tindex[k]
        rows.append({"ts": tindex[min(e_k, nT - 1)], "side": "long" if sgn > 0 else "short",
                     "tf": tf, "R": filled_R, "risk_atr": risk / a,
                     "entry": entry, "stop": stop, "risk_pts": risk, "atr": a,
                     "timed_out": timed_out, "n_conf": n_conf,
                     # target distances in R, so a matched control can reuse
                     # the exact same exit ladder with only the side flipped
                     "tgt_R": tuple(round((t - entry) * sgn / risk, 6) for t in tgts),
                     "hold_min": int((tindex[k] - tindex[e_k]).total_seconds() // 60)})
    return pd.DataFrame(rows)
