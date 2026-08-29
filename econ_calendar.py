"""Rule-based US economic calendar. No API, no key, no staleness.

Major releases follow fixed, publicly published schedules. That makes them
deterministic -- we do not need a live feed, we need a rulebook. This is
strictly better than blanket-blocking 08:30 every day, because it only
blocks when something is actually scheduled.

Covered (all times ET):
  08:30  Non-farm payrolls      first Friday of the month
  08:30  CPI                    ~2nd week, Tue-Thu
  08:30  PPI                    ~day after CPI
  08:30  Jobless claims         EVERY Thursday
  08:30  Retail sales           ~mid-month
  08:30  GDP                    last week of Jan/Apr/Jul/Oct
  10:00  ISM manufacturing      1st business day
  10:00  ISM services           3rd business day
  14:00  FOMC decision          8 scheduled meetings/year

NOT covered, and nothing can cover them: unscheduled events. Geopolitics,
surprise Fed speakers, tariff announcements, an exchange outage. Position
sizing is the only defence against those.
"""
from __future__ import annotations
import pandas as pd

# FOMC decision dates are published years in advance by the Fed.
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
             "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"]
FOMC_2027 = ["2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
             "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15"]
FOMC = set(FOMC_2026 + FOMC_2027)

# minutes-from-midnight ET, (start, end, label)
W_0830 = (8 * 60 + 25, 8 * 60 + 45)
W_1000 = (9 * 60 + 55, 10 * 60 + 15)
W_1400 = (13 * 60 + 55, 14 * 60 + 30)


def _nth_weekday(ts, weekday, n):
    """Is ts the n-th `weekday` of its month? (weekday: Mon=0)"""
    if ts.weekday() != weekday:
        return False
    return (ts.day - 1) // 7 == (n - 1)


def scheduled_events(ts: pd.Timestamp) -> list[str]:
    """Which releases are scheduled on this date."""
    ev = []
    d = ts.strftime("%Y-%m-%d")
    dom, wd = ts.day, ts.weekday()
    if _nth_weekday(ts, 4, 1):
        ev.append("NFP")
    if wd == 3:
        ev.append("jobless claims")
    if 10 <= dom <= 15 and wd <= 3:
        ev.append("CPI window")
    if 12 <= dom <= 17 and wd <= 4:
        ev.append("PPI/retail sales window")
    if ts.month in (1, 4, 7, 10) and dom >= 24 and wd <= 4:
        ev.append("GDP window")
    if dom <= 3 and wd <= 4:
        ev.append("ISM mfg")
    if 3 <= dom <= 6 and wd <= 4:
        ev.append("ISM services")
    if d in FOMC:
        ev.append("FOMC")
    return ev


def blackout(ts: pd.Timestamp):
    """Return (blocked, reason). ts must be ET-localised."""
    ev = scheduled_events(ts)
    if not ev:
        return False, ""
    m = ts.hour * 60 + ts.minute
    morning = [e for e in ev if e not in ("ISM mfg", "ISM services", "FOMC")]
    tenam = [e for e in ev if e in ("ISM mfg", "ISM services")]
    if morning and W_0830[0] <= m <= W_0830[1]:
        return True, ", ".join(morning)
    if tenam and W_1000[0] <= m <= W_1000[1]:
        return True, ", ".join(tenam)
    if "FOMC" in ev and W_1400[0] <= m <= W_1400[1]:
        return True, "FOMC"
    return False, ""


if __name__ == "__main__":
    print("Next 14 days of scheduled releases:")
    now = pd.Timestamp.now(tz="America/New_York").normalize()
    blocked_bars = total_bars = 0
    for i in range(14):
        day = now + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue
        ev = scheduled_events(day)
        print(f"  {day:%a %Y-%m-%d}  {', '.join(ev) if ev else '-'}")
    # how much of the trading day does this actually block?
    for i in range(60):
        day = now - pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for mins in range(120, 960, 5):
            t = day + pd.Timedelta(minutes=mins)
            total_bars += 1
            if blackout(t)[0]:
                blocked_bars += 1
    print(f"\nblocks {blocked_bars/total_bars*100:.1f}% of killzone bars "
          f"(vs ~11% for blanket time-based blocking)")
