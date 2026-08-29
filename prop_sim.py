"""Can a funded account get to $500/day? Simulate the actual rules.

Prop firms remove the CAPITAL constraint -- you trade $50k-150k for a ~$150
fee. They do not remove the EDGE constraint, and their drawdown limits are
tight enough that the difference between edge levels decides everything.

Rules modelled (typical 2026 $50k evaluation):
  profit target      $3,000
  trailing drawdown  $2,500  (trails your PEAK balance, not your start)
  min winning days   5
Then a funded account with the same trailing drawdown.
"""
import numpy as np

TARGET, DD, MIN_WIN_DAYS = 3000.0, 2500.0, 5
TRADES_PER_DAY = 3


def run(edge_R, risk_dollars, n_sims=20000, max_days=120, seed=0):
    """Returns (pass_rate, blow_rate, median_days_to_pass)."""
    rng = np.random.default_rng(seed)
    passed = blown = 0
    days_list = []
    for _ in range(n_sims):
        bal, peak, win_days = 0.0, 0.0, 0
        for day in range(1, max_days + 1):
            start = bal
            for _ in range(TRADES_PER_DAY):
                # R outcomes with the given expectancy; 1R win / 1R loss shape
                p = 0.5 + edge_R / 2.0
                bal += risk_dollars if rng.random() < p else -risk_dollars
                peak = max(peak, bal)
                if bal <= peak - DD:
                    break
            if bal <= peak - DD:
                blown += 1
                break
            if bal - start > 0:
                win_days += 1
            if bal >= TARGET and win_days >= MIN_WIN_DAYS:
                passed += 1
                days_list.append(day)
                break
    return (passed / n_sims, blown / n_sims,
            float(np.median(days_list)) if days_list else float("nan"))


if __name__ == "__main__":
    print("=" * 76)
    print("$50k EVALUATION -- profit target $3,000, trailing drawdown $2,500")
    print("=" * 76)
    print(f"{'edge/trade':<26}{'risk':<9}{'PASS':>8}{'blow up':>10}{'median days':>13}")
    print("-" * 76)
    cases = [
        ("-0.10R  (our TJR bot)", -0.10),
        (" 0.00R  (coin flip)",    0.00),
        ("+0.05R", 0.05),
        ("+0.10R", 0.10),
        ("+0.20R  (elite)", 0.20),
    ]
    for label, e in cases:
        for risk in (250, 500):
            p, b, d = run(e, risk)
            print(f"{label:<26}${risk:<8}{p*100:>7.1f}%{b*100:>9.1f}%{d:>13.0f}")
        print()

    print("=" * 76)
    print("IF FUNDED: what does $500/day require? (funded $50k, DD $2,500)")
    print("=" * 76)
    print("  $500/day at 3 trades/day = $167 net per trade.")
    for e in (0.05, 0.10, 0.20, 0.30):
        need = 167.0 / e if e else float("inf")
        # how many consecutive-loss dollars can you absorb before the DD stops you
        room = DD / need
        print(f"   edge {e:+.2f}R -> risk ${need:6.0f}/trade -> only "
              f"{room:4.1f} losing trades of room before the account is dead")
    print("\n  A 12-trade losing streak is routine at any of these edges.")
