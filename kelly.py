"""WHAT 100%/MONTH ACTUALLY REQUIRES -- and the fastest legitimate path.

Goal stated: compound $10,000 into $10,000/month. That is +100%/month, which
compounds to 4,096x/year (409,500%/yr).

The governing result is the Kelly criterion. For a strategy with excess return
mu and volatility sigma, levered L times, the LOG growth rate is

    g(L) = L*mu - (L*sigma)^2 / 2

This has a maximum. More leverage does not mean more growth -- past the peak,
variance drag eats the mean and growth FALLS, then goes negative. The peak is

    L* = mu / sigma^2        g(L*) = mu^2 / (2*sigma^2) = SHARPE^2 / 2

That last identity is the important one. The maximum compounding rate any
strategy can achieve, at ANY leverage, is Sharpe^2/2 per year. Leverage cannot
get you past it. Only a higher Sharpe can.

So the question "what Sharpe do I need for 100%/month" has an exact answer,
and this file computes it, then simulates the real verified system to show
what actually happens to $10,000 at each leverage level.
"""
import numpy as np

VER_CAGR, VER_SHARPE = 0.1461, 1.14          # measured, mom_12m + calendar
VER_VOL = VER_CAGR/VER_SHARPE
RF = 0.04                                     # financing cost on leverage

def g(L, mu, sd):
    return L*(mu-RF) + RF - (L*sd)**2/2

print("="*72)
print("1. THE REQUIREMENT")
print("="*72)
tgt_m = 1.00
tgt_y = (1+tgt_m)**12 - 1
gr = np.log(1+tgt_y)
print(f"  target                +{tgt_m*100:.0f}%/month = {(1+tgt_m)**12:,.0f}x/yr "
      f"= {tgt_y*100:,.0f}%/yr")
print(f"  required log growth   g = {gr:.2f}/yr")
print(f"  required Sharpe       sqrt(2g) = {np.sqrt(2*gr):.2f}   (at PERFECT Kelly sizing)")
print()
for nm,s in (("my verified system",VER_SHARPE),("a very good hedge fund",2.0),
             ("Renaissance Medallion (best ever)",2.5),("REQUIRED",np.sqrt(2*gr))):
    print(f"    {nm:<34} Sharpe {s:>5.2f}  ->  max growth {(np.exp(s**2/2)-1)*100:>12,.0f}%/yr")

print("\n"+"="*72)
print("2. THE VERIFIED SYSTEM AT EVERY LEVERAGE  (mu=14.6%, sigma=12.8%)")
print("="*72)
Lstar=(VER_CAGR-RF)/VER_VOL**2
print(f"  Kelly-optimal leverage L* = {Lstar:.1f}x     max growth = "
      f"{(np.exp(g(Lstar,VER_CAGR,VER_VOL))-1)*100:.1f}%/yr\n")
print(f"  {'lev':>5}{'growth/yr':>11}{'$10k in 1yr':>14}{'ann.vol':>9}"
      f"{'est maxDD':>11}{'P(ruin)':>9}")
print("  "+"-"*58)
rng=np.random.default_rng(7)
for L in (1,2,3,5,8,10,15,20,30,50):
    gg=g(L,VER_CAGR,VER_VOL); vol=L*VER_VOL
    # simulate: daily steps, ruin = equity < 10% of start at any point
    n=20000; steps=252
    mu_d=(L*(VER_CAGR-RF)+RF)/252; sd_d=vol/np.sqrt(252)
    paths=rng.standard_normal((n,steps))*sd_d+mu_d
    eq=np.cumprod(1+np.clip(paths,-0.99,None),axis=1)
    ruin=(eq.min(axis=1)<0.10).mean()
    dd=np.median((eq/np.maximum.accumulate(eq,axis=1)).min(axis=1)-1)
    print(f"  {L:>4}x{(np.exp(gg)-1)*100:>10.1f}%{10000*np.exp(gg):>14,.0f}"
          f"{vol*100:>8.0f}%{dd*100:>10.0f}%{ruin*100:>8.1f}%")

print("\n"+"="*72)
print("3. P(reaching $10k/month) -- i.e. growing $10k to ~$1.2M")
print("="*72)
print("  ($10k/month at a sustainable 12%/yr withdrawal needs ~$1,000,000)\n")
print(f"  {'lev':>5}{'1 year':>10}{'3 years':>10}{'5 years':>10}{'10 years':>11}"
      f"{'P(ruin) 5y':>12}")
print("  "+"-"*58)
for L in (1,2,3,5,8,10,20):
    row=[]
    for yrs in (1,3,5,10):
        n=20000; steps=252*yrs
        mu_d=(L*(VER_CAGR-RF)+RF)/252; sd_d=L*VER_VOL/np.sqrt(252)
        p=rng.standard_normal((n,steps))*sd_d+mu_d
        eq=10000*np.cumprod(1+np.clip(p,-0.99,None),axis=1)
        alive=eq.min(axis=1)>1000
        row.append(((eq[:,-1]>=1_000_000)&alive).mean()*100)
        if yrs==5: ruin5=(~alive).mean()*100
    print(f"  {L:>4}x{row[0]:>9.1f}%{row[1]:>9.1f}%{row[2]:>9.1f}%{row[3]:>10.1f}%"
          f"{ruin5:>11.1f}%")
