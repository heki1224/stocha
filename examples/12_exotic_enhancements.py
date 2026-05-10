"""Tutorial 12: Exotic Enhancements — BGK, Seasoning, Rebate (v1.7.1)

Demonstrates the v1.7.1 additions to the exotic-option suite:
1. Broadie-Glasserman-Kou (1997) discrete-monitoring continuity correction for
   barrier options.
2. Seasoning (mid-life pricing) for Asian and Lookback options via the
   `running_avg` / `time_elapsed` and `running_max` / `running_min` arguments.
3. Rebate payments for barrier options (Haug §4.17) with both `paid_at_hit`
   and `paid_at_expiry` timing variants.
"""

import math

import stocha

# === 1. Broadie-Glasserman-Kou Continuity Correction =======================
print("=" * 64)
print("1. BGK discrete-monitoring correction for barrier options")
print("=" * 64)

# Continuous-monitoring analytical (the textbook formula)
cont = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
)
# Daily-monitored variant (252 fixings/year): barrier shifted outward by
# H · exp(+β·σ·√(T/n)) with β = 0.5826.
daily = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out", n_monitoring=252,
)
# Monthly fixings — much coarser, larger correction
monthly = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out", n_monitoring=12,
)
print(f"  UO call (continuous):    {cont:.4f}")
print(f"  UO call (daily, n=252):  {daily:.4f}  (Δ={daily-cont:+.4f})")
print(f"  UO call (monthly, n=12): {monthly:.4f}  (Δ={monthly-cont:+.4f})")
print("  → Discrete monitoring: fewer KOs ⇒ higher price.")

# === 2a. Asian Seasoning ====================================================
print()
print("=" * 64)
print("2a. Asian seasoning — mid-life valuation with running average")
print("=" * 64)

# 6 months in, the running arithmetic average so far is 102 (a touch above strike).
fresh = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
)
seasoned_low = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
    running_avg=98, time_elapsed=0.5,
)
seasoned_high = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
    running_avg=105, time_elapsed=0.5,
)
print(f"  Fresh (no seasoning):    {fresh:.4f}")
print(f"  Seasoned, A=98,  t1=0.5: {seasoned_low:.4f}")
print(f"  Seasoned, A=105, t1=0.5: {seasoned_high:.4f}")
print("  → Higher running average lifts the call value.")

# Deep-ITM: running average so high that K* = (T·K - t1·A)/(T-t1) ≤ 0.
deep_itm_call = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
    running_avg=200, time_elapsed=0.9,
)
deep_itm_put = stocha.asian_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    option_type="put", average_type="geometric", method="analytical",
    running_avg=200, time_elapsed=0.9,
)
print(f"  Deep-ITM call (A=200):   {deep_itm_call:.4f}  (deterministic PV)")
print(f"  Deep-ITM put  (A=200):   {deep_itm_put:.4f}  (always OTM)")

# === 2b. Lookback Seasoning =================================================
print()
print("=" * 64)
print("2b. Lookback seasoning — running max / running min")
print("=" * 64)

# Floating put: as the running max grows, payoff (max - S_T) grows, so price ↑.
fresh_put = stocha.lookback_price(
    s=100, r=0.05, sigma=0.2, t=1.0, option_type="put",
)
seasoned_put = stocha.lookback_price(
    s=100, r=0.05, sigma=0.2, t=1.0, option_type="put",
    running_max=120,
)
print(f"  Floating put (fresh):       {fresh_put:.4f}")
print(f"  Floating put (max=120):     {seasoned_put:.4f}")
print("  → Higher historical max widens the floating put payoff.")

# Fixed call seasoning: running max already above strike → deterministic intrinsic.
fixed_call = stocha.lookback_price(
    s=100, r=0.05, sigma=0.2, t=1.0,
    strike_type="fixed", k=100, running_max=120,
)
intrinsic = (120 - 100) * math.exp(-0.05)
print(f"  Fixed call (M=120, K=100):  {fixed_call:.4f}")
print(f"  Discounted intrinsic alone: {intrinsic:.4f}  ⇒ extra option value above")

# === 3. Barrier Rebate =====================================================
print()
print("=" * 64)
print("3. Barrier rebate (Haug §4.17)")
print("=" * 64)

base = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
)
# KO paid at hit: rebate paid the moment the barrier is touched.
hit = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
    rebate=5, rebate_at_hit=True,
)
# KO paid at expiry: rebate discounted to T regardless of when hit occurs.
expiry = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
    rebate=5, rebate_at_hit=False,
)
print(f"  UO call (no rebate):                {base:.4f}")
print(f"  UO call + R=5 paid-at-hit:          {hit:.4f}  (rebate PV {hit-base:.4f})")
print(f"  UO call + R=5 paid-at-expiry:       {expiry:.4f}  (rebate PV {expiry-base:.4f})")
print("  → Paid-at-hit ≥ paid-at-expiry (earlier money is worth more).")

# Knock-in rebate: paid only if barrier was NEVER touched.
di_base = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=80, barrier_type="down-and-in",
)
di_reb = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=80, barrier_type="down-and-in", rebate=5,
)
print(f"  DI put (no rebate):                 {di_base:.4f}")
print(f"  DI put + R=5 paid-at-expiry-if-not-hit: {di_reb:.4f}  (rebate PV {di_reb-di_base:.4f})")

# Edge case: spot already breached the barrier.
breached = stocha.barrier_price(
    s=125, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
    rebate=5, rebate_at_hit=True,
)
print(f"  UO call, S0=125 (already breached), R=5 @hit: {breached:.4f}  (= R, paid now)")
