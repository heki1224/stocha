"""Tutorial 11: Exotic Options — Barrier, Asian, and Lookback

Demonstrates:
1. Barrier options (8 types, analytical vs MC, in-out parity)
2. Asian options (geometric analytical, arithmetic MC with CV)
3. Lookback options (floating & fixed strike, analytical vs MC)
4. Method selection: auto, analytical, mc
"""

import stocha

# === 1. Barrier Options =====================================================
print("=" * 60)
print("1. Barrier Options")
print("=" * 60)

# Up-and-out call: knocked out if S >= 120
uo_call = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-out",
)
print(f"  Up-and-out call (H=120):  {uo_call:.4f}")

# Up-and-in call: activated if S >= 120
ui_call = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-in",
)
print(f"  Up-and-in call  (H=120):  {ui_call:.4f}")
print(f"  In + Out = Vanilla:       {uo_call + ui_call:.4f}")

# Down-and-out put: knocked out if S <= 80
do_put = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=80.0, barrier_type="down-and-out", option_type="put",
)
print(f"  Down-and-out put (H=80):  {do_put:.4f}")

# Down-and-in put
di_put = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=80.0, barrier_type="down-and-in", option_type="put",
)
print(f"  Down-and-in put  (H=80):  {di_put:.4f}")
print(f"  In + Out = Vanilla:       {do_put + di_put:.4f}")

# Analytical vs MC comparison
print("\n  Analytical vs MC:")
analytical = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-out", method="analytical",
)
mc = stocha.barrier_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    barrier=120.0, barrier_type="up-and-out",
    method="mc", n_paths=500_000, n_steps=1000,
)
print(f"  Analytical: {analytical:.4f}")
print(f"  MC:         {mc:.4f}")

# === 2. Asian Options =======================================================
print("\n" + "=" * 60)
print("2. Asian Options")
print("=" * 60)

# Geometric average call (analytical, Kemna-Vorst)
geo_call = stocha.asian_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    average_type="geometric", method="analytical",
)
print(f"  Geometric call (analytical): {geo_call:.4f}")

# Arithmetic average call (MC with geometric CV)
arith_call = stocha.asian_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    average_type="arithmetic", n_paths=200_000,
)
print(f"  Arithmetic call (MC+CV):     {arith_call:.4f}")

# Arithmetic is always >= geometric (AM-GM inequality)
print(f"  Arith >= Geo? {arith_call >= geo_call}")

# Floating strike Asian
float_asian = stocha.asian_price(
    s=100.0, k=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="floating", n_paths=200_000,
)
print(f"  Floating strike call (MC):   {float_asian:.4f}")

# Asian put
asian_put = stocha.asian_price(
    s=100.0, k=105.0, r=0.05, sigma=0.2, t=1.0,
    option_type="put", n_paths=200_000,
)
print(f"  Arithmetic put (K=105, MC):  {asian_put:.4f}")

# === 3. Lookback Options ====================================================
print("\n" + "=" * 60)
print("3. Lookback Options")
print("=" * 60)

# Floating strike call: payoff = S_T - S_min (analytical)
float_call = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="floating", method="analytical",
)
print(f"  Floating call (analytical): {float_call:.4f}")

# Floating strike put: payoff = S_max - S_T
float_put = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="floating", option_type="put", method="analytical",
)
print(f"  Floating put  (analytical): {float_put:.4f}")

# Fixed strike call: payoff = (S_max - K)+
fixed_call = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="fixed", k=100.0, method="analytical",
)
print(f"  Fixed call K=100 (analytical): {fixed_call:.4f}")

# Fixed strike put: payoff = (K - S_min)+
fixed_put = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    strike_type="fixed", option_type="put", k=100.0, method="analytical",
)
print(f"  Fixed put  K=100 (analytical): {fixed_put:.4f}")

# Continuous (analytical) vs discrete (MC) comparison
mc_float = stocha.lookback_price(
    s=100.0, r=0.05, sigma=0.2, t=1.0,
    method="mc", n_paths=500_000,
)
print(f"\n  Continuous vs Discrete monitoring:")
print(f"  Analytical (continuous):  {float_call:.4f}")
print(f"  MC (discrete, 252 steps): {mc_float:.4f}")
print(f"  Discrete underestimates by {(1 - mc_float / float_call) * 100:.1f}%")

# === 4. Effect of Dividends =================================================
print("\n" + "=" * 60)
print("4. Effect of Dividends (q=3%)")
print("=" * 60)

barrier_no_q = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out",
)
barrier_q = stocha.barrier_price(
    s=100, k=100, r=0.05, sigma=0.2, t=1.0,
    barrier=120, barrier_type="up-and-out", q=0.03,
)
print(f"  Barrier (q=0): {barrier_no_q:.4f}  (q=3%): {barrier_q:.4f}")

lookback_no_q = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0)
lookback_q = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0, q=0.03)
print(f"  Lookback (q=0): {lookback_no_q:.4f}  (q=3%): {lookback_q:.4f}")
