"""
benchmark.py — stocha vs NumPy / SciPy performance comparison

Usage:
    maturin develop --release
    python examples/benchmark.py
"""

import sys
import time
import numpy as np
import stocha

# ── helpers ──────────────────────────────────────────────────────────────────

def measure(fn, n_repeat=5):
    """Return minimum wall-clock time over n_repeat runs (seconds)."""
    fn()  # warmup
    return min(time.perf_counter() - (t := time.perf_counter()) or
               (time.perf_counter() - t)
               for _ in [fn() for _ in range(n_repeat)])


def _measure(fn, n_repeat=5):
    fn()  # warmup
    times = []
    for _ in range(n_repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def row(label, n, t_stocha, t_other):
    s_stocha = n / t_stocha / 1e6
    s_other  = n / t_other  / 1e6
    speedup  = t_other / t_stocha
    return f"  {label:<28} {n:>9,}  {s_stocha:>8.2f} M/s  {s_other:>8.2f} M/s  {speedup:>6.2f}x"


def header():
    return (
        f"  {'Benchmark':<28} {'N':>9}  {'stocha':>12}  {'baseline':>12}  {'speedup':>7}\n"
        + "  " + "-" * 75
    )


# ── environment ───────────────────────────────────────────────────────────────

print("\n=== Environment ===")
print(f"  Python  : {sys.version.split()[0]}")
print(f"  NumPy   : {np.__version__}")
try:
    import scipy
    print(f"  SciPy   : {scipy.__version__}")
    HAS_SCIPY = True
except ImportError:
    print("  SciPy   : not installed (Sobol/Halton comparison skipped)")
    HAS_SCIPY = False

# ── benchmarks ────────────────────────────────────────────────────────────────

print("\n=== Results ===")
print(header())

for N in [100_000, 1_000_000]:

    # ── Normal distribution ──────────────────────────────────────────────────
    rng_s = stocha.RNG(seed=42)
    rng_n = np.random.default_rng(42)

    t_s = _measure(lambda: rng_s.normal(size=N))
    t_n = _measure(lambda: rng_n.standard_normal(N))
    print(row("normal()", N, t_s, t_n))

    # ── GBM ──────────────────────────────────────────────────────────────────
    STEPS = 252
    N_PATHS = N // 10  # keep memory reasonable

    def gbm_stocha():
        stocha.gbm(s0=100.0, mu=0.05, sigma=0.20,
                   t=1.0, steps=STEPS, n_paths=N_PATHS, seed=42)

    def gbm_numpy():
        rng = np.random.default_rng(42)
        dt = 1.0 / STEPS
        z = rng.standard_normal((N_PATHS, STEPS))
        log_ret = (0.05 - 0.5 * 0.04) * dt + 0.20 * np.sqrt(dt) * z
        paths = 100.0 * np.exp(np.cumsum(log_ret, axis=1))
        return np.concatenate([np.full((N_PATHS, 1), 100.0), paths], axis=1)

    t_s = _measure(gbm_stocha)
    t_n = _measure(gbm_numpy)
    label = f"gbm({N_PATHS:,} paths, {STEPS} steps)"
    s_stocha = N_PATHS * STEPS / t_s / 1e6
    s_other  = N_PATHS * STEPS / t_n / 1e6
    speedup  = t_n / t_s
    print(f"  {label:<28} {'':>9}  {s_stocha:>8.2f} M/s  {s_other:>8.2f} M/s  {speedup:>6.2f}x")

    # ── Sobol ────────────────────────────────────────────────────────────────
    if HAS_SCIPY:
        from scipy.stats.qmc import Sobol as ScipySobol

        def sobol_stocha():
            stocha.sobol(dim=4, n_samples=N)

        def sobol_scipy():
            sampler = ScipySobol(d=4, scramble=False, seed=42)
            sampler.random(N)

        t_s = _measure(sobol_stocha)
        t_n = _measure(sobol_scipy)
        print(row("sobol(dim=4)", N, t_s, t_n))

    # ── Halton ───────────────────────────────────────────────────────────────
    if HAS_SCIPY:
        from scipy.stats.qmc import Halton as ScipyHalton

        def halton_stocha():
            stocha.halton(dim=4, n_samples=N)

        def halton_scipy():
            sampler = ScipyHalton(d=4, scramble=False)
            sampler.random(N)

        t_s = _measure(halton_stocha)
        t_n = _measure(halton_scipy)
        print(row("halton(dim=4)", N, t_s, t_n))

    print()

print("speedup > 1.0x means stocha is faster.")
