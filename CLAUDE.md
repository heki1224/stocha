# stocha — Project Guide for Claude

## Project Overview

Rust-backed Python library for high-performance random number generation and financial simulation.
PyO3 extension module built with maturin. Python 3.14 / PyO3 0.28 / Rust edition 2021.

## Build & Test

```bash
# Build and install into current Python env (required before pytest)
maturin develop --release

# Run tests
pytest tests/

# DO NOT use `cargo test` — PyO3 cdylib crates fail to link against Python
```

## File Structure

```
src/               # Rust source (PyO3 extension)
  lib.rs           # Module registration + into_py_array2 helper
  prng/            # PCG64DXSM RNG
  qrng/            # Sobol / Halton quasi-random sequences
  finance/         # GBM, Heston, jump_diffusion (Merton), Hull-White, SABR, LSMC
  risk/            # VaR / CVaR
  copula/          # Gaussian / Student-t copulas
  dist/            # Probability distributions (norm_ppf, etc.)
python/stocha/
  __init__.py      # Public Python API (wraps Rust via _stocha)
tests/
  test_rng.py      # RNG, Sobol, Halton (src/prng/, src/qrng/)
  test_models.py   # GBM, Heston, Merton, Hull-White, SABR, LSMC (src/finance/)
  test_risk.py     # VaR/CVaR, copulas (src/risk/, src/copula/)
examples/          # Bilingual tutorials (*.py + *.ja.py), 01–07
tasks/lessons.md   # Accumulated lessons from past sessions
CHANGELOG.md       # Version history (source of truth for release notes)
```

## Key Patterns

### PyO3 rules
- `#[staticmethod]` must use concrete type (e.g., `-> PyResult<RNG>`), not `Self` — `Self` silently drops the method from Python bindings
- When adding Rust `#[pymethods]`, also add the corresponding method to the Python wrapper class in `__init__.py`
- GIL release: wrap Rayon parallel blocks with `py.detach(|| ...)`
- RNG stream isolation: use `advance(i * block_size)`, not `seed.wrapping_add(i)`

### Versioning
- Patch releases (x.x.N): bug fixes, code quality
- Minor releases (x.N.0): new features / models
- CHANGELOG.md is maintained in English; release notes are generated from it

### Release workflow
1. Update `version` in `Cargo.toml` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. `git tag vX.Y.Z && git push origin vX.Y.Z`
4. `gh release create vX.Y.Z --notes "..."` (English only)

## Critical Anti-Patterns (from lessons.md)

- `ln(A/B)` ≠ `ln(A)/B` — verify SABR χ(z) sign on OTM strikes immediately after implementation
- faer API: construction is `Mat::from_fn`, reads via `col_as_slice(col)[row]`, no `write`/`read`
- SABR tests: always use `F >> alpha` + near-the-money strikes (low forward + high alpha breaks the Hagan approximation)
- Crate versions: run `cargo search <crate>` before writing Cargo.toml; faer and rand ecosystems evolve fast
- `.gitignore`: use `/dist/` (root-anchored), not bare `dist/` — the latter silently ignores `src/dist/`
- Doc sync: when adding features or bumping versions, update README.md + README.ja.md (Features, Quick Start, API table, Tutorials) and SECURITY.md (Supported Versions) in the same session
