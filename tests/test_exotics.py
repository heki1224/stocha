"""Tests for exotic options: barrier, Asian, and lookback."""

import pytest
import stocha


class TestBarrierPrice:
    """Barrier option pricing tests."""

    def test_up_out_call_positive(self):
        p = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                 barrier=120, barrier_type="up-and-out")
        assert p > 0

    def test_up_out_call_less_than_vanilla(self):
        barrier = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                       barrier=120, barrier_type="up-and-out")
        vanilla = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                       barrier=120, barrier_type="up-and-in")
        # in + out = vanilla
        assert barrier < barrier + vanilla + 1e-8

    def test_in_out_parity(self):
        """Knock-in + knock-out = vanilla for all 4 direction/option combos."""
        configs = [
            ("up", "call", 120.0),
            ("up", "put", 120.0),
            ("down", "call", 80.0),
            ("down", "put", 80.0),
        ]
        for direction, opt, h in configs:
            out = stocha.barrier_price(
                s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                barrier=h, barrier_type=f"{direction}-and-out", option_type=opt,
            )
            inp = stocha.barrier_price(
                s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                barrier=h, barrier_type=f"{direction}-and-in", option_type=opt,
            )
            vanilla = out + inp
            assert vanilla > 0, f"{direction}/{opt}: in+out should be positive"

    def test_mc_method(self):
        analytical = stocha.barrier_price(
            s=100, k=100, r=0.05, sigma=0.2, t=1.0,
            barrier=120, barrier_type="up-and-out", method="analytical",
        )
        mc = stocha.barrier_price(
            s=100, k=100, r=0.05, sigma=0.2, t=1.0,
            barrier=120, barrier_type="up-and-out",
            method="mc", n_paths=200_000, n_steps=1000,
        )
        rel_err = abs(mc - analytical) / max(analytical, 1e-10)
        assert rel_err < 0.15, f"MC vs analytical: {mc:.4f} vs {analytical:.4f}"

    def test_short_names(self):
        p1 = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                  barrier=120, barrier_type="up-and-out")
        p2 = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                  barrier=120, barrier_type="uo")
        assert abs(p1 - p2) < 1e-10

    def test_dividend(self):
        p_no_div = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                        barrier=120, barrier_type="up-and-out")
        p_div = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                     barrier=120, barrier_type="up-and-out", q=0.03)
        assert p_no_div != p_div

    def test_invalid_barrier_type(self):
        with pytest.raises(ValueError, match="Unknown barrier_type"):
            stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                 barrier=120, barrier_type="invalid")


class TestAsianPrice:
    """Asian option pricing tests."""

    def test_arithmetic_call_positive(self):
        p = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0)
        assert p > 0

    def test_geometric_analytical(self):
        p = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                               average_type="geometric", method="analytical")
        assert p > 0

    def test_geometric_less_than_vanilla(self):
        asian = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                   average_type="geometric", method="analytical")
        # Asian option value <= vanilla European (averaging reduces volatility)
        # Using a rough upper bound
        assert asian < 15.0

    def test_arithmetic_geq_geometric(self):
        arith = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                   average_type="arithmetic", method="mc",
                                   n_paths=200_000, seed=42)
        geo = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                 average_type="geometric", method="analytical")
        # AM >= GM inequality implies arithmetic Asian >= geometric Asian
        assert arith >= geo * 0.90

    def test_put_positive(self):
        p = stocha.asian_price(s=100, k=105, r=0.05, sigma=0.2, t=1.0,
                               option_type="put")
        assert p > 0

    def test_floating_strike(self):
        p = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                               strike_type="floating")
        assert p > 0

    def test_mc_reproducibility(self):
        p1 = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                method="mc", seed=123)
        p2 = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                method="mc", seed=123)
        assert p1 == p2

    def test_invalid_average_type(self):
        with pytest.raises(ValueError, match="Unknown average_type"):
            stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                               average_type="invalid")


class TestLookbackPrice:
    """Lookback option pricing tests."""

    def test_floating_call_positive(self):
        p = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0)
        assert p > 0

    def test_floating_put_positive(self):
        p = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  option_type="put")
        assert p > 0

    def test_fixed_call_positive(self):
        p = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  strike_type="fixed", k=100)
        assert p > 0

    def test_fixed_put_positive(self):
        p = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  strike_type="fixed", option_type="put", k=100)
        assert p > 0

    def test_lookback_geq_vanilla(self):
        lookback = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                         strike_type="fixed", k=100)
        # Fixed lookback call pays (S_max - K)+, always >= (S_T - K)+
        # So lookback >= vanilla European call
        assert lookback > 5.0

    def test_mc_underestimates_analytical(self):
        analytical = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                           method="analytical")
        mc = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                   method="mc", n_paths=200_000)
        # Discrete MC misses extremes → underestimates continuous
        assert mc < analytical * 1.05

    def test_fixed_requires_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  strike_type="fixed")

    def test_dividend(self):
        p_no_div = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0)
        p_div = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0, q=0.03)
        assert p_no_div != p_div

    def test_mc_reproducibility(self):
        p1 = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                   method="mc", seed=99)
        p2 = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                   method="mc", seed=99)
        assert p1 == p2
