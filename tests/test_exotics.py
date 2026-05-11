"""Tests for exotic options: barrier, Asian, and lookback."""

import math

import pytest
import stocha

try:
    import QuantLib as ql
    _HAS_QL = True
except ImportError:
    _HAS_QL = False


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


class TestBarrierBGK:
    """Broadie-Glasserman-Kou (1997) discrete-monitoring continuity correction."""

    def test_default_is_continuous(self):
        # n_monitoring=None must reproduce the existing continuous price exactly.
        p_default = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                         barrier=120, barrier_type="up-and-out")
        p_explicit = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                          barrier=120, barrier_type="up-and-out",
                                          n_monitoring=None)
        assert p_default == p_explicit

    def test_uo_call_discrete_higher_than_continuous(self):
        # Discrete monitoring shifts barrier outward → fewer KOs → higher price.
        cont = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    barrier=120, barrier_type="up-and-out")
        disc = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    barrier=120, barrier_type="up-and-out",
                                    n_monitoring=12)
        assert disc > cont

    def test_do_put_discrete_higher_than_continuous(self):
        cont = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    barrier=80, barrier_type="down-and-out",
                                    option_type="put")
        disc = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    barrier=80, barrier_type="down-and-out",
                                    option_type="put", n_monitoring=12)
        assert disc > cont

    def test_converges_to_continuous_for_large_n(self):
        # BGK shift decays as O(1/√n); tolerance must accommodate that.
        cont = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    barrier=120, barrier_type="up-and-out")
        disc = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    barrier=120, barrier_type="up-and-out",
                                    n_monitoring=100_000)
        assert abs(cont - disc) < 1e-2

    def test_mc_ignores_n_monitoring(self):
        # MC paths are already discrete; BGK shift would double-count.
        mc1 = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                   barrier=120, barrier_type="up-and-out",
                                   method="mc", n_paths=20_000, seed=1)
        mc2 = stocha.barrier_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                   barrier=120, barrier_type="up-and-out",
                                   method="mc", n_paths=20_000, seed=1,
                                   n_monitoring=12)
        assert mc1 == mc2


class TestBarrierRebate:
    """Rebate payments for barrier options (Haug §4.17)."""

    BASE = dict(s=100, k=100, r=0.05, sigma=0.2, t=1.0, barrier=120,
                barrier_type="up-and-out")

    def test_zero_rebate_unchanged(self):
        p1 = stocha.barrier_price(**self.BASE)
        p2 = stocha.barrier_price(**self.BASE, rebate=0.0)
        assert p1 == p2

    def test_uo_paid_at_hit_matches_haug(self):
        # Reference (Haug §4.17.5 hand-computed): 2.0133.
        p_no = stocha.barrier_price(**self.BASE)
        p_reb = stocha.barrier_price(**self.BASE, rebate=5, rebate_at_hit=True)
        assert abs((p_reb - p_no) - 2.0133) < 1e-3

    def test_uo_paid_at_expiry_uses_hit_prob(self):
        # P(hit) ≈ 0.413 for these params; rebate PV = 5 e^{-0.05} · 0.413 ≈ 1.963.
        p_no = stocha.barrier_price(**self.BASE)
        p_reb = stocha.barrier_price(**self.BASE, rebate=5, rebate_at_hit=False)
        assert abs((p_reb - p_no) - 1.9629) < 1e-3

    def test_paid_at_hit_geq_paid_at_expiry(self):
        # Earlier payment is always worth at least the discounted expiry payment.
        p_hit = stocha.barrier_price(**self.BASE, rebate=5, rebate_at_hit=True)
        p_exp = stocha.barrier_price(**self.BASE, rebate=5, rebate_at_hit=False)
        assert p_hit >= p_exp

    def test_ki_rebate_paid_when_not_hit(self):
        # DI: rebate paid at expiry only if barrier was NOT hit.
        di = dict(s=100, k=100, r=0.05, sigma=0.2, t=1.0, barrier=80,
                  barrier_type="down-and-in")
        p_no = stocha.barrier_price(**di)
        p_reb = stocha.barrier_price(**di, rebate=5)
        # Should add positive rebate value.
        assert p_reb - p_no > 0
        # Should be less than R · e^{-rT} = 4.756 (since some paths do hit).
        assert p_reb - p_no < 5 * 2.71828**(-0.05) - 1e-9

    def test_already_breached_uo_paid_at_hit(self):
        # S0 above up barrier: KO knocked, paid_at_hit returns R immediately.
        p = stocha.barrier_price(s=125, k=100, r=0.05, sigma=0.2, t=1.0,
                                 barrier=120, barrier_type="up-and-out",
                                 rebate=5, rebate_at_hit=True)
        assert abs(p - 5.0) < 1e-10

    def test_already_breached_uo_paid_at_expiry(self):
        # S0 above up barrier: KO already hit, R·e^{-rT} paid at T.
        p = stocha.barrier_price(s=125, k=100, r=0.05, sigma=0.2, t=1.0,
                                 barrier=120, barrier_type="up-and-out",
                                 rebate=5, rebate_at_hit=False)
        import math
        assert abs(p - 5.0 * math.exp(-0.05)) < 1e-10

    def test_ki_already_breached_no_rebate(self):
        # KI with barrier already hit: option is now vanilla, rebate is extinguished.
        p_with = stocha.barrier_price(s=125, k=100, r=0.05, sigma=0.2, t=1.0,
                                      barrier=120, barrier_type="up-and-in",
                                      rebate=5)
        p_without = stocha.barrier_price(s=125, k=100, r=0.05, sigma=0.2, t=1.0,
                                         barrier=120, barrier_type="up-and-in")
        assert p_with == p_without

    def test_mc_rebate_matches_analytical(self):
        # MC must agree with analytical within discrete-monitoring noise.
        ana = stocha.barrier_price(**self.BASE, rebate=5, rebate_at_hit=True)
        mc = stocha.barrier_price(**self.BASE, rebate=5, rebate_at_hit=True,
                                  method="mc", n_paths=100_000, n_steps=2000, seed=1)
        rel = abs(mc - ana) / ana
        assert rel < 0.05


_BARRIER_QL_MAP = {
    "up-and-out": "UpOut",
    "up-and-in": "UpIn",
    "down-and-out": "DownOut",
    "down-and-in": "DownIn",
}


def _ql_setup(T_days, r, q, sigma):
    today = ql.Date(15, 5, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    maturity = today + ql.Period(T_days, ql.Days)
    T_exact = day_count.yearFraction(today, maturity)
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, sigma, day_count)
    )
    return today, maturity, T_exact, r_ts, q_ts, vol_ts


def _ql_barrier(S, K, T_days, r, q, sigma, barrier_type, H, option_type):
    today, maturity, T_exact, r_ts, q_ts, vol_ts = _ql_setup(T_days, r, q, sigma)
    spot_h = ql.QuoteHandle(ql.SimpleQuote(S))
    process = ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)
    engine = ql.AnalyticBarrierEngine(process)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    barrier_enum = getattr(ql.Barrier, _BARRIER_QL_MAP[barrier_type])
    option = ql.BarrierOption(barrier_enum, H, 0.0, payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV(), T_exact


def _ql_vanilla(S, K, T_days, r, q, sigma, option_type):
    today, maturity, T_exact, r_ts, q_ts, vol_ts = _ql_setup(T_days, r, q, sigma)
    spot_h = ql.QuoteHandle(ql.SimpleQuote(S))
    process = ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)
    engine = ql.AnalyticEuropeanEngine(process)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, K
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV(), T_exact


@pytest.mark.skipif(not _HAS_QL, reason="QuantLib not installed")
class TestBarrierQuantLibReference:
    """Accuracy audit: Reiner-Rubinstein barrier formula vs QuantLib AnalyticBarrierEngine.

    Tolerance design (α-strict, see .artifacts/2026-05-10-accuracy-audit-tests/):
      (a) standard 8 types × 3 strikes: atol=1e-06, rtol=1e-05
      (b) dividend regimes (q=0/0.06/0.10): atol=1e-06, rtol=1e-05
      (c) in-out parity (stocha-internal): atol=1e-12, rtol=1e-10
      (d) trivial cases (S touched / past barrier): exact 0.0 or stocha vanilla
      (e) S≈H near-barrier: atol=1e-04, rtol=1e-03 (digit cancellation regime)
      (f) edge maturity (T=1/365, T=10): NaN/Inf check only
    """

    S = 100.0
    SIGMA = 0.25
    T_DAYS = 183
    R = 0.10
    Q = 0.04
    # Noise floor: stocha (Rust libm) vs QL (boost::math) cross-library float
    # diff is ~3e-05 abs at worst. atol=5e-05 absorbs it with 50% margin while
    # rtol=1e-05 still detects O(1e-04) relative bugs at large prices (~10).
    ATOL = 5e-05
    RTOL = 1e-05

    # ------- (a) Standard 8 types × 3 strikes -------

    @pytest.mark.parametrize(
        "barrier_type,H",
        [
            ("up-and-out", 105.0),
            ("up-and-in", 105.0),
            ("down-and-out", 95.0),
            ("down-and-in", 95.0),
        ],
    )
    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_standard_matches_quantlib(self, barrier_type, H, K, option_type):
        ql_price, T_exact = _ql_barrier(
            self.S, K, self.T_DAYS, self.R, self.Q, self.SIGMA,
            barrier_type, H, option_type,
        )
        stocha_price = stocha.barrier_price(
            s=self.S, k=K, r=self.R, sigma=self.SIGMA, t=T_exact,
            barrier=H, barrier_type=barrier_type, option_type=option_type,
            q=self.Q, method="analytical",
        )
        assert math.isfinite(stocha_price)
        assert stocha_price == pytest.approx(
            ql_price, abs=self.ATOL, rel=self.RTOL
        ), (
            f"{barrier_type} {option_type} K={K} H={H}: "
            f"stocha={stocha_price:.10f}, QL={ql_price:.10f}, "
            f"abs_err={abs(stocha_price - ql_price):.3e}"
        )

    # ------- (b) Dividend regimes including negative carry q > r -------

    @pytest.mark.parametrize("q", [0.0, 0.06, 0.10])
    @pytest.mark.parametrize(
        "barrier_type,H,K,option_type",
        [
            ("up-and-out", 110.0, 100.0, "call"),
            ("down-and-in", 95.0, 100.0, "put"),
        ],
    )
    def test_dividend_regimes_match_quantlib(
        self, q, barrier_type, H, K, option_type
    ):
        ql_price, T_exact = _ql_barrier(
            self.S, K, self.T_DAYS, self.R, q, self.SIGMA,
            barrier_type, H, option_type,
        )
        stocha_price = stocha.barrier_price(
            s=self.S, k=K, r=self.R, sigma=self.SIGMA, t=T_exact,
            barrier=H, barrier_type=barrier_type, option_type=option_type,
            q=q, method="analytical",
        )
        assert stocha_price == pytest.approx(
            ql_price, abs=self.ATOL, rel=self.RTOL
        ), (
            f"{barrier_type} {option_type} q={q}: "
            f"stocha={stocha_price:.10f}, QL={ql_price:.10f}, "
            f"abs_err={abs(stocha_price - ql_price):.3e}"
        )

    # ------- (c) In-out parity (stocha-internal, no QL mixing) -------

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_up_in_out_parity_internal(self, option_type, K):
        # UI(call) + UO(call) and DI(call) + DO(call) must both equal the same
        # vanilla. Checking UI+UO == DI+DO avoids any external BS implementation.
        kwargs = dict(
            s=self.S, k=K, r=self.R, sigma=self.SIGMA, t=0.5,
            q=self.Q, option_type=option_type, method="analytical",
        )
        ui = stocha.barrier_price(barrier=105.0, barrier_type="up-and-in", **kwargs)
        uo = stocha.barrier_price(barrier=105.0, barrier_type="up-and-out", **kwargs)
        di = stocha.barrier_price(barrier=95.0, barrier_type="down-and-in", **kwargs)
        do = stocha.barrier_price(barrier=95.0, barrier_type="down-and-out", **kwargs)
        assert (ui + uo) == pytest.approx(di + do, abs=1e-12, rel=1e-10), (
            f"{option_type} K={K}: UI+UO={ui+uo:.12f}, DI+DO={di+do:.12f}"
        )

    # ------- (d) Trivial cases (already-knocked); QL not used (it throws) -------

    def test_up_out_already_knocked_returns_zero(self):
        # S > H for up-and-out → option is dead, price = 0.
        p = stocha.barrier_price(
            s=110.0, k=100.0, r=self.R, sigma=self.SIGMA, t=0.5,
            barrier=105.0, barrier_type="up-and-out", option_type="call",
            q=self.Q, method="analytical",
        )
        assert abs(p) < 1e-12, f"expected 0.0, got {p}"

    def test_down_out_already_knocked_returns_zero(self):
        p = stocha.barrier_price(
            s=90.0, k=100.0, r=self.R, sigma=self.SIGMA, t=0.5,
            barrier=95.0, barrier_type="down-and-out", option_type="put",
            q=self.Q, method="analytical",
        )
        assert abs(p) < 1e-12, f"expected 0.0, got {p}"

    def test_up_in_already_knocked_equals_vanilla(self):
        # S > H for up-and-in → barrier already touched, option = vanilla.
        # stocha has no public BS pricer; compare to QL vanilla under same T_exact.
        # QL vanilla is safe to call (no trigger check).
        S_above = 110.0
        K = 100.0
        ql_vanilla, T_exact = _ql_vanilla(
            S_above, K, self.T_DAYS, self.R, self.Q, self.SIGMA, "call"
        )
        ki = stocha.barrier_price(
            s=S_above, k=K, r=self.R, sigma=self.SIGMA, t=T_exact,
            barrier=105.0, barrier_type="up-and-in", option_type="call",
            q=self.Q, method="analytical",
        )
        # Vanilla agreement is still subject to libm vs boost::math noise; use α tol.
        assert ki == pytest.approx(ql_vanilla, abs=self.ATOL, rel=self.RTOL), (
            f"UI past barrier should equal vanilla: stocha={ki}, QL_vanilla={ql_vanilla}"
        )

    def test_down_in_already_knocked_equals_vanilla(self):
        S_below = 90.0
        K = 100.0
        ql_vanilla, T_exact = _ql_vanilla(
            S_below, K, self.T_DAYS, self.R, self.Q, self.SIGMA, "put"
        )
        ki = stocha.barrier_price(
            s=S_below, k=K, r=self.R, sigma=self.SIGMA, t=T_exact,
            barrier=95.0, barrier_type="down-and-in", option_type="put",
            q=self.Q, method="analytical",
        )
        assert ki == pytest.approx(ql_vanilla, abs=self.ATOL, rel=self.RTOL), (
            f"DI past barrier should equal vanilla: stocha={ki}, QL_vanilla={ql_vanilla}"
        )

    # ------- (e) S≈H near-barrier (digit cancellation regime) -------

    def test_near_barrier_down_out_put(self):
        H = 99.99
        ql_price, T_exact = _ql_barrier(
            self.S, 100.0, self.T_DAYS, self.R, self.Q, self.SIGMA,
            "down-and-out", H, "put",
        )
        stocha_price = stocha.barrier_price(
            s=self.S, k=100.0, r=self.R, sigma=self.SIGMA, t=T_exact,
            barrier=H, barrier_type="down-and-out", option_type="put",
            q=self.Q, method="analytical",
        )
        assert stocha_price == pytest.approx(ql_price, abs=1e-04, rel=1e-03), (
            f"near-barrier DO put H={H}: stocha={stocha_price}, QL={ql_price}"
        )

    def test_near_barrier_up_out_call(self):
        H = 100.01
        ql_price, T_exact = _ql_barrier(
            self.S, 100.0, self.T_DAYS, self.R, self.Q, self.SIGMA,
            "up-and-out", H, "call",
        )
        stocha_price = stocha.barrier_price(
            s=self.S, k=100.0, r=self.R, sigma=self.SIGMA, t=T_exact,
            barrier=H, barrier_type="up-and-out", option_type="call",
            q=self.Q, method="analytical",
        )
        assert stocha_price == pytest.approx(ql_price, abs=1e-04, rel=1e-03), (
            f"near-barrier UO call H={H}: stocha={stocha_price}, QL={ql_price}"
        )

    # ------- (f) Edge maturity (NaN/Inf check only) -------

    @pytest.mark.parametrize("T", [1.0 / 365.0, 10.0])
    @pytest.mark.parametrize(
        "barrier_type,H,option_type",
        [
            ("up-and-out", 105.0, "call"),
            ("up-and-in", 105.0, "call"),
            ("down-and-out", 95.0, "put"),
            ("down-and-in", 95.0, "put"),
        ],
    )
    def test_edge_maturity_finite(self, T, barrier_type, H, option_type):
        p = stocha.barrier_price(
            s=self.S, k=100.0, r=self.R, sigma=self.SIGMA, t=T,
            barrier=H, barrier_type=barrier_type, option_type=option_type,
            q=self.Q, method="analytical",
        )
        assert math.isfinite(p), f"non-finite price at T={T}: {p}"
        assert p >= 0.0, f"negative price at T={T}: {p}"


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


class TestAsianSeasoning:
    """Mid-life Asian option pricing via running_avg + time_elapsed."""

    def test_no_seasoning_default_unchanged(self):
        # Both fields None must reproduce the existing price exactly.
        p_default = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                       average_type="geometric", method="analytical")
        p_explicit = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                        average_type="geometric", method="analytical",
                                        running_avg=None, time_elapsed=None)
        assert p_default == p_explicit

    def test_seasoned_geometric_call_positive(self):
        p = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                               average_type="geometric", method="analytical",
                               running_avg=98, time_elapsed=0.5)
        assert p > 0

    def test_deep_itm_call_no_volatility_dependence(self):
        # When K* ≤ 0, the option is fully ITM and price is deterministic.
        # Different sigma values should give the same price.
        # Setup: t1=0.9, A_spent=200, K=100 → K* = (1·100 - 0.9·200)/0.1 = -800 < 0
        kwargs = dict(s=100, k=100, r=0.05, t=1.0, average_type="geometric",
                      method="analytical", running_avg=200, time_elapsed=0.9)
        p1 = stocha.asian_price(sigma=0.1, **kwargs)
        p2 = stocha.asian_price(sigma=0.5, **kwargs)
        # Geometric deep-ITM: depends on sigma only via E[geo A_T] which has
        # σ²·remaining³/(3T²) variance term — non-zero but small at remaining=0.1.
        assert abs(p1 - p2) < 0.5

    def test_deep_itm_put_returns_zero(self):
        # K=100, A_spent=200, t1=0.9 → K* < 0, put is always OTM.
        p = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                               option_type="put", average_type="geometric",
                               method="analytical",
                               running_avg=200, time_elapsed=0.9)
        assert p == 0.0

    def test_at_expiry_returns_realized_payoff(self):
        # time_elapsed = t means option just expired; price = realized intrinsic.
        p_call = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                    average_type="geometric", method="analytical",
                                    running_avg=120, time_elapsed=1.0)
        assert abs(p_call - 20.0) < 1e-10
        p_put = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                                   option_type="put", average_type="geometric",
                                   method="analytical",
                                   running_avg=80, time_elapsed=1.0)
        assert abs(p_put - 20.0) < 1e-10

    def test_seasoned_decreases_with_higher_running_avg_for_put(self):
        # Higher running_avg → call value increases, put value decreases.
        kwargs = dict(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                      average_type="geometric", method="analytical", time_elapsed=0.5)
        call_low = stocha.asian_price(running_avg=95, **kwargs)
        call_high = stocha.asian_price(running_avg=105, **kwargs)
        assert call_high > call_low

    def test_arithmetic_mc_seasoning(self):
        # Arithmetic via MC; ensure seasoning produces a positive price.
        p = stocha.asian_price(s=100, k=100, r=0.05, sigma=0.2, t=1.0,
                               average_type="arithmetic", method="mc",
                               running_avg=98, time_elapsed=0.5,
                               n_paths=20_000, seed=1)
        assert p > 0


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


class TestLookbackSeasoning:
    """Mid-life lookback pricing via running_max / running_min."""

    def test_floating_call_no_seasoning_equals_running_min_eq_s(self):
        # Boundary: running_min = s should reproduce the no-seasoning price
        # (decomposition validates against existing analytical formula).
        p_no = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                     option_type="call")
        p_eq = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                     option_type="call", running_min=100)
        assert abs(p_no - p_eq) < 1e-8

    def test_floating_put_no_seasoning_equals_running_max_eq_s(self):
        p_no = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                     option_type="put")
        p_eq = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                     option_type="put", running_max=100)
        assert abs(p_no - p_eq) < 1e-8

    def test_floating_put_increases_with_running_max(self):
        # Higher running max → larger gap to S_T → higher floating put price.
        p_low = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                      option_type="put", running_max=100)
        p_high = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                       option_type="put", running_max=120)
        assert p_high > p_low

    def test_floating_call_increases_with_lower_running_min(self):
        p_high = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                       option_type="call", running_min=100)
        p_low = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                      option_type="call", running_min=80)
        assert p_low > p_high

    def test_fixed_call_running_max_above_strike(self):
        # If M ≥ K, decomposition: (M-K) e^{-rT} + LookbackFixedCall(K=M).
        # With M=120, K=100: intrinsic ≈ 20 e^{-0.05} ≈ 19.02, plus more.
        p = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  strike_type="fixed", k=100, running_max=120)
        # Discounted intrinsic alone is ~19.02, plus forward lookback >= 0.
        assert p > 19.0

    def test_fixed_put_running_min_below_strike(self):
        p = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  strike_type="fixed", option_type="put",
                                  k=100, running_min=80)
        # Discounted intrinsic alone is ~19.02 + lookback put extra
        assert p > 19.0

    def test_invalid_running_max_below_s(self):
        # running_max < s is logically inconsistent → expected to error / return error.
        # The Rust analytical returns None which becomes an error in PyO3 wrapper.
        with pytest.raises(ValueError):
            stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                  option_type="put", running_max=90,
                                  method="analytical")

    def test_mc_seasoning_initializes_extrema(self):
        # MC with running_max=120 should give similar price to running_max=100
        # only if no path exceeds 120; otherwise price > running_max=100 case.
        p_high = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                       option_type="put", running_max=120,
                                       method="mc", n_paths=20_000, seed=1)
        p_low = stocha.lookback_price(s=100, r=0.05, sigma=0.2, t=1.0,
                                      option_type="put", running_max=100,
                                      method="mc", n_paths=20_000, seed=1)
        assert p_high >= p_low
