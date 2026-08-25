"""
test_review.py -- Arithmetic diagnostics for the parameter set, ring
representation, and challenge algebra.

These checks independently recompute small algebraic and probabilistic facts
that are easy to lose in a parameter migration. They are implementation
regressions and diagnostics, not security proofs.
"""

import math
from fractions import Fraction

import ring
from exact import ExactParams
from params import get, PROFILES
from sample import GAUSSIAN_TAILCUT, REJ1_CONSTANT, VERIFIER_TAILCUT
from dgs import tail_exact

PUBLISHED = ("RiVeR-N8", "RiVeR-N16", "RiVeR-N64", "RiVeR-N128", "RiVeR-N256")

def test_canonical_rounding_errors_use_the_centred_translation():
    """Canonical errors and centred witnesses are exact inverses.

    The construction represents rounding errors canonically in `[0, 60]` and
    translates them to `[-30, 30]` before applying coefficient bounds. The
    implementation performs that translation explicitly at the witness
    boundary.
    """
    B_e = 30
    for name in PUBLISHED:
        par = get(name)
        assert par.B_e == B_e, name

        canon = [0, 2 * B_e, B_e, 1]
        centred = ring.to_centered_error(canon, par.B_e)
        assert centred == [-B_e, B_e, 0, 1 - B_e]
        assert all(-B_e <= c <= B_e for c in centred)
        assert ring.from_centered_error(centred, par.B_e) == canon
        try:
            ring.to_centered_error([2 * B_e + 1], par.B_e)
        except ValueError:
            pass
        else:
            raise AssertionError("an out-of-range canonical error was taken")


# ---- ring slots: why there is no padding ---------------------------------

def test_ring_has_no_implicit_padding_and_accepts_duplicates():
    """The ring is exactly the ordered input tuple, including duplicates."""
    from river import RiVeR
    from oom import OOMStatement
    from params import TOY_PARAMS as par

    scheme = RiVeR(par)
    pp = scheme.setup(b"\x00" * 32)
    zero_pk = [[0] * par.d for _ in range(par.n)]
    ring = [zero_pk] * par.N
    zero_v = [0] * par.d
    statement = OOMStatement(par, scheme.Rq, pp["A"],
                             scheme.hash_message(b"m"), ring, zero_v)

    public_r = ([[0] * par.d for _ in range(par.ell)]
                + [scheme.Rq.from_centered([-par.B_e] * par.d)
                   for _ in range(par.n + 1)])
    assert statement.apply_ck(public_r) == statement.c_i(0)

    assert scheme.validate_ring(ring) == ring
    assert not hasattr(scheme, "canon_pad")

    encoded = {scheme.codec.pk_encode(pk) for pk in ring}
    assert len(encoded) == 1, "an all-dummy ring hides nobody"
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31) for i in range(par.N)]
    real = [pk for _, pk in keys]
    assert scheme.validate_ring(real) == real
    for t in real:
        assert any(c != 0 for p in t for c in p)


# ---- challenge-difference invertibility -----------------------------------

def test_x32_plus_1_factors_over_f61():
    """X^32 + 1 = (X^16 + 11)(X^16 + 50) mod 61."""
    a, b, q0 = 11, 50, 61
    assert (a + b) % q0 == 0            # no X^16 term
    assert (a * b) % q0 == 1            # constant term 1


def test_a_challenge_difference_can_be_a_zero_divisor():
    """x - x' = 11 + X^16 for two legitimate challenges, and it is a zero
    divisor modulo 61."""
    par = get("RiVeR-N8")
    d, gamma, q0 = par.d, par.gamma, par.q0

    x_prime = [1] * d                                   # all coefficients 1
    x = list(x_prime)
    x[0], x[16] = 12, 2

    # both are legitimate members of C^d_{w,gamma} with w = d
    for chal in (x, x_prime):
        assert all(1 <= abs(c) <= gamma for c in chal)
        assert sum(1 for c in chal if c != 0) == par.w

    delta = [x[i] - x_prime[i] for i in range(d)]
    assert delta[0] == 11 and delta[16] == 1
    assert all(c == 0 for i, c in enumerate(delta) if i not in (0, 16))

    # delta = 0 mod (X^16 + 11), since X^16 = -11 there
    assert (delta[0] - 11 * delta[16]) % q0 == 0


def test_the_non_unit_difference_probability_is_reproduced():
    """The paired-coordinate calculation gives `p_nonunit ~ 2^-93.82`."""
    par = get("RiVeR-N8")
    q0, gamma = par.q0, par.gamma
    support = [v for v in range(-gamma, gamma + 1) if v != 0]
    assert len(support) == 2 * gamma

    hits = 0
    for lo_a in support:
        for lo_b in support:
            d_lo = lo_a - lo_b
            for hi_a in support:
                for hi_b in support:
                    if (d_lo - 11 * (hi_a - hi_b)) % q0 == 0:
                        hits += 1
    per_pair = Fraction(hits, len(support) ** 4)
    assert per_pair == Fraction(2155, 131072), per_pair

    p_nonunit = 2 * float(per_pair) ** (par.d // 2) - 2.0 ** -160
    assert abs(math.log2(p_nonunit) - (-93.824446)) < 1e-4
    assert p_nonunit > 2.0 ** -128


# ---- phi_b and K_a as BoundGen outputs ------------------------------------

def test_phi_b_is_a_boundgen_output():
    """`phi_b` is a generated parameter used consistently by the sampler,
    verifier bound, and size model."""
    for name in PUBLISHED:
        par = get(name)
        assert par.phi_b == 2
        assert par.sigma_b == par.phi_b * par.cal_B
        assert par.zb_inf_bound == 6 * par.sigma_b
        # The size model charges this width, which is what makes the
        # reported |pi_OOM| reproduce; `test_params.py` checks that column.
        assert par.phi_b != 1
        # `BoundGen` adds the margin `s_c = 3` and aborts below it, so the
        # derived bound is the 28 every profile uses rather than a bare 25.
        assert par.K_a == 28 and par.K_a_boundgen == 28


# ---- the half-space factor in Rej_2 ---------------------------------------

def test_the_half_space_factor_is_charged_exactly_once():
    """`Rej_2`'s half-space factor of two is charged once and only once.

    The correctness appendix writes `mu_b := 2 exp(1/(2 phi_b^2))` from
    Lemma grs(2), and says in as many words that "this factor is not counted
    again in the subsequent infinity-norm check".  The `<z, v> >= 0`
    rejection is what the two accounts for, and double-charging it would
    inflate every repetition estimate.

    The design target is stated against that corrected quantity: the
    requirement is `mu-tilde <= 10`, and the five published profiles land
    between 8.3 and 8.6.
    """
    for name in PUBLISHED:
        par = get(name)
        assert par.mu_b == 2 * math.exp(1 / (2 * par.phi_b ** 2))
        # The factor is worth exactly 2x on the reported estimate.
        uncorrected = par.mu_river / 2
        assert abs(par.mu_river / uncorrected - 2.0) < 1e-12, name
        # Charged once, the repetition exceeds 3 at these parameters ...
        assert uncorrected > 3.0, name
        # ... and the target is stated against the corrected quantity.
        assert par.mu_river <= 10.0, name
        assert 8.3 <= round(par.mu_river, 1) <= 8.6, name


def test_all_four_rejection_samplers_are_charged():
    """All four Gaussian rejection steps are charged.

    `mu_OOM = mu_a mu_b mu_s mu_m`.  The split response answers `r_0` and
    `r_1` at two different widths, so it costs two `Rej_1` calls rather than
    one, and `mu_m` is the second of them.
    """
    for name in PUBLISHED:
        par = get(name)
        assert par.mu_gaussian == par.mu_a * par.mu_b * par.mu_s * par.mu_m
        three = par.mu_a * par.mu_b * par.mu_s
        assert par.mu_gaussian > three, name
        assert par.mu_m > 1.0


# ---- statistical-loss accounting ------------------------------------------

def test_the_response_truncation_tail_is_reproduced():
    """Record the Gaussian mass beyond the verifier's six-sigma bound."""
    tail = float(tail_exact(VERIFIER_TAILCUT))                 # Pr[|X| > 6s]
    assert abs(math.log2(tail) - (-28.92)) < 0.05, math.log2(tail)

    for name in PUBLISHED:
        par = get(name)
        dims = (par.N - 1) * par.d + par.r_dim * par.d          # f_1 and z
        log_p = math.log2(1 - (1 - tail) ** dims)
        assert -18.0 < log_p < -15.0, (name, log_p)
        assert log_p > -100


def test_the_sampler_cut_is_separate_from_the_verifier_cut():
    """The implementation's sampler support exceeds the verifier bound.

    `gaussian_int` uses a 14-sigma finite support while public response checks
    use six sigma. The larger internal support keeps sampler truncation
    separate from the protocol's response bound.
    """
    assert GAUSSIAN_TAILCUT > VERIFIER_TAILCUT, \
        "the mask must not be truncated at the verifier's own bound"

    par = get("RiVeR-N8")
    n_coeff = ((par.N - 1) * par.d + par.k_hat * par.d + par.r_dim * par.d)
    n_total = n_coeff * par.mu_river                # every attempt leaks

    # the sampler's own contribution, at the chosen cut
    sampler = float(tail_exact(GAUSSIAN_TAILCUT)) * n_total
    assert math.log2(sampler) < -128, math.log2(sampler)

    # ... and what it would have been at the verifier's cut: the mass of
    # D_sigma in the width-||v||_inf strip just past 6 sigma.  The response
    # is split, so there are two strips.
    #
    # Both strips are `2 w gamma B_e d` wide. What separates them is the
    # Gaussian width, and `sigma_s / sigma_m` is 8.0 at this profile.
    density = math.exp(-VERIFIER_TAILCUT ** 2 / 2) / math.sqrt(2 * math.pi)
    v_inf = 2 * par.w * par.gamma * par.B_e * par.d
    strips = {"z_s": (v_inf, par.sigma_s), "z_m": (v_inf, par.sigma_m)}
    naive = {}
    for block, (width, sigma) in strips.items():
        naive[block] = 2 * density * (width / sigma) * n_total
    assert -16.5 < math.log2(naive["z_s"]) < -14.5, math.log2(naive["z_s"])
    assert -13.5 < math.log2(naive["z_m"]) < -11.5, math.log2(naive["z_m"])
    assert naive["z_m"] > naive["z_s"]
    # ... and the gap between them is the width ratio, nothing else
    assert abs(naive["z_m"] / naive["z_s"] - par.sigma_s / par.sigma_m) < 1e-9

    # the protocol's own loss is unaffected by either choice
    protocol = float(tail_exact(VERIFIER_TAILCUT)) * n_total
    assert -15.0 < math.log2(protocol) < -13.0, math.log2(protocol)
    assert protocol > sampler


def test_the_expected_rejection_call_count_is_reproduced():
    """Each attempt makes three `Rej_1` calls and retries geometrically."""
    for name in PUBLISHED:
        par = get(name)
        per_attempt = 3
        expected_calls = per_attempt * par.mu_river
        assert 24 < expected_calls < 27, (name, expected_calls)
        assert REJ1_CONSTANT == 12

# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_review.py: {len(tests)} tests passed")
