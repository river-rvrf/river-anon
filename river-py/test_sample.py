"""
test_sample.py -- Unit tests for the deterministic samplers.

The statistical assertions use fixed seeds, so they are deterministic: a
failure means a real change in the sampler, not bad luck.
"""

import math

import pytest

from sample import (XOF, absorb, hash_bytes, uniform_int,
                    uniform_beta_poly, gaussian_int, sample_challenge,
                    challenge_from_hash, sam_mat, rej1, rej2, exp_threshold,
                    rational_sigma, PROB_ONE, GAUSSIAN_TAILCUT)
from params import TOY_PARAMS, get

PAR = get("RiVeR-N8")


def _centered(poly, q):
    return [c - q if c > q // 2 else c for c in poly]


# ---- XOF -----------------------------------------------------------------

def test_xof_is_deterministic():
    assert XOF(b"d", b"s").read(64) == XOF(b"d", b"s").read(64)


def test_xof_chunking_is_irrelevant():
    whole = XOF(b"d", b"s").read(1000)
    piecewise = XOF(b"d", b"s")
    parts = b"".join(piecewise.read(n) for n in (1, 7, 136, 300, 556))
    assert whole == parts


def test_xof_absorption_is_injective():
    assert absorb(b"d", b"ab", b"c") != absorb(b"d", b"a", b"bc")
    assert absorb(b"d1", b"x") != absorb(b"d2", b"x")


def test_domain_separation_changes_output():
    assert XOF(b"A", b"s").read(32) != XOF(b"B", b"s").read(32)


def test_hash_bytes_length():
    assert len(hash_bytes(48, b"d", b"x")) == 48


# ---- uniform -------------------------------------------------------------

def test_uniform_int_in_range():
    x = XOF(b"t", b"u")
    for modulus in (3, 61, 1000, PAR.q, PAR.q_hat):
        for _ in range(50):
            assert 0 <= uniform_int(x, modulus) < modulus


def test_uniform_int_is_flat():
    x = XOF(b"t", b"flat")
    counts = [0] * 16
    for _ in range(16000):
        counts[uniform_int(x, 16)] += 1
    assert min(counts) > 850 and max(counts) < 1150, counts


def test_uniform_beta_is_ternary_for_beta_1():
    x = XOF(b"t", b"tern")
    poly = uniform_beta_poly(x, 1, PAR.d, PAR.q)
    assert set(_centered(poly, PAR.q)) <= {-1, 0, 1}


def test_sam_mat_shape_and_determinism():
    A = sam_mat(b"seed" * 8, PAR.q, 3, 4, PAR.d, "RiVeR.A")
    assert len(A) == 3 and all(len(row) == 4 for row in A)
    assert all(0 <= c < PAR.q for row in A for p in row for c in p)
    assert A == sam_mat(b"seed" * 8, PAR.q, 3, 4, PAR.d, "RiVeR.A")
    assert A != sam_mat(b"seed" * 8, PAR.q, 3, 4, PAR.d, "G'")


# ---- Gaussian ------------------------------------------------------------

def test_gaussian_moments():
    for sigma in (100, 4096):
        x = XOF(b"t", b"g" + str(sigma).encode())
        samples = [gaussian_int(x, sigma) for _ in range(8000)]
        mean = sum(samples) / len(samples)
        var = sum(s * s for s in samples) / len(samples) - mean ** 2
        assert abs(mean) < 0.1 * sigma, (sigma, mean)
        assert abs(math.sqrt(var) - sigma) / sigma < 0.06, (sigma, var)


def test_gaussian_is_tail_cut():
    x = XOF(b"t", b"tail")
    sigma = 500
    assert all(abs(gaussian_int(x, sigma)) <= GAUSSIAN_TAILCUT * sigma
               for _ in range(3000))


def test_the_three_tail_parameters_are_separate_and_consistent():
    """`GAUSSIAN_TAILCUT`, `VERIFIER_TAILCUT` and `PROB_BITS` are three things.

    The paper requires them to be distinct parameters, requires
    the verifier's bound to be derived separately from the sampler's cut, and
    requires a declared cut past what `PROB_BITS` supports to raise an error
    rather than silently leave the distribution unchanged -- which is what
    `river-py` did at 64 bits.
    """
    from decimal import Decimal
    from sample import (GAUSSIAN_TAILCUT, VERIFIER_TAILCUT, PROB_BITS,
                        _check_probability_width)

    assert (GAUSSIAN_TAILCUT, VERIFIER_TAILCUT, PROB_BITS) == (14, 6, 192)
    assert GAUSSIAN_TAILCUT > VERIFIER_TAILCUT

    effective = _check_probability_width()
    assert Decimal("16.31") < effective < Decimal("16.32")
    assert effective > GAUSSIAN_TAILCUT

    # the historical silent no-op, and one just past the current cut
    with pytest.raises(ValueError):
        _check_probability_width(tailcut=14, bits=64)
    with pytest.raises(ValueError):
        _check_probability_width(tailcut=17, bits=PROB_BITS)
    # and the 32-bit threshold headroom the sampler needs below the cut
    with pytest.raises(ValueError):
        _check_probability_width(tailcut=16, bits=PROB_BITS)


def test_the_integer_and_decimal_width_checks_agree_everywhere():
    """`gaussian_int` calls the integer form; the `Decimal` one is the
    statement it has to match.

    The module docstring claims the two agree at every boundary.  That claim
    was made by an ad-hoc script and not by a test, which is exactly the
    gap this repository records elsewhere as a defect -- so it is a test.
    """
    from sample import (check_probability_width, _check_probability_width,
                        PROB_BITS)

    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    checked = 0
    for bits in (64, 96, 128, 192, 256, 384, 512):
        for tailcut in range(1, 41):
            a = raises(lambda: check_probability_width(bits, tailcut))
            b = raises(lambda: _check_probability_width(tailcut=tailcut,
                                                       bits=bits))
            assert a == b, (bits, tailcut, a, b)
            checked += 1
    assert checked == 280

    # a zero cut is a point mass, and both forms say so
    with pytest.raises(ValueError):
        check_probability_width(PROB_BITS, 0)


def test_gaussian_int_itself_rejects_an_unreachable_cut():
    """The enforcement point is the sampler, not a caller who remembers."""
    x = XOF(b"t", b"reject")
    assert gaussian_int(x, 100, 1, 14) is not None
    for bad in (0, 17, 25, 1000):
        with pytest.raises(ValueError):
            gaussian_int(x, 100, 1, bad)


def test_gaussian_accepts_rational_sigma():
    num, den = rational_sigma(1234.5678)
    x = XOF(b"t", b"rat")
    samples = [gaussian_int(x, num, den) for _ in range(3000)]
    stdev = math.sqrt(sum(s * s for s in samples) / len(samples))
    assert abs(stdev - 1234.5678) / 1234.5678 < 0.08


def test_gaussian_is_symmetric():
    x = XOF(b"t", b"sym")
    samples = [gaussian_int(x, 300) for _ in range(6000)]
    positive = sum(1 for s in samples if s > 0)
    negative = sum(1 for s in samples if s < 0)
    assert abs(positive - negative) < 0.05 * len(samples)


# ---- fixed-point exp -----------------------------------------------------

def test_exp_threshold_matches_math_exp():
    for num, den in ((-1, 2), (-7, 3), (-100, 7), (0, 1)):
        got = exp_threshold(num, den)
        want = math.exp(num / den) * PROB_ONE
        assert abs(got - want) / max(want, 1) < 1e-12, (num, den)


def test_exp_threshold_clamps_and_floors():
    assert exp_threshold(0, 1) == PROB_ONE          # exp(0) = 1
    assert exp_threshold(5, 1) == PROB_ONE          # clamped
    assert exp_threshold(-10 ** 6, 1) == 0          # underflow to zero


def test_exp_threshold_is_reproducible():
    assert exp_threshold(-1234567, 98765) == exp_threshold(-1234567, 98765)


# ---- challenge space -----------------------------------------------------

def test_challenge_has_exact_weight_and_bound():
    x = XOF(b"t", b"chal")
    for _ in range(40):
        c = _centered(sample_challenge(x, PAR.d, PAR.w, PAR.gamma, PAR.q_hat),
                      PAR.q_hat)
        assert sum(1 for v in c if v != 0) == PAR.w
        assert max(abs(v) for v in c) <= PAR.gamma
        assert sum(abs(v) for v in c) <= PAR.w * PAR.gamma


def test_challenge_sparse_case():
    x = XOF(b"t", b"sparse")
    c = _centered(sample_challenge(x, 32, 4, 3, 7681), 7681)
    assert sum(1 for v in c if v != 0) == 4
    assert max(abs(v) for v in c) <= 3


def test_challenge_positions_are_spread():
    """Every coefficient position should be reachable."""
    x = XOF(b"t", b"pos")
    seen = set()
    for _ in range(60):
        c = _centered(sample_challenge(x, 32, 4, 3, 7681), 7681)
        seen |= {i for i, v in enumerate(c) if v != 0}
    assert seen == set(range(32))


def test_challenge_from_hash_is_deterministic():
    a = challenge_from_hash(PAR.d, PAR.w, PAR.gamma, PAR.q_hat, b"ctx")
    b = challenge_from_hash(PAR.d, PAR.w, PAR.gamma, PAR.q_hat, b"ctx")
    c = challenge_from_hash(PAR.d, PAR.w, PAR.gamma, PAR.q_hat, b"ctx2")
    assert a == b and a != c


# ---- rejection sampling --------------------------------------------------

def _rej_rate(fn, phi, T, dim=16, trials=2500, seed=b"r", tau=None):
    """Measured acceptance rate.  `tau` is passed only to `Rej_1`."""
    x = XOF(b"t", seed)
    accepted = 0
    extra = () if tau is None else (tau,)
    for _ in range(trials):
        v = [gaussian_int(x, 30) for _ in range(dim)]
        norm = math.sqrt(sum(c * c for c in v)) or 1.0
        v = [int(c * T / norm) for c in v]
        z = [gaussian_int(x, phi * T) + v[j] for j in range(dim)]
        accepted += 1 - fn(x, z, v, phi, phi * T, 1, *extra)
    return accepted / trials


def test_rej1_acceptance_matches_1_over_M1():
    """The *measured* acceptance rate is `1/M_1`, at the `tau_rej` the
    sampler is handed -- which is the half the repetition report cannot
    check about itself.

    Driven at two values of `tau_rej`, not just the shipped 12: with one
    value a sampler that ignored the argument entirely would still pass,
    since 12 is also the constant it would otherwise hard-code.
    """
    for tau in (12, 8):
        for phi in (6, 10):
            rate = _rej_rate(rej1, phi, 80, seed=b"r1-%d-%d" % (tau, phi),
                             tau=tau)
            expected = 1 / math.exp(tau / phi + 1 / (2 * phi ** 2))
            assert abs(rate - expected) < 0.04, (tau, phi, rate, expected)


def test_rej2_acceptance_matches_1_over_2M2():
    """Lemma "grs" part 2 gives Pr[accept] >= 1/(2 M_2).

    The factor 2 is the half-space condition <z, v> >= 0.  It is the factor
    the reported mu-tilde has to charge exactly once.
    """
    for phi in (3, 8):
        rate = _rej_rate(rej2, phi, 80, seed=b"r2-%d" % phi)
        expected = 1 / (2 * math.exp(1 / (2 * phi ** 2)))
        assert abs(rate - expected) < 0.04, (phi, rate, expected)


def test_rej2_rejects_negative_half_space():
    x = XOF(b"t", b"half")
    v = [1] * 8
    z = [-100] * 8                    # <z, v> < 0
    assert rej2(x, z, v, 3, 300, 1) == 1


def test_rej_rejects_oversized_z():
    x = XOF(b"t", b"big")
    v = [0] * 8
    z = [10 ** 9] + [0] * 7           # way beyond 6 sigma
    assert rej1(x, z, v, 10, 1000, 1, TOY_PARAMS.REJ_TAU) == 1
    assert rej2(x, z, v, 10, 1000, 1) == 1


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_sample.py: {len(tests)} tests passed")
