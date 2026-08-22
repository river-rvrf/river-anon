"""
test_ring.py -- Unit tests for R_q arithmetic, rounding and bit dropping.
"""

import random

from ring import (Ring, round_p, rounding_error, power2round, mod_pm,
                  high_bits, low_bits)
from params import get

SMALL_Q = 7681
PAR = get("RiVeR-N8")


def _rand_poly(rng, q, d):
    return [rng.randrange(q) for _ in range(d)]


# ---- ring axioms ---------------------------------------------------------

def test_identities():
    R = Ring(SMALL_Q, 32)
    a = _rand_poly(random.Random(1), SMALL_Q, 32)
    assert R.add(a, R.zero()) == a
    assert R.mul(a, R.one()) == a
    assert R.add(a, R.neg(a)) == R.zero()


def test_negacyclic():
    """X^d = -1 in R_q."""
    for d in (32, 64):
        R = Ring(SMALL_Q, d)
        x = R.zero()
        x[1] = 1
        prod = x
        for _ in range(d - 1):
            prod = R.mul(prod, x)
        assert prod == R.const(-1)


def test_commutativity_associativity_distributivity():
    R = Ring(PAR.q, PAR.d)
    rng = random.Random(2)
    for _ in range(10):
        a, b, c = (_rand_poly(rng, R.q, R.d) for _ in range(3))
        assert R.mul(a, b) == R.mul(b, a)
        assert R.mul(R.mul(a, b), c) == R.mul(a, R.mul(b, c))
        assert R.mul(a, R.add(b, c)) == R.add(R.mul(a, b), R.mul(a, c))


def test_kronecker_matches_schoolbook():
    for q in (SMALL_Q, PAR.q, PAR.q_hat):
        R = Ring(q, 32)
        rng = random.Random(q & 0xFFFF)
        for _ in range(10):
            a, b = _rand_poly(rng, q, 32), _rand_poly(rng, q, 32)
            assert R.mul(a, b) == R.mul_schoolbook(a, b), q


def test_kronecker_handles_extreme_coefficients():
    R = Ring(PAR.q, PAR.d)
    top = [PAR.q - 1] * PAR.d
    assert R.mul(top, top) == R.mul_schoolbook(top, top)
    assert R.mul(top, R.zero()) == R.zero()


# ---- representatives and norms -------------------------------------------

def test_centered_round_trip():
    R = Ring(PAR.q, PAR.d)
    rng = random.Random(3)
    a = _rand_poly(rng, R.q, R.d)
    assert R.from_centered(R.centered(a)) == a
    assert all(-R.q // 2 < c <= R.q // 2 for c in R.centered(a))


def test_norms_use_centered_form():
    R = Ring(101, 4)
    a = [100, 1, 0, 50]                 # centred: [-1, 1, 0, 50]
    assert R.centered(a) == [-1, 1, 0, 50]
    assert R.inf_norm(a) == 50
    assert R.l1_norm(a) == 52
    assert R.l2_norm_sq(a) == 1 + 1 + 0 + 2500


def test_inner_product_matches_manual_sum():
    R = Ring(PAR.q, PAR.d)
    rng = random.Random(4)
    u = [_rand_poly(rng, R.q, R.d) for _ in range(5)]
    v = [_rand_poly(rng, R.q, R.d) for _ in range(5)]
    manual = R.zero()
    for i in range(5):
        manual = R.add(manual, R.mul(u[i], v[i]))
    assert R.inner(u, v) == manual


def test_mat_vec():
    R = Ring(SMALL_Q, 32)
    rng = random.Random(5)
    M = [[_rand_poly(rng, R.q, R.d) for _ in range(3)] for _ in range(2)]
    v = [_rand_poly(rng, R.q, R.d) for _ in range(3)]
    out = R.mat_vec(M, v)
    assert len(out) == 2
    assert out[0] == R.inner(M[0], v)


# ---- rounding (Fact 1) ---------------------------------------------------

def test_fact1_round_trip():
    """v = floor(u)_p  iff  e = u - q_0 v mod q has coefficients in [0, q_0-1]."""
    R = Ring(PAR.q, PAR.d)
    rng = random.Random(6)
    for _ in range(20):
        u = _rand_poly(rng, R.q, R.d)
        v = round_p(R, u, PAR.q0)
        e = rounding_error(R, u, v, PAR.q0)
        assert all(0 <= c < PAR.p for c in v)
        assert all(0 <= c <= PAR.q0 - 1 for c in e)
        assert all((PAR.q0 * v[i] + e[i]) % R.q == u[i] for i in range(R.d))


def test_round_p_is_integer_division():
    R = Ring(PAR.q, PAR.d)
    u = [0, PAR.q0 - 1, PAR.q0, 2 * PAR.q0 + 7]
    u = u + [0] * (PAR.d - 4)
    v = round_p(R, u, PAR.q0)
    assert v[:4] == [0, 0, 1, 2]


# ---- bit dropping --------------------------------------------------------

def test_power2round_round_trip():
    rng = random.Random(7)
    for K in (5, 13, 28):
        for _ in range(200):
            value = rng.randrange(0, 1 << 45)
            hi, lo = power2round(value, K)
            assert hi * (1 << K) + lo == value
            assert -(1 << (K - 1)) < lo <= 1 << (K - 1)


def test_mod_pm_range():
    for K in (5, 8):
        power = 1 << K
        for v in range(-3 * power, 3 * power):
            lo = mod_pm(v, K)
            assert -(power // 2) < lo <= power // 2
            assert (v - lo) % power == 0


def test_mod_pm_takes_the_exponent_not_the_modulus():
    """The two spellings both run and disagree; pin which one this is.

    `mod_pm(c, 1 << K)` and `mod_pm(c, K)` are both valid Python and give
    different answers, and `river-rs` takes the exponent.  This is the
    guard that keeps the two signatures from drifting apart again.
    """
    assert mod_pm(37, 5) == 5           # 37 mod 32 = 5
    assert mod_pm(37, 32) != 5          # would be 37 mod 2^32
    assert mod_pm(20, 5) == -12         # 20 > 16, so it wraps negative
    assert mod_pm(16, 5) == 16          # the tie stays positive
    assert power2round(37, 5) == (1, 5)


def test_high_low_bits_reconstruct():
    R = Ring(PAR.q_hat, PAR.d)
    rng = random.Random(8)
    a = _rand_poly(rng, R.q, R.d)
    K = PAR.K_b
    hi, lo = high_bits(R, a, K), low_bits(R, a, K)
    cent = R.centered(a)
    assert [hi[i] * (1 << K) + lo[i] for i in range(R.d)] == cent


def test_e_B_bound_from_correctness_proof():
    """||e_B||_inf <= 2^{K_b - 1}: the bound the OOM correctness proof uses."""
    R = Ring(PAR.q_hat, PAR.d)
    rng = random.Random(9)
    for _ in range(20):
        a = _rand_poly(rng, R.q, R.d)
        assert max(abs(c) for c in low_bits(R, a, PAR.K_b)) <= 1 << (PAR.K_b - 1)


def _paper_high(a, K, q):
    """`[[a]]_K` as the paper's Preliminaries define it.

        a mod^pm 2^K := \bar a - 2^K floor((\bar a + 2^{K-1} - 1) / 2^K)
        [[a]]_K      := floor((\bar a + 2^{K-1} - 1) / 2^K)

    on the centred representative `\bar a in (-q/2, q/2]`, with the low part
    in `(-2^{K-1}, 2^{K-1}]`.
    """
    bar = a - q if a > q // 2 else a
    return (bar + (1 << (K - 1)) - 1) // (1 << K)


def test_paper_high_bits_are_power2round_on_the_centred_representative():
    """The paper's definition is `power2round` on the centred representative.

    `mod^pm` is representative-independent, so the tie direction is the only
    convention inside it, and the paper's form lands in
    `(-2^{K-1}, 2^{K-1}]` -- closed at the top, which is what `ring.mod_pm`
    does.  The only thing that distinguishes the two readings is therefore
    which representative goes in.
    """
    q, K = PAR.q_hat, PAR.K_b
    rng = random.Random(11)
    for _ in range(20000):
        a = rng.randrange(q)
        bar = a - q if a > q // 2 else a
        assert _paper_high(a, K, q) == power2round(bar, K)[0]
        low = bar - (_paper_high(a, K, q) << K)
        assert -(1 << (K - 1)) < low <= (1 << (K - 1))
        assert low == mod_pm(bar, K)


def test_the_tie_direction_matches_mod_pm():
    """At `\bar a = 2^{K-1}` the low part is `+2^{K-1}`, not `-2^{K-1}`.

    The tie is closed at the top, so the high part does not carry.
    """
    q, K = PAR.q_hat, PAR.K_b
    tie = 1 << (K - 1)
    assert power2round(tie, K)[1] == tie
    assert _paper_high(tie, K, q) == 0


def test_the_representative_is_what_moved_the_bytes():
    """The representative, not the tie, is what half the coefficients turn on.

    The canonical and centred readings differ on every coefficient whose
    centred representative is negative -- about half of them -- which is why
    the `B` field on the wire is signed.
    """
    q, K = PAR.q_hat, PAR.K_b
    rng = random.Random(11)
    sample = [rng.randrange(q) for _ in range(20000)]
    differ = sum(1 for a in sample
                 if _paper_high(a, K, q) != power2round(a, K)[0])
    assert 0.4 < differ / len(sample) < 0.6, differ / len(sample)


def test_oom_uses_the_centred_representative():
    """`oom._high_low` follows the paper: signed high parts."""
    from oom import OOM, high_bits_bound

    par = PAR
    oom = OOM(par, b"\x05" * 32)
    R = Ring(par.q_hat, par.d)
    rng = random.Random(13)
    vec = [_rand_poly(rng, R.q, R.d) for _ in range(4)]
    highs, lows = oom._high_low(vec, par.K_b)

    assert any(c < 0 for poly in highs for c in poly), \
        "high parts should be signed under the centred convention"
    expect = [[_paper_high(c, par.K_b, R.q) for c in poly] for poly in vec]
    assert highs == expect

    # The reconstruction identity the correctness argument needs.
    centred = R.vec_centered(vec)
    for poly_hi, poly_lo, poly_c in zip(highs, lows, centred):
        for hi, lo, c in zip(poly_hi, poly_lo, poly_c):
            assert c == (hi << par.K_b) + lo
            assert -(1 << (par.K_b - 1)) < lo <= (1 << (par.K_b - 1))

    # ... and every high part fits the codec's declared bound.
    bound = high_bits_bound(par.q_hat, par.K_b)
    assert all(abs(c) <= bound for poly in highs for c in poly)


def test_high_bits_bound_is_tight():
    """The codec's field bound is reached, and not exceeded.

    One too small refuses an honest proof; one too large costs a bit per
    coefficient across `n_hat d` of them.
    """
    from oom import high_bits_bound
    for par in (PAR, get("RiVeR-N8")):
        for K in (par.K_b, par.K_a):
            bound = high_bits_bound(par.q_hat, K)
            top = par.q_hat // 2
            bottom = -(par.q_hat // 2) + (0 if par.q_hat % 2 else 1)
            reached = max(abs(power2round(top, K)[0]),
                          abs(power2round(bottom, K)[0]))
            assert bound == reached
            for a in (top, bottom, 0, 1, -1):
                assert abs(power2round(a, K)[0]) <= bound


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_ring.py: {len(tests)} tests passed")
