"""
test_kat.py -- Known-answer tests for the primitives a port must match first.

`vectors.json` pins whole executions, which is what you want as an acceptance
test but the worst possible thing to debug: a single wrong byte anywhere in the
XOF, a sampler, or the codec moves every field downstream of it, and the diff
says only "proof bytes differ".

These pin the primitives in dependency order instead, so a second
implementation can bisect: XOF, then the samplers built on it, then the
acceptance thresholds, then the bit codec.  The first failing KAT names the
layer at fault.

Every value here was produced by this implementation, so these are
consistency anchors, not independent validation.  What they establish is that
two implementations agree -- and, when one of them changes, exactly where.

`PROB_BITS` is deliberately asserted: it is a wire-visible constant.  Changing
it changes every acceptance decision and therefore every vector, which is
precisely the coupling that let a declared `14 sigma` tail cut silently behave
as `9.42 sigma` (see `sample.py`).
"""

from params import TOY_PARAMS
import lanes_params
import lanes_ring as R
from codec import BitReader, BitWriter, Rice, Signed, Uniform, optimal_rice_k
from sample import (DS_CHALLENGE, DS_COMMIT, GAUSSIAN_TAILCUT, PROB_BITS,
                    XOF, _check_probability_width, exp_accept, exp_threshold,
                    gaussian_int, hash_bytes, rational_sigma, rej1, rej2,
                    sample_challenge, uniform_int)


# ---- wire-visible constants ----------------------------------------------

def test_pinned_constants():
    """These are part of the wire format, not tuning knobs."""
    assert PROB_BITS == 192
    assert GAUSSIAN_TAILCUT == 14
    # ... and the two must remain compatible; see sample.py
    assert float(_check_probability_width()) > GAUSSIAN_TAILCUT


# ---- layer 1: the XOF ----------------------------------------------------

def test_kat_xof_stream():
    assert XOF(b"KAT", b"xof").read(16).hex() == \
        "0e68ff0b1407311065c3f650d2442da7"


def test_kat_hash_bytes():
    assert hash_bytes(16, b"KAT", b"a", b"b").hex() == \
        "641ca3b6c656d248d72526af0cc5b493"


def test_xof_absorption_is_length_prefixed():
    """`("ab", "c")` and `("a", "bc")` must not collide."""
    assert hash_bytes(16, b"KAT", b"ab", b"c") != \
        hash_bytes(16, b"KAT", b"a", b"bc")


# ---- layer 2: samplers ---------------------------------------------------

def test_kat_uniform_int():
    x = XOF(b"KAT", b"uniform")
    assert [uniform_int(x, 61) for _ in range(8)] == \
        [16, 42, 0, 57, 44, 14, 50, 18]


def test_kat_gaussian_small_sigma():
    x = XOF(b"KAT", b"gauss8")
    assert [gaussian_int(x, 8) for _ in range(8)] == \
        [2, -10, 2, 6, 8, -1, 10, 1]


def test_kat_gaussian_lanes_sigma():
    x = XOF(b"KAT", b"gauss352")
    assert [gaussian_int(x, 352) for _ in range(8)] == \
        [-292, 229, 119, 298, 82, 384, -314, -256]


def test_kat_gaussian_rational_sigma():
    """A width pinned through `rational_sigma`, as the OOM layer's are."""
    num, den = rational_sigma(4096.0)
    assert (num, den) == (4096 << 20, 1 << 20)
    x = XOF(b"KAT", b"gaussrat")
    assert [gaussian_int(x, num, den) for _ in range(6)] == \
        [-4672, -3013, -899, -6424, 5142, 5529]


def test_kat_oom_challenge():
    x = XOF(b"KAT", b"chal")
    c = sample_challenge(x, 32, 32, 16, 61)
    assert c[:8] == [8, 16, 51, 12, 46, 58, 56, 49]
    assert all(v != 0 for v in c), "w = d, so every coefficient is nonzero"


def test_kat_lanes_challenge():
    """Pinned in centred form, which `q~` cannot move.

    The residue form is `q~`-dependent, so the modulus change would
    otherwise read as a sampler change.  It is not one: the same XOF bits
    produce the same signs at the same positions, and the assertion below on
    the residues checks that the centring is the only difference.
    """
    x = XOF(b"KAT", b"lchal")
    c = lanes_params.sample_challenge(x)
    centred = [v - R.QTILDE if v > R.QTILDE // 2 else v for v in c]
    assert centred[:8] == [0, -1, 1, 0, 0, 0, 0, 0]
    assert c[:8] == [0, R.QTILDE - 1, 1, 0, 0, 0, 0, 0]
    assert lanes_params.challenge_l1_norm(c) == 44


# ---- layer 3: acceptance thresholds -------------------------------------

def test_kat_exp_threshold():
    assert exp_threshold(-1, 2) == \
        3807254656647399603509906376902738731170839988217701731003


def test_kat_rej1():
    x = XOF(b"KAT", b"r1")
    assert [rej1(x, [1, 2, 3], [1, 0, -1], 20, 1000, 1)
            for _ in range(10)] == \
        [1, 0, 0, 0, 1, 1, 1, 1, 0, 0]


def test_kat_rej2():
    """Exercises both the half-space branch and the exponential one."""
    x = XOF(b"KAT", b"r2")
    assert [rej2(x, [40, 50, 60], [1, 0, 1], 3, 50, 1) for _ in range(10)] == \
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # <z, v> < 0 is rejected outright, with no exponential evaluated
    x = XOF(b"KAT", b"r2neg")
    assert all(rej2(x, [1, 2, 3], [1, 0, -1], 3, 50, 1) == 1
               for _ in range(4))


def test_exp_accept_agrees_with_the_exact_threshold():
    """The fast bracket must never change a decision, only its cost."""
    import random
    rng = random.Random(20260730)
    for _ in range(600):
        den = rng.randrange(1, 10 ** 6)
        num = -rng.randrange(0, 130 * den)
        exact = exp_threshold(num, den)
        for u in (0, max(0, exact - 1), exact, exact + 1,
                  rng.randrange(1 << PROB_BITS)):
            assert exp_accept(u, num, den) == (u < exact), (num, den, u)


# ---- layer 4: the bit codec ---------------------------------------------

def test_kat_rice_parameters():
    """`k` is wire-visible: a different choice is a different encoding."""
    assert optimal_rice_k(352) == 8
    assert optimal_rice_k(4096) == 12
    assert optimal_rice_k(1.5e7) == 24


def test_kat_rice_encoding():
    coder = Rice(352, 4970)
    w = BitWriter()
    for value in (0, 1, -1, 255, -256, 4970, -4970):
        coder.write(w, value)
    blob = w.to_bytes()
    assert blob.hex() == "000208f01f80aafdff1fb5ffff0b"
    r = BitReader(blob)
    assert [coder.read(r) for _ in range(7)] == \
        [0, 1, -1, 255, -256, 4970, -4970]


def test_kat_uniform_and_signed_widths():
    assert Uniform(61).width == 6
    assert Uniform(67112897).width == 27          # the pre-1-Aug q~
    assert Uniform(427634113).width == 29         # q~, just below 2^29
    assert Signed(1).width == 2                   # ternary
    assert Signed(16).width == 6                  # the OOM challenge range


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_kat.py: {len(tests)} tests passed")
