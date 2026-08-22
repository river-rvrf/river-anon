"""
test_lanes_ring.py -- The LANES ring algebra, checked in the test suite.

**Not gated.**  Every property here follows from `(q~, d~, l)` alone, all
three of which the revision supplies; none of it needs the sampler widths,
response bounds or hint rules that were once withheld.  Since the paper
publishes those in closed form the rest of the port constructs too, so
`test_lanes.py`'s module-level skip guard no longer fires -- but the guard
is still there, and this file is the part that must run whether it fires or
not.

It exists because the gate hid a real defect.  `lanes_ring.py`'s
`__main__` already asserted `mul == mul_schoolbook`, which fails instantly
on a wrong twiddle tree; when the whole module was gated, that assertion went with it, and the tree was wrong for exactly
as long.  A self-check that only runs under `make selftest` is one gate
away from silence, so the same algebra is asserted here too.

What made the defect survivable is worth stating, because it will recur:
the transform **round-tripped**.  `intt(ntt(a)) == a` holds for a wrong
tree as long as the inverse repeats the same mistake, so it proves nothing
about whether the slots correspond to `R_q~`.  The properties that do bite
are agreement with schoolbook, and the factorisation identity itself.
"""

import random

import lanes_ring as R
from exact import ExactParams
from params import TOY_PARAMS

EX = ExactParams(TOY_PARAMS)


def _schoolbook(a, b):
    """Negacyclic product, computed without touching the NTT."""
    q, d = R.QTILDE, R.DTILDE
    out = [0] * (2 * d - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % q
    res = out[:d]
    for i in range(d, 2 * d - 1):
        res[i - d] = (res[i - d] - out[i]) % q
    return res


# ---- the ring the paper specifies ----------------------------------------

def test_dimensions_come_from_the_exact_layer():
    """`(d~, l) = (256, 64)`, with the slot degree following."""
    assert (R.DTILDE, R.LSPLIT) == (EX.d_tilde, EX.l_split) == (256, 64)
    assert R.SUBDEG == R.DTILDE // R.LSPLIT == 4
    assert R.LEVELS == R.LSPLIT.bit_length() - 1 == 6
    assert R.QTILDE == EX.q_tilde == 67107713


def test_the_modulus_admits_exactly_this_splitting():
    """`q~ - 1 = 2^7 · odd`, so 128th roots exist and 256th do not.

    That makes `l = 64` the *finest* splitting available: 128 blocks of
    degree 2 would need a primitive 256th root, and there is none.  So
    `SUBDEG = 4` is forced by the modulus rather than chosen.
    """
    n = R.QTILDE - 1
    two_adic = 0
    while n % 2 == 0:
        n //= 2
        two_adic += 1
    assert two_adic == 7
    assert 2 * R.LSPLIT == 128 <= 2 ** two_adic
    assert 4 * R.LSPLIT == 256 > 2 ** two_adic, "l = 64 is maximal"
    assert R.QTILDE % (4 * R.LSPLIT) == 2 * R.LSPLIT + 1 == 129


def test_psi_is_a_primitive_2l_th_root():
    assert R.PSI_ORDER == 2 * R.LSPLIT == 128
    assert pow(R.PSI, R.PSI_ORDER, R.QTILDE) == 1
    assert pow(R.PSI, R.LSPLIT, R.QTILDE) == R.QTILDE - 1, "psi^l = -1"


# ---- the tree ------------------------------------------------------------

def test_leaf_exponents_are_the_odd_residues():
    """One leaf per odd exponent in `[1, 2l)`, each exactly once.

    The tree used to start at the literal `32` and offset by `64 // 2` --
    the values `LSPLIT` and `PSI_ORDER // 2` take at `d~ = 128`.  At
    `d~ = 256` that produced `0..63`, evens included.
    """
    exps = R.LEAF_EXPS
    assert len(exps) == R.LSPLIT
    assert len(set(exps)) == R.LSPLIT
    assert all(e % 2 == 1 for e in exps)
    assert set(exps) == set(range(1, 2 * R.LSPLIT, 2))


def test_the_leaves_multiply_to_the_ring_modulus():
    """`prod_j (X^SUBDEG - zeta_j) == X^{d~} + 1`.

    This is the property that makes the incomplete NTT an isomorphism, and
    the one a wrong tree violates.  It is strictly stronger than checking
    the exponents' parity, and unlike the round trip it cannot be
    satisfied by a matching pair of errors.
    """
    assert R.leaf_product() == R.negacyclic_modulus()


def test_the_tree_is_derived_not_hard_coded():
    """Rebuild it from `LSPLIT` and compare, so a literal cannot creep in."""
    exps = [R.LSPLIT]
    for _ in range(R.LEVELS):
        exps = [x for e in exps for x in (e // 2, e // 2 + R.LSPLIT)]
    assert exps == R.LEAF_EXPS
    assert R.LEAF_ZETA == [pow(R.PSI, e, R.QTILDE) for e in R.LEAF_EXPS]


# ---- multiplication ------------------------------------------------------

def test_ntt_product_agrees_with_schoolbook():
    """The check the gate switched off.

    Dense, sparse and structured operands: a wrong tree disagrees on
    essentially every coefficient, so this fails loudly rather than
    subtly.
    """
    rng = random.Random(7)
    for _ in range(12):
        a = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
        b = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
        assert R.mul(a, b) == _schoolbook(a, b)
        assert R.mul(a, b) == R.mul_schoolbook(a, b)

    for _ in range(8):
        a = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
        b = [0] * R.DTILDE
        for _ in range(4):
            b[rng.randrange(R.DTILDE)] = rng.randrange(R.QTILDE)
        assert R.mul(a, b) == _schoolbook(a, b)


def test_the_algebraic_laws_do_not_detect_a_wrong_tree():
    """The trap, stated as a property and measured.

    Reintroducing the old hard-coded tree and re-running this file, 5 of
    its 10 tests fail.  The five that **pass** are worth naming, because
    they are the ones a reviewer reaches for first:

      * `intt(ntt(a)) == a` -- the inverse repeats whatever the forward
        transform did, wrong or not;
      * slot independence -- a property of any *diagonal* transform;
      * associativity, distributivity, commutativity -- likewise.

    All of these hold for a transform that is not the ring isomorphism at
    all.  Only two things bite: agreement with an independent schoolbook
    product, and the factorisation identity
    `prod_j (X^SUBDEG - zeta_j) == X^{d~} + 1`.  Everything else is
    necessary and nowhere near sufficient.
    """
    rng = random.Random(3)
    for _ in range(8):
        a = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
        assert R.intt(R.ntt(a)) == a

    a = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
    b = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
    c = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
    assert R.mul(R.mul(a, b), c) == R.mul(a, R.mul(b, c))
    assert R.mul(a, R.add(b, c)) == R.add(R.mul(a, b), R.mul(a, c))
    assert R.mul(a, b) == R.mul(b, a)


def test_ring_identities():
    """`1`, `X`, and `X^{d~} = -1`."""
    one = [0] * R.DTILDE
    one[0] = 1
    a = list(range(1, R.DTILDE + 1))
    assert R.mul(one, a) == a

    x = [0] * R.DTILDE
    x[1] = 1
    top = [0] * R.DTILDE
    top[R.DTILDE - 1] = 1
    minus_one = [0] * R.DTILDE
    minus_one[0] = R.QTILDE - 1
    assert R.mul(x, top) == minus_one


# ---- slots ---------------------------------------------------------------

def test_slots_are_independent_under_multiplication():
    """Slot `j` is NTT index `j·SUBDEG`, and slots do not mix.

    `[ENS20]` relies on this throughout: it is what lets one committed
    scalar per block be multiplied independently.

    Note this is *not* a check on the tree: it holds for any diagonal
    transform, wrong twiddles included, and it passed throughout the
    period the tree was broken.  See
    `test_the_algebraic_laws_do_not_detect_a_wrong_tree`.
    """
    rng = random.Random(5)
    u = [rng.randrange(R.QTILDE) for _ in range(R.LSPLIT)]
    v = [rng.randrange(R.QTILDE) for _ in range(R.LSPLIT)]
    assert R.ntt_to_slots(R.slots_to_ntt(u)) == u

    prod = R.mul(R.intt(R.slots_to_ntt(u)), R.intt(R.slots_to_ntt(v)))
    got = R.ntt_to_slots(R.ntt(prod))
    assert got == [(a * b) % R.QTILDE for a, b in zip(u, v)]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_lanes_ring.py: {len(tests)} tests passed")
