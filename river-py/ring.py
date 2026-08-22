"""
ring.py -- Arithmetic in R_q = Z_q[X] / (X^d + 1), plus the rounding and
bit-dropping primitives RiVeR builds on.

A polynomial is a plain `list[int]` of length `d` holding *unsigned canonical*
coefficients in `[0, q)`.  Centred representatives in `(-q/2, q/2]` are used
for every norm and for the exact-link equation, and are produced explicitly by
`centered()`.

Vectors are `list[poly]`; matrices are `list[list[poly]]` (row major).

Two multiplication paths, both exact and interchangeable:

  * `mul_schoolbook` -- the definition, negacyclic convolution.  Reference.
  * `mul` -- Kronecker substitution: pack both operands into single big
    integers with a limb wide enough that no coefficient of the integer
    product can carry into the next limb, multiply once using Python's
    bignum, then unpack and fold `X^d = -1`.  Same result, ~3x faster for the
    sizes used here, and short enough to read.

`test_ring.py` checks the two against each other on random inputs.

No NTT.  None of RiVeR's own moduli admit one -- `X^32 + 1` splits into just
two factors modulo `p`, `q_0` and `q_hat` -- but that is not the whole reason:
the product could be carried over auxiliary NTT-friendly primes and reduced
afterwards.  That is worthwhile in compiled code and a regression here (an
isolated CRT-NTT multiply measures 7.3x slower than Kronecker at `d = 32` in
CPython, since the gain comes only from amortising transforms across a matrix
product).
"""


class Ring:
    """Arithmetic in R_q = Z_q[X] / (X^d + 1)."""

    def __init__(self, q, d):
        self.q = q
        self.d = d
        self.half_q = q // 2
        # Kronecker limb: every product coefficient is < d * (q-1)^2.
        max_coeff = d * (q - 1) ** 2
        self._limb = (max_coeff.bit_length() + 7) // 8      # bytes per limb
        self._mask = (1 << (8 * self._limb)) - 1

    def __repr__(self):
        return f"Ring(q={self.q}, d={self.d})"

    # ---- element creation ------------------------------------------------

    def zero(self):
        return [0] * self.d

    def one(self):
        r = [0] * self.d
        r[0] = 1
        return r

    def const(self, c):
        """Constant polynomial c mod q."""
        r = [0] * self.d
        r[0] = int(c) % self.q
        return r

    def all_ones(self):
        """The polynomial 1 + X + ... + X^{d-1} (the public centring shift)."""
        return [1] * self.d

    # ---- representative conversions --------------------------------------

    def reduce(self, a):
        q = self.q
        return [int(c) % q for c in a]

    def centered(self, a):
        """Unsigned [0,q) -> centred (-q/2, q/2]."""
        h, q = self.half_q, self.q
        return [c - q if c > h else c for c in a]

    def from_centered(self, a):
        q = self.q
        return [c % q for c in a]

    def vec_centered(self, v):
        return [self.centered(a) for a in v]

    # ---- element-wise arithmetic -----------------------------------------

    def add(self, a, b):
        q = self.q
        return [(a[i] + b[i]) % q for i in range(self.d)]

    def sub(self, a, b):
        q = self.q
        return [(a[i] - b[i]) % q for i in range(self.d)]

    def neg(self, a):
        q = self.q
        return [(-c) % q for c in a]

    def scale(self, c, a):
        """Integer scalar times polynomial."""
        q = self.q
        c = int(c) % q
        return [(c * ai) % q for ai in a]

    # ---- multiplication ---------------------------------------------------

    def mul_schoolbook(self, a, b):
        """Negacyclic convolution, straight from the definition."""
        d, q = self.d, self.q
        c = [0] * d
        for i in range(d):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(d):
                k = i + j
                if k < d:
                    c[k] += ai * b[j]
                else:
                    c[k - d] -= ai * b[j]
        return [x % q for x in c]

    def mul(self, a, b):
        """Negacyclic convolution via Kronecker substitution."""
        d, q, limb = self.d, self.q, self._limb
        pa = int.from_bytes(b"".join(x.to_bytes(limb, "little") for x in a),
                            "little")
        pb = int.from_bytes(b"".join(x.to_bytes(limb, "little") for x in b),
                            "little")
        prod = (pa * pb).to_bytes((2 * d) * limb, "little")
        out = [0] * d
        for k in range(d):
            lo = int.from_bytes(prod[k * limb:(k + 1) * limb], "little")
            hi = int.from_bytes(prod[(k + d) * limb:(k + d + 1) * limb],
                                "little")
            out[k] = (lo - hi) % q                 # X^d = -1
        return out

    # ---- norms (always on the centred representation) --------------------

    def inf_norm(self, a):
        h, q = self.half_q, self.q
        return max((q - c if c > h else c) for c in a)

    def l2_norm_sq(self, a):
        h, q = self.half_q, self.q
        return sum((c - q) ** 2 if c > h else c * c for c in a)

    def l1_norm(self, a):
        h, q = self.half_q, self.q
        return sum((q - c if c > h else c) for c in a)

    # ---- vector operations -----------------------------------------------

    def vec_zero(self, n):
        return [self.zero() for _ in range(n)]

    def vec_add(self, u, v):
        return [self.add(u[i], v[i]) for i in range(len(u))]

    def vec_sub(self, u, v):
        return [self.sub(u[i], v[i]) for i in range(len(u))]

    def vec_neg(self, v):
        return [self.neg(x) for x in v]

    def vec_mul(self, c_poly, v):
        """Ring element c_poly times each entry of v."""
        return [self.mul(c_poly, x) for x in v]

    def vec_scale(self, c, v):
        """Integer scalar times each entry of v."""
        return [self.scale(c, x) for x in v]

    def inner(self, u, v):
        """<u, v> = sum_i u_i v_i in R_q."""
        q, d = self.q, self.d
        acc = [0] * d
        for i in range(len(u)):
            t = self.mul(u[i], v[i])
            for j in range(d):
                acc[j] += t[j]
        return [x % q for x in acc]

    def vec_inf_norm(self, v):
        return max((self.inf_norm(x) for x in v), default=0)

    def vec_l2_norm_sq(self, v):
        return sum(self.l2_norm_sq(x) for x in v)

    def vec_inner_int(self, u, v):
        """<vec(u), vec(v)> over Z, on centred coefficients.

        This is the inner product the rejection samplers use: the vectors are
        flattened coefficient vectors, not ring elements.
        """
        total = 0
        for a, b in zip(u, v):
            ca, cb = self.centered(a), self.centered(b)
            total += sum(x * y for x, y in zip(ca, cb))
        return total

    # ---- matrix operations -----------------------------------------------

    def mat_vec(self, M, v):
        return [self.inner(row, v) for row in M]

    # ---- helpers ---------------------------------------------------------

    def vec_concat(self, *vecs):
        out = []
        for v in vecs:
            out.extend(v)
        return out


# ---- rounding (Fact 1) ---------------------------------------------------

def round_p(ring_q, a, q0):
    """`floor(a)_p`: coefficient-wise canonical rep in [0,q), integer-divided
    by q_0 = q/p.  Returns the quotient coefficients (an element of R_p)."""
    return [c // q0 for c in a]


def round_p_vec(ring_q, v, q0):
    return [round_p(ring_q, a, q0) for a in v]


def rounding_error(ring_q, a, rounded, q0):
    """`a - q_0 * rounded mod q`, the canonical rounding error in [0, q_0-1].

    Together with `round_p` this is exactly Fact 1: `v = floor(u)_p` iff there
    is an `e` with coefficients in `[0, q/p - 1]` and `e = u - (q/p) v mod q`.
    """
    q = ring_q.q
    return [(a[i] - q0 * rounded[i]) % q for i in range(ring_q.d)]


# ---- bit dropping --------------------------------------------------------
# `[[u]]_K` (high bits) and `u mod^pm 2^K` (centred low bits).  The paper's
# Preliminaries define both, on the *centred* representative:
#
#   a mod^pm 2^K := \bar a - 2^K floor((\bar a + 2^{K-1} - 1) / 2^K)
#   [[a]]_K      := (\bar a - (a mod^pm 2^K)) / 2^K
#
# with the low part in (-2^{K-1}, 2^{K-1}] -- closed at the top, which is the
# tie `mod_pm` below has always used.  (The form was the other way
# round; that difference is gone.)  `mod_pm` is representative-independent,
# so the only thing the definition fixes is which representative goes in.
#
# `oom.py::OOM._high_low` centres before calling `power2round`, so about
# half the high parts are negative and the transmitted `B` field is signed;
# `test_ring.py::test_paper_high_bits_are_power2round_on_the_centred_representative`
# pins it.

# ---- the centred range shift --------------------------------------------
#
# REPAIR.  The rounding relations are
# written throughout the paper with errors in `[0, q_0-1]`, and the `Eval`
# figure builds the OOM targets as `c_i = (q_0 t_i, q_0 v)` with no offset.
# The parameter derivation nevertheless uses `B_e = floor(q_0/2) = 30`,
# saying only that "the range proved by the underlying proof system can be
# translated so that it is centered at zero".  The algorithms never define
# that translation, and without it 30 is not a valid norm bound on a
# coefficient that ranges over `[0, 60]`.
#
# It is not presentational.  The selected LANES modulus clears
# `24 phi_m eta_m` by 0.56% with `B_e = 30`; with the literal 60 the
# requirement doubles and the modulus fails outright
# (`exact.ExactParams.q_tilde_clears`).
#
# So the shift is carried explicitly, and only here: the OOM witness is
# `e^c = e - B_e in [-B_e, B_e]`, every public target gains `+B_e`, and the
# exact relation proves `e^c + B_e = d_0 + 3d_1 + 9d_2 + 17d_3` with digits
# in `{0,1,2}`.  Keeping it behind these two names means a later
# clarification from the authors changes one boundary rather than the scheme.


def to_centered_error(coeffs, B_e):
    """`[0, q_0-1]` rounding error -> the centred OOM witness `[-B_e, B_e]`."""
    out = [c - B_e for c in coeffs]
    for c in out:
        if not -B_e <= c <= B_e:
            raise ValueError(f"centred error {c} outside [-{B_e}, {B_e}]")
    return out


def from_centered_error(coeffs, B_e):
    """The inverse: centred witness -> the `[0, q_0-1]` range the relation
    states, which is what the radix-3 decomposition consumes."""
    out = [c + B_e for c in coeffs]
    for c in out:
        if not 0 <= c <= 2 * B_e:
            raise ValueError(f"canonical error {c} outside [0, {2 * B_e}]")
    return out


def mod_pm(value, K):
    """`value mod^pm 2^K`: the centred representative in (-2^{K-1}, 2^{K-1}].

    The argument is the **exponent** `K`, not the modulus `2^K`.  It used
    to be the modulus, which reads identically at every call site --
    `mod_pm(c, 1 << K_a)` and `mod_pm(c, K_a)` both run -- and gives
    silently different answers.  The port takes `K`, so the two agreed
    only by the accident of nobody writing the shorter one; the KAT now
    pins it, and this signature makes the two spellings impossible to
    confuse.  No value changes: every caller passed `1 << K`.
    """
    power = 1 << K
    low = value % power
    if low > power // 2:
        low -= power
    return low


def power2round(value, K):
    """Return (high, low) with value = high * 2^K + low and |low| <= 2^{K-1}."""
    low = mod_pm(value, K)
    return (value - low) >> K, low


def high_bits(ring, a, K):
    """`[[a]]_K` applied coefficient-wise to the *centred* representation."""
    return [power2round(c, K)[0] for c in ring.centered(a)]


def low_bits(ring, a, K):
    """`a mod^pm 2^K` applied coefficient-wise to the centred representation."""
    return [power2round(c, K)[1] for c in ring.centered(a)]


def high_bits_vec(ring, v, K):
    return [high_bits(ring, a, K) for a in v]


def low_bits_vec(ring, v, K):
    return [low_bits(ring, a, K) for a in v]


def int_vec_inf_norm(v):
    """||.||_inf of a list of plain integer lists (already centred)."""
    return max((max(abs(c) for c in a) for a in v), default=0)


def negacyclic_mul_int(a, b):
    """Product in Z[X]/(X^d + 1) over the *integers* -- no modular reduction.

    The exact relation R^_ex requires `z_eval = x e_eval + y_eval` as an
    equality over Z rather than modulo any protocol modulus (Section 2.7:
    "an exact equality over the canonical coefficient representatives, rather
    than an equality only modulo q").  Every modular `mul` in this file would
    silently satisfy a weaker statement, so the exact layer uses this instead.
    """
    d = len(a)
    out = [0] * d
    for i in range(d):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(d):
            k = i + j
            if k < d:
                out[k] += ai * b[j]
            else:
                out[k - d] -= ai * b[j]
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    R = Ring(1073123348676497, 32)          # q = 61 * (2^44-ish prime)

    assert R.mul(R.one(), R.one()) == R.one()
    assert R.add(R.zero(), R.one()) == R.one()

    # X^d == -1
    x = R.zero()
    x[1] = 1
    prod = x
    for _ in range(R.d - 1):
        prod = R.mul(prod, x)
    assert prod == R.const(-1), "X^d == -1"

    rng = random.Random(1)
    def rand_poly():
        return [rng.randrange(R.q) for _ in range(R.d)]

    for _ in range(20):
        a, b, c = rand_poly(), rand_poly(), rand_poly()
        assert R.mul(a, b) == R.mul_schoolbook(a, b), "Kronecker == schoolbook"
        assert R.mul(a, b) == R.mul(b, a)
        assert R.mul(R.mul(a, b), c) == R.mul(a, R.mul(b, c))
        assert R.mul(a, R.add(b, c)) == R.add(R.mul(a, b), R.mul(a, c))

    # power2round round-trip
    for _ in range(200):
        v = rng.randrange(-(1 << 40), 1 << 40)
        hi, lo = power2round(v, 13)
        assert hi * (1 << 13) + lo == v and abs(lo) <= 1 << 12

    # Fact 1 round-trip
    q0 = 61
    a = rand_poly()
    v = round_p(R, a, q0)
    e = rounding_error(R, a, v, q0)
    assert all(0 <= c <= q0 - 1 for c in e), "canonical rounding error range"
    assert all((q0 * v[i] + e[i]) % R.q == a[i] for i in range(R.d))

    print("ring.py: all self-tests passed")
