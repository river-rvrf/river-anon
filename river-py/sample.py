"""
sample.py -- Deterministic XOF-driven samplers for RiVeR.

Everything random in the scheme is drawn from a SHAKE-256 stream, so a full
transcript is reproducible from its seeds.  That is what makes the test-vector
generator (`vectors.py`) meaningful across implementations.

Determinism notes
-----------------
* The paper says `Expand` / `SamMat` derive matrices from the public seed
  `rho` but never names an XOF.  We use SHAKE-256 in counter mode:
  `read()` returns SHAKE256(seed || counter_le8) blocks.  Python's `hashlib`
  SHAKE has no streaming squeeze, and counter mode gives an unbounded stream
  with a one-line description.

* Absorption is injective: every part is prefixed with its 8-byte
  little-endian length, so `("ab","c")` and `("a","bc")` cannot collide.

* Acceptance probabilities (Gaussian sampling, `Rej_1`, `Rej_2`) are compared
  in **fixed-point integer** form, with the threshold computed through
  `decimal` at a pinned precision.  `Decimal.exp()` is correctly rounded by
  specification, so the accept/reject decision is bit-identical on every
  platform -- unlike a `math.exp` comparison, which can differ in the last
  ulp and silently fork a test vector.
"""

import hashlib
from decimal import Decimal, localcontext
from functools import lru_cache

SHAKE_BLOCK = 136                # SHAKE-256 rate, in bytes

#: Fixed-point width of every acceptance test.
#:
#: This is coupled to `GAUSSIAN_TAILCUT` and the coupling is easy to miss.
#: `gaussian_int` accepts when `unit_fixed() < floor(2^PROB_BITS * exp(-z^2 /
#: 2 sigma^2))`.  Once that floor reaches zero the value can never be
#: accepted, so the *effective* tail cut is
#:
#:     |z| / sigma  <=  sqrt(2 * PROB_BITS * ln 2),
#:
#: whatever `GAUSSIAN_TAILCUT` says.  At the original 64 bits that is
#: 9.4193 sigma: declaring a 14 sigma cut bought nothing beyond 9.42, and the
#: real per-transcript statistical distance was `2^-52.9`, not `2^-128`.
#:
#: 192 bits puts the hard zero at 16.3 sigma, comfortably past the declared
#: cut, and leaves the threshold at `14 sigma` around `2^50` so the *floor*
#: itself is not the dominant error either.  Quantisation inside the support
#: then contributes about `2^-174` over a whole transcript.
#:
#: `_check_probability_width()` asserts the relationship rather than leaving
#: it to a comment; it runs in `__main__` and in `test_sample.py`.
PROB_BITS = 192
PROB_ONE = 1 << PROB_BITS

#: Decimal digits for `exp()`.  Must exceed `PROB_BITS * log10(2)` = 57.8 by
#: enough that the last fixed-point bit is correct; 80 leaves 22 digits.
EXP_PRECISION = 80

#: Verifier bound, as a multiple of sigma: `Rej` returns `bot` when
#: `||z||_inf > 6 sigma`.  This one is the paper's -- Figure 2, and the
#: verifier checks of Subsection J.1 (`2.pre.tex:298`, `7.suppl.tex:3373`).
#: It is a *protocol* constant: `beta_sel,inf` and the MSIS estimate are
#: derived from it, so it cannot be changed without redoing the parameters.
VERIFIER_TAILCUT = 6

#: Tail cut for the discrete Gaussian sampler, as a multiple of sigma.
#:
#: NOT the same constant as `VERIFIER_TAILCUT`, and deliberately not the same
#: value.  That one truncates the *response* `z = y + v` and is part of the
#: protocol; this one truncates the *mask* `y` and is purely an artefact of
#: how `gaussian_int` is implemented.  A mask just outside the cut can be
#: shifted back inside by `v`, so the two truncations are not interchangeable:
#: sampling `y` at the verifier's own `6 sigma` makes those transcripts
#: unreachable and shifts the honest distribution away from the one the
#: protocol defines.
#:
#: Sizing it: `dgs.statistical_tailcut` union-bounds the per-coefficient tail
#: mass over every Gaussian coefficient a transcript publishes -- about
#: `2^14.7` of them at `RiVeR-N8` -- and requires the total below `2^-128`.
#: That returns 14 for all five published profiles, landing at `2^-130.8`.
#: `python dgs.py` prints the table and re-checks this constant against it.
#:
#: Why not the smaller cut available from the HPRR19 Rényi route
#: (`dgs.renyi_tailcut`, which lands on 5): it
#: needs the tail below `1/(8 Q)` rather than `2^-lambda`, and returns 6 here.
#: It is sound for *search* problems -- unforgeability -- because that is
#: where Rényi divergence has probability preservation.  The mask this sampler
#: produces protects **anonymity**, a decision problem, so the statistical
#: route is the defensible one for these masking widths.
#:
#: The protocol uses the concrete rejection constant 12 and states the
#: corresponding statistical loss through the standard parameterised
#: rejection-sampling argument.  The concrete `6 sigma` truncation therefore
#: no longer
#: contradicts a distance the lemma asserts.
#:
#: The reason for raising this cut is unchanged, and is local: the
#: protocol truncates the *response* at `VERIFIER_TAILCUT` sigma, which
#: costs `2^-14.2` per transcript however `y` is sampled.  Sampling the
#: mask at the verifier's own cut would add a second, avoidable loss of the
#: same order. `test_review.py` pins both quantities independently.
GAUSSIAN_TAILCUT = 14

#: The fixed concrete constant in ``M_1 = exp(12/phi + 1/(2 phi^2))``.
#: It is an internal part of the concrete Rej_1 definition, not a profile
#: parameter or a transcript field.
REJ1_CONSTANT = 12


# ---- domain separators ---------------------------------------------------
# The paper requires all oracles to be pairwise domain-separated.

DS_EXPAND = b"RiVeR.Expand"        # SamMat / matrix derivation from rho
DS_G = b"RiVeR.G"                  # G : {0,1}* -> R_q^ell
DS_CHALLENGE = b"RiVeR.H"          # H : {0,1}* -> C^d_{w,gamma}
DS_COMMIT = b"RiVeR.Com"           # OOM commitment randomness
DS_EXACT = b"RiVeR.Exact"          # exact-layer randomness / commitment
DS_DUMMY = b"RiVeR.dummy"          # CanonPad dummy public keys
DS_KEYGEN = b"RiVeR.KeyGen"


# ---- XOF -----------------------------------------------------------------

def absorb(domain, *parts):
    """Injectively hash a domain separator and byte parts to a 64-byte seed."""
    h = hashlib.shake_256()
    h.update(len(domain).to_bytes(8, "little"))
    h.update(domain)
    for part in parts:
        if isinstance(part, int):
            part = part.to_bytes(8, "little")
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.digest(64)


def hash_bytes(length, domain, *parts):
    """One-shot domain-separated hash to `length` bytes."""
    return hashlib.shake_256(absorb(domain, *parts)).digest(length)


class XOF:
    """Unbounded deterministic byte stream: SHAKE256(seed || counter)."""

    __slots__ = ("_seed", "_counter", "_buf", "_pos")

    def __init__(self, domain, *parts):
        self._seed = absorb(domain, *parts)
        self._counter = 0
        self._buf = b""
        self._pos = 0

    def _refill(self, needed):
        blocks = []
        have = len(self._buf) - self._pos
        if have:
            blocks.append(self._buf[self._pos:])
        while have < needed:
            block = hashlib.shake_256(
                self._seed + self._counter.to_bytes(8, "little")
            ).digest(SHAKE_BLOCK)
            self._counter += 1
            blocks.append(block)
            have += SHAKE_BLOCK
        self._buf = b"".join(blocks)
        self._pos = 0

    def read(self, n):
        if n <= 0:
            return b""
        if len(self._buf) - self._pos < n:
            self._refill(n)
        out = self._buf[self._pos:self._pos + n]
        self._pos += n
        return out

    def uint(self, nbytes):
        return int.from_bytes(self.read(nbytes), "little")

    def unit_fixed(self):
        """Uniform value in [0, 2^PROB_BITS), the left side of every
        acceptance comparison."""
        return self.uint(PROB_BITS // 8)

    def bit(self):
        return self.read(1)[0] & 1


# ---- fixed-point exp -----------------------------------------------------

def exp_threshold(numerator, denominator, scale=None):
    """floor(scale * exp(numerator / denominator)) as an exact integer.

    `numerator / denominator` is an exact rational; the exponential is
    evaluated with `decimal` at `EXP_PRECISION` digits, which is reproducible
    on every platform.  Values >= 1 are clamped to `scale` (always-accept),
    which is what the callers want.
    """
    if scale is None:
        scale = PROB_ONE
    # exp(x) * scale floors to 0 once x < -ln(scale); decide that exactly,
    # in integers, without touching decimal.
    if denominator > 0 and numerator < -(scale.bit_length() + 1) * denominator:
        return 0
    with localcontext() as ctx:
        ctx.prec = EXP_PRECISION
        value = (Decimal(numerator) / Decimal(denominator)).exp()
        if value >= 1:
            return scale
        return int(Decimal(scale) * value)


#: Working precision for the cheap bracket in `exp_accept`.  Only the width of
#: the resulting interval depends on it, never the decision.
EXP_FAST_PRECISION = 30


def exp_accept(u, numerator, denominator, scale=None):
    """`u < floor(scale * exp(numerator / denominator))`, decided exactly.

    Same predicate as comparing against `exp_threshold`, and always the same
    answer -- but evaluated in two stages, because `EXP_PRECISION` has to be
    wide enough for a `2^192` fixed point and `Decimal.exp` at 80 digits is
    5.5x the cost of the same call at 30.

    Stage one evaluates at `EXP_FAST_PRECISION` and brackets the true
    threshold.  `Decimal.exp` is correctly rounded, so a relative slack of
    `10^-(prec - 2)` is a sound enclosure.  If `u` falls outside the bracket
    the comparison is already settled; only when it lands inside -- about
    `2^-79` of the time, since `u` is uniform over `2^PROB_BITS` and the
    bracket spans roughly `2^113` -- does the exact path run.
    """
    if scale is None:
        scale = PROB_ONE
    if denominator > 0 and numerator < -(scale.bit_length() + 1) * denominator:
        return False                       # threshold is 0: never accept
    with localcontext() as ctx:
        ctx.prec = EXP_FAST_PRECISION
        value = (Decimal(numerator) / Decimal(denominator)).exp()
        if value >= 1:
            return u < scale               # clamped: exact, no bracket needed
        slack = Decimal(10) ** -(EXP_FAST_PRECISION - 2)
        lo = int(Decimal(scale) * value * (1 - slack))
        hi = int(Decimal(scale) * value * (1 + slack)) + 1
    if u < lo:
        return True                        # u < lo <= threshold
    if u >= hi - 1:
        return False                       # threshold <= hi - 1 <= u
    return u < exp_threshold(numerator, denominator, scale)


# ---- uniform samplers ----------------------------------------------------

def uniform_int(xof, modulus):
    """Uniform in [0, modulus) by rejection on whole bytes."""
    bits = modulus.bit_length()
    nbytes = (bits + 7) // 8
    mask = (1 << bits) - 1
    while True:
        value = int.from_bytes(xof.read(nbytes), "little") & mask
        if value < modulus:
            return value


def uniform_poly(xof, modulus, d):
    return [uniform_int(xof, modulus) for _ in range(d)]


def uniform_vec(xof, modulus, d, length):
    return [uniform_poly(xof, modulus, d) for _ in range(length)]


def uniform_matrix(xof, modulus, d, rows, cols):
    return [[uniform_poly(xof, modulus, d) for _ in range(cols)]
            for _ in range(rows)]


def sam_mat(seed, modulus, rows, cols, d, label):
    """`SamMat(rho, q, n, m, str)` of Figure 3.

    Deterministic in (seed, modulus, shape, label); both prover and verifier
    call it, and nothing about it is transmitted.
    """
    xof = XOF(DS_EXPAND, seed, label.encode(),
              modulus.to_bytes((modulus.bit_length() + 7) // 8, "little"),
              rows.to_bytes(4, "little"), cols.to_bytes(4, "little"),
              d.to_bytes(4, "little"))
    return uniform_matrix(xof, modulus, d, rows, cols)


def uniform_beta_poly(xof, beta, d, modulus):
    """One draw from `U_beta`: coefficients uniform in [-beta, beta]."""
    span = 2 * beta + 1
    return [(uniform_int(xof, span) - beta) % modulus for _ in range(d)]


def uniform_beta_vec(xof, beta, d, length, modulus):
    return [uniform_beta_poly(xof, beta, d, modulus) for _ in range(length)]


# ---- discrete Gaussian ---------------------------------------------------

def gaussian_int(xof, sigma_num, sigma_den=1, tailcut=GAUSSIAN_TAILCUT):
    """One sample from D_sigma, truncated at `tailcut` * sigma.

    Uniform proposal on the truncated support, accepted with probability
    exp(-z^2 / (2 sigma^2)).  The acceptance rate is about
    sqrt(2 pi) / (2 * tailcut) ~= 0.09 at the default 14, so roughly eleven
    draws per sample.  Simple, exact in the accept/reject decision, and --
    like every sampler here -- not constant time.

    The cost is linear in `tailcut`, which is why a production sampler would
    use a CDT or Karney rather than a uniform proposal; see `GAUSSIAN_TAILCUT`
    for why the cut cannot simply be lowered to compensate.
    """
    check_probability_width(PROB_BITS, tailcut)
    bound = (tailcut * sigma_num) // sigma_den
    span = 2 * bound + 1
    # 2 sigma^2 as an exact rational  two_s2_num / two_s2_den
    two_s2_num = 2 * sigma_num * sigma_num
    two_s2_den = sigma_den * sigma_den
    while True:
        z = uniform_int(xof, span) - bound
        if z == 0:
            return 0
        if exp_accept(xof.unit_fixed(), -(z * z) * two_s2_den, two_s2_num):
            return z


#: `ln 2` as an exact rational, so `check_probability_width` decides in
#: integers.  `river-rs::sample` uses the same two constants.
_LN2_NUM = 693_147_180_559_945_309
_LN2_DEN = 1_000_000_000_000_000_000

#: Bits of threshold that must remain below the declared cut, or the *floor*
#: rather than the Gaussian decides the last few sigma.
_MIN_HEADROOM_BITS = 32


@lru_cache(maxsize=None)
def check_probability_width(bits, tailcut):
    """Reject a sampler width that `bits` of threshold cannot reach.

    `gaussian_int` accepts when `u < floor(2^bits * exp(-z^2 / 2 sigma^2))`.
    Past `|z|/sigma = sqrt(2 bits ln 2)` that floor is zero, nothing is ever
    accepted there, and a larger declared cut is silently the same
    distribution -- which is how a declared `14 sigma` once behaved as
    `9.42 sigma`.

    The paper requires this to be an **error** rather than a
    no-op, so `gaussian_int` calls it on every width it is handed rather than
    leaving it to a test.  Decided in integers and memoised, so the cost is a
    dictionary lookup per sample; `_check_probability_width` below is the
    exact-`Decimal` statement of the same predicate, and `test_sample.py`
    checks the two agree at every boundary.

    `river-rs::sample::check_probability_width` is this function, called from
    `GaussCtx::new` for the same reason.
    """
    if tailcut <= 0:
        raise ValueError(f"GAUSSIAN_TAILCUT={tailcut} is a point mass, "
                         "not a width")
    spent = -((tailcut * tailcut * _LN2_DEN) // -(2 * _LN2_NUM))     # ceil
    if spent >= bits:
        raise ValueError(
            f"PROB_BITS={bits} caps the sampler below "
            f"GAUSSIAN_TAILCUT={tailcut} (the cut needs {spent} bits)")
    if bits - spent < _MIN_HEADROOM_BITS:
        raise ValueError(
            f"PROB_BITS={bits} leaves only {bits - spent} bits of threshold "
            f"at {tailcut} sigma, below the {_MIN_HEADROOM_BITS} required")


def _check_probability_width(tailcut=None, bits=None):
    """The `Decimal` statement of `check_probability_width`.

    Returns the effective cut `sqrt(2 * bits * ln 2)`, in units of sigma, as
    an exact `Decimal`.  Raises if it does not clear the declared cut with at
    least 32 bits of headroom on the threshold, which is what keeps the
    *floor* from becoming the dominant error term.

    This exists because the two constants are independent in the source and
    coupled in the mathematics: raising `GAUSSIAN_TAILCUT` alone is a silent
    no-op past `sqrt(2 * PROB_BITS * ln 2)`.  It is the reference the integer
    form is checked against; the sampler calls the integer form.
    """
    tailcut = GAUSSIAN_TAILCUT if tailcut is None else tailcut
    bits = PROB_BITS if bits is None else bits
    with localcontext() as ctx:
        ctx.prec = 40
        ln2 = Decimal(2).ln()
        effective = (2 * Decimal(bits) * ln2).sqrt()
        # bits the threshold still has at the declared cut
        headroom = Decimal(bits) - Decimal(tailcut) ** 2 / (2 * ln2)
    if effective <= tailcut:
        raise ValueError(
            f"PROB_BITS={bits} caps the sampler at {effective:.4f} sigma, "
            f"below GAUSSIAN_TAILCUT={tailcut}")
    if headroom < 32:
        raise ValueError(
            f"PROB_BITS={bits} leaves only {headroom:.1f} bits of threshold "
            f"at {tailcut} sigma; the floor would dominate")
    return effective


def gaussian_poly(xof, sigma_num, d, modulus, sigma_den=1):
    return [gaussian_int(xof, sigma_num, sigma_den) % modulus
            for _ in range(d)]


def gaussian_vec(xof, sigma_num, d, length, modulus, sigma_den=1):
    return [gaussian_poly(xof, sigma_num, d, modulus, sigma_den)
            for _ in range(length)]


#: Denominator every Gaussian width is pinned to.  Wire-visible: it fixes
#: the sampler's accept/reject decisions and, through `optimal_rice_k`, the
#: encoding.  A port must use this value, not merely "a rational".
SIGMA_DEN = 1 << 20


def rational_sigma(sigma):
    """Pin a width to the rational `round(sigma * 2^20) / 2^20`.

    Not "sigma exactly", which is what this used to claim: the paper's
    widths are irrational (`B_a = gamma sqrt(2w)`, `eta_m = w gamma B_e
    sqrt d`), so no rational is exact, and the input is already a binary
    float that has rounded once.  What this does is pin *one* rational,
    deterministically, so that the sampler's accept/reject decisions and
    the Rice parameter are the same on both sides of a port rather than
    depending on the last ulp of a `sqrt`.

    Two consequences worth naming, since a port inherits both:

      * the input must be computed in the same operation order, because
        `round` here only removes the *final* float error, not one that has
        already accumulated;
      * `2^20` is part of the wire format.  `manifest.py` freezes the
        resulting `(num, den)` per field and profile, and
        `test_manifest.py` pins them, so a change is a test failure rather
        than a silent re-encoding.
    """
    num = int(round(sigma * SIGMA_DEN))
    return num, SIGMA_DEN


# ---- challenge space -----------------------------------------------------

def sample_challenge(xof, d, w, gamma, modulus):
    """Uniform element of `C^d_{w,gamma}`: exactly `w` of the `d` coefficients
    are nonzero, each uniform in `+-[1, gamma]`.

    The paper fixes the *size* of this set (log2 |C| = log2 C(d,w) +
    w log2(2 gamma) = 160) but never gives a hash-to-C procedure; this is one,
    and it is uniform over the set.  Positions are drawn by a partial
    Fisher-Yates shuffle so the sampler always terminates.
    """
    positions = list(range(d))
    chosen = []
    for i in range(w):
        j = i + uniform_int(xof, d - i)
        positions[i], positions[j] = positions[j], positions[i]
        chosen.append(positions[i])
    poly = [0] * d
    for pos in chosen:
        magnitude = 1 + uniform_int(xof, gamma)        # in [1, gamma]
        sign = -1 if xof.bit() else 1
        poly[pos] = (sign * magnitude) % modulus
    return poly


def challenge_from_hash(d, w, gamma, modulus, *parts):
    """`H(...) -> C^d_{w,gamma}`, the OOM Fiat-Shamir challenge."""
    return sample_challenge(XOF(DS_CHALLENGE, *parts), d, w, gamma, modulus)


# ---- rejection sampling (Figure 2) ---------------------------------------
# Return convention follows the paper: 1 = reject, 0 = accept.

def _p_acc(u, inner_zv, norm_v_sq, sigma_num, sigma_den, M_num, M_den):
    """`u < floor(2^PROB_BITS / M * exp((-2<z,v> + ||v||^2) / (2 sigma^2)))`."""
    exponent_num = (-2 * inner_zv + norm_v_sq) * sigma_den * sigma_den
    exponent_den = 2 * sigma_num * sigma_num
    scale = (PROB_ONE * M_den) // M_num
    return exp_accept(u, exponent_num, exponent_den, scale)


def rej1(xof, z_coeffs, v_coeffs, phi, sigma_num, sigma_den):
    """`Rej_1(z, v, phi, T)`, sigma = phi * T as a rational.

    `z_coeffs` and `v_coeffs` are flat lists of *centred* integer
    coefficients.  Returns 1 to reject, 0 to accept.

    The concrete rejection constant is the internal fixed value 12.  Thus the
    exponent numerator is `24 phi + 1`, and the sampler and repetition model
    share [`REJ1_CONSTANT`] rather than exposing a per-profile argument.
    """
    inner_zv = sum(a * b for a, b in zip(z_coeffs, v_coeffs))
    norm_v_sq = sum(b * b for b in v_coeffs)
    # M = exp(12/phi + 1/(2 phi^2)); over the common denominator `2 phi^2`
    # that exponent is `-(24 phi + 1)`.  Fold 1/M into the threshold.
    m_scale = exp_threshold(-(2 * REJ1_CONSTANT * phi + 1),
                            2 * phi * phi)                 # = 1/M
    if not _p_acc(xof.unit_fixed(), inner_zv, norm_v_sq, sigma_num, sigma_den,
                  PROB_ONE, m_scale):
        return 1
    if _inf_norm(z_coeffs) * sigma_den > VERIFIER_TAILCUT * sigma_num:
        return 1
    return 0


def rej2(xof, z_coeffs, v_coeffs, phi, sigma_num, sigma_den):
    """`Rej_2(z, v, phi, T)`, the optimised sampler.

    Rejects outright when `<z, v> < 0`, which is exactly the half-space hint
    the Extended M-LWE assumption accounts for.
    """
    inner_zv = sum(a * b for a, b in zip(z_coeffs, v_coeffs))
    if inner_zv < 0:
        return 1
    norm_v_sq = sum(b * b for b in v_coeffs)
    m_scale = exp_threshold(-1, 2 * phi * phi)                # = 1/M_2
    if not _p_acc(xof.unit_fixed(), inner_zv, norm_v_sq, sigma_num, sigma_den,
                  PROB_ONE, m_scale):
        return 1
    if _inf_norm(z_coeffs) * sigma_den > VERIFIER_TAILCUT * sigma_num:
        return 1
    return 0


def _inf_norm(coeffs):
    return max((abs(c) for c in coeffs), default=0)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    # XOF determinism and independence of chunking
    a = XOF(b"t", b"seed")
    b = XOF(b"t", b"seed")
    assert a.read(300) == b.read(100) + b.read(200)
    assert XOF(b"t", b"ab", b"c").read(16) != XOF(b"t", b"a", b"bc").read(16)

    # uniform_int is in range and reasonably flat
    x = XOF(b"t", b"u")
    vals = [uniform_int(x, 61) for _ in range(12200)]      # 200 per bin
    assert all(0 <= v < 61 for v in vals)
    assert 150 < min(vals.count(k) for k in range(61))

    # Gaussian: empirical moments close to target, support truncated
    sigma = 1000
    x = XOF(b"t", b"g")
    samples = [gaussian_int(x, sigma) for _ in range(20000)]
    mean = sum(samples) / len(samples)
    stdev = math.sqrt(sum(s * s for s in samples) / len(samples) - mean ** 2)
    assert abs(mean) < 40, mean
    assert abs(stdev - sigma) / sigma < 0.05, stdev
    assert max(abs(s) for s in samples) <= 6 * sigma

    # challenge weight and coefficient bounds
    q = 1073123348676497
    x = XOF(b"t", b"c")
    for _ in range(50):
        c = sample_challenge(x, 32, 32, 16, q)
        cent = [v - q if v > q // 2 else v for v in c]
        assert sum(1 for v in cent if v != 0) == 32
        assert max(abs(v) for v in cent) <= 16

    c = sample_challenge(XOF(b"t", b"c2"), 32, 8, 16, q)
    cent = [v - q if v > q // 2 else v for v in c]
    assert sum(1 for v in cent if v != 0) == 8

    # Rej_1 acceptance rate matches 1/M_1 to within sampling noise
    phi = 8
    T = 100
    accepted = 0
    trials = 3000
    x = XOF(b"t", b"r")
    for i in range(trials):
        v = [gaussian_int(x, 30) for _ in range(16)]
        norm_v = math.sqrt(sum(c * c for c in v)) or 1
        v = [int(c * T / norm_v) for c in v]
        z = [gaussian_int(x, phi * T) + v[j] for j in range(16)]
        accepted += 1 - rej1(x, z, v, phi, phi * T, 1)
    expected = 1 / math.exp(12 / phi + 1 / (2 * phi ** 2))
    assert abs(accepted / trials - expected) < 0.05, (accepted / trials, expected)

    print("sample.py: all self-tests passed")
