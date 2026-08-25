"""
params.py -- Parameter profiles and derived bounds for RiVeR.

Base parameters follow Table `tab:river-final-all-params` of the
paper (Appendix "Detailed Parameter Setting").  Every derived bound
is computed as `BoundGen` specifies, so `test_params.py` can check the whole
table numerically.

Provenance
----------
Every constant here carries one of three provenance labels:

  * **Paper** -- printed in the current PDF or TeX.
  * **Derived** -- deterministically derived from paper values by a documented
    convention, but not printed.
  * **Repair** -- an implementation choice needed to make an ambiguous or
    inconsistent part of the paper executable.

Two things the paper leaves open, both **Derived**:

  * **Concrete moduli.**  The paper reports only bit lengths for `p` and
    `q_hat`.  We take the largest prime below `2^bits` congruent to
    `5 mod 8`; that congruence is what makes `X^d + 1` split into exactly
    two irreducible factors for `d = 32` (Lemma "q inv"), which the
    challenge-difference invertibility argument needs.  `q_0 = 61` already
    satisfies it, and the paper states `q_hat = 5 mod 8` explicitly.
    `verify_moduli()` re-derives the pinned values.

  * **Concrete moduli** are the one place the paper reports bit lengths
    rather than values; `(tau_g0, tau_g1)` is **not** -- the table prints
    two decimals and says so in a note, so those are **Paper** and are
    carried as exact rationals.  `test_params.py` checks that one decimal
    would *not* reproduce the table's own `B_g0` column, which is why the
    second decimal is load-bearing.

What the paper closed here
------------------------------------
`r' = 1`, and `beta_SIS,2` is now taken over the response the protocol
actually transmits, so the model/protocol split --
two `B_rs`, two `beta_SIS,2`, and a repaired `q~` -- has no remaining reason
to exist and is gone.  `phi_b` is now a `BoundGen` output rather than a symbol
the algorithms used and the parameter generator never produced, and the
single outer response width is replaced by the split `(sigma_s, sigma_m)`.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

from sample import REJ1_CONSTANT



# ---- prime search --------------------------------------------------------
# Deterministic and reproducible; used only to pin the moduli below.

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24 (the bases below suffice)."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL_PRIMES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def largest_prime_below(bits, residue=5, modulus=8):
    """Largest prime < 2^bits congruent to `residue` mod `modulus`."""
    n = (1 << bits) - 1
    n -= (n - residue) % modulus
    while n > 1:
        if is_prime(n):
            return n
        n -= modulus
    raise ValueError("no prime found")


def smallest_prime_above(bits, residue=5, modulus=8):
    """Smallest prime > 2^bits congruent to `residue` mod `modulus`."""
    n = (1 << bits) + 1
    n += (residue - n) % modulus
    while True:
        if is_prime(n):
            return n
        n += modulus


# Pinned moduli.  Keys are the bit lengths quoted in the paper's table.
#
# **Paper**: the concrete moduli are printed in the parameter table. Both
# tables are reproduced by a rule, and the two rules are
# *different*, which is exactly why guessing was not safe:
#
#   `p`      is the largest prime *below* `2^bits`   that is 5 mod 8;
#   `q_hat`  is the smallest prime *above* `2^{bits-1}` that is 5 mod 8.
#
# This tree previously derived `q_hat` by the `p` rule and so used a value
# roughly twice the paper's at every profile -- admissible against the
# `hat-q` condition, but not the published one, and `q_hat` enters `b_B` and
# hence the wire.
P_BY_BITS = {
    44: 17592186043877,
    48: 281474976710597,
}
QHAT_BY_BITS = {
    44: 8796093022237,
    46: 35184372088997,
    48: 140737488355333,
    49: 281474976710677,
}


def verify_moduli():
    """Re-derive every pinned modulus.  Returns a list of error strings."""
    errors = []
    for table, name, rule in ((P_BY_BITS, "p", "below"),
                              (QHAT_BY_BITS, "q_hat", "above")):
        for bits, value in table.items():
            if rule == "below":
                expect = largest_prime_below(bits, 5, 8)
            else:
                expect = smallest_prime_above(bits - 1, 5, 8)
            if expect != value:
                errors.append(f"{name}[{bits}]: pinned {value}, derived {expect}")
            if value % 8 != 5:
                errors.append(f"{name}[{bits}] = {value} is not 5 mod 8")
            if value.bit_length() != bits:
                errors.append(f"{name}[{bits}] is not a {bits}-bit value")
    return errors


def _show(value, digits=24):
    """A bounded rendering of any value, for a diagnostic string.

    `check()` reports the offending value, and an arbitrary caller-supplied
    integer can be arbitrarily long -- past 4300 digits Python refuses to
    convert it at all, so *formatting the error* raised `ValueError` and a
    domain error surfaced as the outer guard catching an exception.  The
    diagnostic must not be able to fail where the check itself cannot.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() > 80:
            return f"<{value.bit_length()}-bit integer>"
        return str(value)
    if isinstance(value, Fraction):
        if (value.numerator.bit_length() > 80
                or value.denominator.bit_length() > 80):
            return (f"<Fraction, {value.numerator.bit_length()}/"
                    f"{value.denominator.bit_length()} bits>")
        return str(value)
    text = repr(value)
    return text if len(text) <= digits else text[:digits] + "..."


def _isqrt_exact(n):
    """`sqrt(n)` as an int when `n` is a perfect square, else `None`."""
    root = math.isqrt(n)
    return root if root * root == n else None


# ---- parameter set -------------------------------------------------------

@dataclass(frozen=True)
class RiVeRParams:
    """
    One RiVeR parameter profile.

    Field names follow the paper.  Derived quantities are properties so a
    profile is fully described by the literals in `PROFILES`.
    """

    # -- identifier --------------------------------------------------------
    name: str

    # -- ring geometry (fixed across all profiles) -------------------------
    d: int                  # ring dimension of R_q, R_qhat
    q0: int                 # q / p; also the rounding-error range size
    p: int                  # rounding modulus, value space R_p
    q_hat: int              # selector modulus

    # -- module ranks ------------------------------------------------------
    n: int                  # rows of A   (public-key rank)
    ell: int                # cols of A   (secret-key rank)
    n_hat: int              # rows of G'
    k_hat: int              # randomness columns of G'

    # -- ring size ---------------------------------------------------------
    N: int                  # size of the ring (exactly N keys; no padding)

    # -- challenge / key ---------------------------------------------------
    w: int                  # number of nonzero challenge coefficients
    gamma: int              # challenge coefficient bound
    beta: int               # secret-key bound (beta = 1 => ternary)

    # -- rejection sampling ------------------------------------------------
    # The outer response is split into `(z_s, z_key)`, with `ell+n` ring
    # elements at the short-response width, and the one-element `z_eval`
    # block at the error-response width. `phi_m` and `phi_b` are shared by
    # every profile.
    phi_a: float            # slack for the selector response f_1
    phi_s: float            # slack for the short response z_s
    phi_m: float = 32       # slack for the error response z_m       (Paper)
    phi_b: float = 2        # slack for the binary response z_b      (Paper)

    # -- product-bound calibration (exact rationals) ----------------------
    tau_g0: Fraction = Fraction(0)
    tau_g1: Fraction = Fraction(0)

    # -- repetition calibration exported by the parameter search ---------
    epsilon_g_u: float = 0.0   # held-out upper bound for the product check

    # -- bit dropping ------------------------------------------------------
    K_b: int = 5
    K_a: int = 28
    s_cmp: int = 3          # `s_c`, the compression margin BoundGen checks

    # -- reduction-only auxiliary block -----------------------------------
    r_prime: int = 1        # Paper: r' = 1 for every final profile

    # -- tunables ----------------------------------------------------------
    lam: int = 128
    max_attempts: int = 1000

    # Marks profiles that deliberately violate the security-side modulus
    # conditions in order to run fast.  Only TOY sets this.
    insecure_toy: bool = False

    # ---- moduli ----------------------------------------------------------

    @property
    def q(self):
        """Outer modulus q = q_0 * p (composite; R_q = R_p x R_{q_0})."""
        return self.q0 * self.p

    # ---- BoundGen --------------------------------------------------------
    # Each bound below is one named entry of BoundGen's output tuple.  The
    # names keep callers independent of positional unpacking conventions.

    @property
    def B_e(self):
        """`floor(q_0 / 2)`; bounds the *centred* rounding error.

        The rounding relation itself keeps errors in `[0, q_0-1]`; this is
        the centred range used for the concrete norm bounds, and the
        implementation carries the shift explicitly.  See `ring.to_centered_error`.
        """
        return self.q0 // 2

    @property
    def B_a(self):
        """`gamma sqrt(2w)`; scale of the selector masks `a_i`.

        Exactly 128 for every published profile, and returned as an `int`
        whenever `2w` is a perfect square, so `phi_a B_a` and the bounds
        derived from it stay exact.
        """
        root = _isqrt_exact(2 * self.w)
        if root is not None:
            return self.gamma * root
        return self.gamma * math.sqrt(2 * self.w)

    @property
    def cal_B(self):
        """`B = w gamma beta sqrt(d k_hat)`; scale of the binary mask `r_a`."""
        return self.w * self.gamma * self.beta * math.sqrt(self.d * self.k_hat)

    @property
    def B_s(self):
        """`B_s = w gamma B_e sqrt(d(ell+n))`; scale of the short response.

        The mask block is `r_0 = (s, e_key)`, so `B_s` covers `ell + n`
        ring elements and carries `B_e`, which also dominates the bound
        `beta` on `s`.
        """
        return self.w * self.gamma * self.B_e \
            * math.sqrt(self.d * (self.ell + self.n))

    @property
    def eta_m(self):
        """`eta_m = w gamma B_e sqrt(d)`; scale of the error response.

        Independent of the profile: 86889.3 for all five.
        """
        return self.w * self.gamma * self.B_e * math.sqrt(self.d)

    @property
    def B_g0(self):
        """Product-check threshold for `g_0` (carries the `(N-1)` variance).

        Exact: `tau_g0` is a `Fraction` and `phi_a B_a` is an integer, so the
        verifier compares integers against an exact rational rather than
        against a float whose last ulp could fork a transcript.
        """
        return (self.tau_g0 * Fraction(self.d * (self.N - 1), 3)
                * Fraction(self.phi_a * self.B_a) ** 2)

    @property
    def B_g1(self):
        """Product-check threshold for `g_1`.  Exact, as `B_g0`."""
        return (self.tau_g1 * Fraction(self.d, 2)
                * Fraction(self.phi_a * self.B_a) ** 2)

    @property
    def K_a_boundgen(self):
        """`BoundGen`'s lower bound: `K_b + ceil(log2(w g n_hat d)) + s_c`.

        28 for every published profile, because `ceil(log2(w gamma n_hat d))`
        is 20 for all four values of `n_hat` in use (42, 43, 49, 50).
        `BoundGen` aborts when `K_a` is below this; `check()` reproduces that.
        """
        return (self.K_b
                + math.ceil(math.log2(self.w * self.gamma * self.n_hat * self.d))
                + self.s_cmp)

    # ---- Gaussian widths -------------------------------------------------

    @property
    def sigma_a(self):
        """Width of the selector masks: `phi_a B_a`."""
        return self.phi_a * self.B_a

    @property
    def sigma_b(self):
        """Width of the binary mask `r_a`: `phi_b B`.

        **Repair.**  The `Com` figure samples `r_a <- D_B` while its `Rej_2`
        call and the communication formula both use `phi_b B`.  A rejection
        sampler is only correct when the mask width equals the sigma in its
        acceptance test, so we sample at `phi_b B`.
        """
        return self.phi_b * self.cal_B

    @property
    def sigma_s(self):
        """Width of the short response: `sigma_s = phi_s B_s`."""
        return self.phi_s * self.B_s

    @property
    def sigma_m(self):
        """Width of the error response: `sigma_m = phi_m eta_m`."""
        return self.phi_m * self.eta_m

    # ---- verifier bounds -------------------------------------------------
    #
    # Every one of these is an *acceptance* decision, so it is decided
    # exactly.  Each bound has the shape `K sqrt(M)` with `K` rational and
    # `M` a positive integer, so squaring turns the comparison into one
    # between exact rationals and removes the `sqrt` -- and with it the last
    # place where two implementations could disagree on a coefficient
    # sitting on the boundary.  `Fraction(float)` is exact, so this holds
    # even if a profile carries a non-integral `phi`.
    #
    # The `*_inf_bound` floats below are kept for reporting and for the
    # codec's field cap, never for the accept/reject test.

    @property
    def _wgb(self):
        """`w gamma beta`, the scale of `B` (the binary mask `r_a`)."""
        return self.w * self.gamma * self.beta

    @property
    def _wgB(self):
        """`w gamma B_e`, the scale shared by `B_s` and `eta_m`.

        Both response blocks bound a rounding error, so both carry `B_e`.
        Before the paper only `eta_m` did.
        """
        return self.w * self.gamma * self.B_e

    @property
    def f1_inf_bound(self):
        """`||f_1||_inf <= 6 phi_a B_a`.

        Already exact when `2w` is a perfect square, which it is at every
        profile: `B_a = gamma sqrt(2w) = 128`.
        """
        return 6 * self.sigma_a

    @property
    def f1_inf_bound_sq(self):
        """`(6 phi_a B_a)^2`, exactly."""
        return Fraction(6 * self.phi_a * self.gamma) ** 2 * (2 * self.w)

    @property
    def zb_inf_bound(self):
        """`||z_b||_inf <= 6 phi_b B`."""
        return 6 * self.sigma_b

    @property
    def zb_inf_bound_sq(self):
        """`(6 phi_b B)^2 = (6 phi_b w gamma beta)^2 d k_hat`, exactly."""
        return (Fraction(6 * self.phi_b) * self._wgb) ** 2 \
            * (self.d * self.k_hat)

    @property
    def zs_inf_bound(self):
        """`||(z_s, z_key)||_inf <= 6 sigma_s`."""
        return 6 * self.sigma_s

    @property
    def zs_inf_bound_sq(self):
        """`(6 sigma_s)^2 = (6 phi_s w gamma B_e)^2 d(ell+n)`, exactly."""
        return (Fraction(6 * self.phi_s) * self._wgB) ** 2 \
            * (self.d * (self.ell + self.n))

    @property
    def zm_inf_bound(self):
        """`||z_eval||_inf <= 6 sigma_m`."""
        return 6 * self.sigma_m

    @property
    def zm_inf_bound_sq(self):
        """`(6 sigma_m)^2 = (6 phi_m w gamma B_e)^2 d`, exactly.

        Unchanged in form by the paper, but it now caps a single ring
        element (`z_eval`) rather than the `n + 1` of `(z_key, z_eval)`.
        """
        return (Fraction(6 * self.phi_m) * self._wgB) ** 2 * self.d

    @property
    def z_l2_bound(self):
        """`||z||_2 <= 1.2 sqrt(sigma_s^2 d(ell+n) + sigma_m^2 d)`."""
        return 1.2 * math.sqrt(
            self.sigma_s ** 2 * self.d * (self.ell + self.n)
            + self.sigma_m ** 2 * self.d)

    @property
    def sigma_s_sq(self):
        """`sigma_s^2 = (phi_s w gamma B_e)^2 d(ell+n)`, exactly."""
        return (Fraction(self.phi_s) * self._wgB) ** 2 \
            * (self.d * (self.ell + self.n))

    @property
    def sigma_m_sq(self):
        """`sigma_m^2 = (phi_m w gamma B_e)^2 d`, exactly."""
        return (Fraction(self.phi_m) * self._wgB) ** 2 * self.d

    @property
    def z_l2_bound_sq(self):
        """`1.44 (sigma_s^2 d(ell+n) + sigma_m^2 d)`, exactly.

        `1.2^2` is `36/25` as a rational, not as the binary float `1.44`.
        """
        return Fraction(36, 25) * (
            self.sigma_s_sq * self.d * (self.ell + self.n)
            + self.sigma_m_sq * self.d)

    @property
    def T_cmp(self):
        """Compression restart threshold `2^{K_a-1} - w gamma 2^{K_b-1}`."""
        return 2 ** (self.K_a - 1) - self.w * self.gamma * 2 ** (self.K_b - 1)

    # ---- M-SIS / A-MSIS bounds ------------------------------------------

    @property
    def beta_sis_1(self):
        """`2.4 sqrt(sigma_s^2 d(ell+n))`.

        The paper: the `sigma_m` term is gone.  The two accepting forks
        now differ only in the `(s, e_key)` block for this bound.
        """
        return 2.4 * math.sqrt(
            self.sigma_s ** 2 * self.d * (self.ell + self.n))

    @property
    def beta_sis_2(self):
        """`2.4 sqrt(sigma_s^2 d(ell+n) + sigma_m^2 d)`."""
        return 2.4 * math.sqrt(
            self.sigma_s ** 2 * self.d * (self.ell + self.n)
            + self.sigma_m ** 2 * self.d)

    @property
    def beta_sis(self):
        """`max{4 w gamma beta_SIS,1,
               beta_SIS,1 + 2 w gamma sqrt(d(ell beta^2 + (n+r') B_e^2))}`.

        The paper notes the first term is the larger for all five profiles;
        `test_params.py` checks that rather than assuming it.
        """
        return max(4 * self.w * self.gamma * self.beta_sis_1,
                   self.beta_sis_embedded)

    @property
    def beta_sis_embedded(self):
        """The second term of `beta_SIS`, kept separate so it can be checked."""
        return self.beta_sis_1 + 2 * self.w * self.gamma * math.sqrt(
            self.d * (self.ell * self.beta ** 2
                      + (self.n + self.r_prime) * self.B_e ** 2))

    @property
    def beta_sel(self):
        """The six A-MSIS block bounds, in extraction block order.

        Widths `(k_hat, 1, N-1, 1, N-1, n_hat)` for
        `(z_b, f_0, f_1, g_0, g_1, e_c)`.
        """
        s = 4 * self.w * self.gamma
        return (s * 6 * self.sigma_b,
                s * (self.gamma + 6 * (self.N - 1) * self.sigma_a),
                s * 6 * self.sigma_a,
                s * float(self.B_g0),
                s * float(self.B_g1),
                s * 2 ** self.K_a)

    @property
    def beta_sel_merged(self):
        """The five merged bounds used for the A-MSIS *estimate*.

        `f_0` and `f_1` are combined into one width-`N` entry carrying the
        larger of the two bounds, which can only enlarge the admissible
        solution set.
        """
        s = 4 * self.w * self.gamma
        return (s * 6 * self.sigma_b,
                s * (self.gamma + 6 * (self.N - 1) * self.sigma_a),
                s * float(self.B_g0),
                s * float(self.B_g1),
                s * 2 ** self.K_a)

    @property
    def beta_sel_inf(self):
        return max(self.beta_sel)

    # ---- repetition estimate (Appendix "Correctness") -------------------

    @property
    def mu_a(self):
        """`exp(12/phi_a + 1/(2 phi_a^2))`, from Lemma grs(1)."""
        return math.exp(REJ1_CONSTANT / self.phi_a
                        + 1 / (2 * self.phi_a ** 2))

    @property
    def mu_s(self):
        """`exp(12/phi_s + 1/(2 phi_s^2))`, from Lemma grs(1)."""
        return math.exp(REJ1_CONSTANT / self.phi_s
                        + 1 / (2 * self.phi_s ** 2))

    @property
    def mu_m(self):
        """`exp(12/phi_m + 1/(2 phi_m^2))`, from Lemma grs(1)."""
        return math.exp(REJ1_CONSTANT / self.phi_m
                        + 1 / (2 * self.phi_m ** 2))

    @property
    def mu_b(self):
        """`2 exp(1/(2 phi_b^2))`, from Lemma grs(2).

        The factor 2 is the `<z, v> >= 0` half-space rejection, and the
        appendix is explicit that it is therefore not charged again by the
        subsequent infinity-norm check.
        """
        return 2 * math.exp(1 / (2 * self.phi_b ** 2))

    @property
    def eps_tail(self):
        """The four `2 d <width> exp(-18)` tail terms, in `(a, b, s, m)` order.

        The `s` and `m` widths follow the response split: `(ell+n)` and `1`,
        matching the algorithm's two transmitted response blocks.
        """
        t = 2 * self.d * math.exp(-18)
        return (t * (self.N - 1), t * self.k_hat, t * (self.ell + self.n),
                t * 1)

    @property
    def compression_pass_residues(self):
        """Number of residues satisfying both compression predicates.

        A coefficient must simultaneously stay away from the centred
        ``q_hat`` boundary and have a signed ``2^K_a`` remainder of magnitude
        strictly below ``T_cmp``.  Both predicates inspect the same residue,
        so multiplying their marginal probabilities would assume an
        independence that is only approximately true.  Count their
        intersection over one complete ``Z_q_hat`` representative set.
        """
        integer_fields = (self.K_a, self.K_b, self.q_hat, self.w, self.gamma)
        if (any(not isinstance(value, int) or isinstance(value, bool)
                or value <= 0 for value in integer_fields)
                or not self.K_a < 127 or not self.K_b < 127):
            return 0
        modulus = 1 << self.K_a
        perturbation = self.w * self.gamma * (1 << (self.K_b - 1))
        threshold = modulus // 2 - perturbation
        q_threshold = ((self.q_hat - 1) // 2
                       - perturbation)
        if not 0 < threshold <= modulus // 2 or q_threshold <= 0:
            return 0

        # F(n) counts accepted integers in [0,n), and remains valid for
        # negative n because divmod uses a non-negative remainder.  Accepted
        # residues are [0,T-1] U [M-T+1,M-1].
        def prefix(n):
            periods, remainder = divmod(n, modulus)
            return (periods * (2 * threshold - 1)
                    + min(remainder, threshold)
                    + max(0, remainder - (modulus - threshold + 1)))

        lo = -(q_threshold - 1)
        hi = q_threshold - 1
        return prefix(hi + 1) - prefix(lo)

    @property
    def p_cmp_uniform(self):
        """Uniform-residue success model for all compression coefficients."""
        accepted = self.compression_pass_residues
        dimensions = (self.n_hat, self.d)
        if (accepted == 0
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value <= 0 for value in dimensions)):
            return 0.0
        coefficient_pass = accepted / self.q_hat
        return coefficient_pass ** (self.n_hat * self.d)

    @property
    def mu_river(self):
        """The table's "Repeat bound" column, from the appendix's formula.

        `mu_a mu_b mu_s mu_m` over
        `(1-eps_a)(1-eps_b)((1-eps_s)(1-eps_m) - eps_2)(1-eps_g)(1-eps_c)`,
        which is `eq:river-repeat-bound`.

        Every component is computed from the profile and the recorded
        product-check confidence bound; `test_params.py` pins agreement with
        all five printed rows.
        """
        return self.mu_gaussian / self.c_pub_from_components

    @property
    def mu_gaussian(self):
        """`mu_a mu_b mu_s mu_m`: the four Gaussian rejection samplers."""
        return self.mu_a * self.mu_b * self.mu_s * self.mu_m

    @property
    def eps_euclidean(self):
        """`eps_2`: the joint Euclidean response check's failure probability.

        The appendix bounds it by dominating all
        `d(ell+n+1)` response coefficients with a width-`sigma_s` Gaussian
        (sound because every profile requires `sigma_s >= sigma_m`) and
        applying the Euclidean tail bound at ratio

            rho = 1.2 sqrt((ell+n+(sigma_m/sigma_s)^2)/(ell+n+1)),

        giving `rho^M exp(M(1-rho^2)/2)` for `M = d(ell+n+1)`.  The paper
        states this is below `2^-150` for every final profile; it is, by
        about a hundred orders of magnitude, so it does not move the
        repetition estimate at all.  It is computed rather than assumed so
        that a profile which *did* move it would say so.
        """
        m = self.d * (self.ell + self.n + 1)
        ratio = 1.2 * math.sqrt(
            (self.ell + self.n + float(self.sigma_m_sq / self.sigma_s_sq))
            / (self.ell + self.n + 1))
        # In logs: the direct form underflows to exactly 0.0 well before
        # the bound stops being informative.
        log2_eps = (m * math.log2(ratio)
                    + m * (1 - ratio ** 2) / 2 * math.log2(math.e))
        return 2.0 ** log2_eps if log2_eps > -1000 else 0.0

    @property
    def c_pub_from_components(self):
        """The denominator of `eq:river-repeat-bound`, from the components.

        `(1-eps_a)(1-eps_b)((1-eps_s)(1-eps_m) - eps_2)(1-eps_g)(1-eps_c)`.
        The `eps_2` term entered; before it the four tails
        multiplied in flat.
        """
        eps_a, eps_b, eps_s, eps_m = self.eps_tail
        return ((1 - eps_a) * (1 - eps_b)
                * ((1 - eps_s) * (1 - eps_m) - self.eps_euclidean)
                * (1 - self.epsilon_g_u) * self.p_cmp_uniform)

    # ---- size estimate ---------------------------------------------------

    #: Paper: the exact proof has a 13.5 KB entropy estimate at every profile.
    EXACT_PROOF_KB = 13.5

    @property
    def b_B(self):
        """`b_B = ceil(log2(ceil(q_hat / 2^{K_b})))`."""
        return math.ceil(math.log2(-(-self.q_hat // 2 ** self.K_b)))

    @staticmethod
    def _h(sigma):
        """`h(sigma) = log2(4.13 sigma)`: the entropy model of [C:ESLR23] 2.4."""
        return math.log2(4.13 * sigma)

    @property
    def _proof_size_oom_common_bits(self):
        """The `B`, `x`, `f_1` and `z_b` terms, shared by both layouts."""
        return (self.n_hat * self.d * self.b_B
                + self.challenge_entropy
                + (self.N - 1) * self.d * self._h(self.sigma_a)
                + self.k_hat * self.d * self._h(self.sigma_b))

    @property
    def proof_size_oom_kb(self):
        """`|pi_OOM| = L_OOM / 8192` KiB, from the communication formula.

        The six terms are the contributions of `B`, `x`, `f_1`, `z_b`,
        `(z_s, z_key)` and `z_eval`.  The last two are charged at the
        dimensions the algorithm actually transmits: `(ell+n) d h(sigma_s)`
        and `d h(sigma_m)`.

        The manuscript's displayed formula and final table now use this same
        response split, so this value directly reproduces the published OOM
        column.
        """
        return (self._proof_size_oom_common_bits
                + self.s_dim * self.d * self._h(self.sigma_s)
                + self.m_dim * self.d * self._h(self.sigma_m)) / 8192

    @property
    def proof_size_total_kb(self):
        """Total using the paper's entropy estimate for `|pi_ex|`."""
        return self.proof_size_oom_kb + self.EXACT_PROOF_KB

    @property
    def challenge_entropy(self):
        """log2 |C^d_{w,gamma}| = log2 C(d,w) + w log2(2 gamma)."""
        return (math.log2(math.comb(self.d, self.w))
                + self.w * math.log2(2 * self.gamma))

    # ---- dimensions ------------------------------------------------------

    @property
    def r_dim(self):
        """Length of the OOM opening `r = (r_0, r_1) = ((s, e_key), e_eval)`.

        The concatenation is `s`, then `e_key`, then `e_eval`,
        `ell + n + 1` ring elements, with `s_dim + m_dim = r_dim`.
        """
        return self.ell + self.n + 1

    @property
    def s_dim(self):
        """Length of `r_0 = (s, e_key)`, responded to at width `sigma_s`."""
        return self.ell + self.n

    @property
    def m_dim(self):
        """Length of `r_1 = e_eval`, one ring element at `sigma_m`."""
        return 1

    @property
    def c_dim(self):
        """Length of each derived vector `c_i = (q_0 t_i, q_0 v)`."""
        return self.n + 1

    @property
    def gprime_cols(self):
        return self.k_hat + 2 * self.N

    # ---- consistency -----------------------------------------------------

    # ---- domain validation ----------------------------------------------

    #: `(field, kind, predicate, description)` for every profile literal, in
    #: dependency order: nothing here reads a derived property, so it is safe
    #: to run *before* `check()` evaluates any of them.
    def _domain(self):
        """Type and range errors in the profile's own literals.

        `check()` runs this first and returns early if it finds anything,
        because every derived property below assumes these hold.  Without it
        `check()` was neither total nor fail-closed: `d = 0` raised out of
        `K_a_boundgen`, while `beta = 0`, `N = 0`, `max_attempts = 0`,
        `phi_a = 0` and even a NaN width all returned "no errors".
        """
        errors = []

        def want_int(name, minimum, maximum=None):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{name} is not an int")
                return False
            if value < minimum:
                errors.append(f"{name} = {_show(value)} < {minimum}")
                return False
            if maximum is not None and value > maximum:
                errors.append(f"{name} = {_show(value)} > {maximum}")
                return False
            return True

        # Ceilings.  The domain pass exists so `_conditions` can evaluate
        # its derived properties safely, and those go through `math.log2`,
        # `math.sqrt` and float multiplication -- all of which raise on an
        # int too large for a double.  A value above these is not a profile
        # anyone could mean, and saying so by name beats letting the outer
        # guard report an `OverflowError`.
        MAX_DIM = 1 << 20        # ring dimension, module ranks, ring size
        MAX_MODULUS = 1 << 256   # the moduli are ~2^49 today
        MAX_BITS = 1 << 10       # K_a, K_b, lam, s_cmp
        MAX_WIDTH = 1 << 20      # phi_*, tau_*
        MAX_ATTEMPTS = 1 << 40

        def want_positive_real(name, maximum=MAX_WIDTH):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float,
                                                                 Fraction)):
                errors.append(f"{name} is not a real number")
                return False
            # `isfinite` only for floats; see the note in the calibration
            # block below.
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"{name} is not finite")
                return False
            if value <= 0:
                errors.append(f"{name} = {_show(value)} is not positive")
                return False
            if value > maximum:
                errors.append(f"{name} = {_show(value)} exceeds {maximum}")
                return False
            return True

        if not isinstance(self.name, str) or not self.name:
            errors.append("name is not a non-empty string")
        if not isinstance(self.insecure_toy, bool):
            errors.append("insecure_toy is not a bool")

        # Ring geometry and moduli.
        ok_d = want_int("d", 1, MAX_DIM)
        want_int("q0", 2, MAX_MODULUS)
        want_int("p", 2, MAX_MODULUS)
        want_int("q_hat", 2, MAX_MODULUS)

        # Module ranks.  All index into vectors, so all must be positive.
        for field in ("n", "ell", "n_hat", "k_hat", "r_prime"):
            want_int(field, 1, MAX_DIM)

        # A ring needs at least two members to hide anything, and the OOM
        # layer's `f_1` block has `N - 1` entries.
        want_int("N", 2, MAX_DIM)

        # Challenge space: `w` of the `d` coefficients are nonzero.
        ok_w = want_int("w", 1, MAX_DIM)
        if ok_d and ok_w and self.w > self.d:
            errors.append(f"w = {_show(self.w)} > d = {_show(self.d)}")
        want_int("gamma", 1, MAX_DIM)
        want_int("beta", 1, MAX_DIM)

        # Widths.  Zero or non-finite here would divide by zero in the
        # repetition estimate or produce a degenerate sampler.
        for field in ("phi_a", "phi_s", "phi_m", "phi_b",
                      "tau_g0", "tau_g1"):
            want_positive_real(field)

        # Bit dropping.  `K_b < K_a` is what makes the compression margin a
        # margin; `T_cmp` goes negative otherwise.
        ok_kb = want_int("K_b", 1, MAX_BITS)
        ok_ka = want_int("K_a", 1, MAX_BITS)
        want_int("s_cmp", 0, MAX_BITS)
        if ok_kb and ok_ka and self.K_b >= self.K_a:
            errors.append(f"K_b = {_show(self.K_b)} >= K_a = {_show(self.K_a)}")

        # Calibration exports.
        for field, lo, hi, lo_open, hi_open in (
                ("epsilon_g_u", 0, 1, False, True),):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{field} is not a real number")
            # `math.isfinite` converts to float first, so an int too large
            # for a double raises `OverflowError` -- which is how a
            # totality check reintroduced the very failure it was added to
            # remove.  Only floats can be non-finite; an int is finite by
            # construction and the range test below handles any magnitude.
            elif isinstance(value, float) and not math.isfinite(value):
                errors.append(f"{field} is not finite")
            elif (value <= lo if lo_open else value < lo) or \
                 (value >= hi if hi_open else value > hi):
                errors.append(f"{field} = {_show(value)} is outside "
                              f"{'(' if lo_open else '['}{lo}, {hi}"
                              f"{')' if hi_open else ']'}")

        want_int("lam", 1, MAX_BITS)
        want_int("max_attempts", 1, MAX_ATTEMPTS)
        return errors

    def check(self):
        """Return a list of violated parameter conditions (empty == fine).

        **Total**: it never raises, for any field values at all.  A
        validation function that can throw is not one -- and this one used
        to, on `d = 0`, `w = 0` or `gamma = 0`, from inside a derived
        property it evaluated before checking anything.

        Two passes, in dependency order.  `_domain` validates the profile's
        own literals and returns early if any fail, because every derived
        quantity below assumes them.  The rest are the structural and
        security conditions, which are computed inside a guard: reaching it
        means the domain is sound, so an exception there is a defect in this
        module rather than bad input, and it is reported as an error instead
        of escaping.

        Conditions marked "security" are skipped for `insecure_toy` profiles.
        """
        try:
            errors = self._domain()
        except Exception as exc:                     # pragma: no cover
            # The domain pass was outside the guard once, and an integer
            # too large for a double got through `math.isfinite` as an
            # `OverflowError` -- so the guard now covers both passes.  The
            # specific cause is fixed; this is what makes "total" a
            # property of the function rather than of the cases tried.
            return [f"parameter domain check raised "
                    f"{type(exc).__name__}: {exc}"]
        if errors:
            return errors
        try:
            return self._conditions()
        except Exception as exc:                     # pragma: no cover
            return [f"parameter check raised {type(exc).__name__}: {exc}"]

    def _conditions(self):
        """The structural and security conditions.  Assumes `_domain` passed."""
        errors = []
        if self.q0 * self.p != self.q:
            errors.append("q != q_0 * p")
        if self.d & (self.d - 1):
            errors.append("d is not a power of two")
        # Primality is not implied by the congruence, and the congruence is
        # the visible half of the condition: 17592186043869 is 5 mod 8 and
        # composite, and without this check `Setup` would accept it.  The
        # two-factor splitting argument -- and every invertibility claim
        # resting on it -- needs `X^d + 1` over a *field*.
        for label, value in (("p", self.p), ("q_0", self.q0),
                             ("q_hat", self.q_hat)):
            if not is_prime(value):
                errors.append(f"{label} = {value} is not prime")
        for label, value in (("p", self.p), ("q_0", self.q0),
                             ("q_hat", self.q_hat)):
            if value % 8 != 5:
                errors.append(f"{label} = {value} is not 5 mod 8")
        if math.gcd(self.p, self.q0) != 1:
            errors.append("gcd(p, q_0) != 1, CRT does not apply")

        # Correctness-critical: no wraparound in either response block.
        # `q > max{beta_SIS,2, 12 sigma_s, 12 sigma_m}`.
        if self.q <= 12 * self.sigma_s:
            errors.append("q <= 12 sigma_s (short response may wrap mod q)")
        if self.q <= 12 * self.sigma_m:
            errors.append("q <= 12 sigma_m (error response may wrap mod q)")

        # the paper adds `sigma_s >= sigma_m` to Assumption
        # `ass:river-params`.  The joint Euclidean tail argument needs it:
        # it dominates every response coefficient by a width-`sigma_s`
        # Gaussian, which is only sound in this direction.  Compared
        # exactly -- both sides are rationals -- because a profile sitting
        # on the boundary must not turn on a float rounding.
        if self.sigma_s_sq < self.sigma_m_sq:
            errors.append("sigma_s < sigma_m (joint Euclidean tail bound "
                          "dominates by sigma_s; assumption requires "
                          "sigma_s >= sigma_m)")

        # The product-check thresholds have to admit a nonzero `g`.  An
        # arbitrarily small `tau` is inside the domain -- positive, finite,
        # exactly representable -- and yields a bound under 1, which only
        # an all-zero `g` can satisfy.  That is a profile no honest prover
        # can use, so it is a condition rather than a domain rule: it is
        # about what the bound *does*, not about the literal's range.
        for label, bound in (("B_g0", self.B_g0), ("B_g1", self.B_g1)):
            if bound < 1:
                errors.append(f"{label} = {float(bound):.4g} < 1: "
                              "no nonzero g can pass the product check")

        # BoundGen's own abort: the compression margin must leave s_c bits.
        if self.K_a < self.K_a_boundgen:
            errors.append(f"K_a = {self.K_a} < {self.K_a_boundgen} (BoundGen)")

        # Selector modulus condition.
        need = max(2 * (2 * self.gamma + 12 * self.sigma_a) ** 2,
                   2 * self.N ** 2,
                   2 ** (self.K_a + 1))
        if self.q_hat <= need:
            errors.append(f"q_hat <= {need:.4g} (hat-q condition)")

        if not self.insecure_toy:
            if self.q_hat <= self.beta_sel_inf:
                errors.append("q_hat <= ||beta_sel||_inf (security)")
            if self.q <= self.beta_sis:
                errors.append("q <= beta_SIS (security)")
            if self.q <= self.beta_sis_2:
                errors.append("q <= beta_SIS,2 (security)")
            if self.challenge_entropy < 128:
                errors.append("challenge space below 128 bits (security)")
        return errors

    def summary(self):
        """One-line-per-item human summary, used by the CLI and tests."""
        return {
            "name": self.name,
            "N": self.N,
            "(phi_a, phi_s)": (self.phi_a, self.phi_s),
            "(n, ell)": (self.n, self.ell),
            "(log2 p, log2 q)": (round(math.log2(self.p), 2),
                                 round(math.log2(self.q), 2)),
            "(n_hat, k_hat, log2 q_hat)": (self.n_hat, self.k_hat,
                                           round(math.log2(self.q_hat), 2)),
            "B": self.cal_B,
            "B_s": self.B_s,
            "eta_m": self.eta_m,
            "(B_g0, B_g1)": (float(self.B_g0), float(self.B_g1)),
            "beta_SIS,1": self.beta_sis_1,
            "beta_SIS,2": self.beta_sis_2,
            "beta_SIS": self.beta_sis,
            "beta_sel_inf": self.beta_sel_inf,
            "mu_river": self.mu_river,
            "|pi_OOM| KB": self.proof_size_oom_kb,
            "total KB": self.proof_size_total_kb,
        }


# ---- published profiles --------------------------------------------------
# (phi_a, phi_s), (n, ell), log2 p, (n_hat, k_hat, ceil log2 q_hat)
#   -- Table `tab:river-final-all-params`, the paper.  **Paper.**
#
_PUBLISHED = {
    8:   ((32, 26), (44, 54), 44, (42, 46, 44)),
    16:  ((40, 22), (41, 59), 48, (43, 49, 46)),
    64:  ((34, 24), (44, 54), 44, (50, 51, 48)),
    128: ((24, 34), (45, 54), 44, (49, 51, 48)),
    256: ((22, 40), (42, 59), 48, (48, 52, 49)),
}

#: `BoundGen`'s output tuple.  **One order**: the
#: `BoundGen` figure and all three OOM algorithms now agree on
#: `(phi_a, phi_b, phi_s, phi_m)`.
BOUNDGEN_ORDER = (
    "r_prime", "s_cmp", "phi_a", "phi_b", "phi_s", "phi_m",
    "tau_g0", "tau_g1", "cal_B", "B_e", "eta_m", "B_a", "B_s",
    "B_g0", "B_g1", "K_b", "K_a")

#: Descriptive alias used by the OOM layer.
BOUNDGEN_ORDER_OOM = BOUNDGEN_ORDER


#: `(tau_g0, tau_g1)` as exact rationals.  **Paper**.
#:
#: The paper prints two decimals so the associated product bounds can be
#: reproduced.  `test_params.py` also confirms that one decimal would be
#: insufficient for two table entries.
_TAU = {
    8:   (Fraction(314, 100), Fraction(268, 100)),
    16:  (Fraction(309, 100), Fraction(308, 100)),
    64:  (Fraction(305, 100), Fraction(333, 100)),
    128: (Fraction(309, 100), Fraction(358, 100)),
    256: (Fraction(306, 100), Fraction(384, 100)),
}

#: `(tau_g0, tau_g1)` exactly as displayed, to one decimal place.  Retained
#: so `test_params.py` can pin which entries the rounded values reproduce.
_TAU_DISPLAYED = {
    8:   (Fraction(31, 10), Fraction(27, 10)),
    16:  (Fraction(31, 10), Fraction(31, 10)),
    64:  (Fraction(31, 10), Fraction(33, 10)),
    128: (Fraction(31, 10), Fraction(36, 10)),
    256: (Fraction(31, 10), Fraction(38, 10)),
}

#: `(tau_g0, tau_g1)` exactly as the paper prints them, to two decimals.
#: Identical to `_TAU` -- which is the point: this tree recovered them before
#: the paper printed them, and the test compares the two tables.
_TAU_PUBLISHED_2DP = {
    8:   (Fraction(314, 100), Fraction(268, 100)),
    16:  (Fraction(309, 100), Fraction(308, 100)),
    64:  (Fraction(305, 100), Fraction(333, 100)),
    128: (Fraction(309, 100), Fraction(358, 100)),
    256: (Fraction(306, 100), Fraction(384, 100)),
}

#: `epsilon_g^U` per profile.  **Derived** from the one-million-trial counts
#: and one-sided Clopper--Pearson convention in the parameter artifact.  Full
#: recorded precision is retained because this value feeds the unrounded
#: expected-attempt report, though it never affects protocol decisions.
#:
#: The companion `c_pub_model` backsolve is **gone**: the
#: appendix's own component formula now reproduces the printed "Repeat
#: bound" column at all five profiles, so there is nothing left to
#: backsolve.  See `mu_river`.
#:
#: Nothing byte-visible depends on this: it enters only the reported
#: attempt estimate, never the protocol.
_EPSILON_G_U = {
    8:   0.007953415163498056,
    16:  0.00779131496406446,
    64:  0.008953918582094841,
    128: 0.007711269632144487,
    256: 0.00851857096759001,
}


def _published(N):
    (phi_a, phi_s), (n, ell), log_p, (n_hat, k_hat, log_qhat) = _PUBLISHED[N]
    tau_g0, tau_g1 = _TAU[N]
    epsilon_g_u = _EPSILON_G_U[N]
    return RiVeRParams(
        name=f"RiVeR-N{N}",
        d=32, q0=61, p=P_BY_BITS[log_p], q_hat=QHAT_BY_BITS[log_qhat],
        n=n, ell=ell, n_hat=n_hat, k_hat=k_hat, r_prime=1,
        N=N, w=32, gamma=16, beta=1,
        phi_a=phi_a, phi_s=phi_s, phi_m=32, phi_b=2,
        tau_g0=tau_g0, tau_g1=tau_g1,
        epsilon_g_u=epsilon_g_u,
        K_b=5, K_a=28,
    )


PROFILES = {f"RiVeR-N{N}": _published(N) for N in _PUBLISHED}


# ---- toy profile ---------------------------------------------------------
# Structurally identical (same d, q_0, w, gamma, beta, radix encoding, and
# the same split response widths) but with tiny module ranks so the whole
# pipeline runs in seconds.  Deliberately insecure: it does not meet the
# M-SIS / A-MSIS modulus conditions.

TOY_PARAMS = RiVeRParams(
    name="RiVeR-TOY",
    d=32, q0=61,
    p=largest_prime_below(24, 5, 8),
    q_hat=largest_prime_below(40, 5, 8),
    n=4, ell=6, n_hat=4, k_hat=4, r_prime=1,
    N=4, w=32, gamma=16, beta=1,
    phi_a=32, phi_s=26, phi_m=32, phi_b=2,
    tau_g0=Fraction(314, 100), tau_g1=Fraction(268, 100),
    epsilon_g_u=0.01,
    K_b=5, K_a=28,
    insecure_toy=True,
)

PROFILES[TOY_PARAMS.name] = TOY_PARAMS

DEFAULT_PARAMS = PROFILES["RiVeR-N8"]


def get(name):
    """Look up a profile by name, with a helpful error message."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; "
                       f"available: {sorted(PROFILES)}") from None


# --------------------------------------------------------------------------
if __name__ == "__main__":
    errs = verify_moduli()
    assert not errs, errs
    print("moduli re-derived OK")
    for name in sorted(PROFILES):
        par = PROFILES[name]
        bad = par.check()
        status = "ok" if not bad else "; ".join(bad)
        print(f"{name:12s} N={par.N:3d}  mu~={par.mu_river:5.3f}  "
              f"|pi_OOM|={par.proof_size_oom_kb:7.3f} KB  "
              f"total={par.proof_size_total_kb:7.3f} KB   [{status}]")
