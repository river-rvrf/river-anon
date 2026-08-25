"""
lanes_params.py -- Parameters and samplers for the LANES exact proof.

Optional component; see `lanes_backend.py` for how it plugs into `Pi_ex`.

**Every constant below is the paper's**, and nothing here is searched: the
paper gives the whole Hint-MLWE chain as a closed form with no free
constant, and this module re-derives every printed figure from it.

    (ceil(log2 q~), n~, l~, D) = (26, 4, 4, 17),  q~ = 67107713
    d~ = 256,  l = 64,  N_ex = 6,  alpha = 3,  w_hat = 44

    eps = 2^-100
    s_0 = sqrt(ln(2 d~ (1 + 1/eps))) / pi   ~ 2.7668
    s_1 = 2 s_0                              ~ 5.5336    commitment randomness
    s_2 = 2 w_hat s_0                        ~ 243.4775  proof mask
    s   = 2 sqrt(2) w_hat s_0                ~ 344.3291  response

and every published output follows:

    N_z    = (n~ + l~ + N_ex + alpha) d~ = 4352
    beta'  = 2 s sqrt(N_z)               = 45430.5731   (printed 45430.6)
    B_MSIS = 8 w_hat beta'               = 15991561.7   (printed 15991562)
    q~/B_MSIS                            = 4.19645      (printed 4.2)
    delta_MSIS                           = 1.003734     (printed 1.0037)
    D = max{D : 2^D <= w_hat s_1 n~ d~,  q~ > 4 w_hat 2^D} = 17

All of it is re-derived below rather than transcribed, and `test_lanes.py`
pins each against the printed digits.  There is no rejection sampling: the
Hint-MLWE masking replaces it.

Why these widths and not others
-------------------------------
Two identities the paper does not state, and both are load-bearing:

  * `s^2 = s_2^2 + w_hat^2 s_1^2`.  The published `s` is already the
    response width under the *worst-case* l1 challenge bound, not the
    typical one -- so `beta'` is conservative by construction.
  * `sigma_MLWE = s_0` **exactly**, on the unrounded widths.  Substituting
    `s_1 = 2 s_0` and `s_2 = 2 w_hat s_0` into the [KLSS23] reduction
    `1/sigma^2 = 2(1/s_1^2 + w_hat^2/s_2^2)` returns `s_0` identically.
    The widths are chosen so the hint reduction lands back on the smoothing
    parameter for `eps = 2^-100`, which is also why `s_2 = w_hat s_1`.
    `SIGMA_MLWE` below is computed from the *rounded* rationals and so sits
    `2.8e-9` relative away -- that is the `2^-20` rounding, not a gap in
    the identity.

A convention trap: `sigma` in the paper is the [KLSS23] *Gaussian
parameter* and `s` the standard deviation -- the reverse of the usual
reading, and they differ by `sqrt(2 pi)`.  The paper pins which is which by
printing both (`sigma_0 ~ 6.9353` against `s_0 ~ 2.7668`).  This module
works in standard deviations throughout, as `sample.gaussian_int` does.

Artifact status
---------------
The standard-deviation convention above gives
`delta_MLWE = 1.003996`, reproducing the paper's printed `1.0040`.
`estimate_lanes.py` retains an alternate estimator-API conversion only as a
sensitivity diagnostic; it is not used to select these parameters.

The concrete recovery-hint format, response infinity bound, sampler tail
cuts, and wire layout are implementation-level completions.  They are
labelled `Repair` in `lanes_manifest.json`.  Because this artifact does not
supply a reduction for that exact compression/recovery composition, the
tested backend is exposed as `lanes-experimental`; see
`exact.lanes_gate_cause()`.

The verifier's two bounds on `z` are below the width block.  The Euclidean
one is now the paper's `beta'` rule; the per-coefficient one is a Repair,
derived at `2^-128` against the distribution of the *response* `z = y + c r`
rather than of the mask alone.
"""

import math
from decimal import (ROUND_CEILING, ROUND_HALF_EVEN, Decimal as _Decimal,
                     localcontext as _localcontext)
from fractions import Fraction

from exact import ExactParams as _ExactParams
from exact import lanes_rank_roles as _lanes_rank_roles
from lanes_ring import QTILDE, DTILDE, LSPLIT, SUBDEG
from sample import (XOF, uniform_int, gaussian_int, rational_sigma,
                    GAUSSIAN_TAILCUT)
from dgs import (_d as _decimal, chi2_slack, quadratic_form_bound,
                 statistical_tailcut)

# ---- dimensions ----------------------------------------------------------

# Derived from `exact.ExactParams`, not copied.  These were literals that
# duplicated the exact layer's dimensions with nothing tying them
# together; `gen_kat.py` reads these, so a divergence would have been
# invisible until the two layers disagreed at runtime.
N_TILDE = _ExactParams.n_tilde      # the paper's `n~`
ELL_TILDE = _ExactParams.ell_tilde  # the paper's `l~`
N_EX = _ExactParams.N_ex        # message ring elements
AUX = _ExactParams.aux_slots    # g, and the two product-proof commitments
KAPPA = _lanes_rank_roles(N_TILDE, ELL_TILDE, N_EX, AUX)["kappa"]  # 17

# ---- which letter plays which structural role ---------------------------
#
# and the reason it needed a second pass.  The two letters are easy to
# swap and, at every profile so far, impossible to catch numerically: they
# were 7 and 8 in the profile and are both 4 in the one.  This
# module used to be internally inconsistent about them -- `N_LWE` and
# `M_LWE` below read the roles the right way round while `B_0`'s row count
# and `RESPONSE_RANK` read them the other -- and nothing could tell, because
# `KAPPA - n~` and `l~ + N_ex + alpha` were both 17 under the old numbers.
#
# The structure decides, and the paper's own MLWE dimensions state it:
# the coefficient-embedded instance has *secret* dimension `n~ d~` and
# `(l~ + N_ex + alpha) d~` samples.  The secret is the shared random tail,
# and the samples are the rows that touch it -- `B_0`'s `l~` rows plus the
# `N_ex + alpha` commitment rows.  So:
#
#     l~  is the identity rank: rows of `t_0`, width of `B_0`'s `I` block
#     n~  is the shared tail:   columns each `b_i` draws its randomness from
#
# Structural code below uses these role names rather than the letters, so a
# future parameter set with `n~ != l~` cannot silently pick the wrong one.
# `test_exact.py::test_the_lanes_role_names_match_the_exact_layer` checks
# them against `exact.ExactParams` and runs even while this backend is
# gated -- a test inside `test_lanes.py` would be skipped and prove nothing.

# Derived from the single helper, never re-stated here: an alias written
# out by hand is a second place the mapping can be wrong, and at equal
# ranks nothing numeric can tell.  `exact.lanes_rank_roles` is that one
# place, and `test_exact.py` drives it with `(7, 8)` -- where the two
# readings give 16 and 17 -- so it is a discriminating check.
_ROLES = _lanes_rank_roles(N_TILDE, ELL_TILDE, N_EX, AUX)

IDENTITY_RANK = _ROLES["identity_rank"]   # rows of t_0 / B_0's I block
TAIL_RANK = _ROLES["tail_rank"]           # width of the shared random tail

# The Bai--Galbraith response compression used in the LANES size model masks
# and transmits only the part of the opening outside B_0's identity block.
# The commitment still has rank KAPPA: these are response dimensions, not a
# different BDLOP key or a different Hint-MLWE instance.
RESPONSE_RANK = _ROLES["response_rank"]           # 13 = n~ + N_ex + alpha

W_HAT = _ExactParams.w_hat      # challenge weight, ||c||_1 = w_hat
DELTA = SUBDEG                  # 4, the partition stride
W_TILDE = W_HAT // DELTA        # 11 per residue class; `check()` pins DELTA | W_HAT

D_DROP = _ExactParams.D         # commitment compression, = 17
ALPHA = AUX                     # max degree in the exact relation, = 3

# ---- commitment / response recovery -------------------------------------
#
# ENS20 compresses these two objects separately: it sends the high part of
# t_0 (deferring its hint format to Dilithium) and sends only the response to
# B_0's non-identity columns (under a rejection condition).  The RiVeR
# artifact combines that size formula with rejection-free KLSS23 masking in
# one concrete composition and hint format.
#
# This implementation makes that missing wire rule explicit.  It drops D bits
# from t_0 with a centred remainder and quantises the first proof message w
# into RECOVERY_BUCKETS equal torus intervals.  The omitted response and the
# omitted low part perturb the verifier's reconstruction by
#
#     c * (t_0,low - r_identity).
#
# Its coefficient magnitude is at most RECOVERY_ERROR_BOUND.  The largest
# power-of-two bucket count whose *smallest* interval is wider than that bound
# makes the prover/verifier bucket difference cyclically -1, 0, or +1, so one
# Signed(1) hint coefficient recovers it without rejection sampling.  This is
# an implementation choice, not a security theorem supplied by the paper.

T0_SCALE = 1 << D_DROP
T0_LOW_BOUND = T0_SCALE // 2


def _mod_pm(value, power):
    low = value % power
    if low > power // 2:
        low -= power
    return low


def t0_power2round(value):
    """Canonical `value = high * 2^D + low`, with centred `low`."""
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < QTILDE:
        raise ValueError("non-canonical t0 coefficient")
    low = _mod_pm(value, T0_SCALE)
    return (value - low) // T0_SCALE, low


T0_HIGH_MODULUS = t0_power2round(QTILDE - 1)[0] + 1

# ---- Gaussian widths (**Paper**) ---------------------------------------
#
# The paper publishes the Hint-MLWE parameterization outright: a closed form
# with no free constant at all, so nothing here is searched.
#
#     eps  = 2^-100
#     s_0  = sqrt(ln(2 d~ (1 + 1/eps))) / pi          ~ 2.7667894
#     s_1  = 2 s_0                                     ~ 5.5335788   (r)
#     s_2  = 2 w_hat s_0                               ~ 243.4774692 (y)
#     s    = 2 sqrt(2) w_hat s_0                       ~ 344.3291391 (z)
#
# The paper prints all four to four or five places and every one is
# reproduced here to the last printed digit; `test_lanes.py` pins that.
#
# Two identities fall out that the paper does not state, and both are worth
# recording because they explain why these are the widths and not others:
#
#   * `s^2 = s_2^2 + w_hat^2 s_1^2`.  So `s` is the response width under the
#     *worst-case* l1 challenge bound `|c-hat| <= ||c||_1 = w_hat`, not the
#     typical `w_hat sigma_r^2` one.  The paper's `s` is already the
#     conservative one.
#   * `sigma_MLWE = s_0`, exactly on the unrounded widths. Feeding `s_1`
#     and `s_2` into the [KLSS23] reduction
#     `1/sigma^2 = 2(1/s_1^2 + w_hat^2/s_2^2)` returns `s_0` exactly. The
#     widths are chosen precisely so the hint
#     reduction lands back on the smoothing parameter for `eps = 2^-100`.
#     That is the design principle, and it is why `s_2 = w_hat s_1`.
#
# Provenance note: `sigma` in the paper's own sentence is the [KLSS23]
# *Gaussian parameter* and `s` the standard deviation -- the opposite of the
# usual convention, and it matters, because the two differ by `sqrt(2 pi)`.
# The paper pins which is which by printing both (`sigma_0 ~ 6.9353` against
# `s_0 ~ 2.7668`) and by saying "the standard deviation entering the LANES
# communication estimate is therefore `s`".  This module works in standard
# deviations throughout, as `sample.gaussian_int` does.

#: Working precision for the width derivation.  The widths are rounded once,
#: to denominator `2^20`, exactly as the sampler consumes them; everything
#: upstream of that rounding is carried at 60 digits so the rounding is the
#: only approximation.
_PREC = 60
_PI = _Decimal("3.14159265358979323846264338327950288419716939937510582097494")

#: `eps = 2^-100`, the smoothing-parameter target.  **Paper.**
SMOOTHING_EPS_EXP = 100

with _localcontext() as _ctx:
    _ctx.prec = _PREC
    #: `s_0 = sqrt(ln(2 d~ (1 + 1/eps))) / pi`.  **Paper.**
    S0 = ((2 * _Decimal(DTILDE)
           * (1 + _Decimal(2) ** SMOOTHING_EPS_EXP)).ln()).sqrt() / _PI
    S1 = 2 * S0                       # commitment randomness `r`
    S2 = 2 * W_HAT * S0               # proof mask `y`
    S_RESPONSE = 2 * _Decimal(2).sqrt() * W_HAT * S0     # response `z`

#: Denominator of the pinned rationals.  `rational_sigma` uses the same one,
#: so `Fraction`s built here survive a round trip through it unchanged.
SIGMA_DEN = 1 << 20


def _pin(value):
    """Round a `Decimal` width once, to denominator `SIGMA_DEN`."""
    return Fraction(int((value * SIGMA_DEN).to_integral_value(ROUND_HALF_EVEN)),
                    SIGMA_DEN)


SIGMA_R = _pin(S1)          # ~ 5.533579   commitment randomness
SIGMA_Y = _pin(S2)          # ~ 243.477469 proof mask

# `gaussian_int` samples on [-floor(t sigma), +floor(t sigma)].  Spell the
# bound in integers so it is the sampler's support, not a float approximation.
R_INF_SUPPORT = (GAUSSIAN_TAILCUT * SIGMA_R.numerator) // SIGMA_R.denominator
RECOVERY_ERROR_BOUND = W_HAT * (T0_LOW_BOUND + R_INF_SUPPORT)


def _recovery_buckets():
    buckets = 1
    while RECOVERY_ERROR_BOUND < QTILDE // (2 * buckets):
        buckets *= 2
    return buckets


RECOVERY_BUCKETS = _recovery_buckets()
RECOVERY_BITS = RECOVERY_BUCKETS.bit_length() - 1
assert RECOVERY_BUCKETS * 2 > QTILDE // RECOVERY_ERROR_BOUND
assert RECOVERY_ERROR_BOUND < QTILDE // RECOVERY_BUCKETS

# ---- the published security chain, re-derived ---------------------------
#
# Every number the paper prints for LANES follows from the widths above
# and the dimensions, with no further input.  They are derived here rather
# than transcribed so that a change to either end says so.

#: `(n~ + l~ + N_ex + alpha) d~ = 4352`.  **Paper**, printed as "the response
#: coefficient dimension".  This is the *full* rank-`kappa` opening, the
#: object BDLOP's `z = y + c r` bounds.
N_Z_PAPER = KAPPA * DTILDE

with _localcontext() as _ctx:
    _ctx.prec = _PREC
    #: `beta'_BDLOP = 2 s sqrt(N_z)`.  **Paper**: 45430.6.
    BETA_PRIME_BDLOP = 2 * S_RESPONSE * _Decimal(N_Z_PAPER).sqrt()
    #: `B_MSIS = 8 w_hat beta'`.  **Paper**: 15991562.
    B_MSIS = 8 * W_HAT * BETA_PRIME_BDLOP
    #: `q~ / B_MSIS`.  **Paper**: 4.2.
    Q_OVER_B_MSIS = _Decimal(QTILDE) / B_MSIS
    #: Root-Hermite factor for the M-SIS instance, closed form at
    #: `n = n~ d~`.  **Paper**: 1.0037.
    _LOG2 = _Decimal(2).ln()
    DELTA_MSIS = _Decimal(2) ** (
        (B_MSIS.ln() / _LOG2) ** 2
        / (4 * N_TILDE * DTILDE * (_Decimal(QTILDE).ln() / _LOG2)))

#: `sigma_MLWE` from the [KLSS23] hint reduction,
#: `1/sigma^2 = 2(1/s_1^2 + w_hat^2/s_2^2)`.  Equal to `S0` by construction;
#: computed rather than aliased so the identity is a checked property.
SIGMA_MLWE_SQ = 1 / (2 * (1 / SIGMA_R ** 2
                          + Fraction(W_HAT ** 2, 1) / SIGMA_Y ** 2))
SIGMA_MLWE = math.sqrt(float(SIGMA_MLWE_SQ))

#: The commitment compression parameter.  **Paper**: `D = 17`, "the largest
#: value satisfying `2^D <= w_hat s_1 n~ d~` and `q~ > 4 w_hat 2^D`".
#: Re-derived; `test_lanes.py` pins that it lands on 17 and that `D = 18`
#: fails the first inequality.


def largest_compression_exponent():
    """The paper's two inequalities, evaluated exactly."""
    limit = Fraction(W_HAT) * SIGMA_R * N_TILDE * DTILDE
    best = 0
    for exponent in range(1, 64):
        if Fraction(1 << exponent) <= limit and QTILDE > 4 * W_HAT * (1 << exponent):
            best = exponent
        else:
            break
    return best


#: The MLWE width the [KLSS23] reduction runs at, and the instance it runs
#: against.  `sigma_MLWE^2` is rational even though `sigma_MLWE` need not be.
#: Recorded rather than used: no estimator ships here, so this is the input
#: someone else's estimator needs, not a checked security claim.
SIGMA_MLWE_SQ = 1 / (2 * (1 / SIGMA_R ** 2
                           + Fraction(W_HAT ** 2, 1) / SIGMA_Y ** 2))
SIGMA_MLWE = math.sqrt(float(SIGMA_MLWE_SQ))
N_LWE = _ROLES["lwe_secret_rank"] * DTILDE       # 1024: the secret is the tail
M_LWE = _ROLES["lwe_sample_rank"] * DTILDE       # 3328: the rows that touch it

# ---- verifier bounds on the response ------------------------------------
#
# the paper supplies the Euclidean one.  The paper's
#
#     beta'_BDLOP = 2 s sqrt(N_z),      s = 2 sqrt(2) w_hat s_0
#
# is a flat "two standard deviations per coordinate" rule, and it is the
# bound the security claim rests on: `B_MSIS = 8 w_hat beta'` is what the
# extractor gets from two accepting forks, so a verifier that enforced
# anything looser than `beta'` would not support the published `B_MSIS`.
#
# One dimension mismatch has to be named rather than absorbed.  The paper
# bounds the *full* rank-`kappa` opening, `N_z = kappa d~ = 4352`.  This
# implementation applies Bai--Galbraith compression and transmits only the
# `kappa - l~ = 13` non-identity elements, recovering the rest through the
# carry hint, so the verifier has `RESPONSE_RANK d~ = 3328` coefficients in
# front of it.  The per-coordinate rule is applied to the coordinates that
# exist:
#
#     Z_NORM2_BOUND = (2 s)^2 * RESPONSE_RANK * d~.
#
# That is **stricter** than scaling the paper's number down would be, and
# strictly stricter than the paper's own bound, so the published `B_MSIS`
# -- computed at the larger 4352 -- remains an upper bound on what an
# extractor can obtain here.  `test_lanes.py` pins the direction.
#
# The per-coefficient bound is a **Repair**: the paper gives none for LANES.
# It is derived at the same `2^-128` target as everything else in this tree,
# against the distribution of the *response* `z = y + c r` rather than of the
# mask alone.  Getting that wrong is not a tightness question but a
# correctness bug -- an earlier form used `6 sigma_y`, which is 5.93 sd of a
# `z` coefficient and rejected about one honest proof in 119,000.

#: `c` has `w_hat` nonzero coefficients in `{-1,+1}` and `r <- D_{sigma_r}`,
#: so each coefficient of `c r` is a signed sum of `w_hat` independent draws.
VAR_CR = W_HAT * SIGMA_R ** 2                   # ~ 1347.4
VAR_Z = SIGMA_Y ** 2 + VAR_CR                   # ~ 60628.7
N_Z = RESPONSE_RANK * DTILDE                    # transmitted coefficients

#: The paper's rule at the transmitted rank: `(2 s)^2 N_Z`, as an exact
#: integer.
#:
#: Evaluated on the **rounded** widths, through the identity
#: `s^2 = s_2^2 + w_hat^2 s_1^2`, rather than on the 60-digit `S_RESPONSE`.
#: Two reasons, and the first is the binding one:
#:
#:   * `SIGMA_R` and `SIGMA_Y` are what the sampler actually draws at, so
#:     this is the bound that matches the distribution being bounded;
#:   * it is an exact rational computation over `Fraction`, so `river-rs`
#:     reproduces the same integer from the same two pinned rationals.  A
#:     `Decimal` chain at 60 digits would not port -- and this integer is a
#:     verifier decision, so the two implementations have to agree on it
#:     exactly, not to within a rounding.
#:
#: The difference against the unrounded form is under one part in 10^6 and
#: `test_lanes.py` pins that it is.
Z_NORM2_BOUND = -(-(4 * (SIGMA_Y ** 2 + W_HAT ** 2 * SIGMA_R ** 2)
                    * N_Z).numerator
                  // (4 * (SIGMA_Y ** 2 + W_HAT ** 2 * SIGMA_R ** 2)
                     * N_Z).denominator)

#: The paper's own `beta'^2`, at its own dimension, for comparison.
with _localcontext() as _ctx:
    _ctx.prec = _PREC
    Z_NORM2_BOUND_PAPER = int((BETA_PRIME_BDLOP ** 2)
                              .to_integral_value(ROUND_CEILING))

# ---- is the paper's bound satisfiable?  A quadratic form, not a chi-square
#
# Nothing above establishes that an *honest* response passes.  `prove` had no
# retry until the paper, so a bound an honest prover misses is a proof its
# own verifier rejects.  The check below derives what the response
# distribution actually needs at `2^-128` and asserts the paper's bound is
# above it -- so the margin is measured, not assumed.
#
# `z = y + c r` does not have independent coefficients.  Conditioned on `c`,
# the map `r -> c r` is negacyclic multiplication, which diagonalises over the
# primitive `2 d~`-th roots of unity with eigenvalues `|c-hat(zeta_j)|^2`, so
# the covariance of one response polynomial is
#
#     Sigma_1 = sigma_y^2 I + sigma_r^2 M M^T,
#     eig(Sigma_1) = { sigma_y^2 + sigma_r^2 |c-hat(zeta_j)|^2 }_{j < d~},
#
# and `Sigma` is `RESPONSE_RANK` such blocks (the `r_i` are independent, but
# they all meet the *same* `c`).  Two consequences:
#
#   * the diagonal is untouched.  `sum_j |c-hat|^2 = d~ ||c||_2^2`, so the
#     mean eigenvalue is exactly `sigma_y^2 + w_hat sigma_r^2 = VAR_Z` and the
#     per-coefficient variance -- hence `Z_INF_BOUND`'s union bound, which
#     never needed independence -- is unchanged;
#   * the Euclidean tail is not.  `chi2_slack` is the `Sigma = v I` case, and
#     using it here asserts a `2^-128` failure probability that the spectrum
#     does not support.  It comes out about 1.7% low at a typical challenge
#     and more at a bad one.
#
# So the requirement comes from Laurent-Massart, which needs the trace, the
# Frobenius norm and the operator norm.  Only the first is exact; the other
# two are bounded over *every* challenge in `C`, using
# `|c-hat|^2 <= ||c||_1^2 = w_hat^2` and `sum_j |c-hat|^4 <= max |c-hat|^2 *
# sum_j |c-hat|^2`.

#: `tr(Sigma) = n Var[z]`, exactly.
SIGMA_TRACE = N_Z * VAR_Z

#: `||Sigma||_op <= sigma_y^2 + w_hat^2 sigma_r^2`, at `|c-hat|^2 = ||c||_1^2`.
SIGMA_OP = SIGMA_Y ** 2 + W_HAT ** 2 * SIGMA_R ** 2

#: `||Sigma||_F^2 = RESPONSE_RANK sum_j (sigma_y^2 + sigma_r^2 |c-hat_j|^2)^2`,
#: expanded and bounded term by term.
SIGMA_FROB_SQ = N_Z * (SIGMA_Y ** 4
                       + 2 * W_HAT * SIGMA_Y ** 2 * SIGMA_R ** 2
                       + W_HAT ** 3 * SIGMA_R ** 4)

#: The smallest Euclidean bound an honest response can be held to at
#: `2^-128`.  Not what the verifier enforces -- the paper's is -- but what
#: the paper's has to clear.
Z_NORM2_REQUIRED = int(quadratic_form_bound(SIGMA_TRACE, SIGMA_FROB_SQ,
                                            SIGMA_OP).to_integral_value(ROUND_CEILING))

#: What the same target would give under the independence hypothesis, kept
#: so the size of the correction is visible rather than asserted.
NORM2_SLACK = chi2_slack(N_Z)
Z_NORM2_REQUIRED_IID = int((NORM2_SLACK * _decimal(SIGMA_TRACE))
                           .to_integral_value(ROUND_CEILING))

#: The margin the paper's bound leaves over what an honest response needs.
#: Asserted here, not merely reported: a parameter change that inverted it
#: would otherwise surface as a proof the verifier rejects.
assert Z_NORM2_BOUND > Z_NORM2_REQUIRED, (Z_NORM2_BOUND, Z_NORM2_REQUIRED)

#: Per-coefficient bound, at the same `2^-128` target: `statistical_tailcut`
#: union-bounds over all `N_Z` coefficients and returns 14 standard
#: deviations of a `z` coefficient -- not of a `y` coefficient, and not the
#: `6 sigma_y` that was here before, which was 5.93 sd and failed once in
#: 119,000 proofs.
Z_TAILCUT = statistical_tailcut(N_Z)


def _ceil_sqrt(value):
    """Smallest integer `B` with `B^2 >= value`, for an exact `Fraction`."""
    num, den = value.numerator, value.denominator
    b = math.isqrt(num // den)
    while b * b * den < num:
        b += 1
    return b


#: `ceil(t sqrt(Var[z]))`, computed on the exact rational.
#:
#: It used to be `t * isqrt(int(Var[z]))`, which floors the square root
#: *before* multiplying and so loses up to `t` units rather than one.  The
#: shipped bound was therefore 13.9897 sd, not the 14 its own comment
#: claimed -- not a security failure, but a false stated derivation, and one
#: that rejected honest responses in a six-value window.
Z_INF_BOUND = _ceil_sqrt(Fraction(Z_TAILCUT) ** 2 * VAR_Z)

#: Retained for the sampler's own truncation, which is a separate question.
TAIL = 1.05


def message_slot_count():
    """Total scalars the commitment carries: `N_ex * l`."""
    return N_EX * LSPLIT


# ---- samplers ------------------------------------------------------------

def sample_uniform_poly(xof):
    return [uniform_int(xof, QTILDE) for _ in range(DTILDE)]


def as_rational(sigma):
    """`(num, den)` for an `int`, a `Fraction`, or a `float` width."""
    if isinstance(sigma, Fraction):
        return sigma.numerator, sigma.denominator
    if isinstance(sigma, int):
        return sigma, 1
    return rational_sigma(sigma)


def sample_gaussian_poly(xof, sigma):
    num, den = as_rational(sigma)
    return [gaussian_int(xof, num, den) % QTILDE for _ in range(DTILDE)]


def sample_gaussian_vec(xof, sigma, length):
    return [sample_gaussian_poly(xof, sigma) for _ in range(length)]


def sample_challenge(xof):
    """LANES challenge: low-weight ternary with **partitioned support**.

    The challenge space of [ENS20], and the set equation (1) of the RiVeR
    paper's Section 2.3 describes.  For each of the `DELTA = 4` residue
    classes mod 4, a partial Fisher-Yates places exactly `W_TILDE = 11`
    coefficients in `{-1, +1}`, giving total weight `DELTA * W_TILDE = 44`.

    The partition is not cosmetic: spreading the weight evenly across residue
    classes controls the challenge's behaviour in each NTT block, which a
    plain weight-44 ternary polynomial would not.

    Section 2.3 labels this the *OOM* challenge space, but the OOM layer
    actually uses `C^d_{w,gamma}`; it belongs here.
    """
    poly = [0] * DTILDE
    for i in range(DELTA):
        for j in range(LSPLIT - W_TILDE, LSPLIT):
            x = uniform_int(xof, j + 1)                  # x in [0, j]
            poly[j * DELTA + i] = poly[x * DELTA + i]
            poly[x * DELTA + i] = 1 if xof.bit() else QTILDE - 1
    return poly


def challenge_l1_norm(poly):
    h = QTILDE // 2
    return sum(abs(c - QTILDE if c > h else c) for c in poly)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The parameterization below is the paper's. The production backend
    # alias is reserved by the artifact's composition policy, not by these
    # dimensions. See `exact.lanes_gate_cause()`.
    print(f"lanes_params.py: paper widths  s_1={float(SIGMA_R):.6f}"
          f"  s_2={float(SIGMA_Y):.6f}  s={S_RESPONSE:.7f}")
    print(f"  beta'={BETA_PRIME_BDLOP:.4f}  B_MSIS={B_MSIS:.3f}"
          f"  q~/B={Q_OVER_B_MSIS:.5f}  delta_MSIS={DELTA_MSIS:.6f}")

    assert KAPPA == N_TILDE + ELL_TILDE + N_EX + AUX == 17, KAPPA
    assert (N_TILDE, ELL_TILDE) == (4, 4)
    # `kappa - l~`: the identity rank is `l~`, not `n~`.  Both are 4 today,
    # which is exactly why the role names are used rather than the letters.
    assert RESPONSE_RANK == KAPPA - ELL_TILDE == 13
    assert IDENTITY_RANK == ELL_TILDE and TAIL_RANK == N_TILDE
    assert N_Z_PAPER == 4352 and N_Z == 3328
    assert largest_compression_exponent() == D_DROP == 17
    # every message block is padded: `6 d != N_ex l` is intentional, and
    # `lanes_backend.build_linear_system` constrains the padding to zero
    assert message_slot_count() == 384 == N_EX * LSPLIT
    assert Z_NORM2_BOUND > Z_NORM2_REQUIRED
    print(f"  D={D_DROP}  buckets={RECOVERY_BUCKETS}  |z|^2 bound"
          f"={Z_NORM2_BOUND} (honest needs {Z_NORM2_REQUIRED},"
          f" {Z_NORM2_BOUND / Z_NORM2_REQUIRED:.2f}x)")
    print("  self-check ok")
