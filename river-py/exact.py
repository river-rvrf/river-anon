"""
exact.py -- The exact layer `Pi_ex` of RiVeR.

`Pi_ex` is a two-stage commit-and-prove system for

    R^_ex = { ((W, z_eval, x), (e_eval, y_eval)) :
                W  = Com_{ck_ex}(e_eval, y_eval)
              & z_eval = x * e_eval + y_eval
              & coeffs(e_eval + B_e) in [0, q_0 - 1]^d }

The two-stage split matters: `Com` runs *before* the statement exists, so `W`
can be folded into the OOM Fiat-Shamir context and is therefore fixed before
the challenge `x` (Figure 4, and Lemma "exact-delayed-simulation").

What is modelled here
---------------------
The paper treats `Pi_ex` as a black box and instantiates it with LANES
(ENS20 / LANES+ / Hint-MLWE).  It publishes the LANES dimensions, sampler
widths, response-norm model, compression exponent, and a fixed
`|pi_ex| = 13.5` KB entropy estimate.  This module implements those
parameters together with a concrete codec and recovery-hint completion:

  * the exact-backend parameters
    `(n~, l~, d~, w_hat, D, N_ex, q~)
       = (4, 4, 256, 44, 17, 6, 67107713)` with splitting factor `l = 64`;
  * the adjusted radix-3 reconstruction vector `(1, 3, 9, 17)` with digits in
    `{0,1,2}`, which covers `[0, 60]` exactly (max `2(1+3+9+17) = 60`);
  * the six semantic exact messages, in the paper's order
    `(y_eval, e_eval, d_0, d_1, d_2, d_3)`, one per LANES message block:
    each block carries the element's `d = 32` coefficients in its first 32
    slots and is zero-padded to `l = 64`.  The old `6 d == N_ex l` identity is
    gone -- `192 != 384` is intentional;
  * a BDLOP-style commitment over `R_q~` with those ranks
    (`kappa = n~ + l~ + N_ex + alpha = 17`, transmitted response rank 13).

and then plugs in a **proof backend** behind a small interface.  The backend
shipped here, `OpeningBackend`, enforces every clause of `R^_ex` -- including
the integer equality, see below -- but `sigma_ex` *is* the opening, so it is
not zero knowledge.

`OpeningBackend` is a mock, not a stand-in for LANES.  Substituting a real
prover is not confined to `prove`/`verify`: it changes the commitment
randomness distribution, the transcript, the encoding and the size.  Read
`|pi_ex|` from this module as the cost of *this* opening, not as a concrete
encoding of the paper's entropy estimate.

The modulus condition and centred representation
-------------------------------------------------
LANES has a single modulus, so it can only check the link modulo `q~`.  That
pins an integer only when no accepted response can wrap.  The difference of
two accepted error responses is at most `12 sigma_m = 12 phi_m eta_m`, so

    q~ > 24 phi_m eta_m                      (`q_tilde_clears`, below)

makes `z_eval - x e_eval` have a unique centred lift.  With
`eta_m = w gamma B_e sqrt(d)` this is one number for all five profiles:
`66730968.02` against the selected `q~ = 67107713`, a margin of `376744.98`
-- about 0.56%.

The construction explicitly translates the canonical rounding error in
`[0, q_0-1] = [0, 60]` to its centred representation in `[-B_e,B_e]` before
applying the norm bounds.  This implementation follows that translation in
`ring.to_centered_error`.  The distinction is arithmetically load-bearing:
using 60 as a magnitude bound would double the requirement to
`133461936.03`, which the selected modulus would not satisfy.

Because 0.56% is inside what a float `sqrt` and a multiplication chain can
move, `q_tilde_clears` decides the condition over the integers as
`q~^2 > (24 phi_m w gamma B_e)^2 d` rather than in floating point.

`check_relation` still verifies the link over `Z`.  Under the new modulus that
is no longer load-bearing for this backend -- it is redundant with the
commitment -- but it is the clause `R^_ex` actually states, and a backend that
proves it only modulo `q~` should have to say so.
"""

from fractions import Fraction

from params import is_prime
from ring import Ring, from_centered_error, negacyclic_mul_int
from sample import XOF, DS_EXACT, sam_mat, uniform_beta_vec
from codec import Layout, Field, Uniform, Signed, Rice, floor_sqrt


# ---- exact-backend parameters -------------------------------------------

#: Reconstruction weights for the rounding-error range.  Digits in {0,1,2}
#: give exactly [0, 60] = [0, q_0 - 1].
RADIX_WEIGHTS = (1, 3, 9, 17)
RADIX_DIGITS = 3                      # digit alphabet {0, 1, 2}


def lanes_rank_roles(n_tilde, ell_tilde, n_ex, aux_slots):
    """Which of `n~` and `l~` plays which structural role, for any pair.

    **The single place this mapping exists.**  `ExactParams` and
    `lanes_params` both derive from it rather than restating it, so a test
    can drive it with *unequal* ranks and see the mapping the production
    code actually implements.  A numeric check at the published parameters
    cannot distinguish the role names because both ranks are 4.

    The paper's own MLWE dimensions fix the assignment.  The
    coefficient-embedded instance has *secret* dimension `n~ d~` and
    `(l~ + N_ex + alpha) d~` samples; the secret is the shared randomness
    tail, and the samples are the rows that touch it -- `B_0`'s rows plus
    the `N_ex + alpha` commitment rows.  Hence:

        l~   the identity rank: rows of `t_0`, width of `B_0`'s `I` block
        n~   the shared tail:   columns each `b_i` draws randomness from

    Returned as a dict rather than a tuple so callers use the structural
    role names instead of relying on positional conventions.
    """
    kappa = n_tilde + ell_tilde + n_ex + aux_slots
    return {
        "identity_rank": ell_tilde,
        "tail_rank": n_tilde,
        "kappa": kappa,
        # The Bai--Galbraith response drops `B_0`'s identity block, so it
        # is `kappa` minus the *identity* rank -- never minus `n~`.
        "response_rank": kappa - ell_tilde,
        # The two dimensions the paper prints, which is what decides it.
        "lwe_secret_rank": n_tilde,
        "lwe_sample_rank": ell_tilde + n_ex + aux_slots,
    }


class ExactParams:
    """LANES-side parameters, fixed for every RiVeR profile."""

    #: paper, Appendix "Detailed Parameter Setting":
    #: `(n~, l~, d~, w_hat, D) = (4, 4, 256, 44, 17)` with splitting factor
    #: `l = 64`, `N_ex = 6` and `q~ = 67107713`.  All **Paper**.
    d_tilde = 256                     # internal LANES ring dimension
    l_split = 64                      # splitting factor => 64 slots
    q_tilde = 67107713                # Paper; 26 bits, 129 mod 256
    #: The two ranks carry the roles the *structure* gives them, not the
    #: ones the letters suggest, and they are easy to read the wrong way
    #: round.  `B_0 = [I_{l~} | B_0']`, so its
    #: identity block is `l~` wide and `t_0` has `l~` rows; every `b_i`
    #: draws its random tail from the last `n~` columns, so `n~` is the
    #: shared-tail width.  The paper's own MLWE dimensions read the same
    #: way: the coefficient-embedded instance has *secret* dimension
    #: `n~ d~ = 1024` (the tail) against
    #: `(l~ + N_ex + alpha) d~ = 3328` samples (the rows that touch it).
    #:
    #: Both are 4 at this profile, so no expression evaluates differently
    #: and no byte moves -- which is precisely why the labels drifted back.
    #: `test_exact.py` measures each role off a constructed key rather than
    #: comparing the two constants, which would be `4 == 4` and prove
    #: nothing.
    n_tilde = 4                       # shared random tail width
    ell_tilde = 4                     # B_0 identity rank == rows of t_0
    D = 17                            # commitment compression
    w_hat = 44                        # LANES challenge weight
    N_ex = 6                          # message ring elements
    aux_slots = 3                     # g, and the two product-proof commitments

    def __init__(self, par):
        self.par = par                       # the outer RiVeR profile
        self.d = par.d                       # outer ring dimension (32)
        self.q0 = par.q0

        # Commitment shape, following [ENS20] Figure 3 at the paper's
        # LANES parameters, so that a real backend is a drop-in rather than
        # a reshape:
        #   t_0 has l~ = 4 entries
        #   the randomness rank is n~ + l~ + N_ex + alpha = 17
        #   the transmitted response rank is kappa - l~ = 13
        # The "+ alpha" reserves the slots a LANES prover needs for the
        # masking element g and the two product-proof commitments
        # t_{N+2}, t_{N+3}; this backend does not use them, but sizing
        # without them would make the commitment the wrong shape.
        self.Rt = Ring(self.q_tilde, self.d_tilde)     # commitment ring

        # `check()` was a diagnostic nobody called: neither backend's
        # `setup` invoked it, so a stale, composite or independently edited
        # exact parameter survived construction and only surfaced as a wrong
        # proof.  `RiVeR.setup` raises on `par.check()` for the same reason
        #; this is the exact-layer half of it, and it runs wherever the
        # backend is built rather than only where `setup` is called.
        bad = self.check()
        if bad:
            raise ValueError("exact parameters rejected: " + "; ".join(bad))

    #: Coefficients carried by each of the `N_ex` message blocks, and the
    #: zero padding that fills the block out to `l` slots. The paper
    #: gives each of the six outer ring elements its own 64-slot block whose
    #: first `d = 32` slots carry coefficients, so the old `N_ex l == 6 d`
    #: identity no longer holds: 384 != 192, and that is intentional.
    #:
    #: Properties rather than cached values, deliberately: `Verify` runs
    #: against a caller-supplied `pp`, and a mutated `l` or `d~` has to reach
    #: the packing rather than be masked by a value cached at construction.

    @property
    def roles(self):
        """The structural role of each rank; see `lanes_rank_roles`."""
        return lanes_rank_roles(self.n_tilde, self.ell_tilde,
                                self.N_ex, self.aux_slots)

    @property
    def t0_rows(self):
        """Rows of `t_0`, i.e. `B_0`'s identity rank -- which is `l~`.

        Derived, not restated: `lanes_rank_roles` is the only place the
        mapping lives, so driving *it* with unequal ranks tests this too.
        """
        return self.roles["identity_rank"]

    @property
    def response_rank(self):
        """`kappa - l~ = 13`: the part of the opening actually transmitted.

        The Bai--Galbraith compression masks and sends only the response to
        `B_0`'s non-identity columns, so the identity rank comes off.
        """
        return self.roles["response_rank"]

    @property
    def kappa(self):
        """`n~ + l~ + N_ex + alpha = 17`: the commitment randomness rank.

        The `+ alpha` reserves the slots a LANES prover needs for the
        masking element `g` and the two product-proof commitments; this
        backend does not use them, but sizing without them would make the
        commitment the wrong shape.
        """
        return self.roles["kappa"]

    @property
    def fingerprint(self):
        """Every value that defines this parameter set, for comparison.

        `Verify` runs against a caller-supplied `pp`.  Chasing liveness
        attribute by attribute is endless -- one cached derived value and a
        mutated `pp` verifies anyway -- so the verifier compares the whole
        fingerprint against the profile's own instead.  See
        `RiVeR._verify`.
        """
        return (self.d_tilde, self.l_split, self.q_tilde, self.n_tilde,
                self.ell_tilde, self.D, self.w_hat, self.N_ex,
                self.aux_slots, self.d, self.q0)

    @property
    def block_slots(self):
        return self.l_split

    @property
    def block_used(self):
        return self.d

    @property
    def block_pad(self):
        return self.l_split - self.d

    @property
    def slot_stride(self):
        """`d~ / l`: the coefficient index step between adjacent slots."""
        return self.d_tilde // self.l_split

    @property
    def q_tilde_need(self):
        """`24 phi_m eta_m`: the no-wrap bound `q~` has to clear.

        Two accepted responses each satisfy `||z_m||_inf <= 6 sigma_m`, so
        their difference is at most `12 sigma_m = 12 phi_m eta_m`, and a
        unique centred lift of `z_eval - x e_eval` needs `q~` above twice
        that.  `eta_m = w gamma B_e sqrt(d)` is profile-independent, so this
        is one number for all five: 66730968.02, against `q~ = 67107713`.

        The margin is 376744.98, about 0.56%.  It relies on the specified
        centred bound `B_e = 30`; using 60 as a magnitude bound would double
        the requirement.  See `q_tilde_clears` and
        `ring.to_centered_error`.

        The old reconstruction-side floor `w gamma 3^4 = 41472` is kept in the
        max only because it costs nothing; it is three orders of magnitude
        below the response-side term.
        """
        return max(24 * self.par.phi_m * self.par.eta_m,
                   self.par.w * self.par.gamma * 3 ** 4)

    def q_tilde_clears(self, B_e=None):
        """Exact `q~ > 24 phi_m w gamma B_e sqrt(d)`, with no float in it.

        The margin is 0.56%, which is inside what a float `sqrt` plus a
        multiplication chain can move, so the condition the whole exact
        backend rests on is decided over the integers:
        `q~^2 > (24 phi_m w gamma B_e)^2 d`.
        """
        par = self.par
        if B_e is None:
            B_e = par.B_e
        k = 24 * par.phi_m * par.w * par.gamma * B_e
        if k != int(k):
            raise ValueError("non-integral 24 phi_m w gamma B_e")
        return self.q_tilde ** 2 > int(k) ** 2 * par.d

    def check(self):
        errors = []
        # Primality was assumed rather than tested, which left a composite
        # `q~` -- where `R_q~` is not even a domain -- indistinguishable from
        # a good one until a proof failed.  `params.check()` tests the outer
        # moduli the same way.
        if not is_prime(self.q_tilde):
            errors.append(f"q~ = {self.q_tilde} is not prime")
        if self.q_tilde % (4 * self.l_split) != 2 * self.l_split + 1:
            errors.append("q~ != 2l+1 mod 4l (fully-splitting condition)")
        if not 2 ** 25 <= self.q_tilde < 2 ** 26:
            errors.append("ceil(log2 q~) is not 26 as reported")
        # The six exact messages are `(y_eval, e_eval, d_0, d_1, d_2, d_3)`,
        # one per block.  Each block carries `d` coefficients in its first
        # `d` slots and is zero-padded to `l`.
        if 1 + 1 + len(RADIX_WEIGHTS) != self.N_ex:
            errors.append("exact message count != N_ex")
        if self.l_split < self.d:
            errors.append("l < d, so a message block cannot hold d slots")
        if 2 * sum(RADIX_WEIGHTS) != self.q0 - 1:
            errors.append("radix weights do not cover [0, q_0-1] exactly")
        # Decided exactly; `q_tilde_need` is the float form, for reporting.
        if not self.q_tilde_clears():
            errors.append(f"q~ <= {self.q_tilde_need:.7g} "
                          "(internal modulus condition)")
        # Structural invariants the *derived* LANES constants assume.
        # `lanes_ring` computes `SUBDEG = d~ // l` and `LEVELS =
        # l.bit_length() - 1`, and `lanes_params` computes `W_TILDE =
        # w_hat // DELTA` -- each silently truncates, or stops being a
        # logarithm, if these do not hold.  Checking the values are
        # unchanged is not the same as checking they are well formed.
        if self.l_split <= 0:
            errors.append(f"l = {self.l_split} is not positive")
        else:
            if self.d_tilde % self.l_split:
                errors.append(f"l = {self.l_split} does not divide "
                              f"d~ = {self.d_tilde}, so SUBDEG truncates")
            if self.l_split & (self.l_split - 1):
                errors.append(f"l = {self.l_split} is not a power of two, so "
                              "LEVELS = l.bit_length()-1 is not log2(l)")
            delta = self.d_tilde // self.l_split
            if delta == 0:
                errors.append("d~ < l, so the slot stride is 0")
            elif self.w_hat % delta:
                errors.append(f"DELTA = {delta} does not divide "
                              f"w_hat = {self.w_hat}, so W_TILDE truncates")
        return errors


# ---- radix-3 range encoding ---------------------------------------------

def radix_decompose(value):
    """Digits `(a_0, a_1, a_2, a_3)` in {0,1,2} with `sum g_j a_j == value`.

    Greedy from the largest weight.  The encoding is not injective
    (17 = (0,0,0,1) = (2,2,1,0)); that is harmless, soundness only needs the
    reachable set to be exactly [0, 60].
    """
    if not 0 <= value <= 2 * sum(RADIX_WEIGHTS):
        raise ValueError(f"{value} outside the representable range")
    digits = [0] * len(RADIX_WEIGHTS)
    remainder = value
    for idx in range(len(RADIX_WEIGHTS) - 1, -1, -1):
        weight = RADIX_WEIGHTS[idx]
        digit = min(RADIX_DIGITS - 1, remainder // weight)
        digits[idx] = digit
        remainder -= digit * weight
    if remainder != 0:
        raise ValueError(f"greedy decomposition failed for {value}")
    return digits


def radix_recompose(digits):
    return sum(w * a for w, a in zip(RADIX_WEIGHTS, digits))


def decompose_poly(coeffs):
    """Decompose each coefficient; return 4 lists of `d` digits each."""
    per_coeff = [radix_decompose(c) for c in coeffs]
    return [[row[j] for row in per_coeff] for j in range(len(RADIX_WEIGHTS))]


# ---- witness packing -----------------------------------------------------

def pack_witness(ex, e_eval, y_eval_c, digit_polys):
    """Lay the six exact messages into `N_ex` elements of `R_q~`.

    The paper gives **each of the six outer ring elements its
    own LANES message block**: a block holds `l = 64` slots, the element's
    `d = 32` coefficients occupy the first 32, and the remaining 32 are
    explicit zero padding.  So `6 d = 192` scalars sit in `N_ex l = 384`
    slots and the old `6 d == N_ex l` identity is gone -- `192 != 384` is
    intentional, not a shortfall.

    Scalar `j` of message element `i` goes to coefficient `j * slots`, which
    mirrors the NTT-slot layout a real LANES backend uses: [ENS20] commits one
    scalar per NTT block, at index `j * delta` of the transformed array.

    The element order is the paper's, `(y_eval, e_eval, d_0, ..., d_3)`.
    """
    elements = [list(y_eval_c), list(e_eval)] + [list(p) for p in digit_polys]
    # Not `assert`: this validates data, and `python -O` strips asserts.  It
    # is also reachable from `Verify` through a `pp` whose `N_ex` does not
    # match the profile, where an `AssertionError` is the wrong shape of
    # failure -- a verifier returns a bit.
    if len(elements) != ex.N_ex:
        raise ValueError(
            f"witness has {len(elements)} message elements, "
            f"expected {ex.N_ex}")
    for idx, element in enumerate(elements):
        if len(element) != ex.block_used:
            raise ValueError(
                f"witness element {idx} has {len(element)} coefficients, "
                f"expected {ex.block_used}")
    out = []
    for element in elements:
        poly = [0] * ex.d_tilde
        for j in range(ex.block_used):
            poly[j * ex.slot_stride] = element[j] % ex.q_tilde
        # slots `block_used .. block_slots-1` stay zero: that is the padding,
        # and `padding_is_zero` is what makes it a checked property rather
        # than a convention.
        out.append(poly)
    return out


def unpack_witness(ex, message):
    """Inverse of `pack_witness`; the `N_ex * d` carried scalars mod q~.

    The padding slots are *not* returned.  Use `padding_is_zero` to check
    them; a verifier that silently ignored them would accept a witness
    carrying data the relation does not constrain.
    """
    scalars = []
    for poly in message:
        for j in range(ex.block_used):
            scalars.append(poly[j * ex.slot_stride])
    return scalars


def padding_is_zero(ex, message):
    """Every slot past `block_used` in every block is zero mod `q~`.

    The paper makes the padding part of the committed message, so it is
    part of what the exact relation has to pin. `LanesBackend` includes it
    in the proved linear system; here it is also enforced at the boundary
    so the two cannot disagree about which messages are well formed.
    """
    if len(message) != ex.N_ex:
        return False
    for poly in message:
        if len(poly) != ex.d_tilde:
            return False
        for j in range(ex.block_used, ex.block_slots):
            if poly[j * ex.slot_stride] % ex.q_tilde != 0:
                return False
    return True


# ---- commitment ----------------------------------------------------------

class ExactCommitmentKey:
    """BDLOP commitment key `(A_1, A_2)` over `R_q~`.

    `A_1` is `t0_rows x kappa` and `A_2` is `N_ex x kappa`, mirroring `B_0`
    and `(b_j)` of the [BDLOP18] commitment as [ENS20] Figure 3 uses it.

    One deliberate divergence: LANES samples the commitment randomness from a
    discrete Gaussian (and `lanes_commit.py` does), whereas this
    key uses ternary randomness, the BDLOP convention.  The paper does not
    give the LANES-internal Gaussian width, and guessing it would be worse
    than stating the difference.
    """

    def __init__(self, ex, seed):
        self.ex = ex
        self.seed = seed
        self.A1 = sam_mat(seed, ex.q_tilde, ex.t0_rows, ex.kappa,
                          ex.d_tilde, "Pi_ex.A1")
        self.A2 = sam_mat(seed, ex.q_tilde, ex.N_ex, ex.kappa,
                          ex.d_tilde, "Pi_ex.A2")

    def commit(self, message, randomness):
        """`W = (A_1 r, A_2 r + m)`."""
        R = self.ex.Rt
        t0 = R.mat_vec(self.A1, randomness)
        t1 = R.mat_vec(self.A2, randomness)
        t1 = [R.add(t1[i], message[i]) for i in range(self.ex.N_ex)]
        return {"t0": t0, "t1": t1}


# ---- relation ------------------------------------------------------------

def check_relation(ex, statement, witness):
    """Decide `R^_ex` directly.  Returns a list of violated clauses.

    Used by the backend verifier and, independently, by the tests -- the
    relation is small enough to state once and check literally.
    """
    errors = []
    q0, d = ex.q0, ex.d
    e_eval = witness["e_eval"]                  # centred, in [-B_e, B_e]
    y_eval_c = witness["y_eval"]                # centred integers
    digit_polys = witness["digits"]

    B_e = q0 // 2
    if len(e_eval) != d or any(not -B_e <= c <= B_e for c in e_eval):
        errors.append("e_eval outside [-B_e, B_e]^d")

    for j, poly in enumerate(digit_polys):
        if any(a not in (0, 1, 2) for a in poly):
            errors.append(f"digit polynomial {j} is not ternary in {{0,1,2}}")

    for i in range(d):
        if radix_recompose([p[i] for p in digit_polys]) != e_eval[i] + B_e:
            errors.append(f"digit reconstruction fails at coefficient {i}")
            break

    # z_eval = x e_eval + y_eval, as an equality over Z.
    #
    # This must not be checked modulo q~ (or any other protocol modulus).
    # Doing so accepts y_eval + k q~ for any k: the commitment reduces the
    # witness mod q~, so those lifts are indistinguishable to it, and the
    # paper's relation would be satisfied by none of them but the true one.
    # A zero-knowledge backend has to establish the same thing without
    # the integer witness, which is the harder half of the exact layer.
    product = negacyclic_mul_int(statement["x_centered"], e_eval)
    expected = [product[i] + y_eval_c[i] for i in range(d)]
    if statement["z_eval_centered"] != expected:
        errors.append("z_eval != x * e_eval + y_eval over Z")

    return errors


# ---- backend interface ---------------------------------------------------

class ExactBackend:
    """Interface a `Pi_ex` backend must provide.

    A backend never sees the outer secret key: its whole witness is
    `(e_eval, y_eval)` plus the digits derived from `e_eval`.
    """

    name = "abstract"

    def setup(self, par, seed):
        raise NotImplementedError

    def com(self, pp, witness_input, xof):
        raise NotImplementedError

    def prove(self, pp, statement, witness_input, state):
        raise NotImplementedError

    def verify(self, pp, statement, proof):
        raise NotImplementedError


class OpeningBackend(ExactBackend):
    """Honest-opening backend: complete and binding, **not** zero knowledge.

    `Com` is a real BDLOP commitment over `R_q~` with the paper's ranks, and
    `Verify` re-derives the commitment and checks every clause of `R^_ex`.
    What it does not do is hide the witness: `sigma_ex` is the opening.
    """

    name = "opening"

    def setup(self, par, seed):
        ex = ExactParams(par)
        return {"ex": ex, "ck": ExactCommitmentKey(ex, seed), "seed": seed}

    def com(self, pp, witness_input, xof):
        """`(W, st) <- Pi_ex.Com(w_ex)`; the statement is not known yet."""
        ex = pp["ex"]
        e_eval = list(witness_input["e_eval"])
        y_eval_c = list(witness_input["y_eval"])
        # The relation is stated on the canonical range, so undo the
        # centring exactly where the digits are formed.
        digits = decompose_poly(from_centered_error(e_eval, ex.par.B_e))
        message = pack_witness(ex, e_eval, y_eval_c, digits)
        randomness = uniform_beta_vec(xof, 1, ex.d_tilde, ex.kappa, ex.q_tilde)
        W = pp["ck"].commit(message, randomness)
        state = {"e_eval": e_eval, "y_eval": y_eval_c, "digits": digits,
                 "randomness": randomness}
        return W, state

    def prove(self, pp, statement, witness_input, state):
        """Reveal the opening.  A LANES backend proves it in zero knowledge."""
        return {"e_eval": state["e_eval"],
                "y_eval": state["y_eval"],
                "digits": state["digits"],
                "randomness": state["randomness"]}

    def verify(self, pp, statement, proof):
        # `pp["ex"]` is read before the guard on purpose: a `pp` missing
        # it is the caller's error, not a malformed proof, and `RiVeR`'s
        # own boundary is what turns it into a bit.
        ex = pp["ex"]
        try:
            witness = {"e_eval": proof["e_eval"],
                       "y_eval": proof["y_eval"],
                       "digits": proof["digits"]}
            if check_relation(ex, statement, witness):
                return False
            message = pack_witness(ex, proof["e_eval"], proof["y_eval"],
                                   proof["digits"])
            # The padding is part of the committed message, so it is part of
            # what the relation has to pin.  `pack_witness` builds it zero,
            # so this cannot fail here -- it is checked anyway because a
            # LANES backend commits to a message it did not build, and the
            # two backends must agree on which messages are well formed.
            if not padding_is_zero(ex, message):
                return False
            recomputed = pp["ck"].commit(message, proof["randomness"])
            return recomputed == statement["W"]
        except (KeyError, ValueError, IndexError, TypeError, AttributeError,
                OverflowError, ZeroDivisionError):
            return False

    # -- encoding ---------------------------------------------------------

    def __init__(self, par):
        self.par = par
        ex = self.ex = ExactParams(par)
        #: `y_eval` is a coordinate of the OOM mask, so carry the integer,
        #: not a residue, or the integer link equation cannot be stated.
        #:
        #: The bound is **not** the verifier's `6 sigma_m`.  That bounds
        #: `z_eval`, and `y_eval = z_eval - x e_eval`, so an accepted
        #: transcript reaches
        #:
        #:     |y_eval| <= 6 sigma_m + ||x||_1 ||e_eval||_inf
        #:              =  6 sigma_m + w gamma B_e.
        #:
        #: Using `6 sigma_m` alone left the honest prover able to produce a
        #: proof its own serializer refused: `Rice.write` raises past its
        #: bound, so `Eval` would have died with a `ValueError` instead of
        #: restarting.  It needs `|y_eval|` in the last `0.03%` of its range
        #: -- about `6.3e-8` per proof, which is why it was never observed --
        #: but it is reachable, and it is the same shape of defect as an over-tight bound: a
        #: bound derived from the wrong quantity.
        #:
        #: Widening costs nothing on the wire.  A Rice codeword depends only
        #: on `k`, which comes from `sigma`; the bound gates the encoder and
        #: caps the decoder's unary run, so no encodable value moves and only
        #: `proof_bytes`, the worst case, grows -- by 4 bytes at TOY.
        #: `z_eval` is the last coordinate of the `z_m` block, so its width
        #: is `sigma_m` and its verifier bound is `zm_inf_bound`.
        self.bound_y = (floor_sqrt(par.zm_inf_bound_sq)
                        + par.w * par.gamma * par.B_e)

        #: Each field gets the coder its distribution asks for.  `y_eval` is
        #: a Gaussian coordinate of the OOM mask, so Rice; the digits and the
        #: randomness are tiny and uniform, so 2 bits each rather than the
        #: byte a fixed-width field would round up to.
        qt = Uniform(ex.q_tilde)
        self.W_layout = Layout(
            Field("t0", qt, ex.d_tilde, ex.t0_rows),
            Field("t1", qt, ex.d_tilde, ex.N_ex),
        )
        self.proof_layout = Layout(
            Field("t0", qt, ex.d_tilde, ex.t0_rows),
            Field("t1", qt, ex.d_tilde, ex.N_ex),
            Field("e_eval", Signed(par.B_e), ex.d),
            Field("y_eval", Rice(par.sigma_m, self.bound_y), ex.d),
            Field("digits", Uniform(RADIX_DIGITS), ex.d, len(RADIX_WEIGHTS)),
            Field("randomness", Signed(1), ex.d_tilde, ex.kappa, ring=ex.Rt),
        )

    def W_encode(self, W):
        return self.W_layout.encode(W)

    def W_decode(self, data):
        return self.W_layout.decode(data)

    @property
    def W_bytes(self):
        return self.W_layout.max_bytes          # `W` is all uniform: exact

    def proof_encode(self, pi_ex):
        """`pi_ex = (W, sigma_ex)`."""
        flat = dict(pi_ex["sigma"])
        flat["t0"] = pi_ex["W"]["t0"]
        flat["t1"] = pi_ex["W"]["t1"]
        return self.proof_layout.encode(flat)

    def proof_decode(self, data):
        flat = self.proof_layout.decode(data)
        W = {"t0": flat.pop("t0"), "t1": flat.pop("t1")}
        return {"W": W, "sigma": flat}

    @property
    def proof_bytes(self):
        """Worst-case `|pi_ex|`.

        Rice-coding `y_eval` makes the real length sample-dependent, so this
        is an upper bound.  Measure with `len(backend.proof_encode(pi_ex))`.
        """
        return self.proof_layout.max_bytes


BACKENDS = {OpeningBackend.name: OpeningBackend}

#: Optional backends live in the `lanes_*` modules and are imported lazily,
#: both to keep core RiVeR free of any dependency on them and because
#: `lanes_backend` imports from this module.
OPTIONAL_BACKENDS = ("lanes", "lanes-experimental")


#: Kept here rather than in `lanes_backend` so
#: every `lanes_*` module can consult it without importing the backend.
#: Every LANES constant that the gate covers
#: , as `(module attribute, what it is)`.
#:
#: The gate requires a source audit reporting zero constants that have
#: drifted from the paper's closed form.  Listing them here makes that audit
#: executable — `live_lanes_constants()` walks it, and the readiness check
#: refuses to open while any of them carries a value the manifest did not
#: select.
#:
#: The dimensions are deliberately absent.  `d~`, `l`, `q~`, `(n~, l~)` and
#: `D` are read from the current manifest. The production alias is reserved
#: until the artifact's concrete compression/recovery composition has a
#: matching security argument.
#:
#: The list is *closed*: `live_lanes_constants` reports a name it cannot
#: find as an error rather than dropping it, so renaming or deleting one
#: cannot shrink the audit in silence.  It does not, and cannot, detect an
#: equivalent literal moved somewhere else under another name — the
#: durable answer to that is to generate the active constants from the
#: manifest.
#: The wire-visible LANES constants the gate covers, and what each is.
#:
#: Every one is now derived from the paper's closed form.  The audit reads
#: their live values and compares them against `PAPER_LANES_VALUES` below;
#: a mismatch is `constant-changed`, a missing name is `audit-drift`.
GATED_LANES_CONSTANTS = (
    ("SIGMA_R", "commitment randomness width, s_1 rounded to 2^-20"),
    ("SIGMA_Y", "proof mask width, s_2 rounded to 2^-20"),
    ("Z_INF_BOUND", "response infinity bound, derived from those widths"),
    ("Z_NORM2_BOUND", "response Euclidean bound, the paper's (2s)^2 rule"),
    ("RECOVERY_ERROR_BOUND", "hint bound, derived from sigma_r"),
    ("RECOVERY_BUCKETS", "fixed-hint bucket count"),
)

#: What each of those must equal in the paper.
#:
#: Pinned as exact `Fraction`s so a constant that quietly *changed* -- not
#: renamed, not deleted -- is caught too.  These are no longer "the
#: paper's closed form, so the audit requires them to be exactly that.
PAPER_LANES_VALUES = {
    "SIGMA_R": Fraction(2901189, 524288),
    "SIGMA_Y": Fraction(255304631, 1048576),
    "Z_INF_BOUND": Fraction(3448),
    "Z_NORM2_BOUND": Fraction(1578304756),
    "RECOVERY_ERROR_BOUND": Fraction(2886972),
    "RECOVERY_BUCKETS": Fraction(16),
}

#: The inputs a manifest has to account for: the two widths.  The rest
#: are derivations that move with them.
LANES_MANIFEST_INPUTS = ("SIGMA_R", "SIGMA_Y")

#: Every section a frozen LANES manifest must carry, and the keys inside
#: it that must be present and non-empty.
#:
#: The plan's "LANES manifest required to lift the gate" is a list of
#: *data*, not of headings, and this is that list.  An earlier version of
#: this gate checked only that the section names existed, which a table of
#: eight empty dictionaries satisfied — a checklist, not a manifest.
LANES_MANIFEST_SECTIONS = {
    "dimensions": ("d_tilde", "l_split", "q_tilde", "n_tilde", "ell_tilde",
                   "N_ex", "alpha", "D", "w_hat", "n_lwe", "m_lwe",
                   "block_slots", "block_payload"),
    "rank_roles": ("identity_rank", "tail_rank", "kappa", "response_rank"),
    # `K_S1`/`K_S2` are gone: the paper gives the widths in closed form,
    # so what a manifest must now carry is the form's own inputs and
    # outputs -- including which Gaussian convention it read them in, since
    # `s` and `sigma` differ by `sqrt(2 pi)` and the paper prints both.
    "sampler": ("s_0", "s_1", "s_2", "s_response", "convention",
                "epsilon_exponent", "sigma_r", "sigma_y", "sigma_mlwe",
                "tail_cut_r", "tail_cut_y", "prob_bits"),
    "response_bounds": ("inf", "l2", "beta_prime_bdlop", "b_msis",
                        "n_z_paper", "l2_honest_requirement",
                        "shared_by_prover_and_verifier",
                        "comparison", "comparison_note", "population"),
    "recovery": ("d_drop", "rounding", "ties", "omitted_coordinates",
                 "omitted_response_rows", "omitted_response_coefficients",
                 "omitted_t0_low_bits", "recovery_carries",
                 "hint_alphabet", "limit", "failure_rule",
                 "verification_rule", "encoding"),
    # `rounds` and `absorbed_fields` replace the old hand-written `fields`.
    # That list omitted `w_high`, `v` and `v_prime` -- all hashed -- and
    # listed `alpha`, `gamma` and `c`, which are hash outputs.  A port built
    # from it would derive different challenges and never interoperate, so
    # the manifest now reads `lanes_proof.Challenges.ROUNDS` directly and
    # `test_lanes.py` drives a real proof against it.
    "transcript": ("rounds", "absorbed_fields", "derived_not_absorbed",
                   "reconstructed_not_transmitted", "packing",
                   "domain_separators", "order", "hashed_form"),
    "wire": ("fields", "order", "fixed_bits", "kb_convention"),
    # `challenge` is what reproduces the paper's own figures.
    # `5.Parameter.tex`'s footnote gives the LANES challenge-difference
    # noninvertibility probability as 2^-93.5 under the re-optimized
    # parameters (2^-70 under the original) and the outer RVRF figure as
    # 2^-91.5.  Those are the only published quantities separating the
    # LANES parameters from any other set, so a manifest
    # that does not reproduce them has not shown which set it describes.
    # Neither reaches 128 bits.
    "estimator": ("hint_mlwe_inputs", "hint_mlwe_outputs",
                  "msis_inputs", "msis_outputs", "challenge"),
}

#: Labels a manifest constant may carry.
PROVENANCE_LABELS = ("Paper", "Derived", "Repair")

#: The exact-proof entropy estimate the paper reports, in bits, under the KB
#: convention the manifest has to name.  `wire.total_bits` must reproduce
#: it field by field, or the manifest has to say it cannot.
LANES_STATED_KB = Fraction(27, 2)

#: The validated LANES parameter manifest, or `None`.
#:
#: Having one is *not* by itself permission to run the backend — see
#: `LANES_BACKEND_READY`.  Possession of a parameter table is not an
#: implementation.
def _load_lanes_manifest():
    """The frozen `lanes_manifest.json`, or `None` if it is absent.

    Read from disk rather than built here: a manifest regenerated from the
    code on import would agree with the code by construction and certify
    nothing.  `lanes_manifest.py` writes it; `test_lanes_manifest.py`
    fails if the two have drifted.
    """
    try:
        from lanes_manifest import load
    except ImportError:                        # pragma: no cover
        return None
    return load()


LANES_PARAMETER_MANIFEST = _load_lanes_manifest()

#: Diagnostic data associated with the candidate LANES completion.
#:
#: The paper-derived root-Hermite factors reproduce.  The production alias
#: remains disabled because this artifact's concrete compression/recovery
#: completion has no reduction in the artifact; `lanes_security.json`
#: records that scope separately from the parameter manifest.
def _load_lanes_security():
    try:
        from estimate_lanes import load
    except ImportError:                        # pragma: no cover
        return None
    return load()


LANES_SECURITY_EVIDENCE = _load_lanes_security()


class _Unset:
    """Sentinel: "use the module's state", distinct from "absent"."""

    def __repr__(self):                        # pragma: no cover
        return "<unset>"


#: `lanes_unavailable_reason(manifest=None)` has to mean *no manifest*, not
#: *the default one*.  While the default was itself `None` the two readings
#: coincided; now that a manifest ships, conflating them would make it
#: impossible to ask what the gate says without one.
UNSET = _Unset()

#: Whether the LANES *implementation* has passed its interoperability,
#: serialization, negative-test, and vector gates.
#:
#: Deliberately a separate flag from the manifest and from the security
#: evidence.  With one state, obtaining a manifest would have enabled
#: `LanesBackend` immediately -- while the sampler, bounds, hint code,
#: transcript and layout still consumed nothing from it.  A table of the
#: right numbers sitting unused beside the code is not an implementation
#: of them.
#:
#: The implementation gate covers:
#:
#:   * proof and hint rules built *from* the manifest -- `lanes_params`
#:     derives every published figure and `lanes_manifest.json` freezes
#:     what it consumes, in both languages;
#:   * fine-grained KATs for key expansion, commitment, every proof field,
#:     the challenge, hint recovery and the final encoding -- the
#:     `lanes_ring`, `lanes_params` and `lanes_proof` blocks of
#:     `../river-rs/tests/sampler_kat.json`, all three active;
#:   * serializer and verifier green -- `test_lanes.py`, 50 tests;
#:   * negative tests green -- malformed hints in every shape, nonzero
#:     padding, tampered transcript, wrong statement, challenge and
#:     commitment, trailing bytes, edge coefficients;
#:   * the two LANES vector cases restored -- shipped as
#:     `lanes-experimental` and re-derived byte for byte by `river-rs`.
#:
LANES_BACKEND_READY = True


def live_lanes_constants():
    """`({name: (what, value)}, [missing names])` for the gated constants.

    Values are exact `Fraction`s.  A name the module no longer defines is
    *reported*, not skipped: an audit that silently shrinks when a constant
    is renamed is not an audit.
    """
    try:
        import lanes_params as LP
    except Exception as exc:                              # pragma: no cover
        return {}, [f"lanes_params did not import: {exc}"]
    found, missing = {}, []
    for name, what in GATED_LANES_CONSTANTS:
        if hasattr(LP, name):
            found[name] = (what, Fraction(getattr(LP, name)))
        else:
            missing.append(f"{name} not found")
    return found, missing


def lanes_unavailable_reason(manifest=UNSET, backend_ready=UNSET,
                             evidence=UNSET):
    """Why the LANES backend cannot run, or `None` if it can.

    A readiness test with three independent parts:

    1. a frozen parameter manifest exists, carries real data in every
       section, and *selects a value* for every gated constant, matching
       what the code consumes;
    2. the candidate completion's scope is recorded explicitly;
    3. the implementation has passed its own gates and says so.

    Both are required.  A manifest alone would mean a table of the right
    numbers sitting beside the code unused.

    The dimensions and widths do not enter as blockers: `d~`, `l`, `q~`,
    `(n~, l~)`, `D`, `s_1`, `s_2`, `beta'`, `B_MSIS`, `delta_MSIS`, and
    `delta_MLWE` all reproduce the published values.  The remaining gate is
    an artifact-scope decision about the implementation-defined recovery
    composition, not a parameter mismatch.

    """
    if manifest is UNSET:
        manifest = LANES_PARAMETER_MANIFEST
    if backend_ready is UNSET:
        backend_ready = LANES_BACKEND_READY
    if evidence is UNSET:
        evidence = LANES_SECURITY_EVIDENCE
    ready = backend_ready
    live, missing = live_lanes_constants()

    if missing:
        return ("the LANES constant audit has drifted from lanes_params: "
                f"{'; '.join(missing)}. Update "
                "exact.GATED_LANES_CONSTANTS deliberately, or the audit "
                "stops covering what it names")

    # Every gated constant must hold the value the paper's closed form
    # implies.
    wrong = [(n, v) for n, (_, v) in sorted(live.items())
             if v != PAPER_LANES_VALUES[n]]
    if wrong:
        rows = [f"{n} = {v}, expected {PAPER_LANES_VALUES[n]}"
                for n, v in wrong]
        return ("a LANES constant does not match the paper's closed form: "
                + "; ".join(rows)
                + ". The paper derives every one of these from "
                  "s_0 = sqrt(ln(2 d~ (1 + 1/eps)))/pi with no free "
                  "constant, so a mismatch is a port defect or a "
                  "reintroduced selection, not a choice")

    if manifest is None:
        detail = ", ".join(f"{n} ({w})" for n, (w, _) in sorted(live.items()))
        return (
            "no frozen LANES parameter manifest. The paper supplies the "
            "widths and entropy size estimate, while the concrete recovery "
            "and encoding are implementation-level choices, so the wire-visible "
            f"values -- {detail} -- have to be frozen with their provenance "
            "before the production name opens. Supply "
            "exact.LANES_PARAMETER_MANIFEST (see LANES_MANIFEST_SECTIONS), "
            "then set LANES_BACKEND_READY once the implementation gate "
            "passes. "
            "Use exact_backend='opening' or 'lanes-experimental'.")

    bad = validate_lanes_manifest(manifest, live)
    if bad:
        return "the LANES manifest is not usable: " + "; ".join(bad)

    status = manifest.get("status")
    if status != "final":
        return (
            f"the LANES parameter manifest is marked {status!r}, not "
            "'final'. An experimental manifest can never open the "
            "production backend name, whatever else is set. Use "
            "exact_backend='lanes-experimental'")

    if evidence is not None and evidence.get("verdict") != "meets-target":
        blockers = evidence.get("blockers") or []
        return (
            "the production LANES alias is reserved: the parameters "
            "reproduce the paper's printed figures, but the concrete "
            "compression/recovery completion is implementation-defined and "
            "this artifact does not supply a reduction for that exact "
            "composition. "
            + " ".join(f"({i + 1}) {b}." for i, b in enumerate(blockers))
            + " Use exact_backend='lanes-experimental' for the tested "
              "candidate implementation")

    if evidence is None:
        return (
            "the LANES parameter manifest is valid but the artifact has no "
            "scope record for its implementation-defined recovery and "
            "compression completion. Use exact_backend='lanes-experimental'")

    if not ready:
        return (
            "the LANES parameter manifest is present and valid, but the "
            "implementation has not passed its own gate: the sampler, "
            "response bounds, hint rules, transcript and wire layout must "
            "be built *from* it, with fine-grained KATs, serializer, "
            "verifier and negative tests green and both LANES vector cases "
            "shipped. Set exact.LANES_BACKEND_READY only then")
    return None


#: Short, stable tokens for *why* the gate is closed.
#:
#: The prose reason names each language's own API — `exact.LANES_*` here,
#: `exact::LANES_*` there — so the two cannot be compared byte for byte.
#: These can, which is what lets a generated artifact record the cause and
#: a consumer in the other language check it has not drifted.
LANES_GATE_CAUSES = (
    "audit-drift",            # a gated constant was renamed or deleted
    "constant-changed",       # ...or given a different value, unrecorded
    "no-parameter-manifest",  # no frozen table yet
    "manifest-invalid",       # it landed and does not validate
    "manifest-experimental",  # it validates and does not claim to be final
    "no-security-evidence",   # no recorded estimator run yet
    "production-alias-reserved",
    "backend-not-ready",      # everything else; the implementation gate
)


def lanes_gate_cause(manifest=UNSET, backend_ready=UNSET,
                     evidence=UNSET):
    """Which of `LANES_GATE_CAUSES` applies, or `None` if the gate is open.

    The same decision `lanes_unavailable_reason` makes, reported as a
    token rather than as prose.
    """
    if manifest is UNSET:
        manifest = LANES_PARAMETER_MANIFEST
    if backend_ready is UNSET:
        backend_ready = LANES_BACKEND_READY
    if evidence is UNSET:
        evidence = LANES_SECURITY_EVIDENCE
    ready = backend_ready
    live, missing = live_lanes_constants()

    if missing:
        return "audit-drift"
    # The paper's closed form is the baseline, with or without a manifest:
    # a live value that has left it means the code no longer implements the
    # selected parameter set.
    if any(v != PAPER_LANES_VALUES[n] for n, (_, v) in live.items()):
        return "constant-changed"
    if manifest is None:
        return "no-parameter-manifest"
    if validate_lanes_manifest(manifest, live):
        return "manifest-invalid"
    if manifest.get("status") != "final":
        return "manifest-experimental"
    if evidence is None:
        return "no-security-evidence"
    if evidence.get("verdict") != "meets-target":
        return "production-alias-reserved"
    if not ready:
        return "backend-not-ready"
    return None


def manifest_value(entry):
    """The value inside a `{"value": ..., "provenance": ...}` cell.

    Every cell of a LANES manifest carries its own provenance label, so a
    section's keys map to wrappers rather than to bare values.  Anything
    that is not such a wrapper is returned unchanged, which keeps hand-made
    fixtures in tests readable.
    """
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def validate_lanes_manifest(manifest, live=None):
    """Every way a manifest can fail to be one.  `[]` means usable.

    Checks *data*, not headings: each section must carry every key the
    plan names, non-empty, carried in a cell that states its provenance,
    and `constants` must select a value for each gated constant that
    matches what the code consumes.  A manifest that labels a wrong
    width as **Paper** is caught by the value comparison, not by the label.
    """
    if live is None:
        live, _ = live_lanes_constants()

    errors = []
    if not isinstance(manifest, dict):
        return ["not a mapping"]

    for section, keys in sorted(LANES_MANIFEST_SECTIONS.items()):
        body = manifest.get(section)
        if body is None:
            errors.append(f"section {section} is absent")
            continue
        if not isinstance(body, dict) or not body:
            errors.append(f"section {section} is empty")
            continue
        for key in keys:
            if key not in body:
                errors.append(f"{section}.{key} is absent")
                continue
            cell = body[key]
            value = manifest_value(cell)
            if value is None or value in ("", [], {}, ()):
                # an empty container is as empty as an absent key; `0` and
                # `False` are values and are left alone
                errors.append(f"{section}.{key} is empty")
                continue
            # Provenance is *required*, not merely validated when present.
            # This used to read `"provenance" in cell and ... not in
            # LABELS`, so a bare value -- `{"value": 4}`, or just `4` --
            # satisfied it by having no label to check.  The whole point of
            # the table is that every cell says where it came from.
            if not isinstance(cell, dict) or "value" not in cell:
                errors.append(f"{section}.{key} is a bare value with no "
                              "Paper/Derived/Repair provenance")
            elif cell.get("provenance") not in PROVENANCE_LABELS:
                errors.append(f"{section}.{key} carries no "
                              "Paper/Derived/Repair provenance")

    wire = manifest.get("wire")
    if isinstance(wire, dict):
        total = manifest_value(wire.get("total_bits"))
        discrepancy = manifest_value(wire.get("discrepancy"))
        if total is None:
            # A Rice-coded layout has no fixed total, which is a fact about
            # the format and not a gap -- but it has to be *recorded*, or a
            # manifest with no size accounting at all reads the same way.
            if not discrepancy:
                errors.append(
                    "wire.total_bits is absent and wire.discrepancy does "
                    "not say why -- a manifest with no size accounting "
                    "must say so")
        else:
            try:
                total_kb = Fraction(total, 8192)
            except (TypeError, ValueError, ZeroDivisionError):
                errors.append("wire.total_bits is not a bit count")
            else:
                if total_kb != LANES_STATED_KB and not discrepancy:
                    errors.append(
                        f"wire.total_bits is {float(total_kb):.4f} KB "
                        f"against the stated {float(LANES_STATED_KB)} KB, "
                        "and wire.discrepancy does not record it")

    selected = manifest.get("constants")
    if not isinstance(selected, dict):
        errors.append("constants is absent or not a mapping")
        return errors
    for name, (_, value) in sorted(live.items()):
        entry = selected.get(name)
        if entry is None:
            errors.append(f"{name} is still live and the manifest does not "
                          "select it")
            continue
        if not isinstance(entry, dict) or "value" not in entry:
            errors.append(f"{name} carries no value in the manifest")
            continue
        if entry["value"] is None:
            errors.append(f"{name} carries an empty value in the manifest")
            continue
        if entry.get("provenance") not in PROVENANCE_LABELS:
            errors.append(f"{name} carries no Paper/Derived/Repair "
                          "provenance")
        try:
            chosen = Fraction(entry["value"])
        except (TypeError, ValueError):
            errors.append(f"{name}: manifest value is not a number")
            continue
        if chosen != value:
            errors.append(
                f"{name}: the manifest selects {chosen} but the code "
                f"consumes {value}")
    return errors


def note_lanes_gate(module_name):
    """Print why the production `lanes` name is gated, and carry on.

    It used to *skip* the self-check.  That stopped being right at
    The paper: the parameters are the paper's, both implementations run
    the layer end to end and produce byte-identical proofs, and the gate is
    on security *evidence* for the production name.  A self-check that
    skips while the code demonstrably works is one gate away from silence
    -- which is exactly how `lanes_ring`'s twiddle tree stayed wrong for as
    long as it did (`test_lanes_ring.py` documents it).

    So the reason is printed, as context, and the checks run against
    `LanesBackend.experimental`.
    """
    reason = lanes_unavailable_reason()
    if reason:
        print(f"{module_name}: the production `lanes` name is gated "
              f"({lanes_gate_cause()}); running against "
              f"`lanes-experimental`, which is the same code.")


def skip_if_lanes_unavailable(module_name):
    """Retained name; no longer skips.  See :func:`note_lanes_gate`."""
    note_lanes_gate(module_name)


#: Retained under its old name so existing call sites keep working.


def get_backend(name, par):
    if name in BACKENDS:
        return BACKENDS[name](par)
    if name == "lanes":
        from lanes_backend import LanesBackend
        return LanesBackend(par)
    if name == "lanes-experimental":
        # Deliberately a *different* name, not a flag on "lanes".
        #
        # The readiness gate has to go on refusing `"lanes"`, but with no
        # way past it the implementation has no
        # regression coverage at the current dimensions -- which is how an
        # unconstrained message-block padding survived to be found by
        # inspection.  Spelling the name out means every caller, every
        # generated report and every vector case says which one it ran, so
        # an experimental parameter set cannot be mistaken for the paper's.
        from lanes_backend import LanesBackend
        return LanesBackend.experimental(par)
    raise KeyError(f"unknown exact backend {name!r}; available: "
                   f"{sorted(set(BACKENDS) | set(OPTIONAL_BACKENDS))}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    from params import TOY_PARAMS, DEFAULT_PARAMS

    # radix encoding covers [0, 60] and nothing else
    for value in range(61):
        digits = radix_decompose(value)
        assert all(a in (0, 1, 2) for a in digits)
        assert radix_recompose(digits) == value
    for bad in (-1, 61):
        try:
            radix_decompose(bad)
            raise SystemExit(f"accepted out-of-range {bad}")
        except ValueError:
            pass

    reachable = {radix_recompose(d)
                 for d in [(a, b, c, e)
                           for a in range(3) for b in range(3)
                           for c in range(3) for e in range(3)]}
    assert reachable == set(range(61)), "radix set != [0,60]"

    for par in (TOY_PARAMS, DEFAULT_PARAMS):
        ex = ExactParams(par)
        assert not ex.check(), ex.check()

    # commit / prove / verify round trip on a synthetic witness
    import random
    par = TOY_PARAMS
    backend = OpeningBackend(par)
    pp = backend.setup(par, b"\x01" * 32)
    ex = pp["ex"]
    rng = random.Random(3)

    # The witness is the *centred* error `e^c in [-B_e, B_e]`; the relation
    # is stated on `e^c + B_e in [0, q_0-1]`.  This self-check used to draw
    # from the canonical range and hand it in as the centred one, so `com`
    # decomposed values up to 88 -- outside what the radix weights reach --
    # and the check has been failing since the offset was introduced.
    # `ring.from_centered_error` is what now names the range at the boundary.
    e_eval = [rng.randrange(-par.B_e, par.B_e + 1) for _ in range(par.d)]
    y_eval = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(par.d)]
    x_c = [0] * par.d
    for pos in rng.sample(range(par.d), par.w):
        x_c[pos] = rng.choice([-1, 1]) * rng.randint(1, par.gamma)

    from ring import negacyclic_mul_int as _mul
    product = _mul(x_c, e_eval)
    z_eval_c = [product[i] + y_eval[i] for i in range(par.d)]

    w_in = {"e_eval": e_eval, "y_eval": y_eval}
    W, st = backend.com(pp, w_in, XOF(DS_EXACT, b"seed"))
    stmt = {"W": W, "z_eval_centered": z_eval_c, "x_centered": x_c}
    sigma = backend.prove(pp, stmt, w_in, st)
    assert backend.verify(pp, stmt, sigma), "honest proof rejected"

    # tamper detection
    bad = dict(sigma)
    bad["e_eval"] = list(sigma["e_eval"])
    bad["e_eval"][0] = bad["e_eval"][0] + 1 if bad["e_eval"][0] < par.B_e \
        else bad["e_eval"][0] - 1
    assert not backend.verify(pp, stmt, bad), "tampered opening accepted"

    stmt_bad = dict(stmt)
    stmt_bad["z_eval_centered"] = list(z_eval_c)
    stmt_bad["z_eval_centered"][0] += 1
    assert not backend.verify(pp, stmt_bad, sigma), "wrong statement accepted"

    # encoding round trip
    blob = backend.proof_encode({"W": W, "sigma": sigma})
    assert len(blob) <= backend.proof_bytes    # Rice: variable
    again = backend.proof_decode(blob)
    assert again["W"] == W
    assert again["sigma"]["e_eval"] == sigma["e_eval"]
    assert again["sigma"]["digits"] == sigma["digits"]
    assert backend.verify(pp, {"W": again["W"],
                               "z_eval_centered": z_eval_c,
                               "x_centered": x_c}, again["sigma"])

    print(f"exact.py: all self-tests passed "
          f"(|pi_ex| = {len(blob) / 1024:.3f} KB, bound {backend.proof_bytes / 1024:.3f} KB)")
