"""
river.py -- RiVeR: Setup, KeyGen, Eval, Verify (Figures 3, 4 and 5).

A ring VRF: member `j*` of a public-key ring publishes a pseudorandom value
`v` together with a proof that *some* ring member computed it correctly.

The proof has two linked components:

  * the **OOM layer** (`oom.py`) hides the index and proves knowledge of a
    short opening of one derived vector `c_i`;
  * the **exact layer** (`exact.py`) certifies that the centred evaluation
    error becomes canonical after adding the public offset `B_e`, plus the
    equation tying it to the OOM response.

They are bound together by ordering: `Pi_ex.Com` runs first, its output `W`
goes into `rho' = (R~, v, W)`, and `rho'` is an input to the OOM Fiat-Shamir
hash.  So `W` is fixed before the challenge `x`, and the exact proof is then
completed relative to the response `z_eval` that challenge produced.

Implementation choices this module makes that the paper leaves open are
marked `CHOICE:`.
"""

import os

from ring import (Ring, round_p, round_p_vec, rounding_error,
                  to_centered_error)
from sample import (XOF, DS_COMMIT, DS_EXACT, DS_G, DS_KEYGEN,
                    hash_bytes, sam_mat, uniform_beta_vec, uniform_poly)
from codec import RiVeRCodec, ring_digest, statement_digest
from oom import OOM, OOMStatement
from exact import get_backend


#: Every way a malformed public input can surface before a protocol check
#: reaches it.  `Verify` returns a bit, so all of them are a `False`.
#: `OverflowError` is the one that is easy to miss: it is what
#: `int(float("inf"))` raises, and a coefficient arriving as a float is
#: exactly the shape a network peer can send.  `ZeroDivisionError` is its
#: counterpart one layer down: a modulus that arrives as `0` reaches a
#: `%` before any range check does.
MALFORMED = (KeyError, TypeError, IndexError, ValueError, AttributeError,
             OverflowError, ZeroDivisionError)


def _is_coefficient(c, modulus):
    """`c` is an integer in `[0, modulus)`.

    Strict on the type rather than coercing with `int(c)`: a float
    coefficient is malformed input, not input to be rounded, and `int()`
    raises on `inf` / `nan` instead of rejecting them.  `bool` is excluded
    because it is an `int` subclass and `True` is not a coefficient.
    """
    return (isinstance(c, int) and not isinstance(c, bool)
            and 0 <= c < modulus)


class RiVeR:
    """The scheme, bound to one parameter profile and one exact backend."""

    def __init__(self, par, exact_backend="opening"):
        self.par = par
        self.Rq = Ring(par.q, par.d)
        self.Rp = Ring(par.p, par.d)
        self.codec = RiVeRCodec(par)
        self.exact = get_backend(exact_backend, par)

    # ---- Setup (Figure 3) ------------------------------------------------

    def setup(self, seed):
        """`RiVeR.Setup(1^lambda)`, made deterministic in `seed`.

        The paper samples `rho <- {0,1}^256` inside `OM.Setup`; we derive it
        from the caller's seed so a whole run is reproducible.

        `Setup` invokes `BoundGen`, and `BoundGen` aborts on a profile whose
        compression margin is too thin or that violates a modulus
        condition.  `check()` is that abort; raising here is what makes it one
        rather than a diagnostic nobody calls.
        """
        bad = self.par.check()
        if bad:
            raise ValueError(
                f"BoundGen: profile {self.par.name} is not supported: "
                + "; ".join(bad))
        rho = hash_bytes(32, DS_KEYGEN + b".rho", seed)
        pp = {
            "seed": seed,
            "rho": rho,
            "A": sam_mat(rho, self.par.q, self.par.n, self.par.ell,
                         self.par.d, "RiVeR.A"),
            "oom": OOM(self.par, rho),
            "ex": self.exact.setup(self.par,
                                   hash_bytes(32, DS_EXACT + b".ck", rho)),
        }
        return pp

    # ---- KeyGen (Figure 3) -----------------------------------------------

    def keygen(self, pp, seed):
        """`s <- S_beta^ell`, `t <- floor(A s)_p`.  Returns `(sk, pk)`."""
        par = self.par
        xof = XOF(DS_KEYGEN, seed)
        s = uniform_beta_vec(xof, par.beta, par.d, par.ell, par.q)
        As = self.Rq.mat_vec(pp["A"], s)
        t = [round_p(self.Rq, row, par.q0) for row in As]
        return s, t

    # ---- G : {0,1}* -> R_q^ell -------------------------------------------

    def hash_message(self, m):
        """`h_m <- G(m)`."""
        par = self.par
        xof = XOF(DS_G, m)
        return [uniform_poly(xof, par.q, par.d) for _ in range(par.ell)]

    # ---- ring admissibility ----------------------------------------------

    def validate_ring(self, ring_pks):
        """Check the ordered ring `R` and return it unchanged.

        Duplicates are admissible.  If the evaluator's key occurs more than
        once, `ring_index` implements the specified first-occurrence rule
        `j* = min{j in [N] : t_j = pk}`.  Repeated entries do not create
        additional distinct identities.

        The prover and verifier enforce the same admissible domain:

          * exactly `N` keys -- not "at most", and no padding;
          * in caller-supplied order -- the order is part of the statement,
            and two rings with the same members in a different order are
            different statements;
          * every key structurally valid and canonical.

        `Eval` additionally requires the evaluator's key to occur, which
        `ring_index` establishes.  `Verify` does not need to locate an
        evaluator, but applies everything above.

        """
        par = self.par
        if not isinstance(ring_pks, (list, tuple)):
            raise ValueError("ring must be a sequence of public keys")
        if len(ring_pks) != par.N:
            raise ValueError(
                f"ring has {len(ring_pks)} keys, expected exactly {par.N}")
        for pk in ring_pks:
            self._validate_pk(pk)
        return list(ring_pks)

    def _validate_pk(self, pk):
        """Reject a malformed or non-canonical public key.

        "Admissible: non-dummy public keys valid" is an input type in Figure 5,
        not a check, so it has to happen somewhere.  A wrong shape used to
        surface as an `IndexError` out of the middle of `Verify`; raising
        `ValueError` here means `Verify` reports 0 for malformed input rather
        than propagating an exception to the caller.
        """
        par = self.par
        if not isinstance(pk, (list, tuple)) or len(pk) != par.n:
            raise ValueError(f"public key must have {par.n} ring elements")
        for poly in pk:
            if not isinstance(poly, (list, tuple)) or len(poly) != par.d:
                raise ValueError(f"ring element must have {par.d} coefficients")
            if not all(_is_coefficient(c, par.p) for c in poly):
                raise ValueError("public-key coefficient outside [0, p)")

    def _validate_sk(self, sk):
        """Reject a malformed, non-canonical or out-of-range secret key.

        `sk <- U_beta^ell`, so every coefficient is short.  It is *stored*
        as a canonical residue mod `q` -- `keygen` returns what
        `uniform_beta_vec` produces -- so `-1` is `q-1` here, and the range
        check is on the centred representative.

        `Eval` used to take all of this on trust and reach
        `R.inner(h_m, sk)` first, where a wrong shape surfaced as an
        `IndexError` from inside the arithmetic.
        """
        par = self.par
        if not isinstance(sk, (list, tuple)) or len(sk) != par.ell:
            raise ValueError(f"secret key must have {par.ell} ring elements")
        for poly in sk:
            if not isinstance(poly, (list, tuple)) or len(poly) != par.d:
                raise ValueError(f"ring element must have {par.d} coefficients")
            if not all(_is_coefficient(c, par.q) for c in poly):
                raise ValueError("secret-key coefficient outside [0, q)")
            for c in self.Rq.centered(list(poly)):
                if not -par.beta <= c <= par.beta:
                    raise ValueError(
                        f"secret-key coefficient {c} outside "
                        f"[-{par.beta}, {par.beta}] (not in S_beta)")

    def _check_keypair(self, pp, pk, sk):
        """`pk == floor(A s)_p`, i.e. the two halves belong together.

        Without this, a mismatched pair produced a rounding error outside
        `[0, q_0-1]`, and the first thing to notice was either
        `to_centered_error` or the opening invariant -- both of which report
        a broken scheme when the real cause is a caller passing the wrong
        key.  Checking it up front makes that a `ValueError` with a name.
        """
        par = self.par
        R = self.Rq
        self._validate_pk(pk)
        derived = round_p_vec(R, R.mat_vec(pp["A"], sk), par.q0)
        if derived != [list(poly) for poly in pk]:
            raise ValueError("public key is not floor(A s)_p for this "
                             "secret key")

    def ring_index(self, ring, pk):
        """`j* = min{j in [N] : t_j = pk}`, the paper's hidden index.

        Duplicate ring entries are admissible, and the tie-break is the
        first occurrence.

        Comparison is on the canonical encoding, not on the Python objects,
        so two structurally equal keys built by different paths tie-break
        the same way in both implementations.
        """
        target = self.codec.pk_encode(pk)
        for i, t in enumerate(ring):
            if self.codec.pk_encode(t) == target:
                return i
        raise ValueError("public key is not a member of the ring")

    # ---- Eval (Figure 4) -------------------------------------------------

    def eval(self, pp, pk, sk, ring_pks, m, seed=None, collect_stats=False):
        """`(v, pi) <- RiVeR.Eval(pp, pk, sk, R, m)`.

        `seed` is **auxiliary** randomness, and defaults to fresh
        `os.urandom(32)`.  Pass it only to reproduce an execution -- and
        prefer `eval_deterministic`, which says so at the call site.

        The nonce actually used is `H(seed || sk || ring || v || m)`, so a
        caller who reuses a seed across messages, rings or keys still gets
        independent masks.  That derivation is load-bearing rather than
        defensive -- it blocks a nonce-reuse recovery attack; see
        `eval_deterministic` for why the API defaults to fresh coins anyway.

        With `collect_stats` the returned tuple also carries the attempt count
        and the reason each failed attempt aborted, which is what
        `test_e2e.py` uses to compare the measured restart rate against
        `mu-tilde_RiVeR`.
        """
        if seed is None:
            seed = os.urandom(32)
        par = self.par
        R = self.Rq

        ring = self.validate_ring(ring_pks)
        j_star = self.ring_index(ring, pk)
        # Validate the *keypair* before any arithmetic touches it.  `Eval`
        # used to compute `A s` and the rounding errors first and discover a
        # malformed or mismatched key as an `IndexError` or a failed
        # assertion from inside the attempt loop.  A caller that hands in the
        # wrong `sk` for this `pk` is making an input error, so it is a
        # `ValueError` raised up front, from one place, with a reason.
        self._validate_sk(sk)
        self._check_keypair(pp, pk, sk)
        h_m = self.hash_message(m)

        # Rounding errors are canonical, in [0, q_0-1]; the OOM witness is
        # the centred `e^c = e - B_e`, and every public target carries the
        # matching `+B_e` (see `ring.to_centered_error`, and
        # `OOMStatement.c_i`).  Both sides use the one conversion.
        inner = R.inner(h_m, sk)
        v = round_p(R, inner, par.q0)
        epsilon_eval = rounding_error(R, inner, v, par.q0)
        e_eval = to_centered_error(epsilon_eval, par.B_e)
        e_eval_res = R.from_centered(e_eval)

        # Key-side errors use the same centring.
        As = R.mat_vec(pp["A"], sk)
        epsilon_key = [rounding_error(R, As[i], ring[j_star][i], par.q0)
                       for i in range(par.n)]
        e_key = [R.from_centered(to_centered_error(poly, par.B_e))
                 for poly in epsilon_key]

        r = list(sk) + e_key + [e_eval_res]
        if len(r) != par.r_dim:
            raise ValueError(f"opening has {len(r)} rows, expected {par.r_dim}")

        statement = OOMStatement(par, R, pp["A"], h_m, ring, v)
        ck_digest = statement_digest(self.codec, pp["rho"], h_m)

        # The honest opening really does open c_{j*}.  This is the invariant
        # the whole OOM layer rests on, so it is checked once per evaluation
        # -- and with a raise, not an `assert`, because `python -O` strips
        # asserts and this is the one equation that makes the proof mean
        # anything.  `_check_keypair` above catches the reachable cause; if
        # this still fails, the parameters or the statement construction are
        # wrong, not the caller's input.
        if statement.apply_ck(r) != statement.c_i(j_star):
            raise RuntimeError("c_{j*} != Com(0; r): the honest opening does "
                               "not open the claimed ring position")

        # Derandomised nonce, as in Dilithium's `rho' = H(K || mu)`.
        #
        # The masks must NOT come from `seed` alone.  Two evaluations that
        # share a mask `y` but get different challenges publish
        # `z_1 = y + x_1 r` and `z_2 = y + x_2 r`, so
        # `z_1 - z_2 = (x_1 - x_2) r` and the whole witness -- `s`, `e_key`,
        # `e_eval` -- falls out by one linear solve.  Binding the message and
        # the key in here means a caller who reuses `seed` across messages
        # gets independent masks anyway, which is the only form of the
        # guarantee an API can actually enforce.
        #
        # The abort loop makes this sharper than it first looks.  With the
        # mask sequence pinned, the accept/reject decision is very nearly a
        # function of the mask rather than the message, so the index of the
        # first accepting attempt is nearly determined by the seed: measured
        # over 24 messages at one seed, only 3-5 distinct indices occur and a
        # third of message pairs share one.
        nonce = hash_bytes(32, DS_COMMIT + b".nonce",
                           seed,
                           self.codec.sk_encode(sk),
                           ring_digest(self.codec, ring, v),
                           m)

        oom = pp["oom"]
        stats = {"attempts": 0, "aborts": []}
        for attempt in range(par.max_attempts):
            stats["attempts"] += 1
            com_xof = XOF(DS_COMMIT, nonce, attempt.to_bytes(4, "little"))
            ex_xof = XOF(DS_EXACT, nonce, attempt.to_bytes(4, "little"))

            t_oom, st_oom = oom.com(statement, j_star, r, com_xof)

            # w_ex = (e_eval, y_eval), committed *before* the challenge
            y_eval = R.centered(st_oom["y_om"][par.ell + par.n])
            witness_in = {"e_eval": e_eval, "y_eval": y_eval}
            W, st_ex = self.exact.com(pp["ex"], witness_in, ex_xof)

            rho_digest = self._rho_digest(ring, v, W)
            pi_oom = oom.prove(statement, j_star, r, t_oom, st_oom,
                               ck_digest, rho_digest, com_xof)
            if pi_oom is None:
                # The OOM layer aborted.  Nothing from this attempt survives:
                # both XOFs are rebuilt from `(nonce, attempt)` at the top of
                # the next pass, so the masks, the selector state and the
                # exact commitment randomness are all discarded together.
                stats["aborts"].append("oom")
                continue

            # Only now is `z_eval` defined.  The bottom test above happens
            # first, unconditionally, matching the `Eval` figure.
            z_eval = R.centered(pi_oom["z"][par.ell + par.n])
            x_c = pi_oom["x"]

            # Correctness invariant: the response equation is exact over Z,
            # not merely modulo q.  If this ever fails, q is too small for
            # sigma_m -- a parameter defect, not a restartable event, so it
            # raises rather than aborting the attempt.
            x_e = R.centered(R.mul([c % par.q for c in x_c], e_eval_res))
            if z_eval != [x_e[k] + y_eval[k] for k in range(par.d)]:
                raise RuntimeError(
                    "z_eval != x e_eval + y_eval over Z: q is too small "
                    "for sigma_m")

            ex_statement = {"W": W, "z_eval_centered": z_eval,
                            "x_centered": x_c}
            sigma_ex = self.exact.prove(pp["ex"], ex_statement, witness_in,
                                        st_ex)
            if sigma_ex is None:
                # The exact layer aborted.  Its own test, separate from the
                # OOM one.  `W` is
                # already bound into the challenge, so the OOM proof cannot be
                # reused with a fresh exact commitment -- the whole attempt is
                # discarded, which
                # is what `mu_RiVeR = mu_OOM mu_ex` accounts for.  The
                # shipped `opening` backend never aborts, so this path is
                # exercised by a backend that does; see
                # `test_river.py::test_an_exact_abort_restarts_the_attempt`.
                stats["aborts"].append("exact")
                continue

            pi = {"oom": pi_oom, "ex": {"W": W, "sigma": sigma_ex}}
            if collect_stats:
                return v, pi, stats
            return v, pi

        raise RuntimeError(
            f"no accepting attempt in {par.max_attempts} tries")

    def _rho_digest(self, ring, v, W):
        """Digest of `rho' = (R~, v, W)`."""
        base = ring_digest(self.codec, ring, v)
        return hash_bytes(32, DS_COMMIT + b".rho'", base,
                          self.exact.W_encode(W))

    # ---- Verify (Figure 5) -----------------------------------------------

    def eval_deterministic(self, pp, pk, sk, ring_pks, m, seed,
                           collect_stats=False):
        """`eval` with the auxiliary randomness pinned, for test vectors.

        Identical to `eval(..., seed=seed)`.  It exists as a separate name so
        that reproducibility is something a caller asks for explicitly, rather
        than the accident of having passed a positional argument -- the
        distinction the review asked for, and the one that keeps a
        deterministic path available without making it the default.

        Safe against seed reuse for the reason `eval` documents, but there is
        no reason for production code to call it.
        """
        return self.eval(pp, pk, sk, ring_pks, m, seed, collect_stats)

    def verify(self, pp, ring_pks, m, v, pi):
        """`RiVeR.Verify(pp, R, m, v, pi)` in {0, 1}.

        Total on `R`, `m`, `v` and `pi` for *any* value at all.  Those
        four arrive from a peer, so a malformed one has to be a `0` and
        not an exception -- including the shapes Python notices before any
        protocol check does: `None` where a list belongs, a string where a
        coefficient belongs, a `float("inf")` that `int()` refuses to
        convert.  Every stage is guarded by `MALFORMED`, and no unguarded
        work happens between them.

        `pp` is not in that class.  It is the CRS: `Setup`'s output for
        this profile, which every party is assumed to hold the same copy
        of, and which the scheme's security assumes was honestly
        generated.  Validating an adversarially *chosen* `pp` would be
        theatre -- one that passed every structural check would still
        break soundness, for reasons that have nothing to do with
        exceptions.  What the code does guarantee is weaker and worth
        having: a `pp` that is merely stale, from another profile, or
        internally inconsistent is a `0` rather than a traceback.  That is
        the last `except` below, which is deliberately outermost, so the
        per-stage `MALFORMED` guards stay the precise mechanism and this
        one only catches what they were never scoped for -- a modulus
        edited to `0` inside a backend, say.  A bug on the honest path
        cannot hide behind it: `test_e2e.py` and `vectors.json` both
        assert `verify` is `True` on honest proofs, so an exception there
        is a failing test, not a silent `False`.
        """
        try:
            return self._verify(pp, ring_pks, m, v, pi)
        except Exception:
            # The outermost boundary described above.  Broad on purpose:
            # its job is that no input reaches a caller as a traceback,
            # and enumerating exception types is what let a `q_tilde` of
            # `0` through as a `ZeroDivisionError` in the first place.
            return False

    def _verify(self, pp, ring_pks, m, v, pi):
        """`verify` without the outer boundary; see its docstring."""
        par = self.par
        R = self.Rq

        # A `pp` from another profile, or one edited after `Setup`, is a `0`
        # rather than a traceback -- and it is rejected *before* any
        # arithmetic, rather than by whichever derived quantity happens to
        # notice.  Comparing the whole fingerprint is what makes that
        # uniform: a per-attribute liveness rule is only ever as good as the
        # last derived value someone remembered to make live.
        try:
            if pp["ex"]["ex"].fingerprint != self.exact.ex.fingerprint:
                return False
            if pp["ex"]["ex"].check():
                return False
        except MALFORMED:
            return False

        try:
            ring = self.validate_ring(ring_pks)
        except MALFORMED:
            return False                      # inadmissible or malformed ring

        # CHOICE.  Figure 5 still never checks that v is canonically
        # reduced; a non-canonical v changes q_0 v mod q and hence the whole
        # statement, so we check it here.  The ring check above has moved the
        # other way: the paper dropped the admissible-ring set
        # `R_pp` from every experiment, so no admissibility condition is
        # stated anywhere now.  We keep ours.
        try:
            if len(v) != par.d or not all(_is_coefficient(c, par.p) for c in v):
                return False
        except MALFORMED:
            return False

        try:
            h_m = self.hash_message(m)
            statement = OOMStatement(par, R, pp["A"], h_m, ring, v)
            ck_digest = statement_digest(self.codec, pp["rho"], h_m)
        except MALFORMED:
            return False                      # malformed message or parameters

        # `pi` reaches here either from `proof_decode`, which has already
        # validated every field, or straight from a caller's dictionary, which
        # has not.  Re-encoding is the cheapest way to apply exactly the same
        # checks to both: it rejects a wrong shape, a coefficient outside its
        # declared range, and a non-canonical residue -- the review's point
        # that `t_g + q~` or a shifted `z` coefficient must not verify just
        # because it arrived as a dict rather than as bytes.
        #
        # Any failure is `False`, never an exception: `Verify` returns a bit.
        try:
            pi_oom = pi["oom"]
            W = pi["ex"]["W"]
            sigma_ex = pi["ex"]["sigma"]
            self.codec.oom_encode(pi_oom)
            self.exact.proof_encode(pi["ex"])
        except MALFORMED:
            return False

        try:
            rho_digest = self._rho_digest(ring, v, W)
            if not pp["oom"].verify(statement, pi_oom, ck_digest, rho_digest):
                return False

            z_eval = R.centered(pi_oom["z"][par.ell + par.n])
            ex_statement = {"W": W, "z_eval_centered": z_eval,
                            "x_centered": pi_oom["x"]}
            return bool(self.exact.verify(pp["ex"], ex_statement, sigma_ex))
        except MALFORMED:
            return False

    # ---- serialization ---------------------------------------------------

    def proof_encode(self, pi):
        return self.codec.proof_encode(pi, self.exact)

    def proof_decode(self, data):
        return self.codec.proof_decode(data, self.exact)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from params import TOY_PARAMS

    par = TOY_PARAMS
    scheme = RiVeR(par)
    pp = scheme.setup(b"\x00" * 32)

    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(par.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]

    message = b"hello"
    start = time.time()
    v, pi, stats = scheme.eval(pp, pk, sk, ring, message, b"\xAA" * 32,
                               collect_stats=True)
    elapsed = time.time() - start
    assert scheme.verify(pp, ring, message, v, pi), "verify failed"

    blob = scheme.proof_encode(pi)
    assert scheme.verify(pp, ring, message, v, scheme.proof_decode(blob))

    # a different message must not verify against this proof
    assert not scheme.verify(pp, ring, b"other", v, pi)

    print(f"river.py: eval+verify OK  "
          f"({stats['attempts']} attempts, {elapsed:.1f}s, "
          f"proof {len(blob)} bytes)")
