"""
lanes_proof.py -- The LANES exact proof: `Gen`, `Prove`, `Ver`.

Optional component.  Implements Figures 3 and 4 of [ENS20] with three
departures, all of which RiVeR needs:

  * `k = 1`, i.e. no automorphism (`X -> X^sigma`) stage;
  * the Hint-MLWE treatment of [KLSS23], which removes the internal
    rejection sampling and so contributes no repetition multiplier;
  * support for proving only *part* of the message ternary, as the hybrid
    exact/relaxed framework of [ESLR23] requires -- here the radix digits
    are ternary but `e_eval` and `y_eval` are not.

References are listed in `README.md`.

Relation proved
---------------
For a committed message of `N_ex` ring elements carrying `l = 64` scalar
slots each:

  1. every slot of elements `[alpha_lo, alpha_hi)` is ternary (`lanes_mp.py`);
  2. the full slot vector `m in Z_q~^{N_ex l}` satisfies a public linear
     system `A m = u` over `Z_q~`.

Both are checked against the *same* commitment, which is what lets the range
argument and the linking equation refer to one witness.

How the linear part works
-------------------------
The verifier's challenge `gamma` compresses `A m = u` into the single scalar
`gamma^T (A m - u) = 0`.  That scalar is extracted as the constant
coefficient of an NTT-domain element: for slot values `v`,

    constant_coefficient(slots_to_ntt(v)) = (sum_j v_j) / l   (mod q~)

so `phi` carries a compensating factor `l` to cancel the `1/l`.  The
masking element `g` is sampled with constant coefficient zero so it does not
disturb the test, and is committed in `t_{N+1}` so it cannot be chosen later.

What is transmitted, and what is not
-----------------------------------
Three of the prover's messages -- `w`, `v` and `v'` -- are **check targets**:
each appears in exactly one verification equation, on its own, so that
equation determines it from everything else in the proof.  Transmitting them
is transmitting what the verifier can compute.

So `Ver` transmits the challenge `c` instead and *recovers* all three, in the
order the transcript needs them:

    w  := B_0 z - c t_0
    v  := <b_{N+1}, z> - c t_{N+2} + c f_{N+3} + sum_e alpha_e f(f+c)(f-c)
    v' := <b_G, z> + sum_e phi_e . <b_e, z> - c (tau + t_g - h)

then re-derives `alpha`, `gamma` and `c'` over a transcript containing the
recovered values, and accepts iff `c' == c`.  Each equality that used to be
tested directly is now folded into that one comparison, which is the standard
Fiat-Shamir commitment-recovery trade: an adversary who moves any of the three
moves the transcript, hence `c'`.

`c` costs `d~` ternary coefficients, 256 bits, against the 33,408 the three
elements cost. The two further bandwidth optimisations in the paper's
size model are also applied:

  * `t_0` is sent as its coefficient-domain high part after dropping
    `D` low bits (`D = 17`, 13 before it);
  * the mask and response cover only the `kappa - l~` non-identity
    columns -- 13 at these ranks.

Those two omissions make exact recovery of `w` depend on both omitted
quantities.  The verifier first computes

    B_0' z - c (2^D t_0,high) = w + c(t_0,low - r_identity)

and then applies one fixed ternary bucket carry per coefficient.  The carry
is 2,048 bits (`n~ = 4` rows of `d~ = 256` at 2 bits); its derivation and
exact perturbation bound are in `lanes_params.py`.

At the paper's dimensions and widths, a measured proof is about 13.9 KB
beside its 13.5 KB entropy estimate. `lanes_backend.model_bits` evaluates a
field-level entropy model at about 13.4 KB, and the concrete encoding overhead
has three named parts, each pinned by
`test_lanes.py::test_measured_proof_size_is_reported_field_by_field`: the
closed form omits the recovery hint (2,048 bits) and the challenge (512),
codes `z` at its entropy where the serializer uses Rice, and charges
`log2 q~ - D = 9` bits for a `t0` high part that `power2round` leaves in
`[0, 513)` and the serializer therefore writes in 10 (1,024 bits).

Although the displayed perturbation contains `r_identity`, substituting
`t_0 = r_identity + B_0' r_tail` cancels it:

    t_0,low - r_identity = B_0' r_tail - 2^D t_0,high.

The hint therefore does not create a special channel for the omitted identity
block.  It does reveal a deterministic carry depending on the tail opening and
the public compressed commitment; accounting for that leakage is precisely
the fixed-hint composition obligation left open below.

This carry format is an implementation-derived completion of the black-box
exact layer. [ENS20] gives response compression with a rejection condition
and defers commitment-compression hints to Dilithium. This artifact combines
the compression model with rejection-free [KLSS23] masking in one concrete
wire format. Byte interoperability and algebraic correctness are tested here;
the artifact does not supply a security reduction for this exact fixed-hint
composition.
No arbitrary hint-weight cap is imposed: unlike Dilithium's sparse hint
format, this wire format is dense, LANES has no retry at this point, and the
paper supplies neither a cap nor a completeness/security argument for one.

Status
------
Validated behaviourally: honest proofs verify, and every tampering the tests
apply is rejected.  That is **not** a soundness proof.  The RiVeR paper fixes
`Pi_ex`'s parameters but does not restate the protocol, so the construction
here follows [ENS20] directly.
"""

import lanes_mp
import lanes_ring as R
from lanes_commit import (LanesCommitmentKey, commit, expand_t0,
                          B_G, B_MP1, B_MP2)
from lanes_params import (KAPPA, RESPONSE_RANK, IDENTITY_RANK, N_EX, SIGMA_Y,
                          RECOVERY_BUCKETS, Z_NORM2_BOUND, Z_INF_BOUND,
                          sample_uniform_poly, sample_gaussian_vec,
                          sample_challenge)
from sample import uniform_int, XOF, hash_bytes, DS_EXACT

AN = N_EX * R.LSPLIT               # 192 scalar inputs

DS_LANES = DS_EXACT + b".lanes.fs"


class Challenges:
    """Where `alpha`, `gamma` and `c` come from.

    The protocol is three-round: the verifier speaks after the first message
    (`alpha`), after the product commitments (`gamma`), and after the linear
    messages (`c`).  RiVeR needs `Pi_ex` non-interactive, so the default
    binds each challenge to the transcript so far by hashing it.

    The transcript is bound in protocol order, so a challenge can never
    depend on a message that follows it.  Deriving all three from a single
    fixed stream instead -- as a benchmark harness may do, since it only needs
    prover and verifier to agree -- would let the prover choose later messages
    with the challenges already in hand, and is not a proof.
    """

    #: The Fiat-Shamir transcript, as data.  Each entry is
    #: `(challenge, [absorbed field names])`: the fields hashed in before
    #: that challenge is drawn, in order.  `prove` and `verify` name every
    #: `absorb` call, and `record_transcript()` replays a real proof and
    #: compares what was absorbed against this -- so a manifest or a port
    #: reading this cannot drift from what the code does.
    #:
    #: Note which names are *absorbed* and which are *derived*: `alpha`,
    #: `gamma` and `c` are outputs of the hash, never inputs to it, and
    #: `w_high`, `v` and `v_prime` are check targets the verifier
    #: reconstructs rather than reads off the wire -- but they are hashed,
    #: so a port that omits them produces different challenges.
    ROUNDS = (
        ("alpha", ("statement", "t0", "t", "w_high", "t_g")),
        ("gamma", ("t_mp1", "t_mp2", "v")),
        ("c", ("h", "v_prime")),
    )

    def __init__(self, statement_bytes):
        self.statement = statement_bytes
        self.transcript = [statement_bytes]
        #: Names absorbed so far, in order, starting with the statement.
        self.absorbed = ["statement"]

    def absorb(self, *parts, names=()):
        if names and len(names) != len(parts):
            raise ValueError("absorb: one name per part, or none")
        for part in parts:
            self.transcript.append(part)
        self.absorbed.extend(names)

    def _xof(self, label):
        return XOF(DS_LANES + label, *self.transcript)

    def alpha(self, count):
        x = self._xof(b".alpha")
        return [R.ntt(sample_uniform_poly(x)) for _ in range(count)]

    def gamma(self, count):
        x = self._xof(b".gamma")
        return [uniform_int(x, R.QTILDE) for _ in range(count)]

    def challenge(self):
        return sample_challenge(self._xof(b".c"))


def _pack(*elements):
    """Byte image of NTT-domain elements, for transcript hashing."""
    out = bytearray()
    for el in elements:
        for coeff in el:
            out += int(coeff % R.QTILDE).to_bytes(4, "little")
    return bytes(out)


def _add(a, b):
    q = R.QTILDE
    return [(a[i] + b[i]) % q for i in range(R.DTILDE)]


def _sub(a, b):
    q = R.QTILDE
    return [(a[i] - b[i]) % q for i in range(R.DTILDE)]


def recovery_high(poly):
    """Equal-interval torus quotient used in the recoverable transcript."""
    if len(poly) != R.DTILDE:
        raise ValueError("wrong recovery polynomial length")
    if any(not isinstance(v, int) or isinstance(v, bool)
           or not 0 <= v < R.QTILDE for v in poly):
        raise ValueError("non-canonical recovery coefficient")
    return [(v * RECOVERY_BUCKETS) // R.QTILDE for v in poly]


def make_recovery_hint(target, base):
    """Cyclic bucket carry from `base` to `target`, necessarily in {-1,0,1}."""
    if len(target) != len(base):
        raise ValueError("recovery row mismatch")
    out = []
    for want, have in zip(target, base):
        if len(want) != R.DTILDE or len(have) != R.DTILDE:
            raise ValueError("recovery polynomial mismatch")
        row = []
        for a, b in zip(want, have):
            delta = (a - b) % RECOVERY_BUCKETS
            if delta == RECOVERY_BUCKETS - 1:
                delta = -1
            if delta not in (-1, 0, 1):
                raise ValueError("recovery perturbation crossed multiple buckets")
            row.append(delta)
        out.append(row)
    return out


def use_recovery_hint(base, hint):
    """Apply a peer-supplied {-1,0,1} cyclic carry to bucket rows."""
    if len(base) != IDENTITY_RANK or len(hint) != IDENTITY_RANK:
        raise ValueError("wrong recovery hint row count")
    out = []
    for have, carries in zip(base, hint):
        if len(have) != R.DTILDE or len(carries) != R.DTILDE:
            raise ValueError("wrong recovery hint polynomial length")
        if any(not isinstance(v, int) or isinstance(v, bool)
               or v not in (-1, 0, 1) for v in carries):
            raise ValueError("recovery hint is not ternary")
        out.append([(a + h) % RECOVERY_BUCKETS
                    for a, h in zip(have, carries)])
    return out


def _linear_terms(ulp, gamma):
    """`phi` (length `AN`) and `<u, gamma>`, from the compression challenge."""
    q = R.QTILDE
    A, u = ulp["A"], ulp["u"]
    phi = [0] * AN
    for i in range(AN):
        acc = 0
        for k in range(len(u)):
            a = A[k][i]
            if a:
                acc += a * gamma[k]
        phi[i] = R.LSPLIT * acc % q
    u_gamma = sum(u[k] * gamma[k] for k in range(len(u))) % q
    return phi, u_gamma


def _phi_slice(phi, element):
    return phi[element * R.LSPLIT:(element + 1) * R.LSPLIT]


def prove(ck, com_pub, com_sec, message_slots, ternary_slots,
          ulp, alpha_lo, alpha_hi, xof, challenges=None):
    """`LANES.Prove`.  Returns the proof as a dict of NTT-domain elements.

    `xof` supplies the prover's private randomness (`g`, `y`); `challenges`
    supplies the verifier's, defaulting to Fiat-Shamir over the transcript.
    """
    if challenges is None:
        challenges = Challenges(b"")
    q = R.QTILDE
    r_hat = com_sec["r_hat"]

    # g: constant coefficient zero, committed in t_{N+1}
    g_coeff = [0] + [uniform_int(xof, q) for _ in range(R.DTILDE - 1)]
    g = R.ntt(g_coeff)
    t_g = _add(ck.apply_b(B_G, r_hat), g)

    # Bai--Galbraith response compression masks only the non-identity part
    # of B_0.  The committed opening still has KAPPA elements; its first
    # IDENTITY_RANK elements are recovered through the carry hint below.
    y = sample_gaussian_vec(xof, SIGMA_Y, RESPONSE_RANK)
    y_hat = [R.ntt(p) for p in y]
    w = [ck.apply_B0_tail(i, y_hat) for i in range(IDENTITY_RANK)]
    w_high = [recovery_high(R.intt(p)) for p in w]
    b_y = [ck.apply_b_tail(i, y_hat) for i in range(N_EX)]

    challenges.absorb(_pack(*com_pub["t0"]), _pack(*com_pub["t"]),
                      _pack(*w_high), _pack(t_g),
                      names=("t0", "t", "w_high", "t_g"))
    alpha = challenges.alpha(alpha_hi - alpha_lo)

    t_mp1, t_mp2, v = lanes_mp.prove(
        ck, ternary_slots, b_y, r_hat, y_hat, alpha_lo, alpha_hi, alpha)

    challenges.absorb(_pack(t_mp1, t_mp2, v), names=("t_mp1|t_mp2|v",))
    gamma = challenges.gamma(len(ulp["u"]))
    phi, u_gamma = _linear_terms(ulp, gamma)

    # h = g + slotwise( sum_e phi_{e,j} m_{e,j} - <u, gamma> )
    h = list(g)
    for slot in range(R.LSPLIT):
        acc = 0
        for elem in range(N_EX):
            acc += phi[elem * R.LSPLIT + slot] * message_slots[elem][slot]
        idx = slot * R.SUBDEG
        h[idx] = (h[idx] + acc - u_gamma) % q

    # v' = <b_G, y> + sum_e phi_e . <b_e, y>
    v_prime = ck.apply_b_tail(B_G, y_hat)
    for elem in range(N_EX):
        v_prime = _add(v_prime,
                       R.scale_blocks(b_y[elem], _phi_slice(phi, elem)))

    challenges.absorb(_pack(h, v_prime), names=("h|v_prime",))
    c = challenges.challenge()
    z = [R.add(y[i], R.mul(c, com_sec["r"][IDENTITY_RANK + i]))
         for i in range(RESPONSE_RANK)]

    # The prover applies the verifier's own response bounds, and returns
    # bottom rather than a proof that will be rejected.
    #
    # This used to be missing: `prove` formed `z` and returned it, while
    # `verify` enforced both bounds -- so an out-of-bound mask produced a
    # proof that verified as `False` and could not even be serialized
    # (`Rice` rejects a coefficient above its cap with `ValueError`).  A
    # prover that can return a proof its own verifier rejects is a defect
    # regardless of how rarely it fires, and it fires: `Z_INF_BOUND` is a
    # `2^-128` tail bound, not an impossibility.
    #
    # Returning `None` is the existing contract for an exact-layer abort.
    # `RiVeR.Eval` discards the whole attempt on it -- OOM proof included,
    # because `W` is already bound into the OOM challenge -- and retries
    # with fresh randomness.  Both bounds live in `response_within_bounds`
    # so the two sides cannot drift apart.
    if not response_within_bounds(z):
        return None

    # With t0_base = 2^D t0_high and r = r_id || r_tail,
    #
    #   B0' z - c t0_base = w + c (t0_low - r_id).
    #
    # The parameter bound makes the bucket displacement at most one.  The
    # fixed ternary carry is the missing metadata both bandwidth
    # optimisations require; it is deliberately not part of the challenge
    # input it helps reconstruct.
    z_hat = [R.ntt(p) for p in z]
    c_hat = R.ntt(c)
    t0_base = [R.ntt(p) for p in expand_t0(com_pub["t0"])]
    recovered_base = [
        recovery_high(R.intt(_sub(ck.apply_B0_tail(i, z_hat),
                                  R.ntt_mul(c_hat, t0_base[i]))))
        for i in range(IDENTITY_RANK)
    ]
    hint = make_recovery_hint(w_high, recovered_base)

    # `alpha` and `gamma` are recomputed by the verifier, and so are the
    # three check targets `w`, `v` and `v'`; `c` is transmitted because
    # recovering them needs it.  See the module docstring.
    return {"t_g": t_g, "t_mp1": t_mp1, "t_mp2": t_mp2,
            "h": h, "c": c, "hint": hint, "z": z}


def response_within_bounds(z):
    """Both verifier bounds on the response, in one place.

    `prove` calls it before returning and aborts if it fails; `verify` calls
    it before hashing anything.  A single definition is what keeps an honest
    proof from being rejected by its own verifier.

    * per-coefficient: `|z_i| <= Z_INF_BOUND`, an artifact-derived decoder
      and verifier cap;
    * Euclidean: `||z||_2^2 < Z_NORM2_BOUND`, the paper's `2 s sqrt(N_z)`
      rule at the transmitted rank.

    The Euclidean comparison is **strict**: a response whose squared norm
    equals the bound is rejected.  That is a choice, and it is the same
    choice on both sides, which is what matters for interoperability.
    """
    norm_sq = 0
    for poly in z:
        for coeff in R.centered(poly):
            if abs(coeff) > Z_INF_BOUND:
                return False
            norm_sq += coeff * coeff
    return norm_sq < Z_NORM2_BOUND


def verify(ck, com_pub, proof, ulp, alpha_lo, alpha_hi, challenges=None):
    """`LANES.Ver`.  Returns True/False."""
    q = R.QTILDE
    if challenges is None:
        challenges = Challenges(b"")
    try:
        z = proof["z"]
        c = proof["c"]
        hint = proof["hint"]
        if len(z) != RESPONSE_RANK or len(c) != R.DTILDE:
            return False
        if len(com_pub["t0"]) != IDENTITY_RANK or len(com_pub["t"]) != N_EX:
            return False
        t0_base = [R.ntt(p) for p in expand_t0(com_pub["t0"])]

        # Both response bounds, before anything is hashed.  Shared with
        # `prove`, which aborts rather than returning a proof this rejects.
        if not response_within_bounds(z):
            return False

        z_hat = [R.ntt(p) for p in z]
        c_hat = R.ntt(c)

        # Recover the torus quotient of w from the rank-17 response, the
        # compressed commitment, and the fixed ternary carry.
        w_base = [
            recovery_high(R.intt(_sub(ck.apply_B0_tail(i, z_hat),
                                      R.ntt_mul(c_hat, t0_base[i]))))
            for i in range(IDENTITY_RANK)
        ]
        w_high = use_recovery_hint(w_base, hint)

        challenges.absorb(_pack(*com_pub["t0"]), _pack(*com_pub["t"]),
                          _pack(*w_high), _pack(proof["t_g"]),
                          names=("t0", "t", "w_high", "t_g"))
        alpha = challenges.alpha(alpha_hi - alpha_lo)

        b_z = [ck.apply_b_tail(i, z_hat) for i in range(N_EX)]
        v = lanes_mp.recover_v(ck, com_pub["t"], alpha, proof["t_mp1"],
                               proof["t_mp2"], c_hat, z_hat, b_z,
                               alpha_lo, alpha_hi)

        challenges.absorb(_pack(proof["t_mp1"], proof["t_mp2"], v),
                          names=("t_mp1|t_mp2|v",))
        gamma = challenges.gamma(len(ulp["u"]))

        # the compressed linear relation: h_0 == 0.  This one is a genuine
        # check and stays -- `h` is transmitted, and one scalar constraint
        # does not determine 128 coefficients.
        if R.constant_coefficient(proof["h"]) != 0:
            return False

        phi, u_gamma = _linear_terms(ulp, gamma)

        # tau = slotwise(-<u,gamma>) + sum_e phi_e . t_e
        tau = R.slots_to_ntt([(-u_gamma) % q] * R.LSPLIT)
        for elem in range(N_EX):
            tau = _add(tau, R.scale_blocks(com_pub["t"][elem],
                                           _phi_slice(phi, elem)))

        lhs = ck.apply_b_tail(B_G, z_hat)
        for elem in range(N_EX):
            lhs = _add(lhs, R.scale_blocks(b_z[elem], _phi_slice(phi, elem)))

        inner = _sub(_add(tau, proof["t_g"]), proof["h"])
        v_prime = _sub(lhs, R.ntt_mul(c_hat, inner))

        challenges.absorb(_pack(proof["h"], v_prime),
                          names=("h|v_prime",))
        return challenges.challenge() == c
    except (KeyError, TypeError, IndexError, ValueError):
        return False


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The parameters are the paper's. The production alias is reserved by
    # the artifact's concrete-composition policy; see
    # `exact.lanes_unavailable_reason`.
    from exact import skip_if_lanes_unavailable
    skip_if_lanes_unavailable("lanes_proof.py")

    import random
    from sample import XOF, DS_EXACT

    q = R.QTILDE
    rng = random.Random(17)
    ck = LanesCommitmentKey(b"\x07" * 32)
    ALPHA_LO, ALPHA_HI = 2, N_EX

    def build(tamper=None):
        """A witness with ternary digits and one linear constraint per slot."""
        slots = []
        for e in range(N_EX):
            if ALPHA_LO <= e < ALPHA_HI:
                slots.append([rng.choice([-1, 0, 1]) for _ in range(R.LSPLIT)])
            else:
                slots.append([rng.randrange(1000) for _ in range(R.LSPLIT)])
        # constraint: element 0 slot j == sum of the ternary slots at j
        A = [[0] * AN for _ in range(R.LSPLIT)]
        u = [0] * R.LSPLIT
        for j in range(R.LSPLIT):
            A[j][j] = 1
            for e in range(ALPHA_LO, ALPHA_HI):
                A[j][e * R.LSPLIT + j] = q - 1        # -1
            slots[0][j] = sum(slots[e][j] for e in range(ALPHA_LO, ALPHA_HI))
        msg = [[v % q for v in s] for s in slots]
        if tamper == "digit":
            slots[ALPHA_LO][0] += 2
            msg[ALPHA_LO][0] = slots[ALPHA_LO][0] % q
        if tamper == "linear":
            slots[0][0] += 1
            msg[0][0] = slots[0][0] % q
        return msg, slots, {"A": A, "u": u}

    def run(tamper=None, corrupt=None):
        msg, slots, ulp = build(tamper)
        xof = XOF(DS_EXACT, b"proof-test", bytes([rng.randrange(256)]))
        pub, sec = commit(ck, msg, xof)
        pi = prove(ck, pub, sec, msg, slots, ulp, ALPHA_LO, ALPHA_HI, xof,
                   Challenges(b"selftest"))
        if corrupt:
            corrupt(pi)
        return verify(ck, pub, pi, ulp, ALPHA_LO, ALPHA_HI,
                      Challenges(b"selftest"))

    assert run(), "honest proof rejected"
    assert not run(tamper="digit"), "non-ternary digit accepted"
    assert not run(tamper="linear"), "broken linear relation accepted"

    def bump(field):
        def f(pi):
            pi[field] = list(pi[field])
            pi[field][0] = (pi[field][0] + 1) % q
        return f

    for field in ("h", "t_g", "t_mp1", "t_mp2"):
        assert not run(corrupt=bump(field)), f"tampered {field} accepted"

    def bump_stmt(pi):
        pass
    # a proof must not verify against a different statement
    msg, slots, ulp = build()
    xof = XOF(DS_EXACT, b"stmt")
    pub, sec = commit(ck, msg, xof)
    pi = prove(ck, pub, sec, msg, slots, ulp, ALPHA_LO, ALPHA_HI, xof,
               Challenges(b"statement-A"))
    assert verify(ck, pub, pi, ulp, ALPHA_LO, ALPHA_HI, Challenges(b"statement-A"))
    assert not verify(ck, pub, pi, ulp, ALPHA_LO, ALPHA_HI,
                      Challenges(b"statement-B")), "statement not bound"

    def bump_z(pi):
        pi["z"] = [list(p) for p in pi["z"]]
        pi["z"][0][0] = (pi["z"][0][0] + 1) % q
    assert not run(corrupt=bump_z), "tampered z accepted"

    def bump_c(pi):
        pi["c"] = list(pi["c"])
        pi["c"][0] = (pi["c"][0] + 1) % q
    assert not run(corrupt=bump_c), "tampered c accepted"

    def huge_z(pi):
        pi["z"] = [list(p) for p in pi["z"]]
        pi["z"][0][0] = q // 2
    assert not run(corrupt=huge_z), "oversized z accepted"

    print("lanes_proof.py: all self-tests passed "
          "(honest accepted; ternary, linear and transcript tampering rejected)")


def record_transcript(challenges):
    """The absorbed field names of a completed run, flattened.

    `Challenges.ROUNDS` declares the transcript; this reads back what a real
    run actually hashed.  `lanes_manifest` writes the declaration and
    `test_lanes.py` drives a proof and compares the two, so the manifest
    describes the implemented transcript rather than an intended one.

    Packed groups are recorded under a single joined name (`"h|v_prime"`),
    because they are one `absorb` argument and a port must concatenate them
    the same way; the declaration is expanded on `"|"` for comparison.
    """
    return list(challenges.absorbed)


def declared_transcript():
    """`Challenges.ROUNDS` flattened to the absorbed field order."""
    out = []
    for _, fields in Challenges.ROUNDS:
        out.extend(fields)
    return out
