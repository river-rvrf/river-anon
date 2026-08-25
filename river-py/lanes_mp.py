"""
lanes_mp.py -- The cubic product proof: committed slots are ternary.

Optional component.  The product proof of [ALS20], in the form [ENS20]
Figure 3 uses it.

What it proves
--------------
For every slot `m` of the selected message elements, `m(m+1)(m-1) = 0`, i.e.
`m in {-1, 0, 1}`.  That is the whole reason RiVeR's rounding-error range is
encoded in ternary digits: the range proof reduces to this.

How
---
With `f = <b_i, z> - c t_i = b_y - c m` (the masked opening of slot `m`), the
identity

    f^3 - c^2 f = b_y^3 - 3c m b_y^2 + c^2 (3m^2 - 1) b_y - c^3 (m^3 - m)

has a last term that vanishes exactly when `m^3 = m`.  The prover commits to
the two intermediate coefficients (`t_{N+2}`, `t_{N+3}`) before seeing `c`,
and the verifier recombines them; the check passes iff the `m^3 - m` term is
zero.  `alpha` is a random ring element per element, batching the slots.
"""

import lanes_ring as R
from lanes_commit import B_MP1, B_MP2


def _add(a, b):
    q = R.QTILDE
    return [(a[i] + b[i]) % q for i in range(R.DTILDE)]


def _sub(a, b):
    q = R.QTILDE
    return [(a[i] - b[i]) % q for i in range(R.DTILDE)]


def prove(ck, ternary_slots, b_y, r_hat, y_hat, alpha_lo, alpha_hi, alpha):
    """Produce `(t_{N+2}, t_{N+3}, v)`, all NTT domain.

    `ternary_slots` maps element index -> list of `l` slot values as integers
    in `{-1, 0, 1}`; only elements in `[alpha_lo, alpha_hi)` are proved.
    `alpha` is supplied by the caller so it can be Fiat-Shamir derived.
    """
    q = R.QTILDE
    t_mp1 = ck.apply_b(B_MP1, r_hat)
    t_mp2 = ck.apply_b(B_MP2, r_hat)
    v = ck.apply_b_tail(B_MP1, y_hat)

    for idx in range(alpha_lo, alpha_hi):
        a = alpha[idx - alpha_lo]
        by = b_y[idx]
        slots = ternary_slots[idx]

        a_by = R.ntt_mul(a, by)                    # alpha * b_y
        a_by2 = R.ntt_mul(a_by, by)                # alpha * b_y^2
        a_by3 = R.ntt_mul(a_by2, by)               # alpha * b_y^3

        # t_{N+2} -= sum_j 3 m_j * (alpha b_y^2)|_j
        t_mp1 = _sub(t_mp1, R.scale_blocks(a_by2, [3 * m % q for m in slots]))
        # t_{N+3} += sum_j (3 m_j^2 - 1) * (alpha b_y)|_j
        t_mp2 = _add(t_mp2, R.scale_blocks(a_by,
                                           [(3 * m * m - 1) % q for m in slots]))
        v = _add(v, a_by3)

    t_mp1 = _add(t_mp1, ck.apply_b_tail(B_MP2, y_hat))
    return t_mp1, t_mp2, v


def recover_v(ck, com_t, alpha, t_mp1, t_mp2, c_hat, z_hat, b_z,
              alpha_lo, alpha_hi):
    """The value `v` the cubic check equates to, recomputed from the rest.

    This used to *compare* against a transmitted `v` and return a bit.  `v`
    is a check target and so is fully determined by everything else in the
    proof, which makes transmitting it redundant: `lanes_proof` recovers it
    here, feeds it back into the Fiat-Shamir transcript in the position the
    prover put it, and lets `c' == c` decide.  See `lanes_proof`'s module
    docstring for why that is the same test.
    """
    # f_{N+3} = <b_{N+2}, z> - c t_{N+3}
    f_mp2 = _sub(ck.apply_b_tail(B_MP2, z_hat), R.ntt_mul(c_hat, t_mp2))
    total = R.ntt_mul(c_hat, f_mp2)

    for idx in range(alpha_lo, alpha_hi):
        f = _sub(b_z[idx], R.ntt_mul(c_hat, com_t[idx]))
        term = R.ntt_mul(f, _add(f, c_hat))
        term = R.ntt_mul(term, _sub(f, c_hat))
        total = _add(total, R.ntt_mul(alpha[idx - alpha_lo], term))

    # + <b_{N+1}, z> - c t_{N+2}
    total = _add(total, ck.apply_b_tail(B_MP1, z_hat))
    total = _sub(total, R.ntt_mul(c_hat, t_mp1))
    return total


def verify(ck, com_t, alpha, t_mp1, t_mp2, v, c_hat, z_hat, b_z,
           alpha_lo, alpha_hi):
    """`recover_v(...) == v`, for this module's own self-test."""
    return recover_v(ck, com_t, alpha, t_mp1, t_mp2, c_hat, z_hat, b_z,
                     alpha_lo, alpha_hi) == v


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The parameters are the paper's. The production alias is reserved by
    # the artifact's concrete-composition policy; see
    # `exact.lanes_unavailable_reason`.
    from exact import skip_if_lanes_unavailable
    skip_if_lanes_unavailable("lanes_mp.py")

    import random
    from lanes_commit import LanesCommitmentKey, commit
    from lanes_params import (RESPONSE_RANK, IDENTITY_RANK, N_EX, SIGMA_Y,
                              sample_gaussian_vec, sample_challenge,
                              sample_uniform_poly)
    from lanes_ring import LSPLIT, QTILDE
    from sample import XOF, DS_EXACT

    ck = LanesCommitmentKey(b"\x03" * 32)
    rng = random.Random(9)
    ALPHA_LO, ALPHA_HI = 2, N_EX

    def run(ternary):
        slots = [[0] * LSPLIT for _ in range(N_EX)]
        for i in range(ALPHA_LO, ALPHA_HI):
            slots[i] = ternary[i - ALPHA_LO]
        msg = [[v % QTILDE for v in s] for s in slots]

        xof = XOF(DS_EXACT, b"mp-test")
        pub, sec = commit(ck, msg, xof)
        y = sample_gaussian_vec(xof, SIGMA_Y, RESPONSE_RANK)
        y_hat = [R.ntt(p) for p in y]
        b_y = [ck.apply_b_tail(i, y_hat) for i in range(N_EX)]

        alpha = [R.ntt(sample_uniform_poly(xof))
                 for _ in range(ALPHA_HI - ALPHA_LO)]
        t1, t2, v = prove(ck, slots, b_y, sec["r_hat"], y_hat,
                          ALPHA_LO, ALPHA_HI, alpha)
        c = sample_challenge(xof)
        c_hat = R.ntt(c)
        z = [R.add(y[i], R.mul(c, sec["r"][IDENTITY_RANK + i]))
             for i in range(RESPONSE_RANK)]
        z_hat = [R.ntt(p) for p in z]
        b_z = [ck.apply_b_tail(i, z_hat) for i in range(N_EX)]
        return verify(ck, pub["t"], alpha, t1, t2, v, c_hat, z_hat, b_z,
                      ALPHA_LO, ALPHA_HI)

    good = [[rng.choice([-1, 0, 1]) for _ in range(LSPLIT)]
            for _ in range(ALPHA_HI - ALPHA_LO)]
    assert run(good), "honest ternary witness rejected"

    for bad_value in (2, -2, 3):
        bad = [list(s) for s in good]
        bad[0][0] = bad_value
        assert not run(bad), f"non-ternary slot {bad_value} accepted"

    print("lanes_mp.py: all self-tests passed (ternary accepted, non-ternary rejected)")
