"""
lanes_commit.py -- BDLOP commitment used by the LANES exact proof.

Optional component.  The commitment scheme of [BDLOP18] at the shape
[ENS20] Figure 3 uses, exploiting the structure of the key.

Shape
-----
The randomness `r` has `kappa = n~ + l~ + N_ex + alpha = 17` ring elements,
split into three roles:

    r[0 .. l~)                       the identity block of B_0    (4)
    r[l~ .. l~ + N_ex + alpha)       one per committed element    (9)
    r[kappa - n~ .. kappa)           the shared random tail       (4)

`B_0` is `[I_{l~} | random]` and each `b_i` is a unit vector at column
`l~ + i` plus a random tail in the last `n~` columns, so the response rank
is `kappa - l~ = n~ + N_ex + alpha = 13`.

**Which letter is which** is not decidable from this file:
`n~` and `l~` are both 4, so
no dimension check can tell them apart.  The paper's MLWE dimensions settle
it -- secret dimension `n~ d~`, samples `(l~ + N_ex + alpha) d~` -- and
`lanes_params` exposes the outcome as `IDENTITY_RANK` and `TAIL_RANK`.  The
code below uses those names, never the letters, for exactly that reason.  The identity entries are never materialised:
`apply_B0(i, x)` adds `x[i]` directly and `apply_b(i, x)` adds
`x[l~ + i]`, so only the random blocks are stored.

Everything lives in the NTT domain; the message occupies one scalar per NTT
block (`lanes_ring.slots_to_ntt`).  The exception is the public `t_0`: its
full NTT image is converted back to coefficients and only the high part after
`D`-bit power-of-two rounding is transmitted.  The opening randomness stays
at rank `kappa`; only the masked response later drops the `l~` identity
columns.
"""

import lanes_ring as R
from lanes_params import (KAPPA, RESPONSE_RANK, IDENTITY_RANK, TAIL_RANK, N_EX,
                          AUX, SIGMA_R, T0_SCALE, T0_HIGH_MODULUS,
                          t0_power2round, sample_uniform_poly,
                          sample_gaussian_vec)
from sample import XOF, DS_EXACT

#: b-row indices beyond the message elements
B_G = N_EX            # 6: carries the masking element g       (t_{N+1})
B_MP1 = N_EX + 1      # 7: first product-proof commitment      (t_{N+2})
B_MP2 = N_EX + 2      # 8: second product-proof commitment     (t_{N+3})
B_ROWS = N_EX + AUX   # 9


class LanesCommitmentKey:
    """`(B_0, b_0 .. b_8)`, stored as their random blocks only."""

    def __init__(self, seed):
        self.seed = seed
        xof = XOF(DS_EXACT + b".lanes.gen", seed)
        #: B_0 random block: n~ x (kappa - n~)
        self.B0 = [[R.ntt(sample_uniform_poly(xof))
                    for _ in range(KAPPA - IDENTITY_RANK)]
                   for _ in range(IDENTITY_RANK)]
        #: b_i random block: B_ROWS x l~, over the last l~ columns
        self.b = [[R.ntt(sample_uniform_poly(xof)) for _ in range(TAIL_RANK)]
                  for _ in range(B_ROWS)]

    # -- the two structured inner products --------------------------------

    def apply_B0(self, row, r_hat):
        """Row `row` of `B_0 r`: `r[row] + sum_j B0[row][j] r[l~ + j]`."""
        if len(r_hat) != KAPPA:
            raise ValueError(f"B0 input has {len(r_hat)} elements, expected {KAPPA}")
        acc = self.apply_B0_tail(row, r_hat[IDENTITY_RANK:])
        return [(acc[i] + r_hat[row][i]) % R.QTILDE for i in range(R.DTILDE)]

    def apply_B0_tail(self, row, tail_hat):
        """`B_0' tail`, omitting the `I_l~` block used by response compression."""
        if len(tail_hat) != RESPONSE_RANK:
            raise ValueError(
                f"B0 tail has {len(tail_hat)} elements, expected {RESPONSE_RANK}")
        return R.inner_ntt(self.B0[row], tail_hat)

    def apply_b(self, row, r_hat):
        """`<b_row, r> = r[l~ + row] + sum_j b[row][j] r[kappa - n~ + j]`."""
        if len(r_hat) != KAPPA:
            raise ValueError(f"b input has {len(r_hat)} elements, expected {KAPPA}")
        return self.apply_b_tail(row, r_hat[IDENTITY_RANK:])

    def apply_b_tail(self, row, tail_hat):
        """`<b_row, (0_l~ || tail)>`, on the transmitted response rank."""
        if len(tail_hat) != RESPONSE_RANK:
            raise ValueError(
                f"b tail has {len(tail_hat)} elements, expected {RESPONSE_RANK}")
        acc = R.inner_ntt(self.b[row], tail_hat[RESPONSE_RANK - TAIL_RANK:])
        unit = tail_hat[row]
        return [(acc[i] + unit[i]) % R.QTILDE for i in range(R.DTILDE)]


def compress_t0(t0):
    """Drop `D` low bits from coefficient-domain images of NTT `t_0`."""
    if len(t0) != IDENTITY_RANK:
        raise ValueError(f"t0 has {len(t0)} elements, expected {IDENTITY_RANK}")
    out = []
    for poly_hat in t0:
        poly = R.intt(poly_hat)
        if len(poly) != R.DTILDE:
            raise ValueError("wrong t0 polynomial length")
        out.append([t0_power2round(v)[0] for v in poly])
    return out


def expand_t0(t0_high):
    """Coefficient-domain representative `2^D t_0,high` used by verification."""
    if len(t0_high) != IDENTITY_RANK:
        raise ValueError(f"t0 has {len(t0_high)} elements, expected {IDENTITY_RANK}")
    out = []
    for poly in t0_high:
        if len(poly) != R.DTILDE:
            raise ValueError("wrong compressed t0 polynomial length")
        if any(not isinstance(v, int) or isinstance(v, bool)
               or not 0 <= v < T0_HIGH_MODULUS for v in poly):
            raise ValueError("non-canonical compressed t0 coefficient")
        out.append([(v * T0_SCALE) % R.QTILDE for v in poly])
    return out


def commit(ck, message_slots, xof):
    """`LANES.Com`.

    `message_slots` is a list of `N_ex` slot vectors, each of length
    `l = 64` -- one entry per NTT block of `R_q~`.
    Returns the public part `(t0_high, t)` and the secret randomness.  `t0`
    carries the integer high parts after dropping `D` bits; `t` and the
    randomness remain in the NTT domain.
    """
    if len(message_slots) != N_EX:
        raise ValueError(f"expected {N_EX} message elements")
    r = sample_gaussian_vec(xof, SIGMA_R, KAPPA)
    r_hat = [R.ntt(p) for p in r]

    t0_full = [ck.apply_B0(i, r_hat) for i in range(IDENTITY_RANK)]
    t0 = compress_t0(t0_full)
    t = []
    for i in range(N_EX):
        base = ck.apply_b(i, r_hat)
        t.append(R.add_slots_inplace(list(base), message_slots[i]))

    return {"t0": t0, "t": t}, {"r": r, "r_hat": r_hat}


def open_check(ck, public, secret, message_slots):
    """Recompute the commitment; used only by tests."""
    again, _ = None, None
    t0 = compress_t0([ck.apply_B0(i, secret["r_hat"])
                      for i in range(IDENTITY_RANK)])
    t = []
    for i in range(N_EX):
        base = ck.apply_b(i, secret["r_hat"])
        t.append(R.add_slots_inplace(list(base), message_slots[i]))
    return t0 == public["t0"] and t == public["t"]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The parameters are the paper's. The production alias is reserved by
    # the artifact's concrete-composition policy; see
    # `exact.lanes_unavailable_reason`.
    from exact import skip_if_lanes_unavailable
    skip_if_lanes_unavailable("lanes_commit.py")

    import random
    from lanes_ring import LSPLIT, QTILDE

    ck = LanesCommitmentKey(b"\x01" * 32)
    rng = random.Random(4)
    msg = [[rng.randrange(QTILDE) for _ in range(LSPLIT)] for _ in range(N_EX)]

    pub, sec = commit(ck, msg, XOF(DS_EXACT, b"commit-test"))
    assert len(pub["t0"]) == IDENTITY_RANK and len(pub["t"]) == N_EX
    assert open_check(ck, pub, sec, msg), "commitment does not reopen"

    bad = [list(v) for v in msg]
    bad[0][0] = (bad[0][0] + 1) % QTILDE
    assert not open_check(ck, pub, sec, bad), "opened to a different message"

    # determinism in the XOF, and dependence on the key
    pub2, _ = commit(ck, msg, XOF(DS_EXACT, b"commit-test"))
    assert pub2 == pub
    pub3, _ = commit(LanesCommitmentKey(b"\x02" * 32), msg,
                     XOF(DS_EXACT, b"commit-test"))
    assert pub3 != pub

    # the message really does land in the slots
    zero_msg = [[0] * LSPLIT for _ in range(N_EX)]
    pub0, sec0 = commit(ck, zero_msg, XOF(DS_EXACT, b"z"))
    pubm, _ = commit(ck, msg, XOF(DS_EXACT, b"z"))
    for i in range(N_EX):
        diff = [(pubm["t"][i][k] - pub0["t"][i][k]) % QTILDE
                for k in range(R.DTILDE)]
        assert R.ntt_to_slots(diff) == [v % QTILDE for v in msg[i]]

    print("lanes_commit.py: all self-tests passed")
