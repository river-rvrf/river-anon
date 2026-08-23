"""
test_lanes.py -- Tests for the optional LANES exact backend.

The parameters are the paper-derived LANES parameters.

Structure and widths both come from the paper: `d~ = 256`, `l = 64`,
`q~ = 67107713` (26 bits), `(n~, l~) = (4, 4)`, `D = 17`, six separately
padded 64-slot message blocks, and the closed-form Gaussian widths
`s_1 = 2 s_0`, `s_2 = 2 w_hat s_0` with
`s_0 = sqrt(ln(2 d~ (1 + 1/eps)))/pi` at `eps = 2^-100`.  `lanes_params`
re-derives every figure the paper prints -- `beta' = 45430.6`,
`B_MSIS = 15991562`, `q~/B_MSIS = 4.2`, `delta_MSIS = 1.0037`, `D = 17` --
and this suite pins each against the printed digits.

What is still this repository's, and is labelled Repair everywhere: the
whole recovery-hint construction, the response *infinity* bound (the paper
states none), the wire layout, and the sampler tail cuts.

The standard-deviation convention reproduces the published
`delta_MLWE = 1.0040`.  The production alias remains reserved because the
concrete compression/recovery completion is implementation-defined and this
artifact does not supply a reduction for that exact composition.

So this suite reaches the code through
`exact_backend="lanes-experimental"`, a different backend name from
`"lanes"`: the readiness gate goes on refusing `"lanes"`,
and nothing here lifts it.  The alternative -- no coverage at all behind
the gate -- is how an unconstrained message-block padding survived to be
found by inspection instead of by a test.

"""

from params import TOY_PARAMS as _TOY

#: Backend name every test here uses.  Never `"lanes"`; see the docstring.
EXPERIMENTAL = "lanes-experimental"

try:
    from lanes_backend import LanesBackend as _LB
    _LB.experimental(_TOY)
    _SKIP = None
except Exception as _exc:                      # pragma: no cover
    _SKIP = f"the LANES port does not construct: {_exc!r}"

if _SKIP:
    if __name__ == "__main__":
        print("test_lanes.py: SKIPPED -- " + _SKIP)
        raise SystemExit(0)
    import pytest as _pytest
    _pytest.skip(_SKIP, allow_module_level=True)


import math
from fractions import Fraction
import random

import pytest

import lanes_mp
import lanes_proof
import lanes_ring as R
from lanes_backend import (LanesBackend, ALPHA_LO, ALPHA_HI, IDX_E, IDX_Y,
                           IDX_DIGITS, WEIGHT_SUM, build_linear_system)
from lanes_commit import (LanesCommitmentKey, commit, open_check, expand_t0,
                          B_G)
from lanes_params import (KAPPA, RESPONSE_RANK, IDENTITY_RANK, ELL_TILDE,
                          N_TILDE, N_EX,
                          AUX, W_HAT, DELTA, W_TILDE, D_DROP, T0_SCALE,
                          T0_LOW_BOUND, T0_HIGH_MODULUS, RECOVERY_BUCKETS,
                          RECOVERY_ERROR_BOUND, SIGMA_R, SIGMA_Y,
                          Z_NORM2_BOUND, Z_NORM2_REQUIRED,
                          Z_NORM2_REQUIRED_IID, Z_INF_BOUND,
                          VAR_Z, N_Z, sample_challenge, sample_gaussian_vec,
                          message_slot_count, t0_power2round)
from exact import ExactParams, RADIX_WEIGHTS, get_backend
from params import TOY_PARAMS, get
from ring import negacyclic_mul_int
from river import RiVeR
from sample import XOF, DS_EXACT, VERIFIER_TAILCUT

PAR = TOY_PARAMS
CK = LanesCommitmentKey(b"\x11" * 32)


# ---- ring ----------------------------------------------------------------

def test_ring_splits_into_64_blocks():
    assert R.LSPLIT == 64 and R.SUBDEG == 4 and R.DTILDE == 256
    assert len(R.LEAF_EXPS) == 64
    assert all(e % 2 == 1 for e in R.LEAF_EXPS)
    assert len(set(R.LEAF_EXPS)) == 64


def test_ntt_round_trip_and_schoolbook():
    rng = random.Random(1)
    for _ in range(8):
        a = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
        b = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
        assert R.intt(R.ntt(a)) == a
        assert R.mul(a, b) == R.mul_schoolbook(a, b)


def test_negacyclic_property():
    x = [0] * R.DTILDE
    x[1] = 1
    prod = x
    for _ in range(R.DTILDE - 1):
        prod = R.mul(prod, x)
    expect = [0] * R.DTILDE
    expect[0] = R.QTILDE - 1
    assert prod == expect


def test_slot_helpers_are_ntt_domain():
    rng = random.Random(2)
    vals = [rng.randrange(R.QTILDE) for _ in range(R.LSPLIT)]
    assert R.ntt_to_slots(R.slots_to_ntt(vals)) == vals


def test_constant_coefficient_identity():
    """The identity the linear proof relies on: const coeff == (sum slots)/l."""
    rng = random.Random(3)
    inv_l = pow(R.LSPLIT, R.QTILDE - 2, R.QTILDE)
    for _ in range(5):
        vals = [rng.randrange(R.QTILDE) for _ in range(R.LSPLIT)]
        assert R.constant_coefficient(R.slots_to_ntt(vals)) == \
            sum(vals) % R.QTILDE * inv_l % R.QTILDE


def test_scale_blocks_is_slot_diagonal():
    rng = random.Random(4)
    hat = [rng.randrange(R.QTILDE) for _ in range(R.DTILDE)]
    scal = [rng.randrange(R.QTILDE) for _ in range(R.LSPLIT)]
    out = R.scale_blocks(hat, scal)
    for j in range(R.LSPLIT):
        for k in range(R.SUBDEG):
            idx = j * R.SUBDEG + k
            assert out[idx] == scal[j] * hat[idx] % R.QTILDE


# ---- parameters ----------------------------------------------------------

def test_dimensions_match_the_paper():
    ex = ExactParams(get("RiVeR-N8"))
    assert (ex.d_tilde, ex.l_split, ex.n_tilde, ex.ell_tilde, ex.N_ex) == \
        (256, 64, 4, 4, 6)
    assert KAPPA == N_TILDE + ELL_TILDE + N_EX + AUX == 17
    assert RESPONSE_RANK == KAPPA - N_TILDE == 13
    # `6 outer_d != N_ex l` (192 != 384) is intentional: every block is
    # padded, and `build_linear_system` constrains the padding to zero.
    assert message_slot_count() == 384 == N_EX * R.LSPLIT
    assert W_HAT == 44 and DELTA * W_TILDE == W_HAT


def test_every_published_lanes_figure_is_reproduced():
    """the paper prints the whole chain; each printed digit is pinned.

    This is the check that the implementation reads the paper's closed form
    the way the paper means it -- including the Gaussian convention, where
    `s` and `sigma` differ by `sqrt(2 pi)` and reading the wrong one is 18
    bits of security.
    """
    import lanes_params as lp

    # the widths, each to the precision the paper prints
    for value, printed in ((lp.S0, "2.7668"), (lp.S1, "5.5336"),
                           (lp.S2, "243.4775"),
                           (lp.S_RESPONSE, "344.3291")):
        places = len(printed.split(".")[1])
        assert f"{float(value):.{places}f}" == printed, (printed, value)

    # ...and the Gaussian *parameters* the paper prints beside them, which
    # is what pins the convention: sigma = s sqrt(2 pi), not s / sqrt(2 pi)
    root = math.sqrt(2 * math.pi)
    for value, printed in ((lp.S0, "6.9353"), (lp.S1, "13.8706"),
                           (lp.S2, "610.3075")):
        assert f"{float(value) * root:.4f}" == printed, printed

    # the security chain
    assert f"{float(lp.BETA_PRIME_BDLOP):.1f}" == "45430.6"
    assert round(float(lp.B_MSIS)) == 15991562
    assert f"{float(lp.Q_OVER_B_MSIS):.1f}" == "4.2"
    assert f"{float(lp.DELTA_MSIS):.4f}" == "1.0037"
    assert lp.N_Z_PAPER == 4352 == (lp.N_TILDE + lp.ELL_TILDE + lp.N_EX
                                    + lp.ALPHA) * R.DTILDE

    # `D = 17`, and the two inequalities that pick it
    assert lp.largest_compression_exponent() == lp.D_DROP == 17
    assert 2 ** 17 <= Fraction(W_HAT) * lp.SIGMA_R * lp.N_TILDE * R.DTILDE
    assert not (2 ** 18 <= Fraction(W_HAT) * lp.SIGMA_R
                * lp.N_TILDE * R.DTILDE)
    assert R.QTILDE > 4 * W_HAT * 2 ** 17


def test_the_widths_land_on_the_smoothing_parameter_by_construction():
    """`sigma_MLWE = s_0` exactly -- the design principle behind the widths.

    The paper does not state this, but it is why `s_1 = 2 s_0` and
    `s_2 = 2 w_hat s_0` rather than any other pair with the same ratio:
    substituting them into the [KLSS23] reduction returns the smoothing
    parameter for `eps = 2^-100`.  Pinned because a port that rounded the
    widths differently, or dropped the factor 2, would break the identity
    and nothing else would notice.

    `s_2 = w_hat s_1`, so the reduction *could* be collapsed here -- but it
    is evaluated in the unsimplified form, which is what a future width
    pair without that relation would need.
    """
    from lanes_params import SIGMA_MLWE, SIGMA_MLWE_SQ, N_LWE, M_LWE, ALPHA
    import lanes_params as lp

    assert (N_TILDE, ELL_TILDE) == (4, 4)
    inverse_sq = 2 * (1 / SIGMA_R ** 2
                      + Fraction(W_HAT) ** 2 / SIGMA_Y ** 2)
    assert inverse_sq == 1 / SIGMA_MLWE_SQ
    assert math.isclose(SIGMA_MLWE ** 2, float(SIGMA_MLWE_SQ), rel_tol=1e-15)

    # the identity, to the rounding of the widths themselves
    assert math.isclose(SIGMA_MLWE, float(lp.S0), rel_tol=1e-6)
    # and the relation that makes it hold
    assert math.isclose(float(SIGMA_Y), W_HAT * float(SIGMA_R), rel_tol=1e-6)
    # ...which also identifies the response width
    assert math.isclose(
        float(lp.S_RESPONSE),
        math.sqrt(float(SIGMA_Y) ** 2 + W_HAT ** 2 * float(SIGMA_R) ** 2),
        rel_tol=1e-6)

    assert (N_LWE, M_LWE) == (N_TILDE * R.DTILDE,
                              (ELL_TILDE + N_EX + ALPHA) * R.DTILDE)
    assert (N_LWE, M_LWE) == (1024, 3328)      # the paper's, and reproduced


def test_challenge_is_ternary_with_partitioned_support():
    x = XOF(b"t", b"chal")
    for _ in range(50):
        c = R.centered(sample_challenge(x))
        assert set(c) <= {-1, 0, 1}
        assert sum(1 for v in c if v) == W_HAT
        for i in range(DELTA):
            assert sum(1 for j in range(R.LSPLIT) if c[j * DELTA + i]) == W_TILDE


# ---- commitment ----------------------------------------------------------

def _random_message(seed=7):
    rng = random.Random(seed)
    return [[rng.randrange(R.QTILDE) for _ in range(R.LSPLIT)]
            for _ in range(N_EX)]


def test_commitment_opens_and_is_binding():
    msg = _random_message()
    pub, sec = commit(CK, msg, XOF(DS_EXACT, b"c1"))
    assert open_check(CK, pub, sec, msg)
    bad = [list(v) for v in msg]
    bad[0][0] = (bad[0][0] + 1) % R.QTILDE
    assert not open_check(CK, pub, sec, bad)


def test_commitment_is_deterministic_in_the_xof():
    msg = _random_message()
    a, _ = commit(CK, msg, XOF(DS_EXACT, b"same"))
    b, _ = commit(CK, msg, XOF(DS_EXACT, b"same"))
    c, _ = commit(CK, msg, XOF(DS_EXACT, b"other"))
    assert a == b and a != c


def test_commitment_randomness_is_short():
    _, sec = commit(CK, _random_message(), XOF(DS_EXACT, b"short"))
    assert len(sec["r"]) == KAPPA
    assert max(R.inf_norm(p) for p in sec["r"]) <= 6 * SIGMA_R


def test_t0_drops_exactly_17_bits_with_a_bounded_remainder():
    msg = _random_message(19)
    pub, sec = commit(CK, msg, XOF(DS_EXACT, b"compressed-t0"))
    base = expand_t0(pub["t0"])
    assert D_DROP == 17 and T0_SCALE == 1 << 17
    assert all(0 <= v < T0_HIGH_MODULUS for row in pub["t0"] for v in row)

    for i in range(N_TILDE):
        full = R.intt(CK.apply_B0(i, sec["r_hat"]))
        delta = R.centered([(full[j] - base[i][j]) % R.QTILDE
                            for j in range(R.DTILDE)])
        assert max(abs(v) for v in delta) <= T0_LOW_BOUND


def test_t0_rounding_convention_is_pinned_at_the_tie():
    for value in (0, T0_LOW_BOUND - 1, T0_LOW_BOUND,
                  T0_LOW_BOUND + 1, R.QTILDE - 1):
        high, low = t0_power2round(value)
        assert high * T0_SCALE + low == value
        assert -T0_LOW_BOUND < low <= T0_LOW_BOUND
    assert t0_power2round(T0_LOW_BOUND) == (0, T0_LOW_BOUND)
    assert t0_power2round(T0_LOW_BOUND + 1) == (1, 1 - T0_LOW_BOUND)


def test_recovery_hint_parameters_cover_the_combined_perturbation():
    """The next power-of-two bucket count would make ternary carries unsafe."""
    assert RECOVERY_ERROR_BOUND < R.QTILDE // RECOVERY_BUCKETS
    assert RECOVERY_ERROR_BOUND >= R.QTILDE // (2 * RECOVERY_BUCKETS)

    backend, _, _, sigma, _ = _backend_run()
    assert len(sigma["z"]) == RESPONSE_RANK
    assert len(sigma["hint"]) == N_TILDE
    assert all(v in (-1, 0, 1) for row in sigma["hint"] for v in row)


# ---- product proof -------------------------------------------------------

def _mp_run(ternary_slots):
    slots = [[0] * R.LSPLIT for _ in range(N_EX)]
    for i in range(ALPHA_LO, ALPHA_HI):
        slots[i] = ternary_slots[i - ALPHA_LO]
    msg = [[v % R.QTILDE for v in s] for s in slots]
    xof = XOF(DS_EXACT, b"mp")
    pub, sec = commit(CK, msg, xof)
    y = sample_gaussian_vec(xof, SIGMA_Y, RESPONSE_RANK)
    y_hat = [R.ntt(p) for p in y]
    b_y = [CK.apply_b_tail(i, y_hat) for i in range(N_EX)]
    alpha = [R.ntt([random.Random(i).randrange(R.QTILDE)
                    for _ in range(R.DTILDE)])
             for i in range(ALPHA_HI - ALPHA_LO)]
    t1, t2, v = lanes_mp.prove(CK, slots, b_y, sec["r_hat"], y_hat,
                               ALPHA_LO, ALPHA_HI, alpha)
    c = sample_challenge(xof)
    z = [R.add(y[i], R.mul(c, sec["r"][N_TILDE + i]))
         for i in range(RESPONSE_RANK)]
    z_hat = [R.ntt(p) for p in z]
    b_z = [CK.apply_b_tail(i, z_hat) for i in range(N_EX)]
    return lanes_mp.verify(CK, pub["t"], alpha, t1, t2, v, R.ntt(c), z_hat,
                           b_z, ALPHA_LO, ALPHA_HI)


def test_product_proof_accepts_ternary():
    rng = random.Random(11)
    good = [[rng.choice([-1, 0, 1]) for _ in range(R.LSPLIT)]
            for _ in range(ALPHA_HI - ALPHA_LO)]
    assert _mp_run(good)


def test_product_proof_rejects_non_ternary():
    rng = random.Random(12)
    good = [[rng.choice([-1, 0, 1]) for _ in range(R.LSPLIT)]
            for _ in range(ALPHA_HI - ALPHA_LO)]
    for bad_value in (2, -2, 3, 100):
        bad = [list(s) for s in good]
        bad[0][0] = bad_value
        assert not _mp_run(bad), bad_value


# ---- the Pi_ex backend ---------------------------------------------------

def _witness(seed=5, par=PAR):
    rng = random.Random(seed)
    e_eval = [rng.randrange(par.q0) - par.B_e for _ in range(par.d)]
    y_eval = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(par.d)]
    x_c = [0] * par.d
    for pos in rng.sample(range(par.d), par.w):
        x_c[pos] = rng.choice([-1, 1]) * rng.randint(1, par.gamma)
    prod = negacyclic_mul_int(x_c, e_eval)
    z_c = [prod[i] + y_eval[i] for i in range(par.d)]
    return e_eval, y_eval, x_c, z_c


def _backend_run(seed=5, par=PAR):
    backend = LanesBackend.experimental(par)
    pp = backend.setup(par, b"\x31" * 32)
    e_eval, y_eval, x_c, z_c = _witness(seed, par)
    w_in = {"e_eval": e_eval, "y_eval": y_eval}
    W, st = backend.com(pp, w_in, XOF(DS_EXACT, b"be", bytes([seed])))
    stmt = {"W": W, "z_eval_centered": z_c, "x_centered": x_c}
    sigma = backend.prove(pp, stmt, w_in, st)
    return backend, pp, stmt, sigma, st


def test_backend_accepts_honest_proof():
    backend, pp, stmt, sigma, _ = _backend_run()
    assert backend.verify(pp, stmt, sigma)


def test_backend_digits_are_centred_ternary_and_reconstruct():
    backend, pp, stmt, sigma, st = _backend_run()
    e_eval = st["e_eval"]
    for j in range(len(RADIX_WEIGHTS)):
        assert all(a in (-1, 0, 1) for a in st["slots"][IDX_DIGITS + j])
    for i in range(PAR.d):
        acc = sum(w * st["slots"][IDX_DIGITS + j][i]
                  for j, w in enumerate(RADIX_WEIGHTS))
        assert acc == e_eval[i]


def test_backend_rejects_wrong_statement():
    backend, pp, stmt, sigma, _ = _backend_run()
    bad = dict(stmt, z_eval_centered=list(stmt["z_eval_centered"]))
    bad["z_eval_centered"][0] += 1
    assert not backend.verify(pp, bad, sigma)


def test_backend_rejects_wrong_challenge():
    backend, pp, stmt, sigma, _ = _backend_run()
    bad = dict(stmt, x_centered=list(stmt["x_centered"]))
    idx = next(i for i, v in enumerate(bad["x_centered"]) if v)
    bad["x_centered"][idx] = -bad["x_centered"][idx]
    assert not backend.verify(pp, bad, sigma)


def test_backend_rejects_wrong_commitment():
    backend, pp, stmt, sigma, _ = _backend_run(seed=5)
    other = _backend_run(seed=6)[2]["W"]
    assert not backend.verify(pp, dict(stmt, W=other), sigma)


def test_backend_rejects_tampered_transcript():
    backend, pp, stmt, sigma, _ = _backend_run()
    for field in ("t_g", "t_mp1", "t_mp2", "h", "c"):
        bad = dict(sigma)
        bad[field] = list(sigma[field])
        bad[field][0] = (bad[field][0] + 1) % R.QTILDE
        assert not backend.verify(pp, stmt, bad), field

    bad = dict(sigma, z=[list(p) for p in sigma["z"]])
    bad["z"][0][0] = (bad["z"][0][0] + 1) % R.QTILDE
    assert not backend.verify(pp, stmt, bad)

    bad = dict(sigma, hint=[list(p) for p in sigma["hint"]])
    bad["hint"][0][0] = {-1: 0, 0: 1, 1: 0}[bad["hint"][0][0]]
    assert not backend.verify(pp, stmt, bad), "recovery carry"

    # `w`, `v` and `v'` are no longer transmitted, so tampering with them
    # is not a thing an adversary can do -- what replaces those cases is
    # that the *recovered* values feed the transcript, so moving anything
    # they depend on moves `c'`.  `z` above is one such; so is `t_0`, which
    # only `w`'s recovery reads.
    bad_w = dict(stmt)
    bad_w["W"] = dict(stmt["W"], t0=[list(p) for p in stmt["W"]["t0"]])
    bad_w["W"]["t0"][0][0] = \
        (bad_w["W"]["t0"][0][0] + 1) % T0_HIGH_MODULUS
    assert not backend.verify(pp, bad_w, sigma), "t0 moved under w's recovery"


def test_backend_rejects_malformed_proof():
    backend, pp, stmt, _, _ = _backend_run()
    assert not backend.verify(pp, stmt, {})


def test_backend_rejects_every_shape_of_malformed_hint():
    """The hint is the one field with no algebraic check of its own.

    `z`, `t0`, `t_g` and the rest are all bound by the transcript or by a
    ring equation.  The recovery carry is *metadata*: it moves the
    reconstructed `w_high` by one bucket, and nothing else constrains it.
    So every way of getting it wrong has to be rejected by the recovery
    rule itself, not by the algebra downstream.
    """
    backend, pp, stmt, sigma, _ = _backend_run()
    assert backend.verify(pp, stmt, sigma)

    def rejected(hint, why):
        assert not backend.verify(pp, dict(stmt), dict(sigma, hint=hint)), why

    # out of the ternary alphabet, in both directions
    for value in (2, -2, R.QTILDE - 1, 7):
        bad = [list(p) for p in sigma["hint"]]
        bad[0][0] = value
        rejected(bad, f"hint coefficient {value}")

    # wrong number of rows, and wrong length within a row
    rejected(sigma["hint"][:-1], "one row short")
    rejected(list(sigma["hint"]) + [list(sigma["hint"][0])], "one row long")
    bad = [list(p) for p in sigma["hint"]]
    bad[0] = bad[0][:-1]
    rejected(bad, "one coefficient short")

    # structurally wrong types -- `verify` is total, so these are `False`
    # rather than exceptions
    for shape in (None, [], "hint", [None] * IDENTITY_RANK, 0):
        rejected(shape, f"hint shaped {shape!r:.20}")

    # an all-zero hint is well-formed and still wrong wherever the honest
    # carry was nonzero -- which is the case a domain check alone misses
    if any(v for row in sigma["hint"] for v in row):
        rejected([[0] * R.DTILDE for _ in sigma["hint"]], "all-zero carry")


def test_prove_returns_bottom_rather_than_a_proof_verify_would_reject():
    """A prover that can return a proof its own verifier rejects is a defect.

    `prove` used to form `z` and return it while `verify` enforced both
    response bounds, so an out-of-bound mask produced a proof that verified
    as `False` and could not even be serialized (`Rice` raises above its
    cap).  Both bounds now live in `response_within_bounds`, `prove` calls
    it, and bottom is the existing contract for an exact-layer abort --
    `RiVeR.Eval` discards the whole attempt on it.

    Driven by forcing the mask past the bound rather than by waiting for a
    `2^-128` event.
    """
    import lanes_proof

    backend, pp, stmt, sigma, state = _backend_run()

    # the honest response passes, and is the same object both sides test
    assert lanes_proof.response_within_bounds(sigma["z"])

    # ...and a response one step past either bound does not
    over_inf = [list(p) for p in sigma["z"]]
    over_inf[0][0] = R.QTILDE - (Z_INF_BOUND + 1)      # centred: -(bound+1)
    assert not lanes_proof.response_within_bounds(over_inf)

    # The Euclidean test is strict -- equality is rejected -- and that is
    # a choice, so it is pinned rather than left to whichever side is read
    # first.  Built directly: a response whose squared norm is exactly the
    # bound, spread so no coefficient trips the infinity check.
    at_bound = [[0] * R.DTILDE for _ in range(RESPONSE_RANK)]
    remaining, step = Z_NORM2_BOUND, Z_INF_BOUND
    for row in at_bound:
        for k in range(R.DTILDE):
            if remaining <= 0:
                break
            take = min(step, math.isqrt(remaining))
            row[k] = take % R.QTILDE
            remaining -= take * take
    # top up the last coefficient so the total lands exactly on the bound
    assert remaining >= 0
    norm = sum(c * c for p in at_bound for c in R.centered(p))
    if norm < Z_NORM2_BOUND:
        deficit = Z_NORM2_BOUND - norm
        for row in at_bound:
            for k in range(R.DTILDE):
                if row[k] == 0 and deficit > 0:
                    take = min(Z_INF_BOUND, math.isqrt(deficit))
                    if take * take == deficit:
                        row[k] = take % R.QTILDE
                        deficit = 0
                    break
        norm = sum(c * c for p in at_bound for c in R.centered(p))
    if norm == Z_NORM2_BOUND:
        assert not lanes_proof.response_within_bounds(at_bound), \
            "equality must be rejected: the two sides must agree on <"

    # Now the real path: a sampler that draws far too wide makes `prove`
    # return `None` instead of a proof.
    real_sample = lanes_proof.sample_gaussian_vec

    def wide(xof, sigma_w, length):
        return [[(c * 64) % R.QTILDE for c in poly]
                for poly in real_sample(xof, sigma_w, length)]

    lanes_proof.sample_gaussian_vec = wide
    try:
        out = backend.prove(pp, stmt, None, state)
    finally:
        lanes_proof.sample_gaussian_vec = real_sample
    assert out is None, "prove returned a proof its own verifier rejects"


def test_backend_response_is_short():
    backend, pp, stmt, sigma, _ = _backend_run()
    norm_sq = sum(c * c for poly in sigma["z"] for c in R.centered(poly))
    assert norm_sq < Z_NORM2_BOUND


def test_backend_does_not_transmit_the_witness():
    """The proof carries no witness field.

    This is the whole point of using LANES rather than `OpeningBackend`.  It
    is evidence, not a zero-knowledge proof.
    """
    backend, pp, stmt, sigma, st = _backend_run()
    assert set(sigma) == {"t_g", "t_mp1", "t_mp2", "h", "c", "hint", "z"}
    blob = backend.proof_encode({"W": stmt["W"], "sigma": sigma})
    for name in ("e_eval", "y_eval", "digits"):
        assert name not in sigma
    # the raw witness values do not appear verbatim in the encoding.
    # `e_eval` is *centred*, so shift it
    # back to the canonical range the relation states before making bytes.
    e_bytes = bytes(v + backend.ex.par.B_e for v in st["e_eval"])
    assert e_bytes not in blob


def test_backend_encoding_round_trip():
    backend, pp, stmt, sigma, _ = _backend_run()
    blob = backend.proof_encode({"W": stmt["W"], "sigma": sigma})
    assert len(blob) <= backend.proof_bytes    # Rice: variable
    again = backend.proof_decode(blob)
    assert again["W"] == stmt["W"]
    assert backend.verify(pp, dict(stmt, W=again["W"]), again["sigma"])
    assert backend.proof_encode(again) == blob


def test_backend_rejects_trailing_bytes():
    backend, pp, stmt, sigma, _ = _backend_run()
    blob = backend.proof_encode({"W": stmt["W"], "sigma": sigma})
    try:
        backend.proof_decode(blob + b"\x00")
    except ValueError:
        return
    raise AssertionError("trailing bytes accepted")


# ---- linear system -------------------------------------------------------

def test_linear_system_is_satisfied_by_the_honest_witness():
    ex = ExactParams(PAR)
    e_eval, y_eval, x_c, z_c = _witness()
    ulp = build_linear_system(ex, x_c, z_c)

    from exact import decompose_poly
    digits = [[a - 1 for a in poly]
              for poly in decompose_poly([c + PAR.B_e for c in e_eval])]
    def pad(coeffs):
        assert len(coeffs) == PAR.d
        return list(coeffs) + [0] * (R.LSPLIT - PAR.d)

    slots = [[0] * R.LSPLIT for _ in range(N_EX)]
    slots[IDX_E] = pad(e_eval)
    slots[IDX_Y] = pad(y_eval)
    for j, poly in enumerate(digits):
        slots[IDX_DIGITS + j] = pad(poly)
    flat = [v for s in slots for v in s]

    # reconstruction, link, and one zero equation per padding coordinate
    pad_rows = N_EX * (R.LSPLIT - PAR.d)
    assert pad_rows == 192
    assert len(ulp["A"]) == 2 * PAR.d + pad_rows == 256
    for k, row in enumerate(ulp["A"]):
        acc = sum(row[i] * flat[i] for i in range(len(flat))) % R.QTILDE
        assert acc == ulp["u"][k] % R.QTILDE, k

    # Every column is constrained.  Without the padding rows exactly 192 of
    # the 384 columns were identically zero -- the padding positions -- so
    # a prover could commit to any padding it liked.  This is the check
    # that would have caught it.
    columns = len(ulp["A"][0])
    assert columns == N_EX * R.LSPLIT == 384
    unconstrained = [i for i in range(columns)
                     if all(row[i] == 0 for row in ulp["A"])]
    assert unconstrained == [], unconstrained

    # ...and a nonzero padding coordinate really does break the system
    for element in range(N_EX):
        broken = list(flat)
        broken[element * R.LSPLIT + PAR.d] = 1
        assert any(sum(row[i] * broken[i] for i in range(columns)) % R.QTILDE
                   != ulp["u"][k] % R.QTILDE
                   for k, row in enumerate(ulp["A"])), element


def test_backend_rejects_nonzero_padding():
    """A prover that fills the padding slots must not be believed.

    The commitment carries `N_ex l = 384` message coordinates and the
    semantic witness occupies `d = 32` of each 64-slot block, so 192 of
    them are padding.  The product proof restricts the four digit blocks to
    `{-1, 0, 1}` and says nothing at all about the `y_eval` and `e_eval`
    blocks, so before `build_linear_system` grew its zero equations every
    one of these forgeries **verified**.  The test is written to fail
    loudly if those rows are ever removed again.
    """
    backend = LanesBackend.experimental(PAR)
    pp = backend.setup(PAR, b"\x31" * 32)
    e_eval, y_eval, x_c, z_c = _witness()
    w_in = {"e_eval": e_eval, "y_eval": y_eval}
    stmt_xof = XOF(DS_EXACT, b"pad")
    W, st = backend.com(pp, w_in, stmt_xof)
    stmt = {"W": W, "z_eval_centered": z_c, "x_centered": x_c}
    assert backend.verify(pp, stmt, backend.prove(pp, stmt, w_in, st))

    for element in (IDX_Y, IDX_E, IDX_DIGITS, IDX_DIGITS + 3):
        for value in (1, R.QTILDE - 1):
            slots = [list(block) for block in st["slots"]]
            slots[element][PAR.d] = value          # first padding coordinate
            message = [[v % R.QTILDE for v in block] for block in slots]
            pub, sec = commit(pp["ck"], message, XOF(DS_EXACT, b"pad"))
            bad_st = dict(st, slots=slots, message=message, secret=sec)
            bad_stmt = dict(stmt, W=pub)
            sigma = backend.prove(pp, bad_stmt, w_in, bad_st)
            assert not (sigma is not None
                        and backend.verify(pp, bad_stmt, sigma)), \
                f"nonzero padding accepted at block {element}, value {value}"


def test_message_blocks_must_carry_exactly_d_coefficients():
    """Short or long semantic inputs are refused, not silently padded.

    Accepting anything up to `l` would let a caller place coefficients in
    slots the linear system pins to zero, or leave real ones outside the
    equations that constrain them.
    """
    backend = LanesBackend.experimental(PAR)
    pp = backend.setup(PAR, b"\x31" * 32)
    e_eval, y_eval, _, _ = _witness()
    for bad in (e_eval[:-1], e_eval + [0], e_eval + [0] * (R.LSPLIT - PAR.d)):
        with pytest.raises(ValueError):
            backend.com(pp, {"e_eval": bad, "y_eval": y_eval},
                        XOF(DS_EXACT, b"width"))
        with pytest.raises(ValueError):
            backend.com(pp, {"e_eval": e_eval, "y_eval": bad},
                        XOF(DS_EXACT, b"width"))


# ---- integration with RiVeR ----------------------------------------------

def test_backend_is_selectable():
    assert isinstance(get_backend(EXPERIMENTAL, PAR), LanesBackend)
    assert get_backend(EXPERIMENTAL, PAR).name == EXPERIMENTAL


def test_the_experimental_backend_says_so_and_the_name_round_trips():
    """An experimental run must not report itself as `"lanes"`.

    `scheme.exact.name` is what `vectors.py` stores in a case, and
    `vectors.py` rebuilds a scheme by handing that string straight back to
    `get_backend`.  If the experimental instance
    called itself `"lanes"`, an experimental vector would record the gated
    production name, verification would try to reconstruct the gated
    backend, and that construction refuses -- so the case could never be
    checked.  Benchmarks would meanwhile attribute experimental widths to
    the paper's LANES.
    """
    experimental = LanesBackend.experimental(PAR)
    assert experimental.name == EXPERIMENTAL != LanesBackend.name

    # the class still calls itself "lanes", which is the gated name
    assert LanesBackend.name == "lanes"
    with pytest.raises(NotImplementedError):
        LanesBackend(PAR)

    # and the recorded name reconstructs the same backend, not the gated one
    scheme = RiVeR(PAR, exact_backend=EXPERIMENTAL)
    assert scheme.exact.name == EXPERIMENTAL
    again = RiVeR(PAR, exact_backend=scheme.exact.name)
    assert again.exact.name == EXPERIMENTAL
    assert again.exact.ex.fingerprint == scheme.exact.ex.fingerprint


def test_an_experimental_vector_case_round_trips_under_its_own_name():
    """The real `vectors.py` path, on the name the backend reports.

    `generate_case` writes `scheme.exact.name` into the case;
    `verify_case` hands that string straight back to `RiVeR(...)` and
    re-derives everything from the seeds.  Both are called here rather
    than imitated -- an earlier version of this test rebuilt a scheme by
    hand and so proved nothing about the path that actually ships.

    With a single shared `"lanes"` the reconstruction would have hit the
    readiness gate and raised, so the case could never have been checked.
    """
    import vectors

    case = vectors.generate_case(PAR.name, exact_backend=EXPERIMENTAL)
    assert case["exact_backend"] == EXPERIMENTAL
    assert case["params"] == PAR.name

    errors = vectors.verify_case(case)
    assert errors == [], errors

    # a case that claimed the gated name would not survive the round trip
    forged = dict(case, exact_backend="lanes")
    with pytest.raises(NotImplementedError):
        vectors.verify_case(forged)

    # ...and the gated name is what the shipped set withholds; an
    # experimental case is a new entry under its own name, not one of these
    assert all(backend == "lanes" for _, backend in vectors.WITHHELD_CASES)
    assert EXPERIMENTAL not in {b for _, b in vectors.WITHHELD_CASES}


def test_end_to_end_with_lanes_backend():
    scheme = RiVeR(PAR, exact_backend=EXPERIMENTAL)
    pp = scheme.setup(b"\x41" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(scheme.par.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    v, pi = scheme.eval(pp, pk, sk, ring, b"lanes-e2e", b"\x42" * 32)
    assert scheme.verify(pp, ring, b"lanes-e2e", v, pi)
    blob = scheme.proof_encode(pi)
    assert scheme.verify(pp, ring, b"lanes-e2e", v, scheme.proof_decode(blob))
    assert scheme.proof_encode(scheme.proof_decode(blob)) == blob


def test_end_to_end_rejects_tampering_with_lanes_backend():
    scheme = RiVeR(PAR, exact_backend=EXPERIMENTAL)
    pp = scheme.setup(b"\x43" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(scheme.par.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[0]
    v, pi = scheme.eval(pp, pk, sk, ring, b"m", b"\x44" * 32)
    assert scheme.verify(pp, ring, b"m", v, pi)
    assert not scheme.verify(pp, ring, b"other", v, pi)
    bad = list(v)
    bad[0] = (bad[0] + 1) % PAR.p
    assert not scheme.verify(pp, ring, b"m", bad, pi)


def test_published_profile_with_lanes_backend():
    par = get("RiVeR-N8")
    scheme = RiVeR(par, exact_backend=EXPERIMENTAL)
    pp = scheme.setup(b"\x45" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(scheme.par.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[2]
    v, pi = scheme.eval(pp, pk, sk, ring, b"n8", b"\x46" * 32)
    assert scheme.verify(pp, ring, b"n8", v, pi)


def test_verifier_bounds_hold_for_honest_responses():
    """An honest `z = y + c r` must satisfy the bounds its own verifier checks.

    `lanes_proof.prove` has no retry, so a bound the honest distribution
    exceeds is a plain correctness bug rather than a tightness question.  The
    original bounds were sized against the mask `y` alone with a 1.05 margin,
    which ignored `c r` and used far too small a chi-square margin: about
    1 proof in 584 failed, and 1 in 119,000 broke the per-coefficient bound.

    Both bounds are now derived in `lanes_params` from `Var[z]` at a `2^-128`
    target.  This samples response trajectories directly, which is far cheaper
    than whole proofs and tests exactly the quantity at issue.
    """
    x = XOF(b"test_lanes", b"honest-norms")
    worst_n2 = worst_inf = 0
    for _ in range(120):
        r = sample_gaussian_vec(x, SIGMA_R, RESPONSE_RANK)
        y = sample_gaussian_vec(x, SIGMA_Y, RESPONSE_RANK)
        c_hat = R.ntt(sample_challenge(x))
        n2 = 0
        for i in range(RESPONSE_RANK):
            cr = R.intt(R.ntt_mul(c_hat, R.ntt(r[i])))
            z = R.centered([(y[i][k] + cr[k]) % R.QTILDE
                            for k in range(R.DTILDE)])
            for value in z:
                n2 += value * value
                worst_inf = max(worst_inf, abs(value))
        worst_n2 = max(worst_n2, n2)

    assert worst_n2 < Z_NORM2_BOUND, (worst_n2, Z_NORM2_BOUND)
    assert worst_inf <= Z_INF_BOUND, (worst_inf, Z_INF_BOUND)

    # the bounds must also be derived, not asserted: they have to account for
    # the `c r` term, whose variance is w_hat * sigma_r^2 per coefficient
    assert VAR_Z == SIGMA_Y ** 2 + W_HAT * SIGMA_R ** 2
    assert Z_NORM2_BOUND > N_Z * VAR_Z, "bound below the expected norm"


def test_backend_rejects_a_modified_y_eval():
    """`y_eval` is a committed message, so moving it must break the proof.

    The relation tests each of `y_eval`, `z_eval` and
    `x`; the other two are the wrong-statement and wrong-challenge tests
    above.  `y_eval` is not in the statement, so it has to be moved in the
    witness and the proof re-run against the original commitment.
    """
    backend, pp, stmt, sigma, st = _backend_run()
    e_eval, y_eval, x_c, z_c = _witness(5)
    moved = list(y_eval)
    moved[0] += 1
    w_bad = {"e_eval": e_eval, "y_eval": moved}
    W_bad, st_bad = backend.com(pp, w_bad, XOF(DS_EXACT, b"be", bytes([5])))
    assert W_bad != stmt["W"], "a different message must commit differently"
    sigma_bad = backend.prove(pp, stmt, w_bad, st_bad)
    assert not backend.verify(pp, stmt, sigma_bad)


def test_alternative_integer_lifts_are_excluded_by_the_bound_not_the_proof():
    """`y_eval + q~` is the same residue and a different integer.

    LANES has one modulus, so it cannot tell the two apart: the lifted witness
    commits to the same `W`, satisfies the same linear system, and produces a
    proof that verifies against the *same* statement.  That is asserted here
    rather than papered over -- it is what made the 26-bit modulus unsound.

    What excludes the lift is the response bound.  An accepted `z_eval`
    satisfies `||z_eval||_inf <= 6 sigma_rs`, so with `q~ > 12 sigma_rs` at
    most one integer per residue class is in range, and with the paper's
    `q~ > 24 phi_rs B_rs` the same holds for the difference of two accepted
    responses -- which is the quantity cross-fork extraction needs.
    """
    ex = ExactParams(PAR)
    e_eval, y_eval, _, z_c = _witness(5)

    # the proof system is blind to the lift
    backend, pp, stmt, _, _ = _backend_run()
    w_lift = {"e_eval": e_eval, "y_eval": [v + ex.q_tilde for v in y_eval]}
    W_lift, st_lift = backend.com(pp, w_lift, XOF(DS_EXACT, b"be", bytes([5])))
    assert W_lift == stmt["W"], "a lift by q~ is the same commitment"
    sigma_lift = backend.prove(pp, stmt, w_lift, st_lift)
    assert backend.verify(pp, stmt, sigma_lift)

    # the bound is not.  One lift in range:
    assert abs(z_c[0]) <= VERIFIER_TAILCUT * PAR.sigma_m
    assert abs(z_c[0] + ex.q_tilde) > VERIFIER_TAILCUT * PAR.sigma_m
    assert abs(z_c[0] - ex.q_tilde) > VERIFIER_TAILCUT * PAR.sigma_m

    # and that is a property of every profile, not of this sample
    for name in ("RiVeR-N8", "RiVeR-N16", "RiVeR-N64",
                 "RiVeR-N128", "RiVeR-N256", "RiVeR-TOY"):
        par = get(name)
        q_tilde = ExactParams(par).q_tilde
        assert q_tilde > 2 * VERIFIER_TAILCUT * par.sigma_m      # one response
        assert q_tilde > 4 * VERIFIER_TAILCUT * par.sigma_m      # a difference
        assert 4 * VERIFIER_TAILCUT == 24                          # = 24 phi_rs B_rs


def test_backend_handles_coefficients_at_the_edge_of_the_modulus():
    """Witness coefficients within one of `+-q~/2` must round-trip.

    Centred encoding is where an off-by-one at the modulus boundary shows up,
    and the encoder rejects a non-canonical residue rather than folding it, so
    this exercises both directions at the extreme.
    """
    half = R.QTILDE // 2
    edge = [half, -half, half - 1, -(half - 1), 0, 1, -1]
    poly = [(edge[i % len(edge)]) % R.QTILDE for i in range(R.DTILDE)]
    assert R.centered(poly) == [v if v <= half else v - R.QTILDE
                                for v in poly]
    assert R.from_centered(R.centered(poly)) == poly

    backend = LanesBackend.experimental(PAR)
    layout = backend.W_layout
    high_edge = [0, T0_HIGH_MODULUS - 1, 1, T0_HIGH_MODULUS // 2]
    high_poly = [high_edge[i % len(high_edge)] for i in range(R.DTILDE)]
    W = {"t0": [list(high_poly) for _ in range(N_TILDE)],
         "t": [list(poly) for _ in range(N_EX)]}
    assert layout.decode(layout.encode(W)) == W

    # one past the modulus is not a residue and must not silently fold
    bad = dict(W, t0=[list(high_poly) for _ in range(N_TILDE)])
    bad["t0"][0][0] = T0_HIGH_MODULUS
    with pytest.raises(ValueError):
        layout.encode(bad)


def test_measured_proof_size_is_reported_field_by_field():
    """Report every field from the real serializer.

    `model_bits` puts a comparison-only closed form beside it; see below.
    The two are reported together
    because the port sits above the model and the difference is the
    deliverable.
    """
    backend, pp, stmt, sigma, _ = _backend_run()
    pi_ex = {"W": stmt["W"], "sigma": sigma}
    rows = backend.field_sizes(pi_ex)

    names = [r["name"] for r in rows]
    assert names[:2] == ["t0", "t"] and names[-1] == "z"
    assert {r["name"] for r in rows} == \
        {f.name for f in backend.proof_layout.fields}

    for row in rows:
        assert 0 < row["bits"] <= row["max_bits"]
        assert row["coeffs"] == row["elements"] * R.DTILDE
    # uniform fields are written at exactly ceil(log2 q~) bits, which the
    # modulus makes 26 -- derived here rather than spelled, because
    # the literal 29 outlived the 29-bit modulus it was written for
    q_bits = (R.QTILDE - 1).bit_length()
    assert q_bits == 26
    for row in rows:
        if row["dist"] == "uniform mod q~":
            assert row["bits"] == row["coeffs"] * q_bits

    total = sum(r["bits"] for r in rows)
    assert (total + 7) // 8 == len(backend.proof_encode(pi_ex))

    # The two optimisations and their concrete metadata are wire properties,
    # not just comments in the size report.
    by_name = {row["name"]: row for row in rows}
    assert by_name["t0"]["bits"] == N_TILDE * R.DTILDE * (q_bits - D_DROP + 1)
    assert by_name["hint"]["bits"] == N_TILDE * R.DTILDE * 2 == 2048
    assert by_name["z"]["coeffs"] == RESPONSE_RANK * R.DTILDE == 3328

    # The measured total and a field-level entropy model, together. At the
    # paper's widths the model is 13.30 KB and the concrete encoding is
    # 13.89 KB, shown beside the paper's 13.5 KB entropy estimate.
    model = backend.model_bits()
    model_kb = sum(model.values()) / 8192
    assert 13.2 < model_kb < 13.4, model_kb
    assert 13.8 < total / 8192 < 14.0, total / 8192
    assert total > sum(model.values()), "the port is above the model, not below"
    # and the excess is accounted for, not just bounded: the model omits
    # the recovery hint and the challenge entirely, and codes `z` at its
    # entropy `h(sigma_y)` where the serializer uses Rice
    model_z = next(v for k, v in model.items() if k.startswith("z "))
    omitted = by_name["hint"]["bits"] + by_name["c"]["bits"]
    rice_overhead = by_name["z"]["bits"] - model_z
    # ...and the model charges `log2 q~ - D = 9` bits for a `t0` high part
    # that `power2round` leaves in `[0, 513)`, so the serializer writes 10.
    # One bit per coefficient, reproduced as printed rather than corrected.
    t0_shortfall = N_TILDE * R.DTILDE
    assert t0_shortfall == 1024
    assert abs((total - sum(model.values()))
               - (omitted + rice_overhead + t0_shortfall)) < 1e-6

    # and Rice stays within five percent of the entropy model it
    # approximates -- requirement (c), a concrete entropy-oriented encoding
    z_row = next(r for r in rows if r["name"] == "z")
    per_coeff = z_row["bits"] / z_row["coeffs"]
    ideal = math.log2(4.13 * float(SIGMA_Y))
    assert 1.0 < per_coeff / ideal < 1.05, (per_coeff, ideal)


def test_the_euclidean_bound_accounts_for_the_response_covariance():
    """`z = y + c r` has correlated coefficients, so a chi-square is wrong.

    This is the test the covariance correction needed and did not have: the
    old one checked `chi2_slack(N_Z)` and would have stayed green if
    `Z_NORM2_BOUND` were reverted to the independent-coordinate form.

    Everything below is exact integer arithmetic on real challenges.  The
    covariance of one response polynomial is
    `Sigma_1 = sigma_y^2 I + sigma_r^2 M M^T` with `M` the negacyclic
    multiplication matrix of `c`, and `M M^T` is itself the multiplication
    matrix of `u := c * conj(c)`, where `conj(c)_0 = c_0` and
    `conj(c)_i = -c_{d-i}`.  So

        tr(M M^T)     = d~ u_0     = d~ ||c||_2^2,
        ||M M^T||_F^2 = d~ ||u||_2^2,

    and the whole spectrum reduces to two integers per challenge.
    """
    from fractions import Fraction
    from dgs import quadratic_form_bound
    from ring import negacyclic_mul_int
    from lanes_params import SIGMA_TRACE, SIGMA_FROB_SQ, SIGMA_OP, VAR_CR

    d = R.DTILDE
    x = XOF(b"test_lanes", b"covariance")
    seen = []
    for _ in range(300):
        c = sample_challenge(x)
        cc = [v - R.QTILDE if v > R.QTILDE // 2 else v for v in c]
        conj = [cc[0]] + [-cc[d - i] for i in range(1, d)]
        u = negacyclic_mul_int(cc, conj)

        # the diagonal is exactly ||c||_2^2 = w_hat, which is why the
        # per-coefficient variance -- and the infinity-norm union bound,
        # which never needed independence -- are unaffected
        assert u[0] == sum(v * v for v in cc) == W_HAT
        seen.append(sum(v * v for v in u))

    # ...but the sum of squared eigenvalues is not what equal ones would
    # give.  The IID model assumes every eigenvalue is the mean, i.e.
    # `||u||_2^2 = w_hat^2 = 1936`; over these 300 challenges the real value
    # runs about 1.6x to 2.8x that, and that spread is exactly what the
    # Euclidean tail sees and the chi-square bound does not.
    assert len(seen) == 300
    assert 1.4 * W_HAT ** 2 < min(seen), min(seen)
    assert 2.5 * W_HAT ** 2 < max(seen) < W_HAT ** 3, max(seen)

    # the shipped Frobenius bound is a valid upper bound for every one
    for norm_u_sq in seen:
        frob_block = (d * SIGMA_Y ** 4
                      + 2 * SIGMA_Y ** 2 * SIGMA_R ** 2 * d * W_HAT
                      + SIGMA_R ** 4 * d * norm_u_sq)
        assert Fraction(SIGMA_FROB_SQ) >= RESPONSE_RANK * frob_block
    assert SIGMA_TRACE == N_Z * VAR_Z

    # The honest-response *requirement* really is the quadratic-form bound
    # and not the chi-square one -- reverting would fail here.  Since
    # the paper this is no longer what the verifier enforces (the paper's
    # `(2 s)^2` rule is), but it is what makes that rule usable: it is the
    # smallest bound an honest response can be held to, and the enforced
    # one has to clear it.
    from decimal import ROUND_CEILING
    want = int(quadratic_form_bound(SIGMA_TRACE, SIGMA_FROB_SQ, SIGMA_OP)
               .to_integral_value(ROUND_CEILING))
    assert Z_NORM2_REQUIRED == want
    assert Z_NORM2_REQUIRED > Z_NORM2_REQUIRED_IID
    assert Z_NORM2_BOUND > Z_NORM2_REQUIRED, "the paper's rule must clear it"

    ratio = Z_NORM2_REQUIRED / Z_NORM2_REQUIRED_IID
    # The correction tracks `w_hat sigma_r^2`'s share of `Var[z]`: 2.2% at
    # the paper's widths, against 9% at the reselected ones and 4% at the
    # ones this test was written for.  What is pinned is the relationship
    # and its direction, not a figure that goes stale when the widths move.
    share = float(Fraction(VAR_CR, 1) / VAR_Z)
    assert 0.01 < share < 0.20, share
    assert 1.0 < ratio < 1.35, ratio
    assert ratio > 1 + share / 2, (ratio, share)

    # And the margin the paper's rule leaves over that requirement, which
    # is the number that says whether an honest prover ever aborts.
    assert 5 < Z_NORM2_BOUND / Z_NORM2_REQUIRED < 6


def test_the_operator_bound_holds_and_is_loose():
    """`||Sigma||_op <= sigma_y^2 + w_hat^2 sigma_r^2`, and by how much.

    The Frobenius side is pinned above from exact integers.  The operator
    side needs an actual eigenvalue: `M M^T`'s are `|c-hat(zeta_j)|^2` over
    the primitive `2 d~`-th roots of unity, so this evaluates the transform
    directly rather than asserting the formula against the mean.

    Complex arithmetic here, unlike everywhere else in this repository,
    because nothing downstream of it reaches a wire-visible decision -- the
    shipped bound is the exact integer `sigma_y^2 + w_hat^2 sigma_r^2`, and
    this only checks that it dominates the true value with margin to spare.
    """
    import cmath
    from lanes_params import SIGMA_OP

    d = R.DTILDE
    # the formula itself, exactly
    assert SIGMA_OP == SIGMA_Y ** 2 + W_HAT ** 2 * SIGMA_R ** 2
    assert SIGMA_OP > VAR_Z, "the operator norm must exceed the mean eigenvalue"

    x = XOF(b"test_lanes", b"operator")
    worst = 0.0
    for _ in range(60):
        c = sample_challenge(x)
        cc = [v - R.QTILDE if v > R.QTILDE // 2 else v for v in c]
        for j in range(d):
            zeta = cmath.exp(2j * cmath.pi * (2 * j + 1) / (2 * d))
            worst = max(worst, abs(sum(cc[k] * zeta ** k
                                       for k in range(d))) ** 2)
    # ||c||_1^2 = w_hat^2 is the worst case over *all* challenges; the ones
    # that occur are far below it, so the bound is sound and loose -- about
    # 5.7x on this sample.  That costs margin, not correctness.
    assert worst < W_HAT ** 2, worst
    assert worst > W_HAT, "and it is not trivially small either"
    bound_at_worst = float(SIGMA_Y ** 2) + worst * float(SIGMA_R ** 2)
    assert float(SIGMA_OP) > bound_at_worst


def test_norm_bound_margin_is_sized_for_2_to_the_minus_128():
    """The Euclidean margin comes from a chi-square tail, not a fudge factor.

    Done in `Decimal`: the requirement is met with no slack at all by
    construction, so a float re-derivation lands on the wrong side of `-128`
    in the last bit.
    """
    from decimal import Decimal, localcontext
    from dgs import chi2_slack
    from lanes_params import VAR_CR

    def log2_chernoff(factor):
        """log2 of `exp(-n (eps - ln(1+eps)) / 2)` at `1 + eps = factor`."""
        with localcontext() as ctx:
            ctx.prec = 50
            eps = Decimal(factor) - 1
            return -Decimal(N_Z) * (eps - (1 + eps).ln()) / 2 / Decimal(2).ln()

    assert log2_chernoff(chi2_slack(N_Z)) <= -128

    # The discarded margin -- `1.05^2` against `sigma_y` alone -- is
    # inadequate at every width pair this tree has run, and the *reason*
    # moves with them, which is why it is measured rather than asserted.
    #
    # It ignores the `c r` term, so what it really bounds is
    # `1.05^2 sigma_y^2 / Var[z]`.  At the paper's widths `w_hat sigma_r^2`
    # is only 2.2% of `Var[z]`, so that lands at 1.078 and buys a Chernoff
    # bound of about `2^-6.9` -- a measurable failure rate, not `2^-128`.
    # At the reselected widths the term was 8% and the factor fell to
    # 1.0087, buying `2^-0.07`, i.e. nothing; at other widths it sat
    # above the mean and gave `2^-5.9` (measured: 1 honest proof in 584).
    # No arrangement of it reaches the target.
    old_factor = Decimal(1.05 ** 2 * float(SIGMA_Y) ** 2) / Decimal(float(VAR_Z))
    assert Decimal("1.05") < old_factor < Decimal("1.10"), old_factor
    assert -8 < log2_chernoff(old_factor) < -6
    assert log2_chernoff(old_factor) > -128, "nowhere near the target"

    # ...and the term it drops is small here, which is precisely why the
    # failure is quiet: 2.2% of the variance, and a bound off by 121 bits.
    share = Decimal(float(VAR_CR)) / Decimal(float(VAR_Z))
    assert Decimal("0.02") < share < Decimal("0.03"), share


def test_the_lanes_dimensions_are_the_exact_layers_and_not_copies():
    """Identity, not equality.

    `lanes_ring` and `lanes_params` used to repeat `ExactParams`'
    dimensions as literals that happened to agree.  `gen_kat.py` reads the
    LANES copies, so a divergence would have produced vectors disagreeing
    with the parameters they claim to come from, with nothing to catch it.
    Asserting the *values* would pass just as happily if both sides were
    edited to the same wrong number; assert they are the same object.
    """
    import lanes_ring as R
    import lanes_params as P
    from exact import ExactParams as E

    for name, got, want in [
        ("QTILDE", R.QTILDE, E.q_tilde),
        ("DTILDE", R.DTILDE, E.d_tilde),
        ("LSPLIT", R.LSPLIT, E.l_split),
        ("N_TILDE", P.N_TILDE, E.n_tilde),
        ("ELL_TILDE", P.ELL_TILDE, E.ell_tilde),
        ("N_EX", P.N_EX, E.N_ex),
        ("AUX", P.AUX, E.aux_slots),
        ("W_HAT", P.W_HAT, E.w_hat),
        ("D_DROP", P.D_DROP, E.D),
    ]:
        assert got is want, f"{name} is a copy of {want}, not the same object"

    # and the constants derived from them follow the structure `check()` pins
    assert R.SUBDEG == E.d_tilde // E.l_split
    assert R.LEVELS == E.l_split.bit_length() - 1
    assert 1 << R.LEVELS == R.LSPLIT, "LEVELS is only log2 if LSPLIT is a power of 2"
    assert R.PSI_ORDER == 2 * R.LSPLIT
    assert P.W_TILDE * P.DELTA == P.W_HAT, "W_TILDE truncated"
    assert P.KAPPA == E.n_tilde + E.ell_tilde + E.N_ex + E.aux_slots


def test_check_enforces_what_the_derived_constants_assume():
    """Each mutation stays self-consistent and still breaks a derivation."""
    from exact import ExactParams
    from params import TOY_PARAMS

    for field, value, expect in [
        ("l_split", 48, "does not divide"),
        ("l_split", 24, "power of two"),
        ("w_hat", 43, "W_TILDE truncates"),
    ]:
        ex = ExactParams(TOY_PARAMS)
        setattr(ex, field, value)
        errors = "; ".join(ex.check())
        assert expect in errors, f"{field}={value}: expected {expect!r}, got {errors!r}"


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_lanes.py: {len(tests)} tests passed")
