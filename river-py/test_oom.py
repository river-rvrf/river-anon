"""
test_oom.py -- Unit tests for the one-out-of-many layer.

Includes the correctness invariants of the OOM correctness proof, which is
where a transcription error in Figure 7 would show up first.
"""

import random

from oom import OOMStatement, _inf_int
from ring import round_p, rounding_error
from sample import XOF, DS_COMMIT
from params import TOY_PARAMS
from river import RiVeR

PAR = TOY_PARAMS


def _fixture(seed=b"\x31" * 32, ring_size=None, signer=1):
    """A full OOM statement plus an honest witness, via the scheme."""
    scheme = RiVeR(PAR)
    pp = scheme.setup(seed)
    # Rings are exactly `N` keys; there is no
    # padding, so `ring_size` is `N` or the ring is inadmissible.
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(PAR.N if ring_size is None else ring_size)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[signer]

    ring = scheme.validate_ring(ring)
    j_star = scheme.ring_index(ring, pk)
    h_m = scheme.hash_message(b"oom-test")
    R = scheme.Rq

    inner = R.inner(h_m, sk)
    v = round_p(R, inner, PAR.q0)
    e_eval = [c - PAR.B_e for c in rounding_error(R, inner, v, PAR.q0)]
    As = R.mat_vec(pp["A"], sk)
    e_key = [[c - PAR.B_e
              for c in rounding_error(R, As[i], ring[j_star][i], PAR.q0)]
             for i in range(PAR.n)]
    r = list(sk) + [R.from_centered(e) for e in e_key] \
        + [R.from_centered(e_eval)]

    statement = OOMStatement(PAR, R, pp["A"], h_m, ring, v)
    return scheme, pp, statement, j_star, r


# ---- statement structure -------------------------------------------------

def test_honest_opening_opens_c_jstar():
    """c_{j*} = Com^0_{ck_r}(r).  The invariant everything else rests on."""
    _, _, statement, j_star, r = _fixture()
    assert statement.apply_ck(r) == statement.c_i(j_star)


def test_apply_ck_matches_dense_matrix():
    """The structural ck_r product equals the dense [A | -I | 0 ; h^T | 0 | -1]."""
    scheme, pp, statement, _, _ = _fixture()
    R = scheme.Rq
    rng = random.Random(1)
    y = [[rng.randrange(R.q) for _ in range(PAR.d)] for _ in range(PAR.r_dim)]

    dense = []
    for i in range(PAR.n):
        row = list(pp["A"][i])
        for j in range(PAR.n):
            row.append(R.const(-1) if i == j else R.zero())
        row.append(R.zero())
        dense.append(row)
    row = list(statement.h_m) + [R.zero()] * PAR.n + [R.const(-1)]
    dense.append(row)

    assert statement.apply_ck(y) == R.mat_vec(dense, y)


def test_combine_c_matches_naive_sum():
    scheme, pp, statement, _, _ = _fixture()
    R = scheme.Rq
    rng = random.Random(2)
    coeffs = [[rng.randrange(R.q) for _ in range(PAR.d)] for _ in range(PAR.N)]

    naive = [R.zero() for _ in range(PAR.c_dim)]
    for i in range(PAR.N):
        ci = statement.c_i(i)
        naive = [R.add(naive[j], R.mul(coeffs[i], ci[j]))
                 for j in range(PAR.c_dim)]
    assert statement.combine_c(coeffs) == naive


# ---- Com invariants ------------------------------------------------------

def _commit(scheme, pp, statement, j_star, r, seed=b"c"):
    xof = XOF(DS_COMMIT, seed)
    return pp["oom"].com(statement, j_star, r, xof) + (xof,)


def test_com_selector_invariants():
    scheme, pp, statement, j_star, r = _fixture()
    _, st, _ = _commit(scheme, pp, statement, j_star, r)
    d, N = PAR.d, PAR.N

    # sum_i a_i = 0, because a_0 = -sum_{i>=1} a_i
    assert all(sum(st["a"][i][k] for i in range(N)) == 0 for k in range(d))

    # b is the unit vector at j*
    for i in range(N):
        assert st["b"][i] == ([1] + [0] * (d - 1) if i == j_star else [0] * d)

    # c_sel = a o (1 - 2b)
    for i in range(N):
        sign = -1 if i == j_star else 1
        assert st["c_sel"][i] == [sign * c for c in st["a"][i]]


def test_com_low_bits_bound():
    """||e_B||_inf <= 2^{K_b - 1}, used by the correctness proof."""
    scheme, pp, statement, j_star, r = _fixture()
    _, st, _ = _commit(scheme, pp, statement, j_star, r)
    assert _inf_int(st["e_B"]) <= 1 << (PAR.K_b - 1)


def test_com_high_low_split_reconstructs():
    """`\bar u_B = 2^{K_b} B + e_B`, coefficient by coefficient.

    On the **centred** representative, which is where `[[.]]_K` is defined
   .  `B` is therefore signed.
    """
    scheme, pp, statement, j_star, r = _fixture()
    t, st, _ = _commit(scheme, pp, statement, j_star, r)
    from ring import Ring
    Rqhat = Ring(PAR.q_hat, PAR.d)
    centred = Rqhat.vec_centered(st["u_B"])
    for i in range(PAR.n_hat):
        for k in range(PAR.d):
            assert (t["B"][i][k] * (1 << PAR.K_b) + st["e_B"][i][k]
                    == centred[i][k])
    assert any(c < 0 for poly in t["B"] for c in poly), \
        "high parts should be signed under the centred convention"


def test_com_E_definition():
    """E = ck_r y_OM - sum_i a_i c_i."""
    scheme, pp, statement, j_star, r = _fixture()
    t, st, _ = _commit(scheme, pp, statement, j_star, r)
    R = scheme.Rq
    a_q = [[c % PAR.q for c in ai] for ai in st["a"]]
    expected = [R.sub(lhs, rhs) for lhs, rhs in
                zip(statement.apply_ck(st["y_om"]), statement.combine_c(a_q))]
    assert t["E"] == expected


# ---- Prove / Ver ---------------------------------------------------------

def _full_proof(seed=b"\x31" * 32):
    scheme, pp, statement, j_star, r = _fixture(seed)
    ck_digest = b"\x00" * 32
    rho_digest = b"\x01" * 32
    for attempt in range(PAR.max_attempts):
        xof = XOF(DS_COMMIT, b"p", attempt.to_bytes(4, "little"))
        t, st = pp["oom"].com(statement, j_star, r, xof)
        pi = pp["oom"].prove(statement, j_star, r, t, st,
                             ck_digest, rho_digest, xof)
        if pi is not None:
            return scheme, pp, statement, pi, ck_digest, rho_digest, t
    raise AssertionError("no accepting attempt")


def test_prove_then_verify():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    assert pp["oom"].verify(statement, pi, ck, rho)


def test_sum_of_f_equals_challenge():
    """sum_i f_i = x, because sum b_i = 1 and sum a_i = 0."""
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    d = PAR.d
    head = list(pi["x"])
    for poly in pi["f1"]:
        head = [head[k] - poly[k] for k in range(d)]
    total = [head[k] + sum(p[k] for p in pi["f1"]) for k in range(d)]
    assert total == list(pi["x"])


def test_verifier_reconstructs_A():
    """A' = A whenever the compression margin held (Section 5 invariant)."""
    scheme, pp, statement, pi, ck, rho, t = _full_proof()
    d, N = PAR.d, PAR.N
    Rqh = pp["oom"].Rqhat
    head = list(pi["x"])
    for poly in pi["f1"]:
        head = [head[k] - poly[k] for k in range(d)]
    f = [head] + [list(p) for p in pi["f1"]]
    g = []
    for i in range(N):
        fi = [c % PAR.q_hat for c in f[i]]
        diff = Rqh.sub([c % PAR.q_hat for c in pi["x"]], fi)
        g.append(Rqh.centered(Rqh.mul(fi, diff)))
    u = pp["oom"]._reconstruct_u(pi["B"], pi["zb"], f, g, pi["x"])
    A_prime, _ = pp["oom"]._high_low(u, PAR.K_a)
    assert A_prime == t["A"]


def test_challenge_is_in_the_challenge_space():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    x = pi["x"]
    assert sum(1 for c in x if c != 0) == PAR.w
    assert max(abs(c) for c in x) <= PAR.gamma
    assert sum(abs(c) for c in x) <= PAR.w * PAR.gamma


def test_responses_meet_verifier_bounds():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    R = scheme.Rq
    assert _inf_int(pi["f1"]) <= PAR.f1_inf_bound
    assert _inf_int(pi["zb"]) <= PAR.zb_inf_bound
    z_c = [R.centered(p) for p in pi["z"]]
    assert _inf_int(z_c[:PAR.s_dim]) <= PAR.zs_inf_bound
    assert _inf_int(z_c[PAR.s_dim:]) <= PAR.zm_inf_bound
    norm_sq = sum(c * c for p in z_c for c in p)
    assert norm_sq <= PAR.z_l2_bound ** 2


# ---- rejection of bad proofs ---------------------------------------------

def test_verify_rejects_wrong_rho():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    assert not pp["oom"].verify(statement, pi, ck, b"\x02" * 32)


def test_verify_rejects_wrong_ck_digest():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    assert not pp["oom"].verify(statement, pi, b"\x99" * 32, rho)


def test_verify_rejects_tampered_challenge():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    bad = dict(pi, x=list(pi["x"]))
    idx = next(i for i, c in enumerate(bad["x"]) if c != 0)
    bad["x"][idx] = -bad["x"][idx]
    assert not pp["oom"].verify(statement, bad, ck, rho)


def test_verify_rejects_tampered_response():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    R = scheme.Rq
    bad = dict(pi, z=[list(p) for p in pi["z"]])
    bad["z"][0][0] = (bad["z"][0][0] + 1) % PAR.q
    assert not pp["oom"].verify(statement, bad, ck, rho)


def test_verify_rejects_tampered_f1():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    bad = dict(pi, f1=[list(p) for p in pi["f1"]])
    bad["f1"][0][0] += 1
    assert not pp["oom"].verify(statement, bad, ck, rho)


def test_verify_rejects_tampered_B():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    bad = dict(pi, B=[list(p) for p in pi["B"]])
    bad["B"][0][0] += 1
    assert not pp["oom"].verify(statement, bad, ck, rho)


def test_verify_rejects_oversized_response():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    bad = dict(pi, f1=[list(p) for p in pi["f1"]])
    bad["f1"][0][0] = int(PAR.f1_inf_bound) + 1
    assert not pp["oom"].verify(statement, bad, ck, rho)


def test_verify_rejects_wrong_shapes():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    assert not pp["oom"].verify(statement, dict(pi, f1=pi["f1"][:-1]), ck, rho)
    assert not pp["oom"].verify(statement, {}, ck, rho)


def test_verify_rejects_proof_for_a_different_statement():
    scheme, pp, statement, pi, ck, rho, _ = _full_proof()
    other = _fixture(b"\x32" * 32)[2]
    assert not pp["oom"].verify(other, pi, ck, rho)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_oom.py: {len(tests)} tests passed")
