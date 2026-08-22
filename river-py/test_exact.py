"""
test_exact.py -- Unit tests for the exact layer Pi_ex.
"""

import itertools
import json
import random
from fractions import Fraction

import exact

import pytest

from exact import (ExactParams, OpeningBackend, lanes_rank_roles,
                   RADIX_WEIGHTS, RADIX_DIGITS,
                   radix_decompose, radix_recompose, decompose_poly,
                   pack_witness, unpack_witness, padding_is_zero,
                   check_relation, get_backend)
import lanes_params as lp
from lanes_backend import WEIGHT_SUM
from params import TOY_PARAMS, get
from ring import negacyclic_mul_int
from sample import XOF, DS_EXACT

PAR = TOY_PARAMS
EX = ExactParams(PAR)


# ---- exact-backend parameters --------------------------------------------

def test_exact_dimensions_match_the_paper():
    """`(n~, l~, d~, w_hat, D) = (4, 4, 256, 44, 17)`, `l = 64`, `N_ex = 6`.

    Every one of these is printed by the paper, and the modulus
    with them -- so unlike the profile there is no repair here.  What
    the revision does *not* supply is the LANES sampler widths, response
    bounds, hint rules, or the field-by-field accounting behind its 13.5 KB;
    those gate `lanes_*` rather than this module.
    """
    assert (EX.d_tilde, EX.l_split, EX.n_tilde, EX.ell_tilde, EX.D,
            EX.N_ex, EX.w_hat) == (256, 64, 4, 4, 17, 6, 44)


def test_exact_params_check_passes_for_every_profile():
    for name in ("RiVeR-N8", "RiVeR-N16", "RiVeR-N64",
                 "RiVeR-N128", "RiVeR-N256"):
        ex = ExactParams(get(name))
        assert ex.check() == [], (name, ex.check())


def test_witness_scalar_count():
    """Six message elements of `d` coefficients each, one block apiece.

    The count that used to hold, `6 d == N_ex l`, is deliberately false
    now: the revision pads each block out to `l = 64`, so 192 scalars sit
    in 384 slots.
    """
    assert 1 + 1 + len(RADIX_WEIGHTS) == EX.N_ex == 6
    assert PAR.d * EX.N_ex == 192
    assert EX.N_ex * EX.l_split == 384
    assert EX.N_ex * EX.l_split != PAR.d * EX.N_ex


# ---- radix-3 range encoding ----------------------------------------------

def test_radix_covers_exactly_the_range():
    reachable = set()
    for a in range(RADIX_DIGITS):
        for b in range(RADIX_DIGITS):
            for c in range(RADIX_DIGITS):
                for e in range(RADIX_DIGITS):
                    reachable.add(radix_recompose((a, b, c, e)))
    assert reachable == set(range(PAR.q0))          # exactly [0, 60]
    assert 2 * sum(RADIX_WEIGHTS) == PAR.q0 - 1


def test_radix_decompose_round_trip():
    for value in range(PAR.q0):
        digits = radix_decompose(value)
        assert all(a in (0, 1, 2) for a in digits)
        assert radix_recompose(digits) == value


def test_radix_rejects_out_of_range():
    for bad in (-1, PAR.q0, 1000):
        try:
            radix_decompose(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_radix_encoding_is_not_injective():
    """17 = (0,0,0,1) = (2,2,1,0); harmless, and worth pinning down."""
    assert radix_recompose((0, 0, 0, 1)) == radix_recompose((2, 2, 1, 0)) == 17


def test_the_figures_use_the_centred_error():
    """`e_eval + 30 = sum_j g_j d_j` describes `[-30, 30]`.

    The relation, figures and parameter section all use that reading.
    The implementation follows them.
    """
    reachable = {radix_recompose(d)
                 for d in itertools.product(range(RADIX_DIGITS), repeat=4)}
    assert reachable == set(range(0, 61))

    centred = {value - WEIGHT_SUM for value in reachable}
    assert centred == set(range(-30, 31))
    for e_eval in centred:
        assert radix_recompose(radix_decompose(e_eval + WEIGHT_SUM)) \
            == e_eval + WEIGHT_SUM


def test_decompose_poly_shape():
    coeffs = [i % PAR.q0 for i in range(PAR.d)]
    digits = decompose_poly(coeffs)
    assert len(digits) == len(RADIX_WEIGHTS)
    assert all(len(p) == PAR.d for p in digits)
    for i in range(PAR.d):
        assert radix_recompose([p[i] for p in digits]) == coeffs[i]


# ---- witness packing -----------------------------------------------------

def test_pack_unpack_witness():
    rng = random.Random(1)
    e_eval = [rng.randrange(PAR.q0) - PAR.B_e for _ in range(PAR.d)]
    y_eval = [rng.randrange(-1000, 1000) for _ in range(PAR.d)]
    digits = decompose_poly([c + PAR.B_e for c in e_eval])
    message = pack_witness(EX, e_eval, y_eval, digits)
    assert len(message) == EX.N_ex
    assert all(len(p) == EX.d_tilde for p in message)
    scalars = unpack_witness(EX, message)
    # the paper's element order: (y_eval, e_eval, d_0 .. d_3)
    expected = ([c % EX.q_tilde for c in y_eval]
                + [c % EX.q_tilde for c in e_eval]
                + [c % EX.q_tilde for p in digits for c in p])
    assert scalars == expected


def test_commitment_shape_follows_lanes():
    """t_0 has n~ = 4 entries and the randomness rank is 17.

    The paper selects `(n~, l~, d~, w_hat, D) = (4, 4, 256, 44, 17)`
    with `l = 64`; the "+3" is `g` and the two product-proof commitments, and
    is the `alpha = 3` the revision's MLWE dimensions use:
    `(l~ + N_ex + alpha) d~ = 13 * 256 = 3328` samples against dimension
    `n~ d~ = 1024`, both of which the paper prints.
    """
    assert EX.kappa == EX.n_tilde + EX.ell_tilde + EX.N_ex + 3 == 17
    assert EX.n_tilde * EX.d_tilde == 1024                    # LWE dimension
    assert (EX.ell_tilde + EX.N_ex + EX.aux_slots) * EX.d_tilde == 3328


def test_the_rank_mapping_is_discriminating_at_unequal_ranks():
    """and the reason the previous version of this test was worthless.

    It computed the expected answer locally and compared it to itself: with
    `n~ == l~ == 4` every assertion holds under *either* reading, so
    reversing the production aliases in memory left it green.  A regression
    test that cannot fail is not one.

    So this drives the production helper -- the single place the mapping
    lives -- with the ranks that separate the two readings.  At `(7, 8)`
    the identity rank is 8 and the response rank is `24 - 8 = 16`; the
    reversed reading gives 7 and 17.
    """
    roles = lanes_rank_roles(7, 8, 6, 3)
    assert roles["kappa"] == 24
    assert roles["identity_rank"] == 8, "identity rank is l~, not n~"
    assert roles["tail_rank"] == 7, "the shared tail is n~, not l~"
    assert roles["response_rank"] == 16, "kappa - l~, not kappa - n~"

    # The reversed reading is a *different* number here, which is what
    # makes the assertions above bite.
    assert 24 - 7 == 17 != roles["response_rank"]

    # And the paper's two printed dimensions, which decide the assignment.
    assert roles["lwe_secret_rank"] == 7           # n~ d~ is the secret
    assert roles["lwe_sample_rank"] == 8 + 6 + 3   # (l~ + N_ex + alpha) d~


def test_exact_params_derives_the_roles_at_unequal_ranks():
    """The same, through `ExactParams` itself rather than the helper.

    Subclassing drives the real properties -- `t0_rows`, `kappa`,
    `response_rank` -- at ranks that tell the two readings apart, so this
    covers the derivation path and not only the function it calls.
    """
    class _Unequal(ExactParams):
        n_tilde = 7
        ell_tilde = 8

    ex = _Unequal(PAR)
    assert ex.t0_rows == 8, "t_0 has l~ rows"
    assert ex.kappa == 24
    assert ex.response_rank == 16, "the response drops the identity block"
    assert ex.roles["tail_rank"] == 7

    # The shipped profile still reads the same way, with both ranks equal.
    assert EX.t0_rows == EX.ell_tilde == 4
    assert EX.response_rank == EX.kappa - EX.ell_tilde == 13


def test_the_lanes_module_derives_from_the_same_helper():
    """`lanes_params` must not restate the mapping, only re-export it.

    Two different claims, and only one of them is numeric.

    The *values* below agree at the shipped ranks -- but so would a
    reversed alias, because `n~ == l~ == 4`.  Discrimination comes from
    the two tests above, which drive the helper and `ExactParams` at
    `(7, 8)`.  What this adds is the claim they cannot make: that
    `lanes_params` is **downstream** of the helper rather than a parallel
    copy of it.  A hand-written `IDENTITY_RANK = ELL_TILDE` would satisfy
    every numeric check here forever, so the derivation path is asserted
    on the source itself -- the one property that is textual rather than
    arithmetic.

    This runs even though `test_lanes.py` is skipped: `lanes_params` is
    importable regardless, and the gate window is precisely when the two
    could drift.
    """
    import inspect

    import lanes_params as LP

    source = inspect.getsource(LP)
    for name in ("IDENTITY_RANK", "TAIL_RANK", "RESPONSE_RANK"):
        assert f'{name} = _ROLES[' in source, (
            f"{name} must be derived from the shared helper, not restated; "
            "an alias is a second place the mapping can be wrong and at "
            "equal ranks nothing numeric can tell")

    roles = lanes_rank_roles(LP.N_TILDE, LP.ELL_TILDE, LP.N_EX, LP.AUX)
    assert LP.IDENTITY_RANK == roles["identity_rank"]
    assert LP.TAIL_RANK == roles["tail_rank"]
    assert LP.KAPPA == roles["kappa"] == EX.kappa
    assert LP.RESPONSE_RANK == roles["response_rank"] == EX.response_rank
    assert LP.N_LWE == roles["lwe_secret_rank"] * LP.DTILDE == 1024
    assert LP.M_LWE == roles["lwe_sample_rank"] * LP.DTILDE == 3328


def test_the_two_ranks_carry_the_roles_the_structure_gives_them():
    """The roles measured off a *constructed* key, not off the constants."""
    from exact import ExactCommitmentKey
    ck = ExactCommitmentKey(EX, b"\x21" * 32)

    assert len(ck.A1) == EX.t0_rows == EX.ell_tilde
    assert len(ck.A2) == EX.N_ex
    assert all(len(row) == EX.kappa for row in ck.A1)
    assert all(len(row) == EX.kappa for row in ck.A2)

    e_eval = [1] * PAR.d
    y_eval = [2] * PAR.d
    message = pack_witness(EX, e_eval, y_eval,
                           decompose_poly([c + PAR.B_e for c in e_eval]))
    randomness = [[0] * EX.d_tilde for _ in range(EX.kappa)]
    W = ck.commit(message, randomness)
    assert len(W["t0"]) == EX.ell_tilde
    assert len(W["t1"]) == EX.N_ex
    assert EX.response_rank == EX.kappa - EX.ell_tilde == 13


def test_exact_modulus_is_the_published_one():
    """`q~ = 67107713`: prime, 26 bits, and `2l+1 mod 4l` for `l = 64`."""
    from params import is_prime
    assert EX.q_tilde == 67107713
    assert is_prime(EX.q_tilde)
    assert EX.q_tilde.bit_length() == 26
    assert EX.q_tilde % 256 == 129
    assert EX.q_tilde % (4 * EX.l_split) == 2 * EX.l_split + 1


def test_exact_modulus_clears_only_because_the_range_is_centred():
    """The 0.56% margin is what the centred `B_e = 30` buys.

    With the literal `[0, 60]` bound the requirement doubles and the
    selected modulus fails outright, so the range shift is load-bearing
    rather than presentational.  Decided over the integers, because 0.56%
    is inside what a float `sqrt` chain can move.
    """
    assert EX.q_tilde_clears()
    assert not EX.q_tilde_clears(2 * PAR.B_e)
    margin = EX.q_tilde - EX.q_tilde_need
    assert round(margin, 2) == 376744.98
    assert 0.0056 < margin / EX.q_tilde < 0.0057


def test_packing_uses_slot_stride():
    """Scalars land at coefficient j * (d~ / l), mirroring NTT slots."""
    e_eval = [0] * PAR.d
    y_eval = [1] * PAR.d
    message = pack_witness(EX, e_eval, y_eval,
                           decompose_poly([c + PAR.B_e for c in e_eval]))
    first = message[0]                       # y_eval: element 0
    assert all(first[j * EX.slot_stride] == 1 for j in range(EX.block_used))
    assert all(first[k] == 0 for k in range(EX.d_tilde)
               if k % EX.slot_stride != 0)


def test_each_message_element_gets_its_own_padded_block():
    """Six blocks of `l = 64` slots, `d = 32` used and 32 zero-padded.

    `6 d = 192` scalars in `N_ex l = 384` slots: the old
    `6 d == N_ex l` identity is gone, and `192 != 384` is intentional.
    """
    assert (EX.block_slots, EX.block_used, EX.block_pad) == (64, 32, 32)
    assert EX.N_ex * EX.block_used == 192
    assert EX.N_ex * EX.block_slots == 384
    e_eval = [3] * PAR.d
    y_eval = [5] * PAR.d
    message = pack_witness(EX, e_eval, y_eval,
                           decompose_poly([c + PAR.B_e for c in e_eval]))
    assert padding_is_zero(EX, message)
    for poly in message:
        for j in range(EX.block_used, EX.block_slots):
            assert poly[j * EX.slot_stride] == 0


def test_padding_is_checked_not_assumed():
    """A nonzero padding slot is data the relation does not constrain."""
    e_eval = [3] * PAR.d
    y_eval = [5] * PAR.d
    message = pack_witness(EX, e_eval, y_eval,
                           decompose_poly([c + PAR.B_e for c in e_eval]))
    assert padding_is_zero(EX, message)
    tampered = [list(p) for p in message]
    tampered[2][EX.block_used * EX.slot_stride] = 1
    assert not padding_is_zero(EX, tampered)
    # ... and unpacking still returns only the carried scalars, so a
    # verifier that ignored the padding would not notice.
    assert unpack_witness(EX, tampered) == unpack_witness(EX, message)
    assert not padding_is_zero(EX, message[:-1])


def test_message_element_order_is_the_paper_s():
    """`(y_eval, e_eval, d_0 .. d_3)`, from the paper.

    Stated identically in the supplement's *Exact backend* paragraph and in
    the implementation appendix, and the one thing about the exact backend's
    witness that the paper is unambiguous about.  It fixes `W`, so getting it
    wrong is a different transcript for the same witness.
    """
    e_eval = [7] * PAR.d
    y_eval = [11] * PAR.d
    digits = decompose_poly([c + PAR.B_e for c in e_eval])
    message = pack_witness(EX, e_eval, y_eval, digits)
    at = lambda i: [message[i][j * EX.slot_stride]
                    for j in range(EX.block_used)]
    assert at(0) == y_eval
    assert at(1) == e_eval
    for j, weight in enumerate(RADIX_WEIGHTS):
        assert at(2 + j) == digits[j]
    assert sum(w * digits[j][0] for j, w in enumerate(RADIX_WEIGHTS)) \
        == 7 + PAR.B_e


# ---- relation ------------------------------------------------------------

def _synthetic(seed=2, par=PAR):
    """An honest (statement, witness) pair for R^_ex."""
    ex = ExactParams(par)
    rng = random.Random(seed)
    e_eval = [rng.randrange(par.q0) - par.B_e for _ in range(par.d)]
    y_eval = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(par.d)]
    x_c = [0] * par.d
    for pos in rng.sample(range(par.d), par.w):
        x_c[pos] = rng.choice([-1, 1]) * rng.randint(1, par.gamma)
    product = negacyclic_mul_int(x_c, e_eval)
    z = [product[i] + y_eval[i] for i in range(par.d)]
    statement = {"z_eval_centered": z, "x_centered": x_c}
    witness = {"e_eval": e_eval, "y_eval": y_eval,
               "digits": decompose_poly([c + par.B_e for c in e_eval])}
    return ex, statement, witness


def test_relation_accepts_honest_witness():
    ex, statement, witness = _synthetic()
    assert check_relation(ex, statement, witness) == []


def test_relation_rejects_out_of_range_error():
    ex, statement, witness = _synthetic()
    witness["e_eval"] = list(witness["e_eval"])
    witness["e_eval"][0] = PAR.B_e + 1       # one past the centred range
    assert any("outside" in e for e in check_relation(ex, statement, witness))


def test_relation_rejects_non_ternary_digits():
    ex, statement, witness = _synthetic()
    witness["digits"] = [list(p) for p in witness["digits"]]
    witness["digits"][0][0] = 3
    assert any("ternary" in e for e in check_relation(ex, statement, witness))


def test_relation_rejects_bad_reconstruction():
    ex, statement, witness = _synthetic()
    digits = [list(p) for p in witness["digits"]]
    digits[0][0] = (digits[0][0] + 1) % 3
    witness["digits"] = digits
    errors = check_relation(ex, statement, witness)
    assert any("reconstruction" in e for e in errors) or \
        any("ternary" in e for e in errors)


def test_relation_rejects_broken_link_equation():
    ex, statement, witness = _synthetic()
    statement = dict(statement)
    statement["z_eval_centered"] = list(statement["z_eval_centered"])
    statement["z_eval_centered"][0] += 1
    assert any("z_eval" in e for e in check_relation(ex, statement, witness))


def test_relation_is_checked_over_Z_not_modulo_q_tilde():
    """Adding q~ to a y_eval coefficient must break the relation.

    The commitment reduces the witness mod q~, so it cannot tell these lifts
    apart; only an integer link equation can.
    """
    ex, statement, witness = _synthetic()
    assert check_relation(ex, statement, witness) == []
    lifted = dict(witness, y_eval=list(witness["y_eval"]))
    lifted["y_eval"][0] += ex.q_tilde
    errors = check_relation(ex, statement, lifted)
    assert any("over Z" in e for e in errors), errors


def test_backend_rejects_a_lifted_y_eval():
    backend, pp, statement, sigma = _backend_run()
    assert backend.verify(pp, statement, sigma)
    bad = dict(sigma, y_eval=list(sigma["y_eval"]))
    bad["y_eval"][0] += backend.ex.q_tilde
    assert not backend.verify(pp, statement, bad)


# ---- backend -------------------------------------------------------------

def _backend_run(par=PAR, seed=3):
    backend = OpeningBackend(par)
    pp = backend.setup(par, b"\x05" * 32)
    ex, statement, witness = _synthetic(seed, par)
    w_in = {"e_eval": witness["e_eval"], "y_eval": witness["y_eval"]}
    W, st = backend.com(pp, w_in, XOF(DS_EXACT, b"seed", bytes([seed])))
    statement = dict(statement, W=W)
    sigma = backend.prove(pp, statement, w_in, st)
    return backend, pp, statement, sigma


def test_backend_accepts_honest_proof():
    backend, pp, statement, sigma = _backend_run()
    assert backend.verify(pp, statement, sigma)


def test_backend_commitment_is_deterministic_in_the_xof():
    backend = OpeningBackend(PAR)
    pp = backend.setup(PAR, b"\x05" * 32)
    _, _, witness = _synthetic()
    w_in = {"e_eval": witness["e_eval"], "y_eval": witness["y_eval"]}
    W1, _ = backend.com(pp, w_in, XOF(DS_EXACT, b"fixed"))
    W2, _ = backend.com(pp, w_in, XOF(DS_EXACT, b"fixed"))
    W3, _ = backend.com(pp, w_in, XOF(DS_EXACT, b"other"))
    assert W1 == W2 and W1 != W3


def test_backend_rejects_tampered_opening():
    backend, pp, statement, sigma = _backend_run()
    bad = dict(sigma, e_eval=list(sigma["e_eval"]))
    bad["e_eval"][0] += 1
    assert not backend.verify(pp, statement, bad)


def test_backend_rejects_tampered_randomness():
    backend, pp, statement, sigma = _backend_run()
    bad = dict(sigma, randomness=[list(p) for p in sigma["randomness"]])
    bad["randomness"][0][0] = (bad["randomness"][0][0] + 1) % \
        backend.ex.q_tilde
    assert not backend.verify(pp, statement, bad)


def test_backend_rejects_wrong_statement():
    backend, pp, statement, sigma = _backend_run()
    bad = dict(statement, z_eval_centered=list(statement["z_eval_centered"]))
    bad["z_eval_centered"][0] += 1
    assert not backend.verify(pp, bad, sigma)


def test_backend_rejects_wrong_commitment():
    backend, pp, statement, sigma = _backend_run()
    other = _backend_run(seed=9)[2]["W"]
    assert not backend.verify(pp, dict(statement, W=other), sigma)


def test_backend_rejects_malformed_proof():
    backend, pp, statement, _ = _backend_run()
    assert not backend.verify(pp, statement, {})


def test_backend_proof_encoding_round_trip():
    backend, pp, statement, sigma = _backend_run()
    blob = backend.proof_encode({"W": statement["W"], "sigma": sigma})
    assert len(blob) <= backend.proof_bytes    # Rice: variable
    decoded = backend.proof_decode(blob)
    assert decoded["W"] == statement["W"]
    assert backend.verify(pp, dict(statement, W=decoded["W"]),
                          decoded["sigma"])


def test_backend_lookup():
    assert isinstance(get_backend("opening", PAR), OpeningBackend)
    try:
        get_backend("nope", PAR)
    except KeyError:
        return
    raise AssertionError("unknown backend accepted")


def test_mock_opening_is_smaller_than_the_real_proof():
    """The mock's size says nothing about LANES's, and undershoots it.

    The paper quotes a fixed `|pi_ex| = 13.5 KB` for every
    profile .  Once
    the opening is entropy coded this backend comes in *below* that,
    because what it transmits is mostly ternary commitment randomness at 2
    bits a coefficient -- cheap to send precisely because it is the
    witness.  A zero-knowledge proof has to send masked Gaussians instead,
    which cost an order of magnitude more per coefficient.  So a small
    `|pi_ex|` here is evidence of the leak, not of efficiency.

    It did grow with the new dimensions -- `d~` doubled to 256 and the
    randomness rank is 17 blocks of it -- from about 6.8 KB to about 9.3,
    which is still 30% under the paper's model.

That last claim used to rest on the model alone.  It does not any more:
    the LANES layer runs at the paper's own widths under
    `exact_backend="lanes-experimental"`, so there is a real proof to be
    smaller *than*.  `test_lanes.py` pins its measured size field by field
    (about 13.9 KB); what is checked here is the comparison the docstring
    makes, against the one figure the two backends state the same way --
    `proof_bytes`, the serializer's worst case.
    """
    from params import RiVeRParams
    par = get("RiVeR-N8")
    backend = OpeningBackend(par)
    kb = backend.proof_bytes / 1024
    assert 5.0 < kb < RiVeRParams.EXACT_PROOF_KB, kb
    assert 9.0 < kb < 10.0, kb

    # ...and against the real one, when it is available.
    #
    # `proof_bytes` is the *worst case* on both sides, which is the only
    # figure they express the same way: `opening` is fixed-width, so its
    # worst case is its size, while LANES Rice-codes `z` and comes in well
    # under (13.9 KB measured against 19.0 worst).  Comparing worst to
    # worst understates the gap rather than overstating it, which is the
    # safe direction for a claim that the mock is smaller.
    try:
        from lanes_backend import LanesBackend
        lanes = LanesBackend.experimental(par)
    except Exception:                              # pragma: no cover
        return
    worst_kb = lanes.proof_bytes / 1024
    assert worst_kb > kb, (kb, worst_kb)
    # The mock is under half the zero-knowledge layer's worst case, and
    # under its *measured* size too -- 9.36 against about 13.9.
    assert kb < 0.6 * worst_kb, (kb, worst_kb)
    assert kb < RiVeRParams.EXACT_PROOF_KB, kb


# --------------------------------------------------------------------------

def test_z_no_test_left_a_module_constant_moved():
    """Nothing above may leave a patched constant behind (runs last).

    Tests here move constants in `lanes_params` to prove a check bites.
    The trap is that other modules **bind those names by value at import**,
    and an import that happens inside a patched window captures the moved
    value for the life of the process -- which no `finally` can undo.  It
    happened once.  So this walks every uppercase name the
    two modules share and requires them to still agree.  It sorts and sits
    last on purpose; a failure names the constant, not the test that moved
    it, but it turns a silent poisoning into a red line.
    """
    import lanes_backend as _lb
    import lanes_params as _lp
    for name in dir(_lp):
        if not name.isupper() or not hasattr(_lb, name):
            continue
        assert getattr(_lb, name) == getattr(_lp, name), \
            f"{name} disagrees between lanes_params and lanes_backend"

# ---- the manifest and the provenance audit ------------------------------

def _valid_manifest(live=None, status="final"):
    """A manifest that passes every check, built from the shipped one.

    Derived from `lanes_manifest.json` rather than hand-written: the
    shipped manifest is the shape the validator actually meets, so a
    hand-made fixture drifts from it the moment a section is added -- as
    happened when `dimensions` arrived.

    `status` defaults to `"final"` because most tests here are about what
    happens *after* the status check; the shipped manifest is
    `"experimental"` and can never open the production name.
    """
    import copy


    if live is None:
        live, _ = exact.live_lanes_constants()
    blob = copy.deepcopy(exact.LANES_PARAMETER_MANIFEST)
    assert blob is not None, "the shipped manifest is missing"
    blob["status"] = status
    blob["constants"] = {
        name: {"value": str(value), "provenance": "Repair"}
        for name, (_, value) in live.items()
    }
    return blob


def test_the_gate_cannot_lift_while_a_constant_has_drifted():
    """No *active* constant may have drifted from the paper's closed form.

    The audit is executable rather than narrated: the live values are the
    paper's, and what it enforces is that none has drifted away from them.
    """
    live, missing = exact.live_lanes_constants()
    assert not missing, missing
    assert len(live) == len(exact.GATED_LANES_CONSTANTS)

    # The gate is closed today, and stays closed under every state that
    # falls short of the full list.
    assert exact.lanes_unavailable_reason() is not None

    # With no manifest, the reason must name each gated constant, so the
    # audit and the message cannot drift apart.
    with_none = exact.lanes_unavailable_reason(manifest=None)
    assert with_none is not None
    for name in live:
        assert name in with_none, (
            f"the gate's reason does not name {name}, so the audit and the "
            "message can drift apart")


def test_the_audit_pins_every_gated_value_not_just_its_name():
    """Finding 4: a renamed, deleted *or changed* constant must show up.

    A list of names alone shrinks silently when one is renamed away, and
    says nothing when one is quietly given a different value.  Both are
    pinned.
    """
    live, missing = exact.live_lanes_constants()
    assert not missing

    for name in exact.LANES_MANIFEST_INPUTS:
        assert name in exact.PAPER_LANES_VALUES

    # Every live value is the paper's closed form.
    for name, (_, value) in live.items():
        assert value == exact.PAPER_LANES_VALUES[name], name

    # ...and the manifest selects each one by value, not by label.
    manifest = exact.LANES_PARAMETER_MANIFEST
    assert manifest is not None
    for name, (_, value) in live.items():
        entry = manifest["constants"][name]
        assert Fraction(entry["value"]) == value, name
        assert entry["provenance"] in exact.PROVENANCE_LABELS

    # A live value that moved away from the paper's is `constant-changed`,
    # whether or not a manifest is present.
    import lanes_params as LP
    saved = LP.SIGMA_Y
    try:
        LP.SIGMA_Y = saved + 1
        assert exact.lanes_gate_cause(manifest=None) == "constant-changed"
        assert "SIGMA_Y" in exact.lanes_unavailable_reason(manifest=None)
    finally:
        LP.SIGMA_Y = saved

    # A vanished name is an error, not a smaller audit.
    saved = LP.Z_INF_BOUND
    try:
        delattr(LP, "Z_INF_BOUND")
        _, missing = exact.live_lanes_constants()
        assert missing == ["Z_INF_BOUND not found"]
        assert "drifted" in exact.lanes_unavailable_reason()
    finally:
        LP.Z_INF_BOUND = saved


    # A changed *live* value trips the paper-conformance check, which runs
    # first because it is the more fundamental failure: the code no longer
    # implements the published closed form.
    saved = LP.Z_NORM2_BOUND
    try:
        LP.Z_NORM2_BOUND = saved + 1
        assert exact.lanes_gate_cause() == "constant-changed"
        drift = exact.lanes_unavailable_reason()
        assert "does not match the paper's closed form" in drift
        assert f"Z_NORM2_BOUND = {saved + 1}, expected {saved}" in drift
        # ...and with no manifest at all, the same finding, because it is
        # about the code and not about the table
        assert exact.lanes_gate_cause(manifest=None) == "constant-changed"
    finally:
        LP.Z_NORM2_BOUND = saved

    # A *stale manifest* against correct code is the other direction, and
    # the realistic one: regenerate `lanes_params`, forget
    # `make lanes-manifest-regen`.  It is caught by value, naming both
    # sides -- what the code consumes and what was frozen.
    stale = json.loads(json.dumps(exact.LANES_PARAMETER_MANIFEST))
    stale["constants"]["Z_NORM2_BOUND"]["value"] = int(saved) - 1
    reason = exact.lanes_unavailable_reason(manifest=stale)
    assert (f"the manifest selects {int(saved) - 1} but the code consumes "
            f"{saved}") in reason
    assert exact.lanes_gate_cause(manifest=stale) == "manifest-invalid"


def test_the_gate_is_readiness_not_a_dimension_diff():
    """Moving a dimension must not make the LANES backend available."""
    baseline = exact.lanes_unavailable_reason()
    assert baseline is not None
    for attr, value in (("d_tilde", 128), ("l_split", 32),
                        ("q_tilde", 427634113), ("n_tilde", 7),
                        ("ell_tilde", 8), ("D", 13)):
        moved = exact.ExactParams(TOY_PARAMS)
        object.__setattr__(moved, attr, value)
        assert exact.lanes_unavailable_reason() == baseline, attr
        del moved


def test_an_empty_checklist_is_not_a_manifest():
    """Finding 1: section *names* are not section *data*.

    The predecessor of this gate accepted `{}` for every required section,
    so eight empty dictionaries lifted it -- a checklist that contained
    none of the information it was supposed to certify.
    """
    live, _ = exact.live_lanes_constants()

    hollow = {k: {} for k in exact.LANES_MANIFEST_SECTIONS}
    hollow["constants"] = {
        n: {"value": v, "provenance": "Paper"} for n, (_, v) in live.items()
    }
    bad = exact.validate_lanes_manifest(hollow, live)
    assert bad, "an empty checklist was accepted as a manifest"
    for section in exact.LANES_MANIFEST_SECTIONS:
        assert any(section in b for b in bad), section

    # And every individual key inside a section is required too.
    for section, keys in exact.LANES_MANIFEST_SECTIONS.items():
        for key in keys:
            short = _valid_manifest(live)
            del short[section][key]
            bad = exact.validate_lanes_manifest(short, live)
            assert any(f"{section}.{key}" in b for b in bad), (section, key)
            # present but empty is the same failure -- in every shape an
            # empty value comes in, because the sections carry a mix of
            # strings, lists and mappings and only `""` used to be caught
            for empty in ("", [], {}, (), None):
                blank = _valid_manifest(live)
                blank[section][key] = empty
                bad = exact.validate_lanes_manifest(blank, live)
                assert any(f"{section}.{key}" in b for b in bad), (
                    section, key, empty)


def test_re_selection_compares_values_not_just_labels():
    """Finding 3: a label is not a selection.

    Marking a wrong width as **Paper** does not make the paper have
    printed it.  What the gate compares is the manifest's value against the
    value the code actually consumes, so a manifest that selects a
    *different* value reports that the code has not been updated -- which
    is the useful failure, not a silent pass.
    """
    live, _ = exact.live_lanes_constants()

    good = _valid_manifest(live)
    assert exact.validate_lanes_manifest(good, live) == []

    # a value the code does not consume, relabelled Paper, which is the
    # exact failure mode a label check alone would miss
    wrong = int(exact.PAPER_LANES_VALUES["Z_INF_BOUND"]) * 2
    moved = _valid_manifest(live)
    moved["constants"]["Z_INF_BOUND"] = {"value": int(wrong),
                                         "provenance": "Paper"}
    bad = exact.validate_lanes_manifest(moved, live)
    assert any(f"selects {int(wrong)} but the code consumes "
               f"{lp.Z_INF_BOUND}" in b for b in bad), bad

    # a label with no value behind it
    bare = _valid_manifest(live)
    bare["constants"]["Z_NORM2_BOUND"] = {"provenance": "Paper"}
    assert any("carries no value" in b for b in
               exact.validate_lanes_manifest(bare, live))

    # a value with no label
    unlabelled = _valid_manifest(live)
    unlabelled["constants"]["Z_NORM2_BOUND"] = {"value": int(lp.Z_NORM2_BOUND)}
    assert any("provenance" in b for b in
               exact.validate_lanes_manifest(unlabelled, live))

    # a constant it does not mention at all
    partial = _valid_manifest(live)
    del partial["constants"]["SIGMA_R"]
    assert any("does not select" in b for b in
               exact.validate_lanes_manifest(partial, live))



def test_the_wire_total_must_reproduce_or_record_the_stated_size():
    """`wire.total_bits` reproduces 13.5 KB, or says it cannot."""
    live, _ = exact.live_lanes_constants()

    # A total that misses the stated figure has to be recorded as such.
    silent = _valid_manifest(live)
    silent["wire"]["total_bits"] = {"value": 100000, "provenance": "Repair"}
    del silent["wire"]["discrepancy"]
    bad = exact.validate_lanes_manifest(silent, live)
    assert any("stated 13.5 KB" in b for b in bad), bad

    # A total that hits it needs no discrepancy.
    exact_size = _valid_manifest(live)
    del exact_size["wire"]["discrepancy"]
    exact_size["wire"]["total_bits"] = {
        "value": int(exact.LANES_STATED_KB * 8192), "provenance": "Repair"}
    assert exact.validate_lanes_manifest(exact_size, live) == []

    # And *no* total is allowed only when the manifest says why -- a
    # Rice-coded layout has no fixed size, which is a fact about the
    # format, but a manifest with no size accounting at all reads the same
    # way unless it is written down.  The shipped one does write it down.
    quiet = _valid_manifest(live)
    del quiet["wire"]["discrepancy"]
    quiet["wire"]["total_bits"] = {"value": None, "provenance": "Repair"}
    bad = exact.validate_lanes_manifest(quiet, live)
    assert any("no size accounting" in b for b in bad), bad
    assert exact.validate_lanes_manifest(_valid_manifest(live), live,
                                         ) == []


def test_the_gate_names_the_first_blocker_not_the_last():
    """The token vocabulary the KAT records, and the order it resolves in.

    The order matters: a later blocker must not hide an earlier one.  A
    tree whose gated constants have been quietly retuned should say so,
    not report the missing manifest and let the retune travel with it.

    These tokens are what `river-rs` compares against; the prose reason
    names each language's own API and cannot be compared across the two.
    """
    live, _ = exact.live_lanes_constants()
    final = _valid_manifest(live, status="final")
    experimental = _valid_manifest(live, status="experimental")
    evidence = {"estimator": "placeholder", "verdict": "below-target",
                "best_bits": 126.1, "target_bits": 128}

    cause = exact.lanes_gate_cause

    # no manifest, and the constants are the paper's: nothing has moved,
    # so the missing manifest itself is what is reported
    assert cause(None, backend_ready=False) == "no-parameter-manifest"
    assert cause(None, backend_ready=True) == "no-parameter-manifest"

    # a manifest, then each remaining blocker in turn
    assert cause(experimental, True, evidence) == "manifest-experimental"
    assert cause(final, backend_ready=False, evidence=None) == \
        "no-security-evidence"
    # evidence that exists but does not reach the target is its own state
    assert cause(final, backend_ready=True, evidence=evidence) == \
        "security-evidence-pending"
    passing = dict(evidence, verdict="meets-target")
    assert cause(final, backend_ready=False, evidence=passing) == \
        "backend-not-ready"
    assert cause(final, backend_ready=True, evidence=passing) is None

    bad = dict(final)
    bad["rank_roles"] = dict(bad["rank_roles"], kappa=0)
    assert cause(bad, True, evidence) == "manifest-invalid"

    # What the tree actually ships: a valid, *final* manifest -- the
    # parameters are the paper's, nothing is searched -- an implementation
    # that has passed its gate, and security evidence that does not settle the
    # question.  **One** outstanding condition, which is the point of
    # keeping the flags separate: a gate closed for a reason that is no
    # longer true cannot be told from one closed for a reason that is.
    assert cause() == "security-evidence-pending"
    assert exact.LANES_PARAMETER_MANIFEST["status"] == "final"
    assert exact.LANES_BACKEND_READY is True
    assert exact.LANES_SECURITY_EVIDENCE["verdict"] == "below-target"

    # ...and it is the *only* one: flipping the security flag alone opens
    # the gate, so nothing else is quietly holding it shut.
    passing_now = dict(exact.LANES_SECURITY_EVIDENCE, verdict="meets-target")
    assert exact.lanes_gate_cause(evidence=passing_now) is None

    # a value the manifest does not select outranks everything after it
    import lanes_params as LP
    saved = LP.Z_INF_BOUND
    try:
        LP.Z_INF_BOUND = saved + 1
        # The live value left the paper's closed form, so that is the
        # finding, manifest or no manifest.
        assert cause(final, True, evidence) == "constant-changed"
        assert cause(None, True, evidence) == "constant-changed"
    finally:
        LP.Z_INF_BOUND = saved

    # A stale *manifest* against correct code is the other direction.
    stale = json.loads(json.dumps(final))
    stale["constants"]["Z_INF_BOUND"]["value"] = int(LP.Z_INF_BOUND) - 1
    assert cause(stale, True, evidence) == "manifest-invalid"


    # ...as does a shrunken audit, which Python can reach and Rust cannot:
    # there `live_lanes_constants` names the module's constants at compile
    # time, so a rename is a build failure rather than a smaller audit
    saved_names = exact.GATED_LANES_CONSTANTS
    try:
        exact.GATED_LANES_CONSTANTS = saved_names + (
            ("NOT_A_CONSTANT", "invented"),)
        assert cause(final, True, evidence) == "audit-drift"
    finally:
        exact.GATED_LANES_CONSTANTS = saved_names

    for token in (cause(None, False), cause(experimental, True, evidence),
                  cause(final, False, evidence), cause(bad, True, evidence)):
        assert token in exact.LANES_GATE_CAUSES


def test_a_manifest_alone_does_not_enable_the_backend():
    """The states are separate, and every one of them is required.

    Possession of a parameter table is not by itself a reason to lift the
    runtime gate.  With one flag it would have been: if nothing consumed
    the manifest, the backend would run with a table of the right numbers
    sitting unused beside it.

    There are now four, because a table is not evidence either: the
    manifest must claim to be final, security evidence must exist, and the
    implementation must have passed its gates.  A schema-complete manifest
    satisfies a validator; it does not show that an estimator was run.
    """
    live, _ = exact.live_lanes_constants()
    good = _valid_manifest(live, status="final")
    evidence = {"estimator": "placeholder", "verdict": "meets-target",
                "best_bits": 128.4, "target_bits": 128}

    # valid final manifest, evidence present, implementation not ready
    reason = exact.lanes_unavailable_reason(good, backend_ready=False,
                                            evidence=evidence)
    assert reason is not None
    assert "has not passed its own gate" in reason \
        and "LANES_BACKEND_READY" in reason

    # valid final manifest, ready asserted, but no security evidence
    reason = exact.lanes_unavailable_reason(good, backend_ready=True,
                                            evidence=None)
    assert reason is not None
    assert "LANES_SECURITY_EVIDENCE" in reason and "estimator" in reason

    # everything, but the manifest does not claim to be final
    experimental = _valid_manifest(live, status="experimental")
    reason = exact.lanes_unavailable_reason(experimental, backend_ready=True,
                                            evidence=evidence)
    assert reason is not None
    assert "'final'" in reason and "lanes-experimental" in reason

    # ready asserted without a manifest -> still unavailable
    assert exact.lanes_unavailable_reason(None, backend_ready=True,
                                          evidence=evidence) is not None

    # all four -> available
    assert exact.lanes_unavailable_reason(good, backend_ready=True,
                                          evidence=evidence) is None

    # ...and what this tree actually ships is not that: the manifest is
    # present, valid and final, the implementation has passed its gates,
    # and the security evidence is the one thing outstanding.
    assert exact.LANES_PARAMETER_MANIFEST is not None
    assert exact.LANES_PARAMETER_MANIFEST["status"] == "final"
    assert exact.LANES_BACKEND_READY is True
    assert exact.LANES_SECURITY_EVIDENCE["verdict"] == "below-target"
    shipped = exact.lanes_unavailable_reason()
    assert shipped is not None
    assert "security evidence is pending" in shipped
    assert "has not passed its own gate" not in shipped, \
        "the stale readiness reason is gone"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_exact.py: {len(tests)} tests passed")
