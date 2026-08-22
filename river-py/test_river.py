"""
test_river.py -- Unit tests for setup, key generation, CanonPad and the
scheme-level checks in Verify.
"""

from ring import round_p, rounding_error
from params import TOY_PARAMS
from river import RiVeR

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAR = TOY_PARAMS
SEED = b"\x40" * 32


def _scheme(n_keys=None):
    """A scheme, public parameters, and `n_keys` fresh key pairs.

    The default is `N`, because a ring is exactly `N` keys:
    there is no padding and no canonical reordering.
    """
    scheme = RiVeR(PAR)
    pp = scheme.setup(SEED)
    if n_keys is None:
        n_keys = PAR.N
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31) for i in range(n_keys)]
    return scheme, pp, keys


# ---- setup ---------------------------------------------------------------

def test_setup_is_deterministic():
    a = RiVeR(PAR).setup(SEED)
    b = RiVeR(PAR).setup(SEED)
    assert a["rho"] == b["rho"]
    assert a["A"] == b["A"]


def test_setup_seed_changes_everything():
    a = RiVeR(PAR).setup(SEED)
    b = RiVeR(PAR).setup(b"\x41" * 32)
    assert a["rho"] != b["rho"] and a["A"] != b["A"]


def test_matrix_A_has_the_right_shape():
    _, pp, _ = _scheme()
    assert len(pp["A"]) == PAR.n
    assert all(len(row) == PAR.ell for row in pp["A"])
    assert all(0 <= c < PAR.q for row in pp["A"] for p in row for c in p)


# ---- key generation ------------------------------------------------------

def test_keygen_is_deterministic():
    scheme, pp, _ = _scheme()
    a = scheme.keygen(pp, b"\x07" * 32)
    b = scheme.keygen(pp, b"\x07" * 32)
    assert a == b


def test_secret_key_is_ternary():
    scheme, pp, keys = _scheme()
    sk, _ = keys[0]
    centered = [scheme.Rq.centered(p) for p in sk]
    assert all(abs(c) <= PAR.beta for p in centered for c in p)
    assert len(sk) == PAR.ell


def test_public_key_is_the_rounded_product():
    """t = floor(A s)_p, and the rounding error is canonical."""
    scheme, pp, keys = _scheme()
    sk, pk = keys[0]
    As = scheme.Rq.mat_vec(pp["A"], sk)
    assert pk == [round_p(scheme.Rq, row, PAR.q0) for row in As]
    for i in range(PAR.n):
        e = rounding_error(scheme.Rq, As[i], pk[i], PAR.q0)
        assert all(0 <= c <= PAR.q0 - 1 for c in e)


def test_public_key_lives_in_Rp():
    scheme, pp, keys = _scheme()
    _, pk = keys[0]
    assert len(pk) == PAR.n
    assert all(0 <= c < PAR.p for p in pk for c in p)


# ---- G -------------------------------------------------------------------

def test_hash_message_shape_and_determinism():
    scheme, _, _ = _scheme()
    h = scheme.hash_message(b"m")
    assert len(h) == PAR.ell
    assert all(0 <= c < PAR.q for p in h for c in p)
    assert h == scheme.hash_message(b"m")
    assert h != scheme.hash_message(b"m2")


# ---- ring admissibility --------------------------------------------------
#
# `CanonPad` is gone.  A ring is an ordered tuple of exactly `N` distinct,
# structurally valid public keys, and both `Eval` and `Verify` enforce that
# so the two hash the same admissible domain.  See `RiVeR.validate_ring`.

def test_ring_must_be_exactly_N_keys():
    scheme, pp, keys = _scheme(n_keys=PAR.N + 1)
    ring = [pk for _, pk in keys]
    assert len(scheme.validate_ring(ring[:PAR.N])) == PAR.N
    for bad in (ring[:PAR.N - 1], ring, []):
        try:
            scheme.validate_ring(bad)
        except ValueError:
            continue
        raise AssertionError(f"ring of size {len(bad)} accepted")


def test_ring_order_is_part_of_the_statement():
    """The ring is no longer canonically reordered, so two rings with the
    same members in a different order are different statements and hash
    differently."""
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    reversed_ring = list(reversed(ring))
    assert scheme.validate_ring(ring) == ring
    assert scheme.validate_ring(reversed_ring) == reversed_ring

    from codec import ring_digest
    v = [0] * PAR.d
    assert (ring_digest(scheme.codec, ring, v)
            != ring_digest(scheme.codec, reversed_ring, v))


def test_ring_order_changes_the_proof_but_not_the_value():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    v_a, pi_a = scheme.eval(pp, pk, sk, ring, b"m", b"\x07" * 32)
    v_b, pi_b = scheme.eval(pp, pk, sk, list(reversed(ring)), b"m",
                            b"\x07" * 32)
    assert v_a == v_b                       # the value is ring-independent
    assert pi_a["oom"]["z"] != pi_b["oom"]["z"]
    assert scheme.verify(pp, ring, b"m", v_a, pi_a)
    assert scheme.verify(pp, list(reversed(ring)), b"m", v_b, pi_b)
    # ... and neither proof transfers to the other ordering.
    assert not scheme.verify(pp, list(reversed(ring)), b"m", v_a, pi_a)


def test_ring_admits_duplicates():
    """the paper: "we ... do not require its entries to be distinct".

    This tree rejected duplicates in the paper, which said `Eval`
    used "the unique index" while the preliminaries admitted any tuple.  The
    revision resolves it the other way, so the rejection is gone -- and a
    duplicated ring must now prove and verify end to end, not merely parse.
    """
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    ring[2] = ring[0]
    assert scheme.validate_ring(ring) == ring

    sk, pk = keys[0]
    v, pi = scheme.eval(pp, pk, sk, ring, b"m", b"\x07" * 32)
    assert scheme.verify(pp, ring, b"m", v, pi)


def test_a_duplicated_evaluator_key_uses_the_first_occurrence():
    """`j* = min{j in [N] : t_j = pk}`.

    Which position is used is not observable from the proof -- that is what
    the OOM layer hides -- so this pins `ring_index` directly, and pins that
    the proof is the *same* one an unduplicated ring at position 0 gives.
    Both rings put the evaluator's key at index 0; the second merely repeats
    it later, and the paper says a repeat is not a distinct identity.
    """
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    dup = list(ring)
    dup[2] = ring[0]
    assert scheme.ring_index(scheme.validate_ring(dup), ring[0]) == 0

    # ... and index 2 is genuinely occupied by the same key, so "first"
    # is a choice being made, not the only occurrence.
    assert dup[2] == dup[0]


def test_a_duplicated_key_shrinks_the_anonymity_set():
    """The cost of admitting duplicates, stated as a test rather than prose.

    The paper says repeated occurrences "do not represent distinct key
    identities".  A ring of `N` positions carrying `k` copies of one key
    therefore hides its evaluator among `N - k + 1` identities, not `N`.
    Nothing rejects this -- the paper defines it as admissible -- but a
    caller building a ring should know the number it is getting.
    """
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    assert len({scheme.codec.pk_encode(pk) for pk in ring}) == PAR.N

    dup = list(ring)
    dup[2] = dup[3] = ring[0]           # three copies of one key
    scheme.validate_ring(dup)
    identities = len({scheme.codec.pk_encode(pk) for pk in dup})
    assert identities == PAR.N - 2


def test_ring_rejects_malformed_keys():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    for bad in (b"nope", [], [[0] * PAR.d] * (PAR.n + 1),
                [[0] * (PAR.d + 1)] * PAR.n,
                [[PAR.p] + [0] * (PAR.d - 1)] * PAR.n):
        try:
            scheme.validate_ring(ring[:-1] + [bad])
        except ValueError:
            continue
        raise AssertionError(f"malformed key accepted: {bad!r:.40}")


def test_eval_rejects_a_ring_without_the_evaluator():
    scheme, pp, keys = _scheme(n_keys=PAR.N + 1)
    ring = [pk for _, pk in keys[:PAR.N]]
    sk_out, pk_out = keys[PAR.N]
    try:
        scheme.eval(pp, pk_out, sk_out, ring, b"m", b"\x02" * 32)
    except ValueError:
        return
    raise AssertionError("non-member evaluated")


def test_there_are_no_dummy_slots_left():
    """The zero-witness forgery is dissolved rather than patched: every ring
    position is a caller-supplied key, so no position has a public opening."""
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    assert scheme.validate_ring(ring) == ring
    assert not hasattr(scheme, "canon_pad")


def test_ring_index_finds_the_signer():
    scheme, pp, keys = _scheme()
    ring = scheme.validate_ring([pk for _, pk in keys])
    for _, pk in keys:
        assert ring[scheme.ring_index(ring, pk)] == pk


def test_ring_index_rejects_a_non_member():
    scheme, pp, keys = _scheme(n_keys=PAR.N + 1)
    ring = scheme.validate_ring([pk for _, pk in keys[:PAR.N]])
    try:
        scheme.ring_index(ring, keys[PAR.N][1])
    except ValueError:
        return
    raise AssertionError("non-member accepted")


def test_ring_index_takes_the_minimum_over_every_arrangement():
    """`min{j : t_j = pk}` for a key placed at each pair of positions.

    Exhaustive over the ring rather than one arrangement, because "first
    occurrence" and "any occurrence" agree on most inputs and differ only
    when a later duplicate could be picked up instead.
    """
    scheme, pp, keys = _scheme()
    base = [pk for _, pk in keys]
    for i in range(PAR.N):
        for j in range(PAR.N):
            if i == j:
                continue
            ring = list(base)
            ring[i] = ring[j] = base[i]
            found = scheme.ring_index(scheme.validate_ring(ring), base[i])
            assert found == min(i, j), (i, j, found)


# ---- Verify-side checks --------------------------------------------------

def _proof():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    v, pi = scheme.eval(pp, pk, sk, ring, b"msg", b"\x55" * 32)
    return scheme, pp, ring, v, pi


# ---- restart semantics ---------------------------------------------------

class _AbortingExactBackend:
    """An exact backend whose `prove` returns bottom the first `n` times.

    The shipped `opening` backend never aborts, which is exactly why the
    exact-side restart path went untested: `Eval` could place a `None`
    straight into the proof and nothing would notice.  This wraps the real
    backend and refuses a fixed number of times, so the loop has to discard
    a *complete* attempt -- OOM proof included -- and start over.
    """

    def __init__(self, inner, aborts):
        self._inner = inner
        self._remaining = aborts
        self.commitments = 0
        self.name = inner.name

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def com(self, pp, witness_input, xof):
        self.commitments += 1
        return self._inner.com(pp, witness_input, xof)

    def prove(self, pp, statement, witness_input, state):
        if self._remaining > 0:
            self._remaining -= 1
            return None
        return self._inner.prove(pp, statement, witness_input, state)


def test_an_exact_abort_restarts_the_whole_attempt():
    """A bottom from `Pi_ex.Prove` discards the OOM proof with it.

    `W` is folded into the Fiat-Shamir context before the challenge, so an
    accepted OOM proof is bound to *this* exact commitment and cannot be
    reused with a fresh one.  The attempt therefore has to be discarded
    whole, which is what `mu_RiVeR = mu_OOM mu_ex` accounts for.

    The figure does not say this -- it parses `pi_OOM` and reads `z_eval`
    before testing for bottom at all -- so it is a **Repair**, and this is
    the test that holds it.
    """
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]

    baseline = scheme.eval(pp, pk, sk, ring, b"restart", b"\x44" * 32,
                           collect_stats=True)
    v_ok, pi_ok, stats_ok = baseline
    assert "exact" not in stats_ok["aborts"]

    scheme.exact = _AbortingExactBackend(scheme.exact, aborts=2)
    v, pi, stats = scheme.eval(pp, pk, sk, ring, b"restart", b"\x44" * 32,
                               collect_stats=True)

    # Two exact aborts happened, and each cost a whole attempt.  Not
    # "baseline + 2": the OOM layer's accept/reject decision is fixed per
    # attempt index, so discarding the first two OOM-accepting attempts
    # lands on the *third* one, and the indices in between are not
    # consecutive.  What does hold is the bookkeeping.
    assert stats["aborts"].count("exact") == 2
    assert stats["attempts"] > stats_ok["attempts"]
    assert stats["attempts"] == len(stats["aborts"]) + 1
    # Exactly three attempts reached the exact stage: two aborted, one won.
    assert stats["attempts"] - stats["aborts"].count("oom") == 3

    # A fresh exact commitment per attempt, not a reused one.
    assert scheme.exact.commitments == stats["attempts"]

    # The proof that comes out is well formed and verifies -- in particular
    # `sigma` is not `None`, which is what the old code would have shipped.
    assert pi["ex"]["sigma"] is not None
    assert v == v_ok
    scheme.exact = scheme.exact._inner
    assert scheme.verify(pp, ring, b"restart", v, pi)

    # ... and it is a *different* proof from the unaborted one, because the
    # accepting attempt is a later one with different masks.
    assert pi["oom"]["z"] != pi_ok["oom"]["z"]


def test_an_exhausted_attempt_budget_raises_rather_than_returning_bottom():
    """`max_attempts` exceeded is an error, not a `None` in the proof."""
    import dataclasses
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    scheme.exact = _AbortingExactBackend(scheme.exact, aborts=10 ** 6)
    scheme.par = dataclasses.replace(PAR, max_attempts=3)
    try:
        scheme.eval(pp, pk, sk, ring, b"budget", b"\x45" * 32)
    except RuntimeError as exc:
        assert "no accepting attempt" in str(exc)
        return
    raise AssertionError("exhausted attempt budget did not raise")


# ---- key validation ------------------------------------------------------

def test_eval_rejects_a_malformed_secret_key():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    for bad in (None, [], list(sk)[:-1], list(sk) + [sk[0]],
                [[0] * (PAR.d + 1)] * PAR.ell,
                [["x"] * PAR.d] * PAR.ell):
        try:
            scheme.eval(pp, pk, sk=bad, ring_pks=ring, m=b"m",
                        seed=b"\x46" * 32)
        except ValueError:
            continue
        raise AssertionError(f"malformed sk accepted: {bad!r:.30}")


def test_eval_rejects_a_secret_key_outside_S_beta():
    """Short-but-not-ternary keys are caught before any arithmetic."""
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    bad = [list(poly) for poly in sk]
    bad[0][0] = PAR.beta + 1
    try:
        scheme.eval(pp, pk, bad, ring, b"m", b"\x47" * 32)
    except ValueError as exc:
        assert "S_beta" in str(exc)
        return
    raise AssertionError("out-of-range sk accepted")


def test_eval_rejects_a_mismatched_keypair():
    """`pk != floor(A s)_p` is the caller's error, and says so.

    It used to surface from deep inside the attempt loop as a range failure
    on a rounding error, or as a failed opening invariant -- both of which
    report a broken scheme when the cause is a caller pairing the wrong
    halves.
    """
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk = keys[1][0]
    wrong_pk = keys[2][1]
    try:
        scheme.eval(pp, wrong_pk, sk, ring, b"m", b"\x48" * 32)
    except ValueError as exc:
        assert "floor(A s)_p" in str(exc)
        return
    raise AssertionError("mismatched keypair accepted")


def test_the_opening_invariant_is_not_an_assertion():
    """`python -O` strips asserts; this equation is what makes the proof
    mean anything, so it must survive optimisation."""
    import subprocess
    import sys
    code = (
        "from params import TOY_PARAMS as p\n"
        "from river import RiVeR\n"
        "s = RiVeR(p); pp = s.setup(b'\\x00'*32)\n"
        "ks = [s.keygen(pp, bytes([i])+b'\\x00'*31) for i in range(p.N)]\n"
        "sk, pk = ks[1]\n"
        "bad = ks[2][1]\n"
        "try:\n"
        "    s.eval(pp, bad, sk, [k[1] for k in ks], b'm', b'\\x49'*32)\n"
        "except ValueError:\n"
        "    print('rejected')\n"
    )
    out = subprocess.run([sys.executable, "-O", "-c", code],
                         capture_output=True, text=True, cwd=HERE)
    assert out.returncode == 0, out.stderr
    assert "rejected" in out.stdout


def test_verify_accepts_honest_proof():
    scheme, pp, ring, v, pi = _proof()
    assert scheme.verify(pp, ring, b"msg", v, pi)


def test_verify_rejects_wrong_message():
    scheme, pp, ring, v, pi = _proof()
    assert not scheme.verify(pp, ring, b"other", v, pi)


def test_verify_rejects_wrong_value():
    scheme, pp, ring, v, pi = _proof()
    bad = list(v)
    bad[0] = (bad[0] + 1) % PAR.p
    assert not scheme.verify(pp, ring, b"msg", bad, pi)


def test_verify_rejects_non_canonical_value():
    """Figure 5 omits this check; a non-canonical v changes q_0 v mod q."""
    scheme, pp, ring, v, pi = _proof()
    bad = list(v)
    bad[0] = v[0] + PAR.p                    # same residue, different integer
    assert not scheme.verify(pp, ring, b"msg", bad, pi)
    assert not scheme.verify(pp, ring, b"msg", [-1] + list(v[1:]), pi)


def test_verify_rejects_wrong_ring():
    scheme, pp, ring, v, pi = _proof()
    smaller = ring[:-1]
    assert not scheme.verify(pp, smaller, b"msg", v, pi)


def test_verify_rejects_inadmissible_ring():
    scheme, pp, ring, v, pi = _proof()
    assert not scheme.verify(pp, ring + [ring[0]], b"msg", v, pi)


def test_verify_rejects_malformed_public_keys():
    """Malformed input must make Verify return 0, not raise."""
    scheme, pp, ring, v, pi = _proof()
    for bad in ([], [[]], [[0] * PAR.d] * (PAR.n - 1),
                [[0] * (PAR.d - 1)] * PAR.n):
        assert scheme.verify(pp, ring[:-1] + [bad], b"msg", v, pi) is False


def test_verify_is_total_on_malformed_public_input():
    """`Verify` returns a bit for *every* input shape, never an exception.

    Each of these reaches a different unguarded stage: `None` where a list
    belongs, a float coefficient that `int()` refuses to convert (the
    `OverflowError` the guards used to miss), a message that is not bytes,
    and public parameters missing a key.
    """
    scheme, pp, ring, v, pi = _proof()
    inf_key = [[float("inf")] * PAR.d] * PAR.n
    nan_key = [[float("nan")] * PAR.d] * PAR.n
    float_key = [[0.0] * PAR.d] * PAR.n
    cases = [
        (pp, None, b"msg", v, pi),
        (pp, "not a ring", b"msg", v, pi),
        (pp, [None] * PAR.N, b"msg", v, pi),
        (pp, [inf_key] + ring[1:], b"msg", v, pi),
        (pp, [nan_key] + ring[1:], b"msg", v, pi),
        (pp, [float_key] + ring[1:], b"msg", v, pi),
        (pp, ring, b"msg", None, pi),
        (pp, ring, b"msg", [float("inf")] * PAR.d, pi),
        (pp, ring, b"msg", [0.0] * PAR.d, pi),
        (pp, ring, b"msg", v[:-1], pi),
        (pp, ring, None, v, pi),
        (pp, ring, 12345, v, pi),
        ({}, ring, b"msg", v, pi),
        ({"A": pp["A"]}, ring, b"msg", v, pi),
        (pp, ring, b"msg", v, None),
        (pp, ring, b"msg", v, {"oom": None, "ex": None}),
    ]
    for args in cases:
        assert scheme.verify(*args) is False, args[1:2]


def test_verify_survives_an_inconsistent_pp():
    """A stale or edited `pp` is a `0`, not a traceback.

    `pp` is the CRS, so this is not the same guarantee as for `R`, `m`,
    `v` and `pi` -- a validated adversarial `pp` would still break
    soundness.  What it rules out is a mismatched one reaching the caller
    as an exception: before the outer boundary existed, `q_tilde = 0`
    raised `ZeroDivisionError` and `N_ex` raised `AssertionError`, both
    from inside the exact backend and both past every `MALFORMED` guard.
    """
    import copy

    scheme, pp, ring, v, pi = _proof()
    assert scheme.verify(pp, ring, b"msg", v, pi) is True

    def mutated(path, value):
        bad = copy.deepcopy(pp)
        target = bad
        for key in path[:-1]:
            target = target[key]
        if isinstance(target, dict):
            target[path[-1]] = value
        else:
            setattr(target, path[-1], value)
        return bad

    cases = [
        (["ex", "ex", "q_tilde"], 0),        # -> ZeroDivisionError
        (["ex", "ex", "q_tilde"], 1),
        (["ex", "ex", "N_ex"], 1),           # -> AssertionError
        (["ex", "ex", "N_ex"], 0),
        (["ex", "ex", "d_tilde"], 0),
        (["ex", "ex", "l_split"], 0),      # -> ZeroDivisionError in the stride
        (["ex", "ex", "l_split"], 1),      # -> stride past the ring degree
        (["ex", "ex", "n_tilde"], 0),
        (["ex", "ex", "q_tilde"], "not a modulus"),
        (["ex", "ck"], None),
        (["ex"], {}),
        (["oom"], None),
        (["A"], None),
        (["rho"], None),
    ]
    for path, value in cases:
        assert scheme.verify(mutated(path, value), ring, b"msg", v, pi) \
            is False, path


def test_verify_rejects_a_float_that_equals_a_valid_coefficient():
    """`5.0` is malformed input, not input to be rounded.

    Coercing with `int(c)` would have accepted it and then carried a float
    into the arithmetic; the check is on the type as well as the range.
    """
    scheme, pp, ring, v, pi = _proof()
    bad = [list(poly) for poly in ring[0]]
    bad[0][0] = float(bad[0][0])
    assert scheme.verify(pp, [bad] + ring[1:], b"msg", v, pi) is False
    assert scheme.verify(pp, ring, b"msg", [float(c) for c in v], pi) is False


def test_verify_rejects_non_canonical_public_key():
    """A coefficient shifted by p is not the same public key."""
    scheme, pp, ring, v, pi = _proof()
    bad = [list(poly) for poly in ring[0]]
    bad[0][0] += PAR.p
    assert scheme.verify(pp, [bad] + ring[1:], b"msg", v, pi) is False


def test_validate_ring_rejects_malformed_keys_at_every_position():
    scheme, pp, keys = _scheme()
    for bad in ([], [[0] * PAR.d] * (PAR.n + 1), [[0] * PAR.d] * PAR.n
                and [[PAR.p] * PAR.d] * PAR.n):
        try:
            scheme.validate_ring([bad] * PAR.N)
        except ValueError:
            continue
        raise AssertionError(f"accepted malformed key {bad[:1]}")


def test_verify_rejects_malformed_proof():
    scheme, pp, ring, v, pi = _proof()
    assert not scheme.verify(pp, ring, b"msg", v, {})
    assert not scheme.verify(pp, ring, b"msg", v, {"oom": pi["oom"]})


def test_verify_rejects_swapped_exact_component():
    """W is bound into rho', so an exact component from another execution
    of the *same* statement must not transplant."""
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    v1, pi1 = scheme.eval(pp, pk, sk, ring, b"msg", b"\x55" * 32)
    v2, pi2 = scheme.eval(pp, pk, sk, ring, b"msg", b"\x77" * 32)
    assert v1 == v2, "the VRF value must be unique for a fixed key/message"
    assert pi1["ex"]["W"] != pi2["ex"]["W"]
    assert not scheme.verify(pp, ring, b"msg", v1,
                             {"oom": pi1["oom"], "ex": pi2["ex"]})


def test_value_is_unique_per_key_and_message():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    sk, pk = keys[0]
    v1, _ = scheme.eval(pp, pk, sk, ring, b"m", b"\x01" * 32)
    v2, _ = scheme.eval(pp, pk, sk, ring, b"m", b"\x02" * 32)
    assert v1 == v2


def test_value_depends_on_the_signer():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    v0, _ = scheme.eval(pp, keys[0][1], keys[0][0], ring, b"m", b"\x01" * 32)
    v1, _ = scheme.eval(pp, keys[1][1], keys[1][0], ring, b"m", b"\x01" * 32)
    assert v0 != v1


def test_value_is_independent_of_the_ring():
    """v = floor(<G(m), s>)_p depends on the key and message only."""
    scheme, pp, keys = _scheme(n_keys=PAR.N + 1)
    sk, pk = keys[0]
    # Two admissible rings of the required size, sharing only the evaluator.
    ring_a = [pk for _, pk in keys[:PAR.N]]
    ring_b = [pk] + [pk_i for _, pk_i in keys[2:PAR.N + 1]]
    assert ring_a != ring_b
    va, _ = scheme.eval(pp, pk, sk, ring_a, b"m", b"\x01" * 32)
    vb, _ = scheme.eval(pp, pk, sk, ring_b, b"m", b"\x01" * 32)
    assert va == vb


def test_any_member_can_evaluate():
    scheme, pp, keys = _scheme()
    ring = [pk for _, pk in keys]
    for sk, pk in keys:
        v, pi = scheme.eval(pp, pk, sk, ring, b"m", b"\x03" * 32)
        assert scheme.verify(pp, ring, b"m", v, pi)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_river.py: {len(tests)} tests passed")
