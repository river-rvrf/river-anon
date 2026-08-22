"""
test_e2e.py -- End-to-end tests across the whole pipeline.

Covers the toy profile in full and the smallest published profile
(`RiVeR-N8`) for the paths that matter, plus the shipped test-vector file.
"""

import json
import os

import vectors
from params import get, TOY_PARAMS
from river import RiVeR

HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS_PATH = os.path.join(HERE, "vectors.json")


def _available_backends():
    """Exact backends that run at the current parameters."""
    from exact import BACKENDS, OPTIONAL_BACKENDS, get_backend
    from params import TOY_PARAMS
    out = []
    for name in list(BACKENDS) + list(OPTIONAL_BACKENDS):
        try:
            get_backend(name, TOY_PARAMS)
        except NotImplementedError:
            continue
        out.append(name)
    return tuple(out)


AVAILABLE_BACKENDS = _available_backends()


def _run(par, ring_size=None, signer=1, message=b"e2e",
         setup_seed=b"\x60" * 32, eval_seed=b"\x61" * 32):
    scheme = RiVeR(par)
    pp = scheme.setup(setup_seed)
    # A ring is exactly `N` keys.
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(par.N if ring_size is None else ring_size)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[signer]
    v, pi, stats = scheme.eval(pp, pk, sk, ring, message, eval_seed,
                               collect_stats=True)
    return scheme, pp, ring, v, pi, stats


# ---- full pipeline -------------------------------------------------------

def test_toy_pipeline():
    scheme, pp, ring, v, pi, stats = _run(TOY_PARAMS)
    assert scheme.verify(pp, ring, b"e2e", v, pi)
    assert 1 <= stats["attempts"] <= TOY_PARAMS.max_attempts


def test_published_profile_pipeline():
    par = get("RiVeR-N8")
    scheme, pp, ring, v, pi, stats = _run(par)
    assert scheme.verify(pp, ring, b"e2e", v, pi)


def test_serialization_round_trip_both_profiles():
    for par in (TOY_PARAMS, get("RiVeR-N8")):
        scheme, pp, ring, v, pi, _ = _run(par)
        blob = scheme.proof_encode(pi)
        decoded = scheme.proof_decode(blob)
        assert scheme.proof_encode(decoded) == blob, par.name
        assert scheme.verify(pp, ring, b"e2e", v, decoded), par.name


def test_every_ring_position_evaluates():
    """Any of the `N` members can evaluate, and the proof is the same shape.

    A ring is exactly `N` keys, so the sweep is over the signer's position
    -- which is the one thing the proof has to hide.
    """
    par = TOY_PARAMS
    scheme = RiVeR(par)
    pp = scheme.setup(b"\x62" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31) for i in range(par.N)]
    ring = [pk for _, pk in keys]
    for j_star in range(par.N):
        sk, pk = keys[j_star]
        v, pi = scheme.eval(pp, pk, sk, ring, b"sizes", b"\x63" * 32)
        assert scheme.verify(pp, ring, b"sizes", v, pi), j_star

    # An under- or over-sized ring is inadmissible, not a smaller anonymity
    # set: there is no padding left to absorb it.
    for bad in (ring[:1], ring[:par.N - 1], ring + ring[:1]):
        try:
            scheme.eval(pp, keys[0][1], keys[0][0], bad, b"sizes",
                        b"\x63" * 32)
        except ValueError:
            continue
        raise AssertionError(f"ring of size {len(bad)} accepted")


def test_proof_length_smoke_test_shows_no_signer_dependence():
    """Smoke test: length shows no signer dependence at this sample size.

    It does **not** establish signer independence.  Twelve proofs per backend
    sit in a band a tenth of a percent wide and one signer alone spans most of
    it; four samples per signer cannot separate three *distributions*, and a
    small signer-dependent shift would pass both inequalities.  Establishing
    the property wants pairwise distribution tests at a far larger sample
    count, or fixed-length padding if the hiding has to be unconditional.
    This catches a gross regression -- a field whose width tracks the witness
    -- and is not the evidence.

    The *argument* is structural, and differs by backend:

      * `opening` Rice-codes `y_eval`, a coordinate of the OOM mask.  After
        rejection sampling it is distributed as `D_sigma` independently of the
        witness, which is what makes its length witness-independent.
      * `lanes` Rice-codes `z = y + c r` and has **no** rejection sampling at
        all -- the [KLSS23] Hint-MLWE treatment is what removes it, so citing
        rejection here (as an earlier version of this docstring did) cites a
        mechanism the backend deletes.  The argument is instead that `y` and
        `r` are drawn independently of the witness, and the witness reaches
        `z` only through `c`, which ranges over a challenge space of fixed
        weight `w_hat` whatever was committed.  Conditioned on `c`, the law of
        `z` does not move.

    Both backends, because each has its own variable-length field and
    `lanes`'s is the larger; the shipped vectors have one signer per case and
    so cannot cover this at all.
    """
    # The production `lanes` name is gated on security evidence, so
    # `AVAILABLE_BACKENDS` carries `lanes-experimental` instead -- the same
    # code under the name an artifact can honestly record.
    for backend in AVAILABLE_BACKENDS:
        _proof_length_smoke_for_backend(backend)


def _proof_length_smoke_for_backend(backend):
    par = TOY_PARAMS
    scheme = RiVeR(par, exact_backend=backend)
    pp = scheme.setup(b"\x64" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(par.N)]
    ring = [pk for _, pk in keys]

    per_signer = []
    for sk, pk in keys:
        lengths = []
        for seed in range(4):
            v, pi = scheme.eval(pp, pk, sk, ring, b"anon", bytes([seed]) * 32)
            assert scheme.verify(pp, ring, b"anon", v, pi)
            lengths.append(len(scheme.proof_encode(pi)))
        per_signer.append(lengths)

    every = [n for row in per_signer for n in row]
    spread = max(every) - min(every)
    mean = sum(every) / len(every)

    # a band far too narrow to be a per-signer offset -- necessary,
    # nowhere near sufficient (measured: 0.100% opening, 0.087% lanes)
    assert spread < 0.01 * mean, (backend, spread, mean)

    # and one signer alone already covers most of it, so the variation is
    # randomness rather than identity
    widest = max(max(row) - min(row) for row in per_signer)
    assert widest >= spread / 2, (backend, widest, spread)


def test_seed_reuse_across_messages_does_not_reuse_masks():
    """Reusing `seed` across messages must not reuse the mask `y`.

    If it did, two proofs would publish `z_1 = y + x_1 r` and
    `z_2 = y + x_2 r`, so `z_1 - z_2 = (x_1 - x_2) r` and one linear solve
    recovers the whole witness -- the ternary key `s`, `e_key` and `e_eval`.
    That is generic Fiat-Shamir-with-aborts nonce reuse, and it is why `eval`
    derives its nonce from `(seed, sk, ring, m)` rather than `seed` alone.

    Checked directly on the masks: `y = z - x s` component by component, which
    needs only the key the test already holds.  Masks that differ kill the
    relation whether or not the two runs abort the same number of times.
    """
    par = TOY_PARAMS
    scheme = RiVeR(par)
    R = scheme.Rq
    pp = scheme.setup(b"\x00" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(par.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]

    seed = b"\xAA" * 32
    masks = []
    for msg in range(4):
        v, pi = scheme.eval(pp, pk, sk, ring, bytes([msg]), seed)
        assert scheme.verify(pp, ring, bytes([msg]), v, pi)
        x = [c % par.q for c in pi["oom"]["x"]]      # x is carried centred
        z = pi["oom"]["z"]
        rows = []
        for i in range(par.ell):
            xs = R.mul(x, sk[i])
            rows.append(tuple(R.centered(
                [(z[i][c] - xs[c]) % par.q for c in range(par.d)])))
        masks.append(tuple(rows))

    assert len(set(masks)) == len(masks), "mask reused across messages"


def test_evaluation_is_deterministic_in_its_seed():
    a = _run(TOY_PARAMS)
    b = _run(TOY_PARAMS)
    assert a[0].proof_encode(a[4]) == b[0].proof_encode(b[4])
    assert a[3] == b[3]


def test_different_eval_seeds_give_different_proofs_same_value():
    par = TOY_PARAMS
    x = _run(par, eval_seed=b"\x01" * 32)
    y = _run(par, eval_seed=b"\x02" * 32)
    assert x[3] == y[3], "the VRF value must not depend on the eval seed"
    assert x[0].proof_encode(x[4]) != y[0].proof_encode(y[4])


# ---- cross-profile isolation ---------------------------------------------

def test_proof_does_not_verify_under_a_different_setup():
    par = TOY_PARAMS
    scheme, pp, ring, v, pi, _ = _run(par, setup_seed=b"\x70" * 32)
    other_pp = scheme.setup(b"\x71" * 32)
    assert not scheme.verify(other_pp, ring, b"e2e", v, pi)


# ---- restart behaviour ---------------------------------------------------

def test_measured_attempts_track_the_corrected_estimate():
    """Measured restarts land near mu-tilde_RiVeR.

    They did not until the paper: the appendix charged
    `mu_bin = M_2` for the optimised sampler while Lemma "grs" part 2 only
    guarantees `Pr[accept] >= 1/(2 M_2)`, the missing factor being the
    half-space condition `<z, v> >= 0`.  The correctness appendix
    now names `mu_b := 2 exp(1/(2 phi_b^2))` and states that the factor is
    not charged again by the infinity-norm check, so `mu_river` is the
    estimate to measure against.
    """
    par = TOY_PARAMS
    scheme = RiVeR(par)
    pp = scheme.setup(b"\x80" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(par.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[0]

    trials = 12
    total = 0
    for i in range(trials):
        _, _, stats = scheme.eval(pp, pk, sk, ring, b"m%d" % i,
                                  bytes([i]) + b"\x90" * 31,
                                  collect_stats=True)
        total += stats["attempts"]
    mean = total / trials
    # A band on the *ratio*, not a ceiling.  The old `mean <= 4 mu` bound
    # survived mu doubling unchanged, so it could not have caught the factor
    # coming back out; this can.  The sample is deterministic in the seeds
    # above and gives 8.583 / 7.762 = 1.106; dropping the half-space factor
    # would halve mu to 3.881 and put the ratio at 2.211, outside the band.
    # The spread is wide -- individual counts run 1 to 42 -- so the band is
    # on the mean of twelve and is not tight.
    ratio = mean / par.mu_river
    assert mean >= 1.0, mean
    assert 0.5 <= ratio <= 1.4, (mean, par.mu_river, ratio)


def test_the_defensive_euclidean_check_is_free():
    """The prover applies a bound the paper leaves commented out.

    `OOM.Ver` enforces `||z||_2 <= 1.2 sqrt(sigma_s^2 d ell +
    sigma_m^2 d (n+1))`; the corresponding prover check is commented out in
    the figure, and the commented form is a different, much smaller bound.
    `oom.prove` applies the verifier's, so a returned proof always passes the
    checks `Verify` will apply.

    That is a check the paper's attempt estimate does not charge, so it is
    worth asking whether it moves the observed rate.  It does not:
    with the same seeds, disabling it leaves every attempt count identical,
    because it never fires on an attempt the four infinity-norm checks let
    through.  Measured, not argued -- and re-measured here, so it stays true.
    """
    import math
    from params import RiVeRParams

    def counts(trials=6):
        par = TOY_PARAMS
        scheme = RiVeR(par)
        pp = scheme.setup(b"\x80" * 32)
        keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
                for i in range(par.N)]
        ring = [pk for _, pk in keys]
        sk, pk = keys[0]
        out = []
        for i in range(trials):
            _, _, st = scheme.eval(pp, pk, sk, ring, b"m%d" % i,
                                   bytes([i]) + b"\x90" * 31,
                                   collect_stats=True)
            out.append(st["attempts"])
        return out

    enabled = counts()
    real = RiVeRParams.z_l2_bound
    RiVeRParams.z_l2_bound = property(lambda self: math.inf)
    try:
        disabled = counts()
    finally:
        RiVeRParams.z_l2_bound = real
    assert enabled == disabled, (enabled, disabled)


def test_all_published_profiles_end_to_end():
    """Every published profile, not just N8.

    Skipped by default because N=256 alone takes ~10 s; `make test-all` sets
    RIVER_SLOW_TESTS=1.  The committed fast suite covers TOY and N8.
    """
    if not os.environ.get("RIVER_SLOW_TESTS"):
        return
    for name in ("RiVeR-N8", "RiVeR-N16", "RiVeR-N64",
                 "RiVeR-N128", "RiVeR-N256"):
        par = get(name)
        scheme, pp, ring, v, pi, stats = _run(par)
        assert scheme.verify(pp, ring, b"e2e", v, pi), name
        blob = scheme.proof_encode(pi)
        assert scheme.verify(pp, ring, b"e2e", v,
                             scheme.proof_decode(blob)), name


# ---- shipped test vectors ------------------------------------------------

def test_shipped_vectors_verify():
    with open(VECTORS_PATH) as handle:
        blob = json.load(handle)
    ok, errors = vectors.verify_vectors(blob)
    assert ok, errors


def test_vectors_regenerate_identically():
    with open(VECTORS_PATH) as handle:
        stored = json.load(handle)
    fresh = vectors.generate(tuple((c["params"], c["exact_backend"])
                                   for c in stored["cases"]))
    assert fresh == stored, "regenerated vectors differ from the shipped file"


def _blob_with(blob, cases):
    """The shipped file's metadata, with `cases` swapped in.

    Carrying *all* the metadata matters: the corruption test used to build
    `{schema_version, cases}` only, so `verify_vectors` returned early on
    the missing `paper_revision` and never looked at the corrupted proof at
    all.  It passed for the wrong reason, and would have kept passing if
    the proof check had been deleted.
    """
    out = {k: v for k, v in blob.items() if k != "cases"}
    out["cases"] = cases
    return out


def test_vector_verification_catches_corruption():
    with open(VECTORS_PATH) as handle:
        blob = json.load(handle)
    case = dict(blob["cases"][0])
    proof = dict(case["proof"])
    proof["bytes"] = ("00" if proof["bytes"][:2] != "00" else "01") \
        + proof["bytes"][2:]
    case["proof"] = proof
    intact = [c for c in blob["cases"][1:]]
    ok, errors = vectors.verify_vectors(_blob_with(blob, [case] + intact))
    assert not ok and errors
    # It must fail on the *proof*, not on metadata it never got past.
    joined = " ".join(errors)
    assert "proof bytes differ" in joined, errors
    assert "paper_revision" not in joined and "schema_version" not in joined

    # The control: the same blob, uncorrupted, verifies.
    ok_ctrl, _ = vectors.verify_vectors(_blob_with(blob, blob["cases"]))
    assert ok_ctrl


def test_vector_metadata_gate_rejects_a_substituted_file():
    """The file-level checks are load-bearing, not decoration.

    Without them `verify_vectors` passed on an empty `cases` array and on a
    changed `generator`, so a truncated or substituted file reported
    success.
    """
    with open(VECTORS_PATH) as handle:
        blob = json.load(handle)
    assert vectors.verify_vectors(blob)[0]

    mutated = dict(blob, generator="somebody-else")
    ok, errors = vectors.verify_vectors(mutated)
    assert not ok, "a substituted generator was accepted"
    assert any("generator" in e for e in errors), errors

    # A missing or empty case list is not a pass either.
    for bad in ({k: v for k, v in blob.items() if k != "cases"},
                dict(blob, cases=[]),
                dict(blob, cases=blob["cases"][:1])):
        assert not vectors.verify_vectors(bad)[0], "a short case set passed"


def test_withheld_cases_are_accounted_for():
    """The coverage claim is enforced, not narrated.

    The READMEs say the vector accounting cannot shrink in silence.  That
    was true of the *shipped* set -- `REQUIRED_CASES` is checked -- but the
    withheld `lanes` cases were only a comment, so one quietly reappearing
    or the gate quietly lifting would have gone unnoticed.
    """
    # The four tuples are frozen *here*, independently of `vectors.py`.
    # Deriving them from `CASE_PROFILES` -- which is what this test used to
    # do -- makes every assertion below vacuous: delete both `RiVeR-N8`
    # entries, shipped and withheld together, and a self-referential check
    # still passes while coverage has silently halved.  Reproduced before
    # fixing.
    PROFILES_COVERED = ("RiVeR-TOY", "RiVeR-N8")
    EXPECTED_SHIPPED = {(p, b) for p in PROFILES_COVERED
                        for b in ("opening", "lanes-experimental")}
    EXPECTED_WITHHELD = {(p, "lanes") for p in PROFILES_COVERED}

    cover = vectors.coverage()
    assert set(cover["shipped"]) == EXPECTED_SHIPPED, cover["shipped"]
    assert set(cover["withheld"]) == EXPECTED_WITHHELD, cover["withheld"]
    assert set(vectors.REQUIRED_CASES) == EXPECTED_SHIPPED
    assert EXPECTED_SHIPPED.isdisjoint(EXPECTED_WITHHELD)
    # Together: both profiles under all three backend names, nothing
    # unaccounted for.
    assert EXPECTED_SHIPPED | EXPECTED_WITHHELD == \
        {(p, b) for p in PROFILES_COVERED
         for b in ("opening", "lanes-experimental", "lanes")}
    # The production `lanes` name is withheld precisely because it cannot
    # run; `lanes-experimental` is the same code and does.
    assert cover["withheld_backends"] == ("lanes",)
    assert "lanes" not in cover["backends"]
    assert "lanes-experimental" in cover["backends"]
    assert "lanes" not in AVAILABLE_BACKENDS
    assert "lanes-experimental" in AVAILABLE_BACKENDS

    # ... and the reason is the readiness gate, not an unrelated failure.
    from exact import lanes_unavailable_reason
    assert lanes_unavailable_reason()

    # The shipped file contains exactly the shipped set.
    with open(VECTORS_PATH) as handle:
        blob = json.load(handle)
    present = {(c["params"], c["exact_backend"]) for c in blob["cases"]}
    assert present == EXPECTED_SHIPPED


def test_vector_schema_version_is_checked():
    ok, errors = vectors.verify_vectors({"schema_version": 999, "cases": []})
    assert not ok and errors


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_e2e.py: {len(tests)} tests passed")
