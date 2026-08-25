"""
test_lanes_manifest.py -- the frozen LANES manifest against the live code.

`lanes_manifest.json` is data, and data goes stale.  These tests fail if
any value the implementation consumes has moved away from the frozen one,
in either direction, and if the manifest's provenance labels stop being
honest.

The direction matters: the file is the record and the code is checked
against it.  Regenerating the file from the code on every run would make
them agree by construction and certify nothing, which is why
`make lanes-manifest-regen` is a deliberate act like `make vectors`.
"""

from fractions import Fraction

import exact
import lanes_manifest
import lanes_backend as lb
import lanes_params as lp
import lanes_ring as R


def _value(section, key):
    return exact.manifest_value(lanes_manifest.load()[section][key])


def _provenance(section, key):
    return lanes_manifest.load()[section][key]["provenance"]


def test_the_frozen_file_matches_the_modules():
    """The whole file, section by section."""
    assert lanes_manifest.check() == [], lanes_manifest.check()
    assert lanes_manifest.load() is not None


def test_the_gate_reads_the_frozen_file():
    """`exact` loads the file rather than rebuilding it."""
    assert exact.LANES_PARAMETER_MANIFEST == lanes_manifest.load()
    assert exact.validate_lanes_manifest(exact.LANES_PARAMETER_MANIFEST) == []


def test_the_table_is_final_and_the_gate_is_still_shut():
    """The manifest is final while the candidate alias remains explicit.

    The manifest's status is about *this table* -- every wire- and
    security-visible value frozen with a provenance -- and it can now
    honestly be "final", because nothing in it is searched any more.  What
    keeps `LanesBackend` under the experimental name is the artifact-scope
    gate for the implementation-defined recovery composition.
    """
    blob = lanes_manifest.load()
    assert blob["status"] == "final"
    assert "the paper's" in blob["note"]
    assert "lanes-experimental" in blob["note"]

    # The live gate: evidence, not the manifest.
    assert exact.lanes_gate_cause() == "production-alias-reserved"

    # ...and an experimental manifest would still shut it on its own,
    # whatever else were set -- the check has not been weakened, only
    # satisfied.
    experimental = dict(blob, status="experimental")
    assert exact.lanes_gate_cause(
        experimental, backend_ready=True,
        evidence={"verdict": "meets-target"}) == "manifest-experimental"


def test_every_gated_constant_is_selected_by_value():
    """A label is not a selection; the value has to match what runs."""
    live, missing = exact.live_lanes_constants()
    assert not missing
    constants = lanes_manifest.load()["constants"]
    assert set(constants) == set(live)
    for name, (_, value) in live.items():
        assert Fraction(constants[name]["value"]) == value, name
        assert constants[name]["provenance"] in ("Derived", "Repair"), name
        # ...and the value is the paper's closed form.  A relabelled
        # constant would pass the name check and fail here.
        assert value == exact.PAPER_LANES_VALUES[name], name


def test_the_widths_are_the_papers_closed_form():
    """Manifest, module and the paper's printed digits are one value."""
    assert Fraction(_value("sampler", "sigma_r")) == lp.SIGMA_R
    assert Fraction(_value("sampler", "sigma_y")) == lp.SIGMA_Y
    assert _value("sampler", "sigma_denominator") == lp.SIGMA_DEN
    assert _value("sampler", "epsilon_exponent") == -100

    # Each width, as the paper prints it.
    for key, printed in (("s_0", "2.7668"), ("s_1", "5.5336"),
                         ("s_2", "243.4775"), ("s_response", "344.3291")):
        value = float(_value("sampler", key))
        places = len(printed.split(".")[1])
        assert f"{value:.{places}f}" == printed, (key, value)

    # The convention is recorded, because `s` and `sigma` differ by
    # sqrt(2 pi) and a port that read the wrong one would be 18 bits out.
    assert _value("sampler", "convention") == "standard deviation"
    assert _provenance("sampler", "s_1") == "Paper"
    assert _provenance("sampler", "sigma_r") == "Derived"


def test_the_dimensions_are_the_papers_and_are_labelled_so():
    """Structure is Paper; the widths on top of it are not."""
    dims = lanes_manifest.load()["dimensions"]
    for key, value in (("d_tilde", R.DTILDE), ("l_split", R.LSPLIT),
                       ("q_tilde", R.QTILDE), ("n_tilde", lp.N_TILDE),
                       ("ell_tilde", lp.ELL_TILDE), ("N_ex", lp.N_EX),
                       ("alpha", lp.ALPHA), ("D", lp.D_DROP),
                       ("w_hat", lp.W_HAT), ("n_lwe", lp.N_LWE),
                       ("m_lwe", lp.M_LWE)):
        assert exact.manifest_value(dims[key]) == value, key
        assert dims[key]["provenance"] == "Paper", key

    # the padding is a consequence, and is recorded as such
    assert _value("dimensions", "block_slots") == R.LSPLIT == 64
    assert _value("dimensions", "block_payload") == 32
    assert _provenance("dimensions", "block_payload") == "Derived"


def test_nothing_this_tree_invented_is_labelled_paper():
    """The one relabelling that would make the manifest a lie.

    The widths *are* Paper now, so this list shrank -- which is the labels
    working, not the check weakening.  What remains this tree's own is the
    recovery-hint construction end to end, the response infinity bound (the
    paper states none), and the sampler tail cuts.  None may read as
    printed.
    """
    blob = lanes_manifest.load()
    for section, keys in (("sampler", ("tail_cut_r", "tail_cut_y")),
                          ("response_bounds", ("inf",)),
                          ("recovery", ("rounding", "ties", "hint_alphabet",
                                        "buckets", "error_bound",
                                        "encoding"))):
        for key in keys:
            assert blob[section][key]["provenance"] == "Repair", (section, key)


def test_the_bounds_agree_with_what_the_verifier_enforces():
    assert _value("response_bounds", "inf") == lp.Z_INF_BOUND
    assert _value("response_bounds", "l2") == lp.Z_NORM2_BOUND
    assert _value("response_bounds", "population") == lp.N_Z == 3328
    assert Fraction(_value("response_bounds", "var_z")) == lp.VAR_Z

    # The paper's own figures, and the direction of the one inequality
    # between them: our enforced bound is over 3328 coefficients and the
    # paper's beta' over 4352, so ours is strictly the stricter.
    assert f"{float(_value('response_bounds', 'beta_prime_bdlop')):.1f}" \
        == "45430.6"
    assert round(float(_value("response_bounds", "b_msis"))) == 15991562
    assert _value("response_bounds", "n_z_paper") == 4352
    assert lp.Z_NORM2_BOUND < lp.Z_NORM2_BOUND_PAPER

    # ...and it is above what an honest response actually needs.
    assert _value("response_bounds", "l2_honest_requirement") \
        == lp.Z_NORM2_REQUIRED < lp.Z_NORM2_BOUND

    # One definition, shared by both sides.
    assert _value("response_bounds", "shared_by_prover_and_verifier") \
        == "lanes_proof.response_within_bounds"


def test_the_transcript_section_describes_the_implemented_transcript():
    """It used to describe an intended one, and they differed.

    The old list omitted `w_high`, `v` and `v_prime` -- all three are
    hashed -- and included `alpha`, `gamma` and `c`, which are outputs of
    the hash.  A port built from it would derive different challenges.
    """
    import lanes_proof
    blob = lanes_manifest.load()["transcript"]
    assert exact.manifest_value(blob["absorbed_fields"]) \
        == lanes_proof.declared_transcript()

    absorbed = exact.manifest_value(blob["absorbed_fields"])
    for name in ("w_high", "v", "v_prime"):
        assert name in absorbed, name
    for name in exact.manifest_value(blob["derived_not_absorbed"]):
        assert name not in absorbed, name
    assert exact.manifest_value(blob["derived_not_absorbed"]) \
        == ["alpha", "gamma", "c"]


def test_the_rank_roles_name_the_right_letters():
    """`n~ == l~ == 4`, so only the prose can be wrong -- and it was.

    The identity rank is `l~` and the tail rank is `n~`; the table said
    `n~` and `kappa - n~`.  Nothing numeric can catch it at these
    parameters, which is why the manifest records that fact too.
    """
    roles = lanes_manifest.load()["rank_roles"]
    assert "l~" in roles["identity_rank"]["how"]
    assert "n~" in roles["tail_rank"]["how"]
    assert "kappa - l~" in roles["response_rank"]["how"]
    assert exact.manifest_value(roles["roles_are_distinguishable"]) is False
    assert roles["identity_rank"]["value"] == lp.ELL_TILDE
    assert roles["tail_rank"]["value"] == lp.N_TILDE


def test_the_recovery_section_admits_it_has_no_security_argument():
    """The one place a manifest could quietly imply analysis that is absent.

    The fixed-hint composition is this repository's.  It has no leakage or
    extraction analysis, and a manifest that listed its parameters without
    saying so would read as though it did.
    """
    recovery = lanes_manifest.load()["recovery"]
    assert "NONE" in exact.manifest_value(recovery["security_argument"])
    assert _value("recovery", "buckets") == lp.RECOVERY_BUCKETS
    assert _value("recovery", "error_bound") == lp.RECOVERY_ERROR_BOUND
    assert _value("recovery", "d_drop") == lp.D_DROP == 17
    assert _value("recovery", "scale") == lp.T0_SCALE

    # The four counted cells against the dimensions, not against
    # each other's prose.  `omitted_coordinates` is the *retained* name --
    # it is the cell that read 0 while four ring elements were being
    # dropped -- and it has to keep meaning the same thing as the cell
    # that replaced it, or the table would carry two answers.
    rows = _value("recovery", "omitted_response_rows")
    assert rows == lp.KAPPA - lp.RESPONSE_RANK == 4
    coeffs = _value("recovery", "omitted_response_coefficients")
    assert coeffs == rows * R.DTILDE == 1024
    assert _value("recovery", "omitted_coordinates") == coeffs
    assert _value("recovery", "recovery_carries") == R.DTILDE * lp.IDENTITY_RANK
    assert _value("recovery", "omitted_t0_low_bits") == \
        R.DTILDE * lp.IDENTITY_RANK * lp.D_DROP


def test_the_estimator_section_reproduces_the_printed_deltas():
    """The diagnostic run is recorded without becoming a selection rule."""
    est = lanes_manifest.load()["estimator"]

    # MLWE: the normative standard-deviation reading reproduces 1.0040.
    hint_out = exact.manifest_value(est["hint_mlwe_outputs"])
    assert "REPRODUCED" in hint_out["status"]
    assert hint_out["paper_reports"] == "delta_MLWE = 1.0040"
    bits = hint_out["bits_by_reading"]
    assert set(bits) == {"standard-deviation", "gaussian-parameter-as-stddev"}
    assert 116 < bits["standard-deviation"] < 117
    assert 134 < bits["gaussian-parameter-as-stddev"] < 135
    # The alternate API conversion is retained only as sensitivity data.
    assert bits["gaussian-parameter-as-stddev"] - bits["standard-deviation"] > 17
    deltas = hint_out["delta_by_reading"]
    assert round(deltas["standard-deviation"], 4) == 1.0040

    # A reduction term no estimator call reports is retained as data.
    loss = exact.manifest_value(est["hint_mlwe_statistical_loss"])
    assert -95 < loss < -94, loss

    # M-SIS: this one *is* reproduced, twice over.
    msis_out = exact.manifest_value(est["msis_outputs"])
    assert "REPRODUCED" in msis_out["status"]
    assert 128 <= msis_out["bits"] < 128.5
    assert 128 <= msis_out["published_B_MSIS_bits"] < 128.5
    assert msis_out["paper_reports"] == "delta_MSIS = 1.0037"
    assert round(msis_out["delta_closed_form"], 4) == 1.0037
    assert round(float(msis_out["B_MSIS"])) == 15991562

    challenge = exact.manifest_value(est["challenge"])
    assert "optional" in challenge["status"]

    # the inputs are real, and match the module
    hint_in = exact.manifest_value(est["hint_mlwe_inputs"])
    assert hint_in["n"] == lp.N_LWE and hint_in["m"] == lp.M_LWE
    assert Fraction(hint_in["sigma_mlwe_sq"]) == lp.SIGMA_MLWE_SQ


def test_the_candidate_composition_keeps_a_distinct_backend_name():
    """A complete manifest does not relabel the candidate as production."""
    assert exact.LANES_SECURITY_EVIDENCE is not None
    assert exact.LANES_SECURITY_EVIDENCE["verdict"] == "candidate-composition"

    security = lanes_manifest.load()["security"]
    assert security["verdict"] == "candidate-composition"
    assert security["delta_mlwe_reproduced"] is True
    assert security["evidence"] == "lanes_security.json"
    assert len(security["estimator"]) == 40          # the tool's commit
    assert security["still_missing"], "nothing is claimed to be complete"

    # a passing verdict is still not enough on its own
    good = dict(exact.LANES_PARAMETER_MANIFEST, status="final")
    passing = dict(exact.LANES_SECURITY_EVIDENCE, verdict="meets-target")
    assert exact.lanes_gate_cause(good, backend_ready=False,
                                  evidence=passing) == "backend-not-ready"
    assert exact.lanes_gate_cause(good, backend_ready=True,
                                  evidence=exact.LANES_SECURITY_EVIDENCE) \
        == "production-alias-reserved"


def test_the_wire_section_matches_the_serializer():
    from params import TOY_PARAMS
    from lanes_backend import LanesBackend

    backend = LanesBackend.experimental(TOY_PARAMS)
    names = [f.name for f in backend.proof_layout.fields]
    assert _value("wire", "order") == names
    assert [f["name"] for f in _value("wire", "fields")] == names
    assert _value("wire", "total_bits") is None
    detail = _value("wire", "discrepancy")
    assert "13.5 KB" in detail and "entropy estimate" in detail


def test_the_challenge_cell_matches_the_reported_parameter_set():
    """The required challenge cell cannot drift as unchecked prose."""
    challenge = _value("estimator", "challenge")
    assert challenge["paper_lanes_noninvertibility"] == "2^-90.5"
    assert challenge["paper_outer"] == "2^-91.5"


def test_the_generated_kat_records_this_trees_gate():
    """The withheld record in `sampler_kat.json` is *ours*, and current.

    `river-rs` consumes that file and now compares the *cause* directly
    against its own gate, not merely that both are shut.  It could not
    before: this side had a frozen manifest and that side had none, so it
    reported `no-parameter-manifest` for a table it had never been given.
    `river-rs/src/lanes_manifest.rs` is generated from
    `lanes_manifest.json`, so both are gated on the same table and a
    difference in cause is a real divergence.

    The prose `reason` is still compared only here, on the side that
    generated it: each implementation's names its own API.
    """
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "river-rs", "tests", "sampler_kat.json")
    if not os.path.exists(path):                # pragma: no cover
        return
    with open(path) as fh:
        withheld = json.load(fh)["withheld"]

    assert withheld["cause"] == exact.lanes_gate_cause(), (
        "the KAT's withheld cause has drifted from this tree's gate; run "
        "`make -C ../river-rs kat-regen`")
    assert withheld["cause"] in exact.LANES_GATE_CAUSES
    assert withheld["reason"] == exact.lanes_unavailable_reason()
    live, _ = exact.live_lanes_constants()
    assert withheld["constants"] == sorted(live)

    # **No blocks are withheld.**  All three LANES blocks -- ring, params,
    # proof -- are generated and driven, which is what makes the two
    # `lanes-experimental` vector cases bisectable primitive by primitive.
    # The record survives to carry the *cause*, which is about the
    # production backend name rather than about any block.
    assert withheld["blocks"] == []
    with open(path) as fh:
        blob = json.load(fh)
    for block in ("lanes_ring", "lanes_params", "lanes_proof"):
        assert blob.get(block), f"{block} is absent from the KAT"


def test_regenerating_it_is_a_deliberate_act():
    """`--check` catches a moved constant; nothing regenerates silently.

    The constant has to move in **both** places that hold it.  `lp` is
    where it is defined and where the `constants` and `response_bounds`
    cells read it; `lanes_backend` bound its own copy at import time and
    is what the serializer's Rice coder carries, so the `wire` section
    follows that one.  Patching only `lp` would leave the wire section
    reporting the old bound -- and worse, it would *poison* the process:
    `manifest()` imports `lanes_backend` lazily, so if this is the first
    call the import happens inside the patched window and captures the
    moved value permanently, which no `finally` can undo.  Moving both is
    also the truthful mutation: a real change to `Z_INF_BOUND` reaches
    every binding at once.
    """
    saved = lp.Z_INF_BOUND
    try:
        lp.Z_INF_BOUND = lb.Z_INF_BOUND = saved + 1
        bad = lanes_manifest.check()
        assert bad, "a moved constant did not show up as manifest drift"
        assert any("constants" in b for b in bad), bad
        assert any("response_bounds" in b for b in bad), bad
        assert any("wire" in b for b in bad), bad
    finally:
        lp.Z_INF_BOUND = lb.Z_INF_BOUND = saved
    assert lanes_manifest.check() == []


def test_the_serializer_and_the_table_state_one_infinity_bound():
    """The Rice cap on `z` *is* `response_bounds.inf`, not a second bound.

    Two modules hold the constant, and the manifest reports each of them
    from its own binding: `response_bounds.inf` from `lanes_params`, the
    `z` field's Rice `bound` from whatever `lanes_backend` bound at import.
    Nothing else in the table compares the two, so a divergence between
    them would be a manifest that contradicts itself while every section
    matched the module it was built from.  It is also exactly what the
    infinity bound means on the wire: a coefficient the coder would accept
    but the verifier rejects is a proof that serializes and does not
    verify -- the coder and the verifier disagreeing, the other way round.
    """
    blob = lanes_manifest.load()
    z = [f for f in blob["wire"]["fields"]["value"] if f["name"] == "z"]
    assert len(z) == 1, "no single `z` field in the layout"
    assert z[0]["coder"] == "Rice"
    assert z[0]["bound"] == blob["response_bounds"]["inf"]["value"]
    assert z[0]["bound"] == lp.Z_INF_BOUND == lb.Z_INF_BOUND



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

# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_lanes_manifest.py: {len(tests)} tests passed")
