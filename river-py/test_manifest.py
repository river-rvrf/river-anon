"""
test_manifest.py -- Freeze the wire-visible numeric choices.

`manifest.py` collects every value two implementations have to agree on
exactly: paper-derived parameters together with the pinned Gaussian
rationals, Rice parameters, response bounds, and fixed field widths.  This
pins their concrete wire representation.

The point is the *order* in which a mismatch is discovered.  Without this,
a changed sampler width or a one-off Rice parameter first shows up as
"proof bytes differ at byte 4" in a cross-language vector, which names
neither the field nor the cause.  Here it names both.

Regenerating this table is a deliberate act, exactly like regenerating
`vectors.json`: `python3 manifest.py --json` prints the current values, and
a diff against `FROZEN` below is the list of fields whose encoding moved.
"""

import json
import re
import math
from fractions import Fraction

import manifest
from codec import (RICE_CONST_DEN, RICE_CONST_NUM, RiVeRCodec, floor_sqrt,
                   optimal_rice_k)
from params import PROFILES, get
from sample import SIGMA_DEN, rational_sigma

#: `(sigma_num, rice_k, bound, width_bits)` for every wire field of
#: every profile.  `None` where the entry does not apply.
FROZEN = {
    "RiVeR-N128": {
        "B": (None, None, 2199023255552, 43),
        "f1": (3221225472, 11, 18432, None),
        "x": (None, None, 16, 6),
        "y_eval": (2915520479977, 21, 16698102, None),
        "zb": (43377043416, 15, 248205, None),
        "zm": (2915520479977, 21, 16682742, None),
        "zs": (30822128908419, 25, 176365636, None),
    },
    "RiVeR-N16": {
        "B": (None, None, 549755813891, 41),
        "f1": (5368709120, 12, 30720, None),
        "x": (None, None, 16, 6),
        "y_eval": (2915520479977, 21, 16698102, None),
        "zb": (42518007000, 15, 243289, None),
        "zm": (2915520479977, 21, 16682742, None),
        "zs": (20044203299842, 24, 114693851, None),
    },
    "RiVeR-N256": {
        "B": (None, None, 4398046511104, 44),
        "f1": (2952790016, 11, 16896, None),
        "x": (None, None, 16, 6),
        "y_eval": (2915520479977, 21, 16698102, None),
        "zb": (43800244105, 15, 250627, None),
        "zm": (2915520479977, 21, 16682742, None),
        "zs": (36625772743250, 25, 209574352, None),
    },
    "RiVeR-N64": {
        "B": (None, None, 2199023255552, 43),
        "f1": (4563402752, 12, 26112, None),
        "x": (None, None, 16, 6),
        "y_eval": (2915520479977, 21, 16698102, None),
        "zb": (43377043416, 15, 248205, None),
        "zm": (2915520479977, 21, 16682742, None),
        "zs": (21646635171840, 24, 123863040, None),
    },
    "RiVeR-N8": {
        "B": (None, None, 137438953472, 39),
        "f1": (4294967296, 12, 24576, None),
        "x": (None, None, 16, 6),
        "y_eval": (2915520479977, 21, 16698102, None),
        "zb": (41195879100, 15, 235724, None),
        "zm": (2915520479977, 21, 16682742, None),
        "zs": (23450521436160, 24, 134184960, None),
    },
    "RiVeR-TOY": {
        "B": (None, None, 17179869181, 35),
        "f1": (4294967296, 12, 24576, None),
        "x": (None, None, 16, 6),
        "y_eval": (2915520479977, 21, 16698102, None),
        "zb": (12148002000, 13, 69511, None),
        "zm": (2915520479977, 21, 16682742, None),
        "zs": (7490994291296, 23, 42863813, None),
    },
}



#: `(field order, max_bytes, min_bytes)` per layout per profile.
#: The order is the wire order; a port that permutes it produces a
#: well-formed blob that decodes to different values.
FROZEN_LAYOUTS = {
    "RiVeR-N128": {
        "exact_W": (['t0', 't1'], 8320, 8320),
        "exact_opening": (['t0', 't1', 'e_eval', 'y_eval', 'digits', 'randomness'], 9584, 9552),
        "oom": (['B', 'x', 'f1', 'zb', 'zs', 'zm'], 37316, 28196),
    },
    "RiVeR-N16": {
        "exact_W": (['t0', 't1'], 8320, 8320),
        "exact_opening": (['t0', 't1', 'e_eval', 'y_eval', 'digits', 'randomness'], 9584, 9552),
        "oom": (['B', 'x', 'f1', 'zb', 'zs', 'zm'], 25960, 21080),
    },
    "RiVeR-N256": {
        "exact_W": (['t0', 't1'], 8320, 8320),
        "exact_opening": (['t0', 't1', 'e_eval', 'y_eval', 'digits', 'randomness'], 9584, 9552),
        "oom": (['B', 'x', 'f1', 'zb', 'zs', 'zm'], 48336, 34632),
    },
    "RiVeR-N64": {
        "exact_W": (['t0', 't1'], 8320, 8320),
        "exact_opening": (['t0', 't1', 'e_eval', 'y_eval', 'digits', 'randomness'], 9584, 9552),
        "oom": (['B', 'x', 'f1', 'zb', 'zs', 'zm'], 31616, 25052),
    },
    "RiVeR-N8": {
        "exact_W": (['t0', 't1'], 8320, 8320),
        "exact_opening": (['t0', 't1', 'e_eval', 'y_eval', 'digits', 'randomness'], 9584, 9552),
        "oom": (['B', 'x', 'f1', 'zb', 'zs', 'zm'], 24608, 19772),
    },
    "RiVeR-TOY": {
        "exact_W": (['t0', 't1'], 8320, 8320),
        "exact_opening": (['t0', 't1', 'e_eval', 'y_eval', 'digits', 'randomness'], 9584, 9552),
        "oom": (['B', 'x', 'f1', 'zb', 'zs', 'zm'], 2512, 2012),
    },
}

def test_global_constants_are_pinned():
    """These are wire format, not tuning knobs."""
    g = manifest.global_constants()
    assert g["sigma_den"] == SIGMA_DEN == 1 << 20
    assert g["prob_bits"] == 192
    assert g["gaussian_tailcut"] == 14
    assert g["verifier_tailcut"] == 6
    assert g["rice_const_den"] == 10 ** 30
    assert g["rice_const_num"] == 1177410022515474691011569326460


def test_the_manifest_matches_the_checked_in_file_byte_for_byte():
    """The whole table is frozen, not the fields someone asserted.

    Field-by-field checks are only ever as complete as the list of fields
    in them: pinning order and aggregate sizes while merely *checking for
    the presence* of coder parameters let a `Uniform` modulus be swapped
    for another of the same bit width with every assertion still green.
    Reproduced before this test existed.

    A byte-for-byte comparison has no such list.  Any change to any value
    -- a modulus, a Rice parameter, a bound, a row count, the framing --
    is a diff.  `python3 manifest.py --write` regenerates it, and that is
    a deliberate act exactly like `make vectors`: the diff is the list of
    fields whose encoding moved.
    """
    with open(manifest.MANIFEST_PATH) as handle:
        frozen = handle.read()
    current = manifest.canonical_json()
    if frozen != current:
        # Name the first divergence rather than dumping 38 KB of JSON.
        import difflib
        diff = list(difflib.unified_diff(frozen.splitlines(),
                                         current.splitlines(),
                                         "manifest.json", "computed", n=1))
        raise AssertionError(
            "the wire manifest moved; run `python3 manifest.py --write` if "
            "that was deliberate:\n" + "\n".join(diff[:20]))


def test_the_frozen_file_is_canonical():
    """Sorted keys, fixed indent, trailing newline.

    Otherwise "byte for byte" would be a comparison of formatting, and a
    reordering would read as a change.
    """
    with open(manifest.MANIFEST_PATH) as handle:
        raw = handle.read()
    assert raw.endswith("\n")
    assert raw == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"


def test_the_framing_is_stated_not_summarised():
    """A port needs the block order and prefix encoding, not a byte count.

    `4 + |pi_OOM| + 4 + |pi_ex|` is eight bytes of overhead under two
    different layouts; only one of them decodes.
    """
    blob = manifest.manifest()
    for name, entry in blob["profiles"].items():
        fr = entry["framing"]
        assert fr["block_order"] == ["oom", "exact"], name
        assert fr["length_prefix_bytes"] == 4
        assert fr["length_prefix_endian"] == "little"
        assert fr["total_framing_bytes"] == 8
        assert fr["prefix_bounded_by_layout"] is True

    # ... and the framing the codec actually writes agrees.
    from river import RiVeR
    from params import TOY_PARAMS
    scheme = RiVeR(TOY_PARAMS)
    pp = scheme.setup(b"\x00" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(TOY_PARAMS.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[1]
    v, pi = scheme.eval(pp, pk, sk, ring, b"framing", b"\x51" * 32)
    blob_bytes = scheme.proof_encode(pi)
    oom = scheme.codec.oom_encode(pi["oom"])
    ex = scheme.exact.proof_encode(pi["ex"])
    assert len(blob_bytes) == 4 + len(oom) + 4 + len(ex)
    assert blob_bytes[:4] == len(oom).to_bytes(4, "little")
    assert blob_bytes[4:4 + len(oom)] == oom
    assert blob_bytes[4 + len(oom):8 + len(oom)] == \
        len(ex).to_bytes(4, "little")


def test_the_manifest_is_readable_standalone():
    """Every section the port needs is present, with no outside reference.

    The manifest is the handoff artifact between the two implementations,
    so it has to carry the whole wire-visible table itself: a consumer that
    had to look something up elsewhere would be reading a different table.
    """
    blob = manifest.manifest()
    assert set(blob) == {"global", "profiles"}
    assert blob["global"]
    assert set(blob["profiles"]) == set(PROFILES)
    for name, prof in blob["profiles"].items():
        assert prof, name


def test_layouts_are_frozen_in_wire_order():
    """Field order is wire format: a permutation decodes to other values.

    The layouts are *walked*, not restated, so this pins what the encoder
    actually does rather than a second copy of it.
    """
    blob = manifest.manifest()
    assert sorted(blob["profiles"]) == sorted(FROZEN_LAYOUTS)
    for name, expected in FROZEN_LAYOUTS.items():
        layouts = blob["profiles"][name]["layouts"]
        assert sorted(layouts) == sorted(expected), name
        for key, (order, max_bytes, min_bytes) in expected.items():
            got = layouts[key]
            assert got["order"] == order, (name, key, got["order"])
            assert got["max_bytes"] == max_bytes, (name, key)
            assert got["min_bytes"] == min_bytes, (name, key)


def test_every_layout_field_declares_a_complete_coder():
    """Nothing a port would have to guess is left off.

    Every field names its coder and carries the parameters that reach the
    wire; a Rice field carries `k` and its bound, a fixed-width field its
    width, and a ring-valued field the modulus it is centred against.
    """
    blob = manifest.manifest()
    for name, entry in blob["profiles"].items():
        for key, layout in entry["layouts"].items():
            assert layout["order"], (name, key)
            for field, spec in layout["fields"].items():
                where = (name, key, field)
                assert spec["coder"] in ("rice", "signed", "uniform"), where
                assert spec["count"] == spec["cols"] * (spec["rows"] or 1), where
                if spec["coder"] == "rice":
                    assert "k" in spec and "bound" in spec, where
                else:
                    assert "width" in spec, where


def test_the_exact_layer_dimensions_are_carried():
    """Including which rank plays which role -- this table's whole subject."""
    from exact import ExactParams
    blob = manifest.manifest()
    for name, entry in blob["profiles"].items():
        ex = ExactParams(get(name))
        spec = entry["exact"]
        assert spec["identity_rank"] == ex.ell_tilde == ex.t0_rows
        assert spec["tail_rank"] == ex.n_tilde
        assert spec["response_rank"] == ex.kappa - ex.ell_tilde == 13
        assert spec["q_tilde"] == 67107713
        assert (spec["d_tilde"], spec["l_split"]) == (256, 64)
        assert spec["block_used"] == 32 and spec["block_slots"] == 64
        assert spec["radix_weights"] == [1, 3, 9, 17]
        assert entry["framing"]["total_framing_bytes"] == 8


def test_every_field_of_every_profile_is_frozen():
    blob = manifest.manifest()
    assert sorted(blob["profiles"]) == sorted(FROZEN)
    for name, expected in FROZEN.items():
        fields = blob["profiles"][name]["fields"]
        assert sorted(fields) == sorted(expected), name
        for field, want in expected.items():
            spec = fields[field]
            got = (spec.get("sigma_num"), spec.get("rice_k"),
                   spec.get("bound"), spec.get("width_bits"))
            assert got == want, (name, field, got, want)


def test_the_rice_constant_is_far_from_a_boundary():
    """`k` moves when `sqrt(2 ln 2) sigma` crosses a power of two.

    The constant used to be `11774/10000`, a relative error of `8.5e-6`,
    which is fine exactly as long as no field sits within that of a
    boundary -- something nothing checked.  This measures the actual
    distance at every Gaussian field of every profile, so "unlikely" is
    replaced by a number.
    """
    worst, worst_at = 1.0, None
    for name, par in sorted(PROFILES.items()):
        for _, width_attr, _ in manifest.GAUSSIAN_FIELDS:
            num, den = rational_sigma(getattr(par, width_attr))
            scaled = Fraction(RICE_CONST_NUM * num, RICE_CONST_DEN * den)
            k = optimal_rice_k((num, den))
            assert 2 ** k <= scaled < 2 ** (k + 1), (name, width_attr)
            # Relative distance to whichever boundary is nearer.
            below = (scaled - 2 ** k) / scaled
            above = (2 ** (k + 1) - scaled) / scaled
            d = float(min(below, above))
            if d < worst:
                worst, worst_at = d, (name, width_attr)

    # What has to hold is that no field is within the constant's own error
    # of a boundary -- otherwise a differently-rounded `sqrt(2 ln 2)` picks
    # a different `k` and the two implementations disagree on the wire.
    # The 30-digit constant is good to about 4e-31, so the margin is vast;
    # the number is reported so a future profile that eroded it would say
    # so rather than merely staying above a round threshold.
    #
    # the paper cut the worst case from 1.9% to 0.27% (`zs` at
    # `RiVeR-TOY`, whose `sigma_s` moved when `B_s` was redefined).  Still
    # 25 orders of magnitude of headroom, but no longer 1%.
    assert worst > 1e-3, (worst, worst_at)
    assert worst_at[1] == "sigma_s", worst_at

    # The margin against the constant itself, which is the real property.
    const_error = abs(Fraction(RICE_CONST_NUM, RICE_CONST_DEN)
                      - Fraction(11774100225154746910115693264601, 10 ** 31))
    assert worst > 10 ** 20 * float(const_error)


def test_the_four_digit_constant_would_have_agreed():
    """Pins that the constant change moved no encoding at these profiles."""
    for name, par in sorted(PROFILES.items()):
        for _, width_attr, _ in manifest.GAUSSIAN_FIELDS:
            num, den = rational_sigma(getattr(par, width_attr))
            old = (11774 * num) // (10000 * den)
            old_k = max(0, old.bit_length() - 1)
            assert optimal_rice_k((num, den)) == old_k, (name, width_attr)


def test_bounds_are_the_largest_value_that_can_pass():
    """The encoder's cap and the acceptance test agree by construction.

    `bound = floor(sqrt(bound_sq))`, so `bound` passes and `bound + 1` does
    not -- no `ceil` on a float that could land either side.
    """
    for name, par in sorted(PROFILES.items()):
        codec = RiVeRCodec(par)
        for field, _, bound_attr in manifest.GAUSSIAN_FIELDS:
            bound_sq = getattr(par, bound_attr)
            cap = getattr(codec, f"bound_{field}")
            assert cap == floor_sqrt(bound_sq), (name, field)
            assert cap ** 2 <= bound_sq, (name, field)
            assert (cap + 1) ** 2 > bound_sq, (name, field)


def test_sigma_pinning_is_reproducible_from_the_manifest():
    """A port can rebuild each sampler from `(num, den)` alone."""
    blob = manifest.manifest()
    for name, entry in blob["profiles"].items():
        par = get(name)
        for field, width_attr, _ in manifest.GAUSSIAN_FIELDS:
            spec = entry["fields"][field]
            assert spec["sigma_den"] == SIGMA_DEN
            assert (spec["sigma_num"], spec["sigma_den"]) == \
                rational_sigma(getattr(par, width_attr))
            # The pinned rational is within half an ulp of 2^-20 of the
            # width it stands in for -- that is all `round` promises.
            exact = Fraction(spec["sigma_num"], spec["sigma_den"])
            assert abs(float(exact) - getattr(par, width_attr)) <= 0.5 / SIGMA_DEN


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_manifest.py: {len(tests)} tests passed")
