#!/usr/bin/env python3
"""gen_lanes_manifest.py -- `../river-py/lanes_manifest.json` -> `src/lanes_manifest.rs`.

The companion of `gen_manifest.py`, for the LANES parameter table.

`river-py/lanes_manifest.py` produces the frozen table: every wire- and
security-visible LANES value frozen with a Paper/Derived/Repair label. This
script transcribes it into a Rust
`LanesManifest` const so that

  * `exact::LANES_PARAMETER_MANIFEST` is `Some(&…)` here for the same
    reason it is not `None` there -- the two implementations are gated on
    the same table rather than each on its own opinion of one;
  * `make lanes-manifest-check` regenerates and requires an empty diff, so
    the table cannot drift from the reference it was copied from;
  * `exact::validate_lanes_manifest` compares every constant against what
    `lanes::params` actually consumes, so it cannot drift from *this* crate
    either.

The gate is unaffected: possessing a table is not permission to use the
reserved production alias. The two implementations validate the same table
and report the same outstanding composition-policy condition through the
shared `LANES_GATE_CAUSES` vocabulary.

Deliberate act, like `make kat-regen`.  Run `make lanes-manifest-regen`.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.normpath(os.path.join(HERE, "..", "..", "river-py",
                                           "lanes_manifest.json"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "src",
                                            "lanes_manifest.rs"))


def value_of(cell):
    """The value inside a `{"value": …, "provenance": …}` cell."""
    if isinstance(cell, dict) and "value" in cell:
        return cell["value"]
    return cell


def rational(cell):
    """A `(num, den)` Rust literal from a manifest rational."""
    raw = value_of(cell)
    f = Fraction(raw) if isinstance(raw, str) else Fraction(raw)
    return f"({f.numerator}, {f.denominator})"


def rs_str(text):
    """A Rust string literal.

    The manifest carries prose -- `how` strings, failure rules, the
    discrepancy note -- and some of it contains quotes and backslashes.
    """
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(blob, source_digest):
    out = []
    w = out.append

    w("//! LANES parameter manifest -- **generated**, do not edit.")
    w("//!")
    w("//! `scripts/gen_lanes_manifest.py` writes this from")
    w("//! `../river-py/lanes_manifest.json`, the frozen parameter table.")
    w("//! `make lanes-manifest-check` requires the two to agree.")
    w("//!")
    w("//! This is a **typed projection**, not a copy: the source carries")
    w("//! prose (`how` strings, provenance notes, the security summary)")
    w("//! that has no place in a Rust const.  So `LANES_MANIFEST` also")
    w("//! carries `source_sha256`, the digest of the canonical JSON it was")
    w("//! projected from -- which is what makes projection *drift*")
    w("//! detectable: if the source moves and this file does not, the")
    w("//! digest differs and `make lanes-manifest-check` fails.")
    w("//!")
    w("//! [`crate::exact::validate_lanes_manifest`] compares each constant")
    w("//! against what [`crate::lanes::params`] consumes, so a table that")
    w("//! described a different parameter set would fail there rather than")
    w("//! as \"proof bytes differ\" in a cross-language vector.")
    w("//!")
    w("//! Possessing this table is **not** permission to run the backend;")
    w("//! see [`crate::exact::lanes_unavailable_reason`].")
    w("")
    w("use crate::exact::{")
    w("    DimensionSpec, EstimatorSpec, LanesManifest, ManifestConstant, RankRoleSpec,")
    w("    RecoverySpec, ResponseBoundSpec, SamplerSpec, TranscriptField, TranscriptRound,")
    w("    WireField, WireSpec,")
    w("};")
    w("")

    tr = blob["transcript"]
    absorbed = value_of(tr["absorbed_fields"])
    seps = value_of(tr["domain_separators"])
    reconstructed = set(value_of(tr["reconstructed_not_transmitted"]))
    rounds = value_of(tr["rounds"])

    def form_of(name):
        base = name.split("|")[0]
        if base == "statement":
            return "statement digest"
        return "reconstructed" if base in reconstructed else "transmitted"

    w("/// The absorbed fields, flattened, in hash order.")
    w(f"pub static TRANSCRIPT: [TranscriptField; {len(absorbed)}] = [")
    for name in absorbed:
        w("    TranscriptField {")
        w(f"        name: {rs_str(name)},")
        w(f"        domain_separator: {rs_str(seps.get('lanes', 'LANES'))},")
        w(f"        hashed_form: {rs_str(form_of(name))},")
        w("    },")
    w("];")
    w("")

    # ---- rounds, with their absorb groups -------------------------------
    for k, rnd in enumerate(rounds):
        names = ", ".join(rs_str(a) for a in rnd["absorbs"])
        w(f"static ROUND_{k}: [&str; {len(rnd['absorbs'])}] = [{names}];")
    w("")
    w("/// The three Fiat-Shamir rounds, each with what precedes its")
    w("/// challenge.  A `|`-joined name is one `absorb` argument: the parts")
    w("/// are concatenated before hashing, and a port that hashed them")
    w("/// separately would derive different challenges.")
    w(f"pub static ROUNDS: [TranscriptRound; {len(rounds)}] = [")
    for k, rnd in enumerate(rounds):
        name = rnd["challenge"]
        w("    TranscriptRound {")
        w(f"        challenge: {rs_str(name)},")
        w(f"        separator: {rs_str(seps.get(_sep_key(name), '.' + name))},")
        w(f"        absorbs: &ROUND_{k},")
        w("    },")
    w("];")
    w("")

    # ---- wire ------------------------------------------------------------
    fields = value_of(blob["wire"]["fields"])
    w("/// The serialized proof layout, in wire order, with coder")
    w("/// parameters.  `bits: None` marks the one variable-length field.")
    w(f"pub static WIRE_FIELDS: [WireField; {len(fields)}] = [")
    for f in fields:
        w("    WireField {")
        w(f"        name: {rs_str(f['name'])},")
        w(f"        rows: {f['rows']},")
        w(f"        cols: {f['cols']},")
        w(f"        coder: {rs_str(f['coder'])},")
        w(f"        bits: {_opt(f.get('bits'))},")
        w(f"        modulus: {_opt(f.get('modulus'))},")
        w(f"        bound: {_opt(f.get('bound'))},")
        w(f"        width_bits: {_opt(f.get('width_bits'))},")
        w(f"        rice_k: {_opt(f.get('rice_k'))},")
        w("    },")
    w("];")
    w("")

    # ---- constants -------------------------------------------------------
    consts = blob["constants"]
    order = sorted(consts)
    w("/// The gated constants, each selected *by value*.")
    w("///")
    w("/// A **Paper** label on a retained value does not make the paper")
    w("/// have chosen it, so the gate compares these against what")
    w("/// [`crate::lanes::params`] consumes.")
    w(f"pub static CONSTANTS: [ManifestConstant; {len(order)}] = [")
    for name in order:
        cell = consts[name]
        w("    ManifestConstant {")
        w(f"        name: {rs_str(name)},")
        w(f"        value: {rational(cell)},")
        w(f"        provenance: {rs_str(cell['provenance'])},")
        w("    },")
    w("];")
    w("")

    dim = blob["dimensions"]
    rr = blob["rank_roles"]
    sam = blob["sampler"]
    rb = blob["response_bounds"]
    rec = blob["recovery"]
    est = blob["estimator"]
    wire = blob["wire"]

    w("/// The frozen table itself.")
    w("pub static LANES_MANIFEST: LanesManifest = LanesManifest {")
    w(f"    source_sha256: {rs_str(source_digest)},")
    w("    dimensions: DimensionSpec {")
    for field, key in (("d_tilde", "d_tilde"), ("l_split", "l_split"),
                       ("sub_degree", "sub_degree"), ("q_tilde", "q_tilde"),
                       ("q_tilde_bits", "q_tilde_bits"),
                       ("n_tilde", "n_tilde"), ("ell_tilde", "ell_tilde"),
                       ("n_ex", "N_ex"), ("alpha", "alpha"),
                       ("d_drop", "D"), ("w_hat", "w_hat"),
                       ("w_tilde", "w_tilde"),
                       ("delta_stride", "delta_stride"),
                       ("n_lwe", "n_lwe"), ("m_lwe", "m_lwe"),
                       ("block_slots", "block_slots"),
                       ("block_payload", "block_payload"),
                       ("message_blocks", "message_blocks")):
        w(f"        {field}: {value_of(dim[key])},")
    w("    },")
    w("    rank_roles: RankRoleSpec {")
    for key in ("identity_rank", "tail_rank", "kappa", "response_rank"):
        w(f"        {key}: {value_of(rr[key])},")
    w("    },")
    w("    sampler: SamplerSpec {")
    w(f"        sigma_r: {rational(sam['sigma_r'])},")
    w(f"        sigma_y: {rational(sam['sigma_y'])},")
    w(f"        epsilon_exponent: {abs(int(value_of(sam['epsilon_exponent'])))},")
    w(f"        convention: {rs_str(value_of(sam['convention']))},")
    w(f"        tail_cut_r: {value_of(sam['tail_cut_r'])},")
    w(f"        tail_cut_y: {value_of(sam['tail_cut_y'])},")
    w(f"        prob_bits: {value_of(sam['prob_bits'])},")
    w("    },")
    w("    response_bounds: ResponseBoundSpec {")
    w(f"        inf: {value_of(rb['inf'])},")
    w(f"        l2: {value_of(rb['l2'])},")
    w(f"        comparison: {rs_str(value_of(rb['comparison']))},")
    w(f"        population: {rs_str(value_of(rb['population']))},")
    w("    },")
    w("    recovery: RecoverySpec {")
    w(f"        d_drop: {value_of(rec['d_drop'])},")
    w(f"        rounding: {rs_str(value_of(rec['rounding']))},")
    w(f"        ties: {rs_str(value_of(rec['ties']))},")
    for field in ("omitted_response_rows", "omitted_response_coefficients",
                  "omitted_t0_low_bits", "recovery_carries"):
        w(f"        {field}: {value_of(rec[field])},")
    w(f"        hint_alphabet: {rs_str(value_of(rec['hint_alphabet']))},")
    w(f"        limit: {value_of(rec['error_bound'])},")
    w(f"        failure_rule: {rs_str(value_of(rec['failure_rule']))},")
    w(f"        verification_rule: {rs_str(value_of(rec['verification_rule']))},")
    w(f"        encoding: {rs_str(value_of(rec['encoding']))},")
    w("    },")
    w("    transcript: &TRANSCRIPT,")
    w("    rounds: &ROUNDS,")
    w("    wire: WireSpec {")
    w("        fields: &WIRE_FIELDS,")
    w(f"        total_bits: {_opt(value_of(wire['total_bits']))},")
    w(f"        fixed_bits: {value_of(wire['fixed_bits'])},")
    w(f"        kb_convention: {rs_str(value_of(wire['kb_convention']))},")
    w(f"        discrepancy: Some({rs_str(value_of(wire['discrepancy']))}),")
    w("    },")
    w("    estimator: EstimatorSpec {")
    for key in ("hint_mlwe_inputs", "hint_mlwe_outputs", "msis_inputs",
                "msis_outputs", "challenge"):
        w(f"        {key}: {rs_str(json.dumps(value_of(est[key]), sort_keys=True))},")
    w("    },")
    w("    constants: &CONSTANTS,")
    w("};")
    w("")
    return "\n".join(out) + "\n"


def _opt(value):
    """A Rust `Option<..>` literal from a possibly-`None` manifest value."""
    return "None" if value is None else f"Some({value})"


def _sep_key(challenge):
    """The `domain_separators` key for a challenge name."""
    return {"c": "challenge"}.get(challenge, challenge)


def rustfmt(text):
    """Normalise through `rustfmt`, so `--check` is stable under `cargo fmt`."""
    exe = shutil.which("rustfmt")
    if exe is None:                                   # pragma: no cover
        return text
    done = subprocess.run([exe, "--edition", "2021", "--emit", "stdout"],
                          input=text, capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the checked-in file differs")
    args = ap.parse_args()

    with open(args.input, "rb") as fh:
        raw = fh.read()
    blob = json.loads(raw)
    # The digest is over the *canonical* JSON, not the file bytes: the
    # source is written with `sort_keys=True, indent=1`, but a consumer
    # should not have to reproduce the formatting to check the content.
    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    text = rustfmt(render(blob, digest))

    if args.check:
        try:
            with open(args.out) as fh:
                current = fh.read()
        except FileNotFoundError:
            print(f"{args.out}: missing", file=sys.stderr)
            return 1
        if current != text:
            print(f"{args.out} differs from {args.input}; "
                  f"run `make lanes-manifest-regen`", file=sys.stderr)
            return 1
        print(f"{args.out}: up to date with {args.input}")
        return 0

    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"wrote {args.out} from {args.input}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
