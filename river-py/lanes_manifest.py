"""
lanes_manifest.py -- the frozen LANES parameter manifest, and its generator.

**Experimental.**  `status` is `"experimental"` and nothing here can lift
the production `"lanes"` backend name; see `exact.lanes_unavailable_reason`
for the four independent conditions that would.

What this is
------------
One machine-readable table carrying every wire- or security-visible value
the LANES layer depends on, with a Paper/Derived/Repair label on each.
`lanes_manifest.json` is that table and this module builds it.

The file is the frozen artifact; the code is checked *against* it.  That
direction matters: a manifest regenerated from the code on every import
would agree with the code by construction and certify nothing.  So

    make lanes-manifest        # print it
    make lanes-manifest-regen  # REWRITE lanes_manifest.json -- deliberate
    python3 lanes_manifest.py --check

and `test_lanes_manifest.py` fails if any value the implementation consumes
has moved away from the frozen one.

Provenance
----------
Three labels, and the distinction is load-bearing:

**Paper**
    Printed in the current PDF/TeX.  The dimensions, `w_hat`, `q~`, `D`,
    `(n_LWE, m_LWE)`, the Gaussian widths `s_0, s_1, s_2, s` and the
    convention they are stated in, `beta'_BDLOP`, `B_MSIS`, `delta_MSIS`,
    `delta_MLWE`, `q~/B_MSIS`.

    Most of that became Paper.  Under the previous revision
    only the *outputs* were printed and the widths behind them were this
    tree's own selection; the labels below moved accordingly, which is
    exactly what a provenance table is for.

**Derived**
    Deterministically derived from Paper values by a documented
    convention: `kappa`, the rank roles, `N_Z`, the once-only rounding of
    each width to denominator `2^20` where the sampler consumes it, the
    Euclidean response bound (the paper's `(2 s)^2` rule at the transmitted
    rank), and `sigma_MLWE`.

**Repair**
    An implementation choice needed to make an ambiguous or absent part
    executable.  The whole recovery-hint construction -- bucket count,
    hint alphabet, failure rule, encoding -- which the revision still does
    not specify; the response *infinity* bound, for which it states none;
    the wire layout; and the sampler tail cuts.

A Repair value is never later describable as though the paper printed it.
That is what the labels are for, and `validate_lanes_manifest` compares the
*values*, not just the labels -- a label alone would let a wrong constant
be relabelled **Paper** and travel on.

What it does not contain
------------------------
Estimator *outputs*.  There is no lattice estimator in this repository, so
`estimator.hint_mlwe_outputs` and `estimator.msis_outputs` record the
inputs and say plainly that no run has been performed.  That is deliberately
not enough to open anything: `exact.LANES_SECURITY_EVIDENCE` is a separate
state, and the gate requires it independently of this file.
"""

import json
import os
from fractions import Fraction

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "lanes_manifest.json")

#: What this manifest claims to be.  Only `"final"` can ever open the
#: production backend name; see `exact.lanes_unavailable_reason`.
#:
#: `"final"`, and the word is about *this table*, not
#: about the backend.  Every wire- and security-visible value below is now
#: either printed by the paper or derived from it by a stated convention,
#: with the remaining Repairs (the recovery-hint rules, the wire layout,
#: the infinity bound) labelled as such.  Before that revision the widths
#: were selected here under a predicate the paper does not state, which no
#: labelling could make final.
#:
#: The backend stays shut regardless: the gate has three more conditions
#: after this one, and `LANES_SECURITY_EVIDENCE` is the one that fails.
STATUS = "final"

PAPER, DERIVED, REPAIR = "Paper", "Derived", "Repair"


def _wire_field(f):
    """One serialized field, with everything a port needs to encode it."""
    coder = f.coder
    kind = type(coder).__name__
    rows = 1 if f.rows is None else f.rows
    out = {"name": f.name, "rows": rows, "cols": f.cols, "coder": kind}
    if kind == "Uniform":
        out["modulus"] = coder.modulus
        out["width_bits"] = coder.width
        out["bits"] = rows * f.cols * coder.width
    elif kind == "Signed":
        out["bound"] = coder.bound
        out["width_bits"] = coder.width
        out["bits"] = rows * f.cols * coder.width
    elif kind == "Rice":
        out["rice_k"] = coder.k
        out["bound"] = coder.bound
        out["max_high"] = coder.max_high
        # Variable by construction: `high` is unary.  `None` rather than 0,
        # because 0 would read as "no bits" to a consumer summing them.
        out["bits"] = None
    else:                                          # pragma: no cover
        raise ValueError(f"unhandled coder {kind} on field {f.name}")
    return out


def _fixed_bits(layout):
    """Bits contributed by the fixed-width fields, or `None` if any is not."""
    total = 0
    for f in layout.fields:
        spec = _wire_field(f)
        if spec["bits"] is None:
            continue
        total += spec["bits"]
    return total


def _rat(value):
    """A rational as the canonical string the manifest stores."""
    f = Fraction(value)
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 \
        else str(f.numerator)


def manifest():
    """Build the manifest from the live modules.

    Only `--write` should call this in anger: everything else compares
    against the frozen file.
    """
    import lanes_backend as lb
    import lanes_params as lp
    import lanes_proof
    import lanes_ring as R
    from params import TOY_PARAMS
    from sample import DS_EXACT, GAUSSIAN_TAILCUT, PROB_BITS

    from estimate_lanes import load as _load_security

    backend = lb.LanesBackend.experimental(TOY_PARAMS)
    layout = backend.proof_layout
    evidence = _load_security()

    return {
        "status": STATUS,
        "selected_by": ("the paper: a closed form with no free constant.  "
                        "Nothing in this table is searched"),
        "note": ("The parameterization is the paper's and reproduces every "
                 "figure it prints (beta' = 45430.6, B_MSIS = 15991562, "
                 "q~/B = 4.2, delta_MSIS = 1.0037, D = 17).  What is not "
                 "the paper's: the recovery-hint rules, the wire layout, "
                 "and the response infinity bound, each labelled Repair.  "
                 "The security evidence does not reach the paper's own "
                 "target -- see `security` -- so the backend stays gated."),
        "security": {
            "best_bits": evidence and evidence["best_bits"],
            "target_bits": evidence and evidence["target_bits"],
            "verdict": evidence and evidence["verdict"],
            "evidence": "lanes_security.json",
            "estimator": evidence and evidence["tool"]["commit"],
            "msis_bits": evidence and evidence["msis"]["published"]["bits"],
            "mlwe_bits_by_reading": evidence and {
                r["name"]: r["bits"] for r in evidence["mlwe"]},
            "delta_msis_reproduced": evidence
            and evidence["published_delta"]["msis"]["reproduced"],
            "delta_mlwe_reproduced": evidence
            and evidence["published_delta"]["mlwe"]["reproduced"],
            "hint_mlwe_statistical_loss_log2": evidence
            and evidence["hint_mlwe_statistical_loss_log2"],
            "still_missing": evidence and evidence["blockers"],
        },

        "dimensions": {
            "d_tilde": {"value": R.DTILDE, "provenance": PAPER},
            "l_split": {"value": R.LSPLIT, "provenance": PAPER},
            "sub_degree": {"value": R.SUBDEG, "provenance": DERIVED,
                           "how": "d_tilde / l_split"},
            "q_tilde": {"value": R.QTILDE, "provenance": PAPER},
            "q_tilde_bits": {"value": (R.QTILDE - 1).bit_length(),
                             "provenance": DERIVED},
            "n_tilde": {"value": lp.N_TILDE, "provenance": PAPER},
            "ell_tilde": {"value": lp.ELL_TILDE, "provenance": PAPER},
            "N_ex": {"value": lp.N_EX, "provenance": PAPER},
            "alpha": {"value": lp.ALPHA, "provenance": PAPER,
                      "how": "maximum degree in the exact relation"},
            "D": {"value": lp.D_DROP, "provenance": PAPER},
            "w_hat": {"value": lp.W_HAT, "provenance": PAPER},
            "delta_stride": {"value": lp.DELTA, "provenance": DERIVED,
                             "how": "= sub_degree; challenge partition"},
            "w_tilde": {"value": lp.W_TILDE, "provenance": DERIVED,
                        "how": "w_hat / delta_stride, per residue class"},
            "n_lwe": {"value": lp.N_LWE, "provenance": PAPER,
                      "how": "n~ d~ = 1024, printed at 7.suppl.tex"},
            "m_lwe": {"value": lp.M_LWE, "provenance": PAPER,
                      "how": "(l~ + N_ex + alpha) d~ = 3328, printed"},
            "message_blocks": {"value": lp.N_EX, "provenance": PAPER},
            "block_slots": {"value": R.LSPLIT, "provenance": PAPER},
            "block_payload": {"value": backend.ex.d, "provenance": DERIVED,
                              "how": "outer d; the rest of each block is "
                                     "zero padding, constrained by the "
                                     "linear system"},
        },

        # The two letters are `n~ = l~ = 4` today, so nothing
        # numeric can tell them apart -- which is exactly why the `how`
        # strings have to name the right one.  They previously said `n~`
        # for the identity rank and `kappa - n~` for both the tail and the
        # response, none of which is the definition the code uses.
        "rank_roles": {
            "identity_rank": {"value": lp.IDENTITY_RANK, "provenance": DERIVED,
                              "how": "l~; rows of t_0 and B_0's I block"},
            "tail_rank": {"value": lp.TAIL_RANK, "provenance": DERIVED,
                          "how": "n~; width of the shared random tail, and "
                                 "the MLWE secret rank"},
            "kappa": {"value": lp.KAPPA, "provenance": DERIVED,
                      "how": "n~ + l~ + N_ex + alpha"},
            "response_rank": {"value": lp.RESPONSE_RANK, "provenance": DERIVED,
                              "how": "kappa - l~; transmitted rows of z "
                                     "under Bai-Galbraith compression"},
            "roles_are_distinguishable": {
                "value": False, "provenance": DERIVED,
                "how": "n~ == l~ == 4 at these parameters, so a port that "
                       "swapped them would agree numerically; "
                       "exact.lanes_rank_roles is the single definition and "
                       "test_exact.py drives it at (7, 8) where they differ"},
        },

        "sampler": {
            "epsilon_exponent": {"value": -lp.SMOOTHING_EPS_EXP,
                                 "provenance": PAPER,
                                 "how": "eps = 2^-100"},
            "s_0": {"value": str(lp.S0), "provenance": PAPER,
                    "how": "sqrt(ln(2 d~ (1 + 1/eps))) / pi; printed 2.7668"},
            "s_1": {"value": str(lp.S1), "provenance": PAPER,
                    "how": "2 s_0; printed 5.5336"},
            "s_2": {"value": str(lp.S2), "provenance": PAPER,
                    "how": "2 w_hat s_0; printed 243.4775"},
            "s_response": {"value": str(lp.S_RESPONSE), "provenance": PAPER,
                           "how": "2 sqrt(2) w_hat s_0; printed 344.3291.  "
                                  "Equals sqrt(s_2^2 + w_hat^2 s_1^2): the "
                                  "worst-case l1 challenge bound"},
            "convention": {"value": "standard deviation", "provenance": PAPER,
                           "how": "the paper's `s` is the standard deviation "
                                  "and its `sigma` the [KLSS23] Gaussian "
                                  "parameter, sigma = s sqrt(2 pi); it pins "
                                  "this by printing both"},
            "sigma_r": {"value": _rat(lp.SIGMA_R), "provenance": DERIVED,
                        "how": "s_1 rounded once to denominator 2^20, where "
                               "the sampler consumes it"},
            "sigma_y": {"value": _rat(lp.SIGMA_Y), "provenance": DERIVED,
                        "how": "s_2 rounded once to denominator 2^20"},
            "sigma_mlwe": {"value": _rat(lp.SIGMA_MLWE_SQ),
                           "provenance": DERIVED,
                           "how": "squared; 1/sigma^2 = 2(1/s_1^2 + "
                                  "w_hat^2/s_2^2) per [KLSS23].  Equals "
                                  "s_0^2 by construction"},
            "sigma_denominator": {"value": lp.SIGMA_DEN, "provenance": DERIVED,
                                  "how": "2^20; one rounding, where the "
                                         "sampler consumes it"},
            "tail_cut_r": {"value": GAUSSIAN_TAILCUT, "provenance": REPAIR,
                           "how": "statistical_tailcut over the transcript"},
            "tail_cut_y": {"value": GAUSSIAN_TAILCUT, "provenance": REPAIR,
                           "how": "the same cut; both are gaussian_int"},
            "prob_bits": {"value": PROB_BITS, "provenance": REPAIR,
                          "how": "acceptance-threshold precision"},
            "proposal": {"value": "uniform-proposal rejection, not CDT",
                         "provenance": REPAIR},
            "internal_rejection": {"value": "none -- LANES uses no internal "
                                            "rejection sampling",
                                   "provenance": PAPER},
        },

        "response_bounds": {
            "beta_prime_bdlop": {"value": str(lp.BETA_PRIME_BDLOP),
                                 "provenance": PAPER,
                                 "how": "2 s sqrt(N_z_paper); printed 45430.6"},
            "n_z_paper": {"value": lp.N_Z_PAPER, "provenance": PAPER,
                          "how": "(n~ + l~ + N_ex + alpha) d~ = 4352, the "
                                 "full rank-kappa opening"},
            "b_msis": {"value": str(lp.B_MSIS), "provenance": PAPER,
                       "how": "8 w_hat beta'; printed 15991562"},
            "l2": {"value": lp.Z_NORM2_BOUND, "provenance": DERIVED,
                   "how": "(2 s)^2 * population: the paper's per-coordinate "
                          "rule at the transmitted rank"},
            "l2_vs_paper": {
                "value": lp.Z_NORM2_BOUND_PAPER, "provenance": DERIVED,
                "how": "beta'^2, the paper's bound at its own dimension "
                       "4352.  Ours is over 3328 and is strictly smaller, "
                       "so the published B_MSIS remains an upper bound on "
                       "what an extractor obtains here"},
            "l2_honest_requirement": {
                "value": lp.Z_NORM2_REQUIRED, "provenance": DERIVED,
                "how": "Laurent-Massart on Sigma = sigma_y^2 I + sigma_r^2 "
                       "M M^T at 2^-128: the smallest bound an honest "
                       "response can be held to.  The enforced l2 is above "
                       "it, which is what makes the paper's rule usable"},
            "inf": {"value": lp.Z_INF_BOUND, "provenance": REPAIR,
                    "how": "ceil(Z_TAILCUT sqrt(Var[z])) on the exact "
                           "rational; the paper states no infinity bound"},
            # Two cells, deliberately.  `comparison` is *data* -- the bare
            # operator a port has to implement -- and `comparison_note` is
            # the prose around it.  They were one prose string, which meant
            # `river-rs`'s validator had to grep an English sentence for
            # `<`; a table whose machine-readable fields need parsing is
            # not machine-readable.
            "comparison": {"value": "<", "provenance": DERIVED,
                           "how": "the Euclidean test: ||z||_2^2 < l2, "
                                  "strict, so equality is rejected"},
            "comparison_note": {"value": "squared, over integers, on the "
                                         "centred representative: "
                                         "max|z_i| <= inf, and "
                                         "||z||_2^2 < l2 -- the Euclidean "
                                         "test is STRICT, so equality is "
                                         "rejected",
                                "provenance": DERIVED},
            "shared_by_prover_and_verifier": {
                "value": "lanes_proof.response_within_bounds",
                "provenance": DERIVED,
                "how": "one definition; Prove returns bottom rather than a "
                       "proof Verify would reject"},
            "population": {"value": lp.N_Z, "provenance": DERIVED,
                           "how": "response_rank * d~ = 3328"},
            "tail_cut": {"value": lp.Z_TAILCUT, "provenance": REPAIR,
                         "how": "statistical_tailcut(N_Z) at 2^-128"},
            "var_z": {"value": _rat(lp.VAR_Z), "provenance": DERIVED,
                      "how": "sigma_y^2 + w_hat sigma_r^2"},
        },

        "recovery": {
            "d_drop": {"value": lp.D_DROP, "provenance": PAPER},
            "scale": {"value": lp.T0_SCALE, "provenance": DERIVED,
                      "how": "2^D"},
            "rounding": {"value": "power2round with a centred low part in "
                                  "(-2^(D-1), 2^(D-1)]",
                         "provenance": REPAIR},
            "ties": {"value": "a tie at exactly 2^(D-1) goes to the low "
                              "part, so high is not incremented",
                     "provenance": REPAIR},
            # **Two omissions, counted separately**, because they are
            # different objects and conflating them is how this cell came
            # to read `0`:
            #
            #   * Bai-Galbraith drops `kappa - response_rank = 4` response
            #     ring elements -- `4 d~ = 1024` coefficients -- which the
            #     verifier reconstructs from `t_0` and the carry;
            #   * `power2round` drops `D = 17` low bits from each of the
            #     `l~ d~ = 1024` coefficients of `t_0`.
            #
            # The carry is what *replaces* the first, one ternary
            # coefficient per `t_0` coefficient.  The old value said "none;
            # the fixed-hint design transmits one carry per coefficient",
            # which describes the carry and then reports the omission it
            # compensates for as zero.
            "omitted_response_rows": {
                "value": lp.KAPPA - lp.RESPONSE_RANK, "provenance": DERIVED,
                "how": "kappa - response_rank; B_0's identity block, "
                       "recovered rather than transmitted"},
            "omitted_response_coefficients": {
                "value": (lp.KAPPA - lp.RESPONSE_RANK) * R.DTILDE,
                "provenance": DERIVED, "how": "omitted_response_rows * d~"},
            "omitted_t0_low_bits": {
                "value": lp.IDENTITY_RANK * R.DTILDE * lp.D_DROP,
                "provenance": DERIVED,
                "how": "l~ d~ D; the low part of every t_0 coefficient"},
            "recovery_carries": {
                "value": lp.IDENTITY_RANK * R.DTILDE, "provenance": DERIVED,
                "how": "one ternary carry per t_0 coefficient, which is "
                       "what the two omissions above are replaced by"},
            "omitted_coordinates": {
                "value": (lp.KAPPA - lp.RESPONSE_RANK) * R.DTILDE,
                "provenance": DERIVED,
                "how": "retained name; the response coefficients omitted "
                       "under Bai-Galbraith compression.  It read 0 while "
                       "the transmitted rank was 13 against kappa = 17"},
            "hint_alphabet": {"value": "{-1, 0, 1}, one per t_0 coefficient",
                              "provenance": REPAIR},
            "hint_rows": {"value": lp.IDENTITY_RANK, "provenance": DERIVED},
            "buckets": {"value": lp.RECOVERY_BUCKETS, "provenance": REPAIR,
                        "how": "largest power of two with "
                               "error_bound < q~ / buckets; the doubling "
                               "loop stops one step past that, so the "
                               "returned value satisfies the unbracketed "
                               "form, not the /(2 buckets) one"},
            "bucket_bits": {"value": lp.RECOVERY_BITS, "provenance": DERIVED},
            "error_bound": {"value": lp.RECOVERY_ERROR_BOUND,
                            "provenance": REPAIR,
                            "how": "w_hat (T0_LOW_BOUND + R_INF_SUPPORT), "
                                   "where R_INF_SUPPORT is the sampler's own "
                                   "support floor(GAUSSIAN_TAILCUT sigma_r) "
                                   "-- 14 sigma_r, not 6"},
            "limit": {"value": "error_bound < q~ / buckets, asserted at "
                               "import in lanes_params",
                      "provenance": REPAIR},
            "failure_rule": {"value": "none: the bound is unconditional over "
                                      "the sampler's support, so recovery "
                                      "cannot fail for an honest prover",
                             "provenance": REPAIR},
            "verification_rule": {"value": "apply the carry, then require the "
                                           "recovered w to match the "
                                           "challenge equation exactly",
                                  "provenance": REPAIR},
            "encoding": {"value": "signed 2-bit per coefficient, "
                                  "n~ d~ coefficients, byte-aligned once "
                                  "with the whole layout",
                         "provenance": REPAIR},
            "security_argument": {"value": "NONE.  The fixed-hint composition "
                                           "is this repository's and has no "
                                           "leakage or extraction analysis; "
                                           "see LANES_SECURITY_EVIDENCE",
                                  "provenance": REPAIR},
        },

        # Read off `lanes_proof.Challenges.ROUNDS`, never restated here.
        # The list used to be hand-written and wrong in both directions: it
        # omitted `w_high`, `v` and `v_prime` -- all three *are* hashed,
        # though the verifier reconstructs rather than reads them -- and it
        # listed `alpha`, `gamma` and `c`, which are outputs of the hash and
        # never inputs to it.  A port built from that list would derive
        # different challenges and never interoperate.
        # `test_lanes.py` drives a real proof and compares the recorded
        # absorption against this.
        "transcript": {
            "rounds": {"value": [{"challenge": name, "absorbs": list(fields)}
                                 for name, fields
                                 in lanes_proof.Challenges.ROUNDS],
                       "provenance": DERIVED,
                       "how": "lanes_proof.Challenges.ROUNDS"},
            "absorbed_fields": {"value": lanes_proof.declared_transcript(),
                                "provenance": DERIVED,
                                "how": "the rounds flattened, in hash order"},
            "derived_not_absorbed": {
                "value": ["alpha", "gamma", "c"], "provenance": DERIVED,
                "how": "challenge outputs; hashing one of these in would be "
                       "a different protocol"},
            "reconstructed_not_transmitted": {
                "value": ["w_high", "v", "v_prime"], "provenance": DERIVED,
                "how": "absorbed into the transcript but recovered by the "
                       "verifier rather than carried on the wire"},
            "packing": {"value": "grouped absorbs are one byte string: "
                                 "(t_mp1, t_mp2, v) and (h, v_prime) are "
                                 "each concatenated before hashing",
                        "provenance": DERIVED},
            "order": {"value": "protocol order: each challenge is hashed over "
                               "the messages that precede it and no others",
                      "provenance": DERIVED},
            "domain_separators": {
                "value": {"exact": DS_EXACT.decode(),
                          "lanes": lanes_proof.DS_LANES.decode(),
                          "alpha": ".alpha", "gamma": ".gamma",
                          "challenge": ".c"},
                "provenance": REPAIR},
            "hashed_form": {"value": "canonical centred encodings, absorbed "
                                     "as byte strings in field order",
                            "provenance": REPAIR},
            "statement_binding": {"value": "H(W_encode(W) || x_centered || "
                                           "z_eval_centered)",
                                  "provenance": DERIVED},
        },

        # Each field carries its **coder parameters**, not just the
        # coder's class name.  A port cannot encode from a name: `Uniform`
        # needs its modulus and the resulting width, `Signed` its bound,
        # and `Rice` its `k` *and* its cap -- and `k` is wire-visible, so
        # two implementations that picked it differently would produce
        # different bytes with no other symptom.
        #
        # `rows: None` in the layout means one element, not an absent
        # count; it is recorded as 1 so a consumer does not have to know
        # that convention.  `bits` is `None` for a variable-length field,
        # which is a fact about the format rather than a gap -- see
        # `total_bits`.
        "wire": {
            "fields": {"value": [_wire_field(f) for f in layout.fields],
                       "provenance": REPAIR},
            "order": {"value": [f.name for f in layout.fields],
                      "provenance": REPAIR},
            "total_bits": {"value": None, "provenance": REPAIR,
                           "how": "Rice-coded: sample-dependent, so measured "
                                  "per proof by field_sizes(), not fixed"},
            "fixed_bits": {"value": _fixed_bits(layout), "provenance": REPAIR,
                           "how": "sum over the fixed-width fields; the "
                                  "Rice-coded z is what makes the total "
                                  "variable"},
            "kb_convention": {"value": "1 KB = 8192 bits", "provenance": DERIVED},
            "byte_alignment": {"value": "once, for the whole layout",
                               "provenance": REPAIR},
            "discrepancy": {
                "value": "the paper states 13.5 KB with no field list, so "
                         "the figure is not reproducible from what it "
                         "publishes; what this implementation reports is "
                         "measured, field by field, by "
                         "LanesBackend.field_sizes().",
                "provenance": REPAIR},
        },

        "estimator": {
            "hint_mlwe_inputs": {
                "value": {"n": lp.N_LWE, "m": lp.M_LWE, "q": R.QTILDE,
                          "sigma_mlwe_sq": _rat(lp.SIGMA_MLWE_SQ),
                          "reduction": "1/sigma_MLWE^2 = 2(1/s_1^2 + "
                                       "w_hat^2/s_2^2)  [KLSS23]",
                          "identity": "equals s_0^2: the widths are chosen "
                                      "so the hint reduction returns the "
                                      "smoothing parameter"},
                "provenance": DERIVED},
            "hint_mlwe_outputs": {
                "value": {
                    "bits_by_reading": evidence and {
                        r["name"]: r["bits"] for r in evidence["mlwe"]},
                    "delta_by_reading": evidence and {
                        r["name"]: r.get("delta") for r in evidence["mlwe"]},
                    "paper_reports": "delta_MLWE = 1.0020",
                    "status": "NOT REPRODUCED.  The two defensible readings "
                              "of the paper's own Gaussian convention "
                              "bracket 1.0020 without hitting it, and differ "
                              "by about 18 bits.  See lanes_security.json"},
                "provenance": REPAIR},
            "hint_mlwe_statistical_loss": {
                "value": evidence
                and evidence["hint_mlwe_statistical_loss_log2"],
                "provenance": DERIVED,
                "how": "log2((d+m) 2 eps) from [KLSS23] Thm 1, with "
                       "d+m = kappa module ranks and eps = 2^-100.  No "
                       "estimator call reports this; it is below the "
                       "128-bit target on its own"},
            "msis_inputs": {
                "value": {"rank": lp.N_TILDE * R.DTILDE, "q": R.QTILDE,
                          "m": lp.KAPPA * R.DTILDE,
                          "length_bound": "B_MSIS = 8 w_hat beta'"},
                "provenance": PAPER},
            "msis_outputs": {
                "value": {
                    "B_MSIS": str(lp.B_MSIS),
                    "bits": evidence and evidence["msis"]["derived"]["bits"],
                    "published_B_MSIS_bits": evidence
                    and evidence["msis"]["published"]["bits"],
                    "delta_closed_form": float(lp.DELTA_MSIS),
                    "paper_reports": "delta_MSIS = 1.0037",
                    "status": "REPRODUCED, both by the closed form and by "
                              "the estimator run"},
                "provenance": PAPER},
            "challenge": {
                "value": {"paper_lanes_noninvertibility": "2^-93.5",
                          "paper_outer": "2^-91.5",
                          "status": "NOT REPRODUCED -- neither figure is "
                                    "derived here, and both are below the "
                                    "128-bit benchmark the paper selects "
                                    "against"},
                "provenance": PAPER},
        },

        # The gate's own audit list: `exact.GATED_LANES_CONSTANTS`
        # walks these against the live module, comparing values.  Every one
        # is now Paper or a stated derivation from Paper; `K_S1`/`K_S2` are
        # gone with the search that produced them.
        "constants": {
            "SIGMA_R": {"value": _rat(lp.SIGMA_R), "provenance": DERIVED,
                        "how": "s_1 = 2 s_0, rounded once to 2^-20"},
            "SIGMA_Y": {"value": _rat(lp.SIGMA_Y), "provenance": DERIVED,
                        "how": "s_2 = 2 w_hat s_0, rounded once to 2^-20"},
            "Z_INF_BOUND": {"value": lp.Z_INF_BOUND, "provenance": REPAIR,
                            "how": "the paper states no infinity bound"},
            "Z_NORM2_BOUND": {"value": lp.Z_NORM2_BOUND,
                              "provenance": DERIVED,
                              "how": "(2 s)^2 * response_rank * d~"},
            "RECOVERY_ERROR_BOUND": {"value": lp.RECOVERY_ERROR_BOUND,
                                     "provenance": REPAIR},
            "RECOVERY_BUCKETS": {"value": lp.RECOVERY_BUCKETS,
                                 "provenance": REPAIR},
        },
    }


def load():
    """The frozen manifest, or `None` if it is absent or unreadable.

    Tolerant on purpose: `exact` loads this at import, so a truncated or
    malformed file must degrade to "no manifest" -- which closes the gate --
    rather than making every module in the tree fail to import.
    """
    try:
        with open(PATH) as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return None


def _dumps(blob):
    return json.dumps(blob, indent=1, sort_keys=True) + "\n"


def write():
    # Built before the file is opened: `manifest()` imports `lanes_backend`,
    # which imports `exact`, which loads *this* file.  Truncating it first
    # would have `exact` read an empty manifest mid-build.
    blob = _dumps(manifest())
    with open(PATH, "w") as fh:
        fh.write(blob)
    return PATH


def check():
    """`[]` if the frozen file matches what the modules carry."""
    frozen = load()
    if frozen is None:
        return [f"{os.path.basename(PATH)} is absent"]
    live = json.loads(_dumps(manifest()))
    if frozen == live:
        return []
    out = []
    for section in sorted(set(frozen) | set(live)):
        if frozen.get(section) != live.get(section):
            out.append(f"section {section} differs from the frozen file")
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--write" in sys.argv:
        print(f"rewriting {write()}")
        print("this replaces the frozen LANES manifest; "
              "`make test` diffs against it")
    elif "--check" in sys.argv:
        bad = check()
        if bad:
            print("\n".join(bad))
            raise SystemExit(1)
        print(f"{PATH}: up to date")
    else:
        print(_dumps(manifest()), end="")
