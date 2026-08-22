"""
manifest.py -- The wire-visible numeric choices, frozen per profile.

Everything here is a value two implementations must agree on *exactly* or
they produce different bytes, and none of it is stated by the paper.  It is
collected in one place, as data, so that porting `river-rs` is a matter of
reproducing a table rather than re-deriving a chain of float expressions in
the same order.

What is in it, and why each one is a hazard:

  * **`(sigma_num, sigma_den)`** per Gaussian field.  The paper's widths are
    irrational, so every implementation pins some rational; `sample.rational_sigma`
    pins `round(sigma * 2^20) / 2^20`.  The `round` removes only the *final*
    float error, so the input must be computed in the same operation order.
  * **`rice_k`** per Gaussian field.  `k = floor(log2(sqrt(2 ln 2) sigma))`
    is wire-visible, and one off is a different encoding rather than a
    rounding difference.
  * **`bound`** per response field.  The largest coefficient that can pass
    verification, `floor(sqrt(bound_sq))` -- exact, so the encoder's cap and
    the acceptance test cannot disagree.
  * **`width`** per fixed-width field, in bits.

`test_manifest.py` pins every entry.  A change to a sampler, a width, a
bound or the Rice constant shows up there before it shows up as "proof bytes
differ" in a cross-language vector.

Run `python3 manifest.py` to print the table, or `--json` for the artifact a
port can read.
"""

import json
import os

from codec import RiVeRCodec, RICE_CONST_DEN, RICE_CONST_NUM, floor_sqrt
from exact import ExactParams, RADIX_WEIGHTS, get_backend
from params import PROFILES
from sample import GAUSSIAN_TAILCUT, PROB_BITS, SIGMA_DEN, rational_sigma

#: Gaussian fields, as `(name, width attribute, exact squared bound)`.
GAUSSIAN_FIELDS = (
    ("f1", "sigma_a", "f1_inf_bound_sq"),
    ("zb", "sigma_b", "zb_inf_bound_sq"),
    ("zs", "sigma_s", "zs_inf_bound_sq"),
    ("zm", "sigma_m", "zm_inf_bound_sq"),
)


def describe_coder(coder):
    """The coder's identity and every parameter that reaches the wire."""
    kind = type(coder).__name__.lower()
    spec = {"coder": kind}
    for attr in ("k", "bound", "width", "modulus", "max_high"):
        if hasattr(coder, attr):
            spec[attr] = getattr(coder, attr)
    return spec


def describe_layout(layout):
    """Every field of a `Layout`, in wire order.

    Walks the layout rather than restating it, so this cannot drift from
    what the encoder does -- which is the only reason to have it.
    """
    return {
        "order": [f.name for f in layout.fields],
        "fields": {
            f.name: dict(describe_coder(f.coder),
                         cols=f.cols,
                         rows=f.rows,
                         count=f.count,
                         ring_modulus=(f.ring.q if f.ring is not None
                                       else None))
            for f in layout.fields
        },
        "max_bytes": layout.max_bytes,
        "min_bytes": layout.min_bytes,
    }


def global_constants():
    """The values that are the same at every profile."""
    return {
        "sigma_den": SIGMA_DEN,
        "prob_bits": PROB_BITS,
        "gaussian_tailcut": GAUSSIAN_TAILCUT,
        "rice_const_num": RICE_CONST_NUM,
        "rice_const_den": RICE_CONST_DEN,
        "verifier_tailcut": 6,
    }


def profile_manifest(par):
    """Every wire-visible numeric choice for one profile."""
    codec = RiVeRCodec(par)
    ex = ExactParams(par)
    fields = {}
    for name, width_attr, bound_attr in GAUSSIAN_FIELDS:
        num, den = rational_sigma(getattr(par, width_attr))
        bound = floor_sqrt(getattr(par, bound_attr))
        fields[name] = {
            "coder": "rice",
            "sigma_num": num,
            "sigma_den": den,
            "rice_k": codec.oom_layout_k(name),
            "bound": bound,
        }
    fields["B"] = {
        "coder": "signed",
        "bound": codec.bound_b_hi,
        "width_bits": codec.oom_layout_width("B"),
    }
    fields["x"] = {
        "coder": "signed",
        "bound": codec.bound_x,
        "width_bits": codec.oom_layout_width("x"),
    }
    num, den = rational_sigma(par.sigma_m)
    fields["y_eval"] = {
        "coder": "rice",
        "sigma_num": num,
        "sigma_den": den,
        "rice_k": codec.rice_k_for(par.sigma_m),
        "bound": ex.par.w * ex.par.gamma * ex.par.B_e
                 + floor_sqrt(par.zm_inf_bound_sq),
    }
    backend = get_backend("opening", par)
    return {
        "profile": par.name,
        "rows": {"zs": par.s_dim, "zm": par.m_dim, "zb": par.k_hat,
                 "f1": par.N - 1, "B": par.n_hat},
        "fields": fields,
        "oom_max_bytes": codec.oom_max_bytes,
        # The two layouts, walked rather than restated.  Together with the
        # framing below they are the whole wire format: a port that
        # reproduces these and the field table above has nothing left to
        # infer from the prose.
        "layouts": {
            "oom": describe_layout(codec.oom_layout),
            "exact_opening": describe_layout(backend.proof_layout),
            "exact_W": describe_layout(backend.W_layout),
        },
        "exact": {
            "backend": backend.name,
            "d_tilde": ex.d_tilde,
            "l_split": ex.l_split,
            "q_tilde": ex.q_tilde,
            "identity_rank": ex.t0_rows,
            "tail_rank": ex.n_tilde,
            "kappa": ex.kappa,
            "response_rank": ex.response_rank,
            "N_ex": ex.N_ex,
            "block_slots": ex.block_slots,
            "block_used": ex.block_used,
            "radix_weights": list(RADIX_WEIGHTS),
            "proof_bytes_max": backend.proof_bytes,
        },
        # The framing, stated rather than summarised: a port needs the
        # block order and the prefix encoding, not just how many bytes
        # they cost.  `proof_encode` writes
        #   len(pi_OOM) as 4 LE bytes || pi_OOM || len(pi_ex) as 4 LE || pi_ex
        # and `proof_decode` bounds each claimed length by that block's own
        # layout before slicing, so a hostile prefix cannot become an
        # allocation.
        "framing": {
            "block_order": ["oom", "exact"],
            "length_prefix_bytes": 4,
            "length_prefix_endian": "little",
            "total_framing_bytes": 8,
            "prefix_bounded_by_layout": True,
        },
    }


def manifest():
    """The whole table, for every profile."""
    return {
        "global": global_constants(),
        "profiles": {name: profile_manifest(par)
                     for name, par in sorted(PROFILES.items())},
    }


#: The checked-in canonical rendering, beside this module.
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "manifest.json")


def canonical_json(blob=None):
    """The manifest as one canonical string.

    Sorted keys, fixed indent, trailing newline -- so `manifest.json` can
    be compared **byte for byte** rather than field by field.  That is the
    difference between freezing the table and freezing the fields someone
    remembered to assert: a `Uniform` modulus swapped for another of the
    same bit width passes every structural check and changes this string.
    """
    return json.dumps(manifest() if blob is None else blob,
                      indent=2, sort_keys=True) + "\n"


def write_manifest(path=None):
    """Rewrite `manifest.json`.  A deliberate act, like `make vectors`."""
    path = path or MANIFEST_PATH
    with open(path, "w") as handle:
        handle.write(canonical_json())
    return path


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    blob = manifest()
    if "--write" in sys.argv:
        print(f"rewriting {write_manifest()}")
        print("this replaces the frozen wire manifest; "
              "`make test` diffs against it")
        raise SystemExit(0)
    if "--json" in sys.argv:
        print(canonical_json(blob), end="")
        raise SystemExit(0)

    print(f"paper revision {blob['paper_revision']}, "
          f"vector schema {blob['vector_schema']}")
    print("global: " + ", ".join(f"{k}={v}"
                                 for k, v in blob["global"].items()
                                 if k.startswith(("sigma", "prob", "gauss"))))
    print()
    header = f"{'profile':12s} {'field':8s} {'sigma_num':>14s} {'k':>3s} {'bound':>12s}"
    print(header)
    print("-" * len(header))
    for name, entry in blob["profiles"].items():
        for field, spec in entry["fields"].items():
            print(f"{name:12s} {field:8s} "
                  f"{spec.get('sigma_num', ''):>14} "
                  f"{spec.get('rice_k', ''):>3} "
                  f"{spec.get('bound', ''):>12}")
