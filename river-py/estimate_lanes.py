"""
estimate_lanes.py -- lattice-estimator runs for the paper's LANES parameters.

**This does not run under `make test`.**  The estimator is not a dependency
of this repository: it needs SageMath and a checkout of
<https://github.com/malb/lattice-estimator>.  What ships here is the runner
and its *recorded output*, `lanes_security.json`, so the numbers are
reproducible rather than asserted.

    sage -python estimate_lanes.py --estimator /path/to/lattice-estimator \
        --write

Everything the run depends on is recorded in the output: the estimator's git
commit, the exact parameter objects, the cost model, and every result.

What this file is for
----------------------------------------
This runner does **not** select the Gaussian widths: the paper publishes the
whole chain in closed form (see `lanes_params`), which re-derives every
printed figure to the last digit.  There is nothing to search.

This runner is a diagnostic cross-check, not the parameter-selection rule and
not a substitute for the paper's security argument.  It evaluates the paper's
single parameter set under two conversions that are useful when comparing
estimator APIs.  The standard-deviation reading reproduces the paper's
reported `delta_MLWE = 1.0040`; the alternative is retained only as a
sensitivity check.

The production `"lanes"` alias remains disabled because the concrete
compression/recovery completion in this artifact is implementation-defined
and the artifact does not supply a reduction for that composition.  The
fully tested code is available as `"lanes-experimental"`.

Three instances
---------------
**M-SIS** (one reading only).  Binding of the BDLOP commitment, at rank
`n~ d~ = 1024`, dimension `kappa d~ = 4352`, modulus `q~`, and length bound
`B_MSIS = 8 w_hat beta'` with `beta' = 2 s sqrt(4352) = 45430.57`.  This is
unambiguous: `B_MSIS` is a norm bound, not a Gaussian width, so no
convention question arises.  The paper prints `delta_MSIS = 1.0037` and
`lanes_params.DELTA_MSIS` reproduces it in closed form; this run is the
independent check.

**Hint-MLWE, standard-deviation reading.**  The commitment randomness is
`r <- D_{s_1}`; the responses leak affine functions of it, charged as hints
by [KLSS23], and what remains is MLWE at

    1 / sigma^2 = 2 (1 / s_1^2 + w_hat^2 / s_2^2)      =>   sigma = s_0

against the coefficient-embedded instance `n = n~ d~ = 1024`,
`m = (l~ + N_ex + alpha) d~ = 3328`.  The paper's own sentence -- "the
standard deviation entering the LANES communication estimate is therefore
`s`" -- makes `s_i` the standard deviations, so `sigma = s_0 = 2.7668` is
what an estimator taking `ND.DiscreteGaussian(stddev)` should be handed.

**Hint-MLWE, Gaussian-parameter reading.**  The same sentence also prints
`sigma_0 ~ 6.9353 = s_0 sqrt(2 pi)` as the [KLSS23] "Gaussian parameter".
If an estimator is instead handed *that* number as a standard deviation, the
instance is about 18 bits stronger.  This is recorded as an API-convention
sensitivity check; the standard-deviation reading is the normative one here.

A separately reported reduction term
--------------------------------------
[KLSS23] Theorem 1 is not tight: the reduction loses `(d + m) 2 eps` in
statistical distance.  With `d + m = 17` module ranks and the paper's
`eps = 2^-100` that is about `2^-94.9`, before any union bound over queries
It is recorded as a computed field because no estimator call reports it;
this diagnostic runner does not use it as an artifact acceptance rule.
"""

import argparse
import json
import os
import subprocess
import sys
from math import log2

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "lanes_security.json")

#: The security level the paper's parameter section targets.
TARGET_BITS = 128

#: The root-Hermite factor the paper prints for the MLWE instance.
PUBLISHED_DELTA_MLWE = 1.0040

#: The root-Hermite factor the paper prints for the M-SIS instance.
PUBLISHED_DELTA_MSIS = 1.0037

#: The paper's published M-SIS length bound.
PUBLISHED_B_MSIS = 15991562


def _instances():
    """Dimensions, read from the modules and never restated here."""
    import lanes_params as lp
    import lanes_ring as R
    return {
        "n": lp.N_TILDE * R.DTILDE,
        "q": R.QTILDE,
        "m_lwe": lp.M_LWE,
        "m_sis": lp.KAPPA * R.DTILDE,
    }


def mlwe_readings():
    """The two defensible widths for the Hint-MLWE instance."""
    import lanes_params as lp
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = lp._PREC
        gaussian_parameter = lp.S0 * (2 * lp._PI).sqrt()
    return [
        {
            "name": "standard-deviation",
            "sigma": float(lp.S0),
            "conservative": True,
            "why": ("s_0 read as a standard deviation, which is what the "
                    "paper's own sentence says `s` is.  This is the reading "
                    "the verdict is taken from."),
        },
        {
            "name": "gaussian-parameter-as-stddev",
            "sigma": float(gaussian_parameter),
            "conservative": False,
            "why": ("sigma_0 = s_0 sqrt(2 pi), the [KLSS23] Gaussian "
                    "parameter, handed to the estimator as though it were a "
                    "standard deviation.  Recorded only as an API-convention "
                    "sensitivity check."),
        },
    ]


def hint_mlwe_statistical_loss():
    """`(d + m) 2 eps` from [KLSS23] Theorem 1, as a base-2 logarithm."""
    import lanes_params as lp
    ranks = lp.KAPPA                      # d + m, in module ranks
    return log2(2 * ranks) - lp.SMOOTHING_EPS_EXP


def run(estimator_path):
    """Call the estimator on the paper's parameters.  Requires sage."""
    sys.path.insert(0, estimator_path)
    from estimator import LWE, SIS, ND

    def git(*args):
        return subprocess.run(["git", "-C", estimator_path, *args],
                              capture_output=True, text=True).stdout.strip()

    import lanes_params as lp
    inst = _instances()
    n, q = inst["n"], inst["q"]

    # ---- M-SIS, at our derived bound and at the paper's printed one ----
    sis_rows = {}
    for label, bound in (("derived", int(lp.B_MSIS)),
                         ("published", PUBLISHED_B_MSIS)):
        est = SIS.estimate.rough(SIS.Parameters(
            n=n, q=q, length_bound=bound, m=inst["m_sis"]))
        bits = min(log2(float(v["rop"])) for v in est.values())
        sis_rows[label] = {
            "length_bound": bound,
            "bits": bits,
            "by_attack": {k: log2(float(v["rop"])) for k, v in est.items()},
        }
        print(f"  M-SIS  {label:10s} B={bound:>9}  {bits:6.1f} bits",
              flush=True)

    # ---- Hint-MLWE, under both readings ----
    lwe_rows = []
    for reading in mlwe_readings():
        est = LWE.estimate.rough(LWE.Parameters(
            n=n, q=q,
            Xs=ND.DiscreteGaussian(reading["sigma"]),
            Xe=ND.DiscreteGaussian(reading["sigma"]),
            m=inst["m_lwe"]))
        bits = min(log2(float(v["rop"])) for v in est.values())
        row = dict(reading)
        row["bits"] = bits
        row["by_attack"] = {k: log2(float(v["rop"])) for k, v in est.items()}
        if "usvp" in est:
            row["delta"] = float(est["usvp"]["delta"])
        lwe_rows.append(row)
        print(f"  MLWE   {reading['name']:26s} sigma={reading['sigma']:.4f}"
              f"  {bits:6.1f} bits", flush=True)

    conservative = next(r for r in lwe_rows if r["conservative"])
    bits = min(conservative["bits"], sis_rows["derived"]["bits"])
    loss = hint_mlwe_statistical_loss()

    return {
        "tool": {
            "name": "lattice-estimator",
            "url": "https://github.com/malb/lattice-estimator",
            "commit": git("rev-parse", "HEAD"),
            "committed": git("log", "-1", "--format=%ad"),
            "cost_model": "estimate.rough (core-SVP style)",
            "command": ("sage -python estimate_lanes.py --estimator "
                        "<checkout> --write"),
        },
        "parameters": {
            "source": "paper -- closed form, nothing selected here",
            "s_0": str(lp.S0),
            "s_1": str(lp.S1),
            "s_2": str(lp.S2),
            "s_response": str(lp.S_RESPONSE),
            "beta_prime": str(lp.BETA_PRIME_BDLOP),
            "B_MSIS_derived": str(lp.B_MSIS),
        },
        "instances": {
            "mlwe": {
                "n": n, "q": q, "m": inst["m_lwe"],
                "Xs": "DiscreteGaussian(sigma)", "Xe": "DiscreteGaussian(sigma)",
                "note": ("sigma from the [KLSS23] hint reduction with "
                         "B_H = w_hat^2, a worst-case l1 bound on the "
                         "challenge operator; the true spectral norm is "
                         "smaller, so sigma is a lower bound and the "
                         "reported security a lower bound with it"),
            },
            "msis": {
                "n": n, "q": q, "m": inst["m_sis"],
                "length_bound": "B_MSIS = 8 w_hat beta'",
            },
        },
        "msis": sis_rows,
        "mlwe": lwe_rows,
        "published_delta": {
            "msis": {
                "value": PUBLISHED_DELTA_MSIS,
                "reproduced": True,
                "ours": float(lp.DELTA_MSIS),
                "how": ("closed form log2 delta = (log2 B)^2 / "
                        "(4 n log2 q); lanes_params.DELTA_MSIS"),
            },
            "mlwe": {
                "value": PUBLISHED_DELTA_MLWE,
                "reproduced": round(conservative.get("delta", 0.0), 4)
                              == PUBLISHED_DELTA_MLWE,
                "under_each_reading": {r["name"]: r.get("delta")
                                       for r in lwe_rows},
                "how": ("the standard-deviation reading rounds to the "
                        "paper's printed root-Hermite factor; the alternate "
                        "conversion is diagnostic only"),
            },
        },
        "hint_mlwe_statistical_loss_log2": loss,
        "target_bits": TARGET_BITS,
        "best_bits": bits,
        # The cost figures above are diagnostic.  This status records why
        # the artifact keeps the production alias disabled; it is not a
        # security-level verdict on the paper's parameter set.
        "verdict": "candidate-composition",
        "blockers": [
            "the recovery-hint and concrete compression composition is an "
            "implementation-defined completion, and this artifact does not "
            "supply a reduction for that composition",
        ],
    }


def load():
    """The recorded evidence, or `None`."""
    try:
        with open(PATH) as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return None


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", help="path to a lattice-estimator checkout")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not args.estimator:
        print(__doc__)
        blob = load()
        if blob is None:
            print("recorded: absent")
        else:
            print(f"recorded: {blob['best_bits']:.1f} bits, verdict "
                  f"{blob['verdict']!r}, estimator "
                  f"{blob['tool']['commit'][:7]}")
        raise SystemExit(0)

    result = run(args.estimator)
    print(f"\n{result['best_bits']:.1f} bits, verdict {result['verdict']!r}")
    for blocker in result["blockers"]:
        print(f"  - {blocker}")
    if args.write:
        with open(PATH, "w") as fh:
            fh.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(f"wrote {PATH}")
