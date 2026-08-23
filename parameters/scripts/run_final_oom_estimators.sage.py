#!/usr/bin/env sage
"""Re-run the estimator calls for the final RiVeR OOM rows.

By default this script calls the real `estimator.LWE` API and the local
multi-bound `MSIS_security` backend for the selector A-MSIS check.  This script
is intentionally separate from `river_oom_math_checks.sage.py`, which only
recomputes deterministic formula checks and validates pinned values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from sage.all import RR, ZZ, oo, sqrt

sys.dont_write_bytecode = True

cwd_root = Path.cwd().resolve()
if (cwd_root / "scripts" / "run_final_oom_estimators.sage.py").is_file():
    ROOT = cwd_root
else:
    ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "final_oom_security.json"
DEFAULT_OUTPUT = ROOT / "data" / "final_oom_estimator_rerun.json"
RHF_TARGET = RR("1.004690")


def json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        if value == oo:
            return "Infinity"
        if RR(value) == ZZ(value):
            return int(ZZ(value))
        return float(RR(value))
    except Exception:
        try:
            return float(value)
        except Exception:
            return str(value)


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    return json_value(value)


def add_estimator_path(path_arg: str | None) -> Path:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg).expanduser())
    if os.environ.get("LATTICE_ESTIMATOR_PATH"):
        candidates.append(Path(os.environ["LATTICE_ESTIMATOR_PATH"]).expanduser())
    candidates.append(ROOT / "external" / "lattice-estimator")
    for candidate in candidates:
        if candidate and (candidate / "estimator").is_dir():
            sys.path.insert(0, str(candidate))
            return candidate
    raise SystemExit(
        "Could not locate lattice-estimator.  Pass --estimator-path or set "
        "LATTICE_ESTIMATOR_PATH to the directory containing the estimator package."
    )


def add_msis_path(path_arg: str | None) -> Path:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg).expanduser())
    if os.environ.get("MSIS_SECURITY_PATH"):
        candidates.append(Path(os.environ["MSIS_SECURITY_PATH"]).expanduser())
    candidates.append(ROOT / "external" / "msis-security")
    for candidate in candidates:
        if candidate and candidate.is_file() and candidate.name == "MSIS_security.py":
            sys.path.insert(0, str(candidate.parent))
            return candidate.parent
        if candidate and (candidate / "MSIS_security.py").is_file():
            sys.path.insert(0, str(candidate))
            return candidate
    raise SystemExit(
        "Could not locate MSIS_security.py.  Pass --msis-security-path or set "
        "MSIS_SECURITY_PATH to a directory containing MSIS_security.py."
    )


def cost_bits(cost):
    rop = cost.get("rop", None)
    if rop is None:
        return None
    if rop == oo:
        return oo
    return RR(rop).log() / RR(2).log()


def cost_delta(cost, reduction_model):
    if hasattr(cost, "get") and cost.get("delta", None) is not None:
        return RR(cost["delta"])
    blocksize = cost.get("beta", cost.get("beta_", None))
    if blocksize is None:
        return None
    return RR(reduction_model.delta(int(blocksize)))


def blocksize_of(cost):
    blocksize = cost.get("beta", cost.get("beta_", None))
    return None if blocksize is None else int(blocksize)


def run_outer_mlwr_as_mlwe(row, constants, LWE, ND, RC, run_arora_gb=True):
    inst = row["checked_instances"]["outer_mlwr_as_mlwe"]
    params = LWE.Parameters(
        n=int(inst["expanded_lwe_n"]),
        q=int(inst["q"]),
        Xs=ND.Uniform(-int(constants["beta"]), int(constants["beta"])),
        Xe=ND.Uniform(-int(constants["B_e"]), int(constants["B_e"])),
        m=oo,
    )
    cost = LWE.primal_usvp(
        params,
        red_cost_model=RC.ADPS16,
        red_shape_model="gsa",
    )
    out = {
        "algorithm": "LWE.primal_usvp",
        "delta": json_value(cost_delta(cost, RC.ADPS16)),
        "blocksize": blocksize_of(cost),
        "lattice_bits": json_value(cost_bits(cost)),
        "pinned_delta": inst["delta"],
        "pinned_blocksize": inst["blocksize"],
    }
    if run_arora_gb:
        sample_count = ZZ(constants["d"]) * (ZZ(row["n"]) + ZZ(2) ** ZZ(128))
        gb_params = LWE.Parameters(
            n=int(inst["expanded_lwe_n"]),
            q=int(inst["q"]),
            Xs=ND.Uniform(-int(constants["beta"]), int(constants["beta"])),
            Xe=ND.Uniform(-int(constants["B_e"]), int(constants["B_e"])),
            m=int(sample_count),
        )
        gb_cost = LWE.arora_gb(gb_params)
        out.update(
            {
                "arora_gb_bits": json_value(cost_bits(gb_cost)),
                "arora_gb_samples_used": json_value(gb_cost.get("m", sample_count)),
                "pinned_arora_gb_bits": inst["arora_gb_bits"],
            }
        )
    return out


def run_dual_instance(inst, constants, LWE, ND, RC, secret_bound):
    params = LWE.Parameters(
        n=int(inst["expanded_lwe_n"]),
        q=int(inst["q"]),
        Xs=ND.Uniform(-int(secret_bound), int(secret_bound)),
        Xe=ND.DiscreteGaussian(RR(inst["error_sigma"])),
        m=int(inst["expanded_lwe_m"]),
    )
    cost = LWE.dual(params, red_cost_model=RC.ADPS16)
    return {
        "algorithm": "LWE.dual",
        "delta": json_value(cost_delta(cost, RC.ADPS16)),
        "blocksize": blocksize_of(cost),
        "pinned_delta": inst["delta"],
        "pinned_blocksize": inst.get("blocksize"),
    }


def run_selector_asis_msis(row, constants, msis_security):
    asis = row["selector_asis_bounds"]
    pairs = []
    for bound, width in zip(asis["merged_inf_bounds"], asis["merged_widths"]):
        width = int(width)
        if width > 0:
            pairs.append((int(ZZ(RR(bound).ceil())), width))
    pairs.sort(key=lambda item: item[0], reverse=True)
    while len(pairs) < 5:
        pairs.append((1, 0))
    pairs = pairs[:5]

    msis_security.MIN_b = 300
    msis_security.STEPS_b = 1
    msis_security.STEPS_m = 10
    install_fast_msis_linf_cost(msis_security)
    total_width = sum(width for _, width in pairs)
    ps = msis_security.MSISParameterSet(
        n=int(constants["d"]),
        w=int(total_width),
        h=int(row["hat_n"]),
        B1=pairs[0][0],
        B2=pairs[1][0],
        B3=pairs[2][0],
        B4=pairs[3][0],
        B5=pairs[4][0],
        m1=pairs[0][1],
        m2=pairs[1][1],
        m3=pairs[2][1],
        m4=pairs[3][1],
        m5=pairs[4][1],
        q=int(row["hat_q"]),
        norm="linf",
    )
    validate_fast_msis_linf_cost(msis_security, ps)
    _, blocksize, cost = msis_security.MSIS_summarize_attacks(ps, attack_variant=2)
    delta = msis_security.delta_BKZ(int(blocksize))
    return {
        "algorithm": "MSIS_security.MSIS_summarize_attacks",
        "cost_evaluator": "fast exact equivalent of MSIS_security.SIS_linf_cost for attack_variant=2",
        "attack_variant": 2,
        "acceptance_metric": "root_hermite_factor_delta",
        "delta": json_value(delta),
        "delta_target": json_value(RHF_TARGET),
        "passes_delta_target": bool(RR(delta) <= RHF_TARGET),
        "blocksize": int(blocksize),
        "cost_bits": json_value(cost),
        "cost_bits_diagnostic_only": True,
        "normalized_bounds": [bound for bound, _ in pairs],
        "normalized_widths": [width for _, width in pairs],
        "pinned_delta": row["selector_asis_delta"],
        "pinned_blocksize": None,
    }


def install_fast_msis_linf_cost(msis_security):
    if getattr(msis_security, "_river_fast_linf_installed", False):
        return
    msis_security._river_original_sis_linf_cost = msis_security.SIS_linf_cost

    def fast_sis_linf_cost(
        q,
        w,
        h,
        B1,
        m1,
        B2,
        m2,
        B3,
        m3,
        B4,
        m4,
        B5,
        m5,
        b,
        cost_svp=msis_security.svp_classical,
        verbose=False,
        attack_variant=2,
    ):
        if verbose:
            return msis_security._river_original_sis_linf_cost(
                q,
                w,
                h,
                B1,
                m1,
                B2,
                m2,
                B3,
                m3,
                B4,
                m4,
                B5,
                m5,
                b,
                cost_svp=cost_svp,
                verbose=verbose,
                attack_variant=attack_variant,
            )

        if attack_variant == 2:
            c12 = B1 / B2
            c13 = B1 / B3
            c14 = B1 / B4
            c15 = B1 / B5
        elif attack_variant in (0, 1):
            c12 = c13 = c14 = c15 = 1
        else:
            raise ValueError("Incorrect attack variant!")

        q = int(q)
        w = int(w)
        h = int(h)
        b = int(b)
        m1 = int(m1)
        m2 = int(m2)
        m3 = int(m3)
        m4 = int(m4)
        m5 = int(m5)

        d_total = w
        d1_shape = min(m1, d_total)
        d2_shape = min(max(0, d_total - d1_shape), m2)
        d3_shape = min(max(0, d_total - d1_shape - d2_shape), m3)
        d4_shape = min(max(0, d_total - d1_shape - d2_shape - d3_shape), m4)
        d5_shape = min(max(0, d_total - d1_shape - d2_shape - d3_shape - d4_shape), m5)

        glv = (
            h * math.log(q)
            + d2_shape * math.log(c12)
            + d3_shape * math.log(c13)
            + d4_shape * math.log(c14)
            + d5_shape * math.log(c15)
        )

        step = 2.0 * math.log(msis_security.delta_BKZ(b))
        if step <= 0:
            return msis_security.log_infinity

        raw = (math.sqrt(1.0 + 8.0 * glv / step) - 1.0) / 2.0
        slope_len = max(1, min(d_total, int(math.floor(raw))))
        while slope_len < d_total and step * slope_len * (slope_len + 1) / 2.0 <= glv:
            slope_len += 1
        while slope_len > 1 and step * (slope_len - 1) * slope_len / 2.0 > glv:
            slope_len -= 1

        lv = step * slope_len * (slope_len + 1) / 2.0
        log_l0 = slope_len * step - (lv - glv) / slope_len
        middle_dim = slope_len + 1
        sigma = math.exp(log_l0) / math.sqrt(middle_dim)

        p_middle1 = msis_security.gaussian_center_weight(sigma, B1)
        p_middle2 = msis_security.gaussian_center_weight(sigma, c12 * B2)
        p_middle3 = msis_security.gaussian_center_weight(sigma, c13 * B3)
        p_middle4 = msis_security.gaussian_center_weight(sigma, c14 * B4)
        p_middle5 = msis_security.gaussian_center_weight(sigma, c15 * B5)

        d1 = min(m1, middle_dim)
        d2 = min(max(0, middle_dim - d1), m2)
        d3 = min(max(0, middle_dim - d1 - d2), m3)
        d4 = min(max(0, middle_dim - d1 - d2 - d3), m4)
        d5 = min(max(0, middle_dim - d1 - d2 - d3 - d4), m5)

        log2_eps = (
            d1 * math.log(p_middle1, 2)
            + d2 * math.log(p_middle2, 2)
            + d3 * math.log(p_middle3, 2)
            + d4 * math.log(p_middle4, 2)
            + d5 * math.log(p_middle5, 2)
        )
        log2_R = max(0, -log2_eps - msis_security.nvec_sieve(b))
        return cost_svp(b) + log2_R

    msis_security.SIS_linf_cost = fast_sis_linf_cost
    msis_security._river_fast_linf_installed = True


def validate_fast_msis_linf_cost(msis_security, ps):
    original = getattr(msis_security, "_river_original_sis_linf_cost", None)
    if original is None:
        return
    q = ps.q
    h = int(ps.n) * int(ps.h)
    max_w = int(ps.n) * int(ps.w)
    args = (
        ps.B1,
        int(ps.n) * int(ps.m1),
        ps.B2,
        int(ps.n) * int(ps.m2),
        ps.B3,
        int(ps.n) * int(ps.m3),
        ps.B4,
        int(ps.n) * int(ps.m4),
        ps.B5,
        int(ps.n) * int(ps.m5),
    )
    cases = [
        (int(msis_security.MIN_b), max(h + 1, int(msis_security.MIN_b) + 1)),
        (int(msis_security.MIN_b) + int(msis_security.STEPS_b), max(h + 1, int(msis_security.MIN_b) + int(msis_security.STEPS_b) + 1)),
        (
            min(max_w - 1, int(msis_security.MIN_b) + 7 * int(msis_security.STEPS_b)),
            min(max_w - 1, max(h + 1, int(msis_security.MIN_b) + 13 * int(msis_security.STEPS_m))),
        ),
    ]
    for b, w in cases:
        if b <= 1 or w <= h or w >= max_w:
            continue
        reference = original(
            q,
            w,
            h,
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            b,
            cost_svp=msis_security.svp_quantum,
            attack_variant=2,
        )
        fast = msis_security.SIS_linf_cost(
            q,
            w,
            h,
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            b,
            cost_svp=msis_security.svp_quantum,
            attack_variant=2,
        )
        if abs(reference - fast) > 1e-8 * max(1.0, abs(reference)):
            raise SystemExit(f"fast A-MSIS cost mismatch at b={b}, w={w}: {fast} != {reference}")


def rounded_match(actual, pinned, digits=6):
    if actual in (None, "Infinity"):
        return False
    return round(float(actual), digits) == round(float(pinned), digits)


def check_row(row, results):
    failures = []
    for key, result in results.items():
        if result.get("status") == "not_run":
            failures.append(f"{key}: not run ({result.get('reason', 'no reason given')})")
            continue
        if not rounded_match(result["delta"], result["pinned_delta"]):
            failures.append(f"{key}: delta {result['delta']} != pinned {result['pinned_delta']}")
        if result.get("pinned_blocksize") is not None and result.get("blocksize") != result.get("pinned_blocksize"):
            failures.append(f"{key}: blocksize {result.get('blocksize')} != pinned {result.get('pinned_blocksize')}")
        if RR(result["delta"]) > RHF_TARGET:
            failures.append(f"{key}: delta {result['delta']} exceeds target {float(RHF_TARGET)}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--estimator-path", default=None)
    parser.add_argument("--msis-security-path", default=None)
    parser.add_argument("--skip-arora-gb", action="store_true")
    parser.add_argument("--run-msis", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-msis", action="store_true", help="skip the selector A-MSIS estimator rerun")
    args = parser.parse_args()

    estimator_path = add_estimator_path(args.estimator_path)
    from estimator import LWE, ND, RC
    run_msis = args.run_msis or not args.skip_msis
    msis_path = None
    MSIS_security = None
    if run_msis:
        msis_path = add_msis_path(args.msis_security_path)
        import MSIS_security

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    constants = payload["global_constants"]
    rows = []
    all_failures = []
    for row in payload["rows"]:
        outer_mlwr = run_outer_mlwr_as_mlwe(row, constants, LWE, ND, RC, run_arora_gb=not args.skip_arora_gb)
        outer_hiding = run_dual_instance(
            row["checked_instances"]["outer_hiding_mlwe"],
            constants,
            LWE,
            ND,
            RC,
            secret_bound=constants["beta"],
        )
        selector_hiding = run_dual_instance(
            row["checked_instances"]["selector_hiding_mlwe"],
            constants,
            LWE,
            ND,
            RC,
            secret_bound=constants["beta"],
        )
        if run_msis:
            print(f"running selector A-MSIS estimator for N={row['N']}", file=sys.stderr, flush=True)
            selector_asis = run_selector_asis_msis(row, constants, MSIS_security)
        else:
            selector_asis = {
                "algorithm": "MSIS_security.MSIS_summarize_attacks",
                "status": "not_run",
                "reason": "selector A-MSIS rerun was skipped by --skip-msis",
                "pinned_delta": row["selector_asis_delta"],
            }
        results = {
            "outer_mlwr_as_mlwe": outer_mlwr,
            "outer_hiding_mlwe": outer_hiding,
            "selector_hiding_mlwe": selector_hiding,
            "selector_binding_asis_msis": selector_asis,
        }
        failures = check_row(row, results)
        all_failures.extend(f"N={row['N']}: {failure}" for failure in failures)
        rows.append({"N": row["N"], "results": results, "failures": failures})

    out = {
        "estimator_path": "external/lattice-estimator",
        "msis_security_path": None if msis_path is None else "external/msis-security",
        "red_cost_model": "RC.ADPS16",
        "rhf_target": float(RHF_TARGET),
        "arora_gb_rerun": not args.skip_arora_gb,
        "selector_asis_msis_rerun": bool(run_msis),
        "rows": rows,
        "status": "PASS" if not all_failures else "FAIL",
        "failures": all_failures,
    }
    Path(args.output).write_text(json.dumps(sanitize(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if all_failures:
        raise SystemExit("estimator mismatch:\n" + "\n".join(all_failures))
    print(json.dumps({"rows": len(rows), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
