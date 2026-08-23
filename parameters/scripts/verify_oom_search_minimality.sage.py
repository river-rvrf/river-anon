#!/usr/bin/env sage
"""Finite-grid minimality check for the final RiVeR OOM rows.

This script is intentionally narrower than a full search archive. It
encodes a finite search domain around the final OOM rows and tests whether any
candidate with a strictly smaller OOM communication estimate passes all
deterministic conditions and estimator-delta checks. A failing result remains
an explicit failing diagnostic rather than being normalized into a pass.

The script reuses the artifact-local deterministic formulas from
`scripts/river_oom_math_checks.sage.py` and the artifact-local estimator calls
from `scripts/run_final_oom_estimators.sage.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from sage.all import RR, ZZ, sqrt

sys.dont_write_bytecode = True

cwd_root = Path.cwd().resolve()
if (cwd_root / "scripts" / "verify_oom_search_minimality.sage.py").is_file():
    ROOT = cwd_root
else:
    ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "oom_search_minimality_diagnostic.json"


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oom = load_local_module("river_oom_math_checks", ROOT / "scripts" / "river_oom_math_checks.sage.py")
est_runner = load_local_module("run_final_oom_estimators", ROOT / "scripts" / "run_final_oom_estimators.sage.py")

SMALL_PROFILES = [
    "m3_pa28_ps28_pb2",
    "m3_pa30_ps28_pb2",
    "m3_pa32_ps26_pb2",
    "m3_pa34_ps24_pb2",
    "m3_pa36_ps24_pb2",
    "m3_pa40_ps22_pb2",
]
LARGE_PROFILES = [
    "m3_pa28_ps28_pb2",
    "m3_pa28_ps30_pb2",
    "m3_pa26_ps32_pb2",
    "m3_pa24_ps34_pb2",
    "m3_pa24_ps36_pb2",
    "m3_pa22_ps40_pb2",
]

# This is the finite domain certified by this artifact.  It is deliberately
# explicit so a reviewer can see exactly what is being minimized over.
SEARCH_GRID = {
    8: {
        "n": list(range(41, 45)),
        "ell": [52, 54, 56],
        "p_bits": [44, 48],
        "hat_q_bits": [44, 48],
        "hat_n": [40, 41, 42, 46, 48, 50, 52],
        "hat_k": [44, 45, 46, 48, 50, 52],
        "profiles": SMALL_PROFILES,
    },
    16: {
        "n": list(range(37, 42)),
        "ell": [56, 58, 59, 60],
        "p_bits": [44, 48],
        "hat_q_bits": [46, 48],
        "hat_n": [41, 42, 43, 46, 48, 49, 50, 52],
        "hat_k": [47, 48, 49, 50, 52],
        "profiles": SMALL_PROFILES,
    },
    64: {
        "n": list(range(41, 45)),
        "ell": [52, 54, 56],
        "p_bits": [44, 48],
        "hat_q_bits": [48],
        "hat_n": [48, 49, 50, 51, 52],
        "hat_k": [49, 50, 51, 52],
        "profiles": SMALL_PROFILES,
    },
    128: {
        "n": list(range(44, 51)),
        "ell": [54, 56, 58, 59, 60, 62, 64],
        "p_bits": [44, 48, 52],
        "hat_q_bits": [48],
        "hat_n": [48, 49, 50],
        "hat_k": [49, 50, 51],
        "profiles": LARGE_PROFILES,
    },
    256: {
        "n": list(range(38, 43)),
        "ell": [56, 58, 59, 60],
        "p_bits": [48, 52],
        "hat_q_bits": [49, 52],
        "hat_n": [47, 48, 49, 50, 52, 56],
        "hat_k": [50, 51, 52, 56],
        "profiles": LARGE_PROFILES,
    },
}

INSTANCE_LABELS = {
    "outer_mlwr": "MLWR_{p,q,ell,n,U_beta,U_Be}",
    "outer_hiding": "MLWE_{q,ell+rprime,n,U_Be,U_beta}",
    "selector_hiding": "MLWE_{hat_q,hat_k,hat_n,U_beta,D_{2^K_b/sqrt(12)}}",
    "selector_binding_asis": "A-MSIS^infty_{hat_n,m_sel,hat_q,beta_sel}",
    "aux_msis2_required_delta": "MSIS_{q,n+rprime,ell+n+rprime,beta_SIS2}",
}


def json_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        if RR(value) == ZZ(value):
            return int(ZZ(value))
        return float(RR(value))
    except Exception:
        return str(value)


def profile_parameters(profile: str) -> dict[str, int]:
    match = re.fullmatch(r"m3_pa(\d+)_ps(\d+)_pb(\d+)", profile)
    if not match:
        raise ValueError(f"unrecognized profile name: {profile}")
    return {
        "phi_a": int(match.group(1)),
        "phi_s": int(match.group(2)),
        "phi_b": int(match.group(3)),
        "phi_m": int(oom.PHI_M),
    }


def candidate_rows_for_N(N: int):
    grid = SEARCH_GRID[N]
    for n in grid["n"]:
        for ell in grid["ell"]:
            for p_bits in grid["p_bits"]:
                for hat_q_bits in grid["hat_q_bits"]:
                    for hat_n in grid["hat_n"]:
                        for hat_k in grid["hat_k"]:
                            for profile in grid["profiles"]:
                                row = {
                                    "N": N,
                                    "n": n,
                                    "ell": ell,
                                    "p_bits": p_bits,
                                    "hat_q_bits": hat_q_bits,
                                    "hat_n": hat_n,
                                    "hat_k": hat_k,
                                    "profile": profile,
                                }
                                row.update(profile_parameters(profile))
                                yield row


def row_key(row: dict[str, object]) -> tuple:
    return (
        int(row["N"]),
        int(row["n"]),
        int(row["ell"]),
        int(row["p_bits"]),
        int(row["hat_q_bits"]),
        int(row["hat_n"]),
        int(row["hat_k"]),
        str(row["profile"]),
    )


def attach_final_row_inputs(row: dict[str, object], final_by_N: dict[int, dict[str, object]]) -> dict[str, object]:
    out = dict(row)
    final = final_by_N[int(row["N"])]
    out["tau_g0"] = final["tau_g0"]
    out["tau_g1"] = final["tau_g1"]
    out["epsilon_g_upper"] = final["epsilon_g_upper"]
    out["epsilon_cmp_modelled"] = oom.compression_stability_model(out)["epsilon_cmp_modelled"]
    # Dummy estimator pins are needed only because checked_instances() records
    # the same schema as the final-row security JSON.
    out.update(
        {
            "mlwe_q0_delta": 0.0,
            "mlwe_q0_blocksize": 0,
            "mlwe_q0_gb_bits": 0.0,
            "hiding_mlwe_delta": 0.0,
            "hiding_mlwe_blocksize": 0,
            "selector_mlwe_delta": 0.0,
            "selector_asis_delta": 0.0,
        }
    )
    return out


def enrich_candidate(row: dict[str, object]):
    p = oom.selected_p(oom.D_OUT, int(row["p_bits"]))
    q = p * oom.Q0
    hat_q = oom.selected_hat_q(int(row["hat_q_bits"]), oom.hat_q_lower(row))
    bounds = oom.response_bounds(row)
    selector_bounds = oom.selector_asis_bounds(row)
    size = oom.oom_size_bits(row, hat_q)
    repeat = oom.repeat_accounting(row)
    lhs, rhs = oom.msis_mr09_l2(q, int(row["n"]), bounds["beta_sis"])
    lhs2, rhs2 = oom.msis_mr09_l2(q, ZZ(row["n"]) + oom.EMBEDDED_KEY_RANK, bounds["beta_sis_2"])
    delta_req2 = oom.msis_mr09_required_delta(q, ZZ(row["n"]) + oom.EMBEDDED_KEY_RANK, bounds["beta_sis_2"])
    euclidean_lhs = RR("1.2") * sqrt(
        bounds["sigma_s"] ** 2 * RR(oom.D_OUT) * (RR(row["ell"]) + RR(row["n"]))
        + bounds["sigma_m"] ** 2 * RR(oom.D_OUT)
    )
    euclidean_rhs = RR("1.19") * bounds["sigma_s"] * sqrt(
        RR(oom.D_OUT) * (RR(row["ell"]) + RR(row["n"]) + RR(oom.EMBEDDED_KEY_RANK))
    )
    checks = {
        "p_split_factor_is_2": oom.split_factor_count(oom.D_OUT, p) == 2,
        "p_congruent_5_mod_8": p % ZZ(8) == ZZ(5),
        "q0_congruent_5_mod_8": oom.Q0 % ZZ(8) == ZZ(5),
        "hat_q_congruent_5_mod_8": hat_q % ZZ(8) == ZZ(5),
        "hat_q_exceeds_lower_bound": RR(hat_q) > oom.hat_q_lower(row),
        "q_exceeds_beta_sis": RR(q) > bounds["beta_sis"],
        "q_exceeds_beta_sis_2_requirement": RR(q) > bounds["beta_sis_2_q_requirement"],
        "hat_q_exceeds_beta_sel_inf": RR(hat_q) > RR(selector_bounds["beta_sel_inf"]),
        "outer_msis_mr09_pass": lhs > rhs,
        "outer_auxiliary_msis2_mr09_pass": lhs2 > rhs2,
        "outer_auxiliary_msis2_delta_at_or_below_target": RR(delta_req2) <= oom.RHF_TARGET,
        "repeat_bound_at_or_below_10": RR(repeat["mu_oom"]) <= RR(10),
        "product_bound_below_0p01": RR(row["epsilon_g_upper"]) <= RR("0.01"),
        "euclidean_sigma_m_at_most_sigma_s": bounds["sigma_m"] <= bounds["sigma_s"],
        "euclidean_tail_threshold_condition": euclidean_lhs >= euclidean_rhs,
        "phi_m_is_max_for_eta": ZZ(row["phi_m"]) == oom.max_phi_m(),
        "phi_m_constraint_strict": oom.phi_m_constraint_holds(row["phi_m"]),
        "K_a_matches_boundgen": oom.K_A == oom.k_a_boundgen(row),
    }
    out = dict(row)
    out.update(
        {
            "p": int(p),
            "q": int(q),
            "hat_q": int(hat_q),
            "B_response": float(bounds["B_response"]),
            "beta_sis": float(bounds["beta_sis"]),
            "selector_asis_bounds": selector_bounds,
        }
    )
    out["checked_instances"] = oom.checked_instances(out, p, q, hat_q)
    return out, bounds, selector_bounds, size, repeat, checks, delta_req2


def short_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "N": int(row["N"]),
        "n": int(row["n"]),
        "ell": int(row["ell"]),
        "p_bits": int(row["p_bits"]),
        "hat_q_bits": int(row["hat_q_bits"]),
        "hat_n": int(row["hat_n"]),
        "hat_k": int(row["hat_k"]),
        "profile": str(row["profile"]),
    }


def classify_estimator_failures(deltas: dict[str, object]) -> list[str]:
    return [label for label, value in deltas.items() if RR(value) > oom.RHF_TARGET]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    estimator_path = est_runner.add_estimator_path(None)
    from estimator import LWE, ND, RC

    msis_path = est_runner.add_msis_path(None)
    import MSIS_security

    final_by_N = {int(row["N"]): row for row in oom.FINAL_ROWS}
    constants = {
        "d": int(oom.D_OUT),
        "q0": int(oom.Q0),
        "B_e": int(oom.B_E),
        "beta": int(oom.BETA),
        "w": int(oom.W),
        "gamma": int(oom.GAMMA),
        "embedded_key_rank": int(oom.EMBEDDED_KEY_RANK),
        "phi_m": int(oom.PHI_M),
        "K_b": int(oom.K_B),
        "K_a": int(oom.K_A),
        "s_c": int(oom.S_C),
        "tau_rej": int(oom.TAU_REJ),
        "q_tilde": int(oom.Q_TILDE),
    }
    caches = {"outer_mlwr": {}, "outer_hiding": {}, "selector_hiding": {}, "selector_binding_asis": {}}

    def estimator_deltas(row, delta_req2):
        key = (int(row["ell"]), int(row["q"]))
        if key not in caches["outer_mlwr"]:
            caches["outer_mlwr"][key] = est_runner.run_outer_mlwr_as_mlwe(
                row, constants, LWE, ND, RC, run_arora_gb=False
            )
        key = (int(row["ell"]), int(row["n"]), int(row["q"]))
        if key not in caches["outer_hiding"]:
            caches["outer_hiding"][key] = est_runner.run_dual_instance(
                row["checked_instances"]["outer_hiding_mlwe"], constants, LWE, ND, RC, secret_bound=constants["beta"]
            )
        key = (int(row["hat_k"]), int(row["hat_n"]), int(row["hat_q"]))
        if key not in caches["selector_hiding"]:
            caches["selector_hiding"][key] = est_runner.run_dual_instance(
                row["checked_instances"]["selector_hiding_mlwe"], constants, LWE, ND, RC, secret_bound=constants["beta"]
            )
        key = (
            int(row["N"]),
            int(row["hat_k"]),
            int(row["hat_n"]),
            int(row["hat_q"]),
            int(row["phi_a"]),
            int(row["phi_b"]),
            float(row["tau_g0"]),
            float(row["tau_g1"]),
        )
        if key not in caches["selector_binding_asis"]:
            try:
                result = est_runner.run_selector_asis_msis(row, constants, MSIS_security)
            except ValueError as exc:
                if str(exc) != "MIN_b is too big! Choose smaller MIN_b!":
                    raise
                # The bundled optimizer starts at b=300. If it cannot inspect
                # two block sizes, the instance cannot reach the b>=315
                # required by delta <= 1.004690; record a definite failure at
                # the optimizer's boundary rather than aborting the grid.
                result = {
                    "delta": float(MSIS_security.delta_BKZ(300)),
                    "blocksize": 300,
                    "cost_bits": None,
                    "cost_bits_diagnostic_only": True,
                    "estimator_status": "below-b=300 boundary",
                }
            caches["selector_binding_asis"][key] = result
        return {
            INSTANCE_LABELS["outer_mlwr"]: float(RR(caches["outer_mlwr"][(int(row["ell"]), int(row["q"]))]["delta"])),
            INSTANCE_LABELS["outer_hiding"]: float(RR(caches["outer_hiding"][(int(row["ell"]), int(row["n"]), int(row["q"]))]["delta"])),
            INSTANCE_LABELS["selector_hiding"]: float(
                RR(caches["selector_hiding"][(int(row["hat_k"]), int(row["hat_n"]), int(row["hat_q"]))]["delta"])
            ),
            INSTANCE_LABELS["selector_binding_asis"]: float(RR(caches["selector_binding_asis"][key]["delta"])),
            INSTANCE_LABELS["aux_msis2_required_delta"]: float(RR(delta_req2)),
        }

    summaries = []
    any_smaller_pass = []
    selected_in_grid = True

    for N in sorted(SEARCH_GRID):
        print(f"minimality search N={N}", file=sys.stderr, flush=True)
        final = final_by_N[N]
        final_key = row_key(final)
        grid_rows = [attach_final_row_inputs(row, final_by_N) for row in candidate_rows_for_N(N)]
        if final_key not in {row_key(row) for row in grid_rows}:
            selected_in_grid = False

        smaller_rows = []
        for row in grid_rows:
            enriched, bounds, selector_bounds, size, repeat, checks, delta_req2 = enrich_candidate(row)
            if RR(size["oom_kb"]) < RR(final["oom_kb"]) - RR("1e-12"):
                smaller_rows.append((float(size["oom_kb"]), enriched, bounds, selector_bounds, size, repeat, checks, delta_req2))
        smaller_rows.sort(
            key=lambda item: (
                item[0],
                int(item[1]["ell"]) + int(item[1]["n"]),
                int(item[1]["ell"]),
                int(item[1]["n"]),
                int(item[1]["p_bits"]),
                int(item[1]["hat_q_bits"]),
                int(item[1]["hat_n"]),
                int(item[1]["hat_k"]),
                str(item[1]["profile"]),
            )
        )

        deterministic_fail_reasons = {}
        estimator_fail_reasons = {}
        deterministic_pass_count = 0
        estimator_fail_count = 0
        profile_full_pass = []
        examples = []
        first_deterministic_pass = []

        for _, row, bounds, selector_bounds, size, repeat, checks, delta_req2 in smaller_rows:
            failed_det = tuple(name for name, ok in checks.items() if not ok)
            if failed_det:
                deterministic_fail_reasons[str(failed_det)] = deterministic_fail_reasons.get(str(failed_det), 0) + 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "row": short_row(row),
                            "oom_kb": float(size["oom_kb"]),
                            "failure_type": "deterministic",
                            "failed_checks": list(failed_det),
                        }
                    )
                continue

            deterministic_pass_count += 1
            if len(first_deterministic_pass) < args.max_examples:
                first_deterministic_pass.append({"row": short_row(row), "oom_kb": float(size["oom_kb"])})

            deltas = estimator_deltas(row, delta_req2)
            failed_est = tuple(classify_estimator_failures(deltas))
            if failed_est:
                estimator_fail_count += 1
                estimator_fail_reasons[str(failed_est)] = estimator_fail_reasons.get(str(failed_est), 0) + 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "row": short_row(row),
                            "oom_kb": float(size["oom_kb"]),
                            "failure_type": "estimator_delta",
                            "failed_instances": list(failed_est),
                            "deltas": deltas,
                        }
                    )
                continue

            passing = {
                "row": short_row(row),
                "oom_kb": float(size["oom_kb"]),
                "current_oom_kb": float(final["oom_kb"]),
                "saving_kib": float(RR(final["oom_kb"]) - RR(size["oom_kb"])),
                "deltas": deltas,
            }
            profile_full_pass.append(passing)
            any_smaller_pass.append(passing)

        summaries.append(
            {
                "N": N,
                "selected_row": short_row(final),
                "selected_oom_kb": float(final["oom_kb"]),
                "search_grid_size": len(grid_rows),
                "strictly_smaller_rows": len(smaller_rows),
                "deterministic_pass_smaller_rows": deterministic_pass_count,
                "deterministic_fail_smaller_rows": len(smaller_rows) - deterministic_pass_count,
                "estimator_fail_smaller_rows": estimator_fail_count,
                "full_pass_smaller_rows": len(profile_full_pass),
                "deterministic_failure_reasons": deterministic_fail_reasons,
                "estimator_failure_reasons": estimator_fail_reasons,
                "first_deterministic_pass_examples": first_deterministic_pass,
                "failure_examples": examples,
            }
        )

    payload = {
        "status": "PASS" if selected_in_grid and not any_smaller_pass else "FAIL",
        "claim_tested": "Within the finite search grid recorded in this file, no strictly smaller OOM row passes all deterministic and estimator checks.",
        "scope_note": "This is a finite-grid diagnostic, not a theorem over all integer parameter tuples.",
        "target_delta": float(oom.RHF_TARGET),
        "repeat_target": 10.0,
        "estimator_path": "external/lattice-estimator",
        "msis_security_path": "external/msis-security",
        "selected_rows_in_grid": selected_in_grid,
        "search_grid": SEARCH_GRID,
        "summaries": summaries,
        "smaller_full_pass_rows": any_smaller_pass,
        "cache_sizes": {name: len(cache) for name, cache in caches.items()},
    }

    output = Path(args.output)
    if not args.no_write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if payload["status"] != "PASS":
        raise SystemExit(json.dumps({"status": "FAIL", "smaller_full_pass_rows": len(any_smaller_pass)}, sort_keys=True))
    print(json.dumps({"output": str(output.relative_to(ROOT)), "rows": len(summaries), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
