#!/usr/bin/env python3
"""Emit and verify the final RiVeR OOM parameter table.

This script is intentionally small.  It records only the final rows used by
the manuscript OOM table and checks that the retained rows satisfy the stated
attempt budget.  It does not re-run the lattice estimators or include search
logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"

REPEAT_TARGET = 10.0

GLOBAL_CONSTANTS = {
    "outer_ring_degree": 32,
    "exact_error_bound": 30,
    "secret_bound": 1,
    "challenge_weight": 32,
    "challenge_gamma": 16,
    "embedded_key_rank": 1,
    "phi_m": 32,
    "phi_m_constraint_bits": 26,
    "q_tilde": 67107713,
    "selector_tail_factor": 6,
    "tau_rej": 12,
    "K_b": 5,
    "K_a": 28,
    "s_c": 3,
    "repeat_target": REPEAT_TARGET,
}

ROWS = [
    {
        "N": 8,
        "oom_kb": 20.133209060562596,
        "repeat_bound": 8.34430735637057,
        "n": 44,
        "ell": 54,
        "p": 17592186043877,
        "q": 1073123348676497,
        "p_bits": 44,
        "hat_q": 8796093022237,
        "hat_q_bits": 44,
        "hat_n": 42,
        "hat_k": 46,
        "profile": 'm3_pa32_ps26_pb2',
        "phi_a": 32,
        "phi_s": 26,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 583259.4695605036,
        "beta_sis": 6155761876992.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
    },
    {
        "N": 16,
        "oom_kb": 21.409119640954838,
        "repeat_bound": 8.435196176201655,
        "n": 41,
        "ell": 59,
        "p": 281474976710597,
        "q": 17169973579346417,
        "p_bits": 48,
        "hat_q": 35184372088997,
        "hat_q_bits": 46,
        "hat_n": 43,
        "hat_k": 49,
        "profile": 'm3_pa40_ps22_pb2',
        "phi_a": 40,
        "phi_s": 22,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 563546.1917820047,
        "beta_sis": 5315022028800.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
    },
    {
        "N": 64,
        "oom_kb": 25.535994066823953,
        "repeat_bound": 8.626610904273274,
        "n": 44,
        "ell": 54,
        "p": 17592186043877,
        "q": 1073123348676497,
        "p_bits": 44,
        "hat_q": 140737488355333,
        "hat_q_bits": 48,
        "hat_n": 50,
        "hat_k": 51,
        "profile": 'm3_pa34_ps24_pb2',
        "phi_a": 34,
        "phi_s": 24,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 583259.4695605036,
        "beta_sis": 5682241732608.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
    },
    {
        "N": 128,
        "oom_kb": 29.12017786190496,
        "repeat_bound": 8.616634985923838,
        "n": 45,
        "ell": 54,
        "p": 17592186043877,
        "q": 1073123348676497,
        "p_bits": 44,
        "hat_q": 140737488355333,
        "hat_q_bits": 48,
        "hat_n": 50,
        "hat_k": 51,
        "profile": 'm3_pa24_ps34_pb2',
        "phi_a": 24,
        "phi_s": 34,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 589695.9861080962,
        "beta_sis": 8131983704064.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
    },
    {
        "N": 256,
        "oom_kb": 36.21287399348253,
        "repeat_bound": 8.543595907038434,
        "n": 42,
        "ell": 59,
        "p": 281474976710597,
        "q": 17169973579346417,
        "p_bits": 48,
        "hat_q": 281474976710677,
        "hat_q_bits": 49,
        "hat_n": 49,
        "hat_k": 52,
        "profile": 'm3_pa22_ps40_pb2',
        "phi_a": 22,
        "phi_s": 40,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 570205.2766083457,
        "beta_sis": 9760313180160.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
    },
]

FIELDNAMES = [
    "N",
    "oom_kb",
    "repeat_bound",
    "n",
    "ell",
    "p",
    "q",
    "p_bits",
    "hat_q",
    "hat_q_bits",
    "hat_n",
    "hat_k",
    "K_b",
    "K_a",
    "s_c",
    "tau_rej",
    "q_tilde",
    "profile",
    "phi_a",
    "phi_s",
    "phi_m",
    "phi_b",
    "B_response",
    "beta_sis",
    "beta_sis_active_bound",
]


def rows_with_constants() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in ROWS:
        out = dict(row)
        out["K_b"] = GLOBAL_CONSTANTS["K_b"]
        out["K_a"] = GLOBAL_CONSTANTS["K_a"]
        out["s_c"] = GLOBAL_CONSTANTS["s_c"]
        out["tau_rej"] = GLOBAL_CONSTANTS["tau_rej"]
        out["q_tilde"] = GLOBAL_CONSTANTS["q_tilde"]
        out["phi_m"] = GLOBAL_CONSTANTS["phi_m"]
        rows.append(out)
    return rows


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_tsv(rows: list[dict[str, object]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "final_oom_parameters.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row[key]) for key in FIELDNAMES})
    return path


def write_json(rows: list[dict[str, object]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "final_oom_parameters.json"
    payload = {
        "global_constants": GLOBAL_CONSTANTS,
        "formula_summary": {
            "outer_prime_selection": "p is prime, split_factor(d,p)=2, p == 5 mod 8, and q=p*q0",
            "selector_prime_selection": "hat_q is prime and hat_q == 5 mod 8",
            "B_a": "gamma*sqrt(2*w)",
            "B_s": "w*gamma*B_e*sqrt(d*(n+ell))",
            "B_response": "w*gamma*sqrt(d*(ell*beta^2 + (n+embedded_key_rank)*B_e^2))",
            "eta_m": "w*gamma*B_e*sqrt(d)",
            "phi_m_selection": "largest integer phi_m satisfying q_tilde > 24*phi_m*eta_m",
            "sigma_a": "phi_a*B_a",
            "sigma_s": "phi_s*B_s",
            "sigma_m": "phi_m*eta_m",
            "mathcal_B": "gamma*w*sqrt(d*hat_k)",
            "beta_sis_1": "2.4*sqrt(sigma_s^2*(ell+n)*d)",
            "beta_sis": "max(4*w*gamma*beta_sis_1, beta_sis_1 + 2*B_response)",
            "beta_sis_2": "2.4*sqrt(d*(ell+n)*sigma_s^2 + d*sigma_m^2)",
            "beta_sis_2_q_requirement": "q > max(beta_sis_2, 12*sigma_s, 12*sigma_m)",
            "epsilon_2": "1.19^(d*(ell+n+1))*exp(d*(ell+n+1)*(1-1.19^2)/2)",
            "joint_response_success": "(1-epsilon_s)*(1-epsilon_m)-epsilon_2",
            "B_g_0": "tau_g0*(d*(N-1)/3)*(phi_a*B_a)^2",
            "B_g_1": "tau_g1*(d/2)*(phi_a*B_a)^2",
            "beta_sel_vector": "4*w*gamma*(6*phi_b*mathcal_B, gamma+6*(N-1)*phi_a*B_a, 6*phi_a*B_a, B_g_0, B_g_1, 2^K_a)",
            "K_a": "K_b + ceil(log2(w*gamma*hat_n*d)) + s_c",
            "hat_q_lower_bound": "max(2*(2*gamma+12*phi_a*B_a)^2, 2*N^2, 2^(K_a+1))",
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_report(rows: list[dict[str, object]]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "final_oom_parameters.md"
    lines = [
        "# RiVeR OOM Parameter Verification Report",
        "",
        "This generated report summarizes the final OOM rows retained for the manuscript table.",
        "The complete bound inventory and security-instance data are in `data/final_oom_security.json`.",
        "",
        "## Fixed constants",
        "",
    ]
    for key, value in GLOBAL_CONSTANTS.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Formulas",
            "",
            "- `B_response = w gamma sqrt(d(ell beta^2 + (n + embedded_key_rank) B_e^2))`.",
            "- `eta_m = w gamma B_e sqrt(d)`.",
            "- `mathcal B_s = w gamma B_e sqrt(d(n+ell))`.",
            "- `sigma_s = phi_s mathcal B_s`.",
            "- `phi_m` is the largest integer satisfying `q_tilde > 24 phi_m eta_m`.",
            "- `sigma_m = phi_m eta_m`.",
            "- `beta_sis_1 = 2.4 sqrt(sigma_s^2 (ell+n) d)`.",
            "- `beta_sis = max(4 w gamma beta_sis_1, beta_sis_1 + 2 B_response)`.",
            "- `beta_sis_2 = 2.4 sqrt(d(ell+n) sigma_s^2 + d sigma_m^2)`.",
            "- Auxiliary MSIS2 checks require `q > max(beta_sis_2, 12 sigma_s, 12 sigma_m)` and `delta <= 1.004690`.",
            "- The joint Euclidean response check contributes `epsilon_2 <= 1.19^(d(ell+n+1)) exp(d(ell+n+1)(1-1.19^2)/2)` to repeat accounting.",
            "- The sequential response-block success term is `(1-epsilon_s)(1-epsilon_m)-epsilon_2`.",
            "- Selector A-MSIS beta-vector components and merged estimator widths are stored in `selector_asis_bounds`.",
            "- Repeat accounting is recomputed in `data/final_oom_security.json` and checked against `repeat_bound`.",
            "- Outer `p` is selected with split factor 2 and `p == 5 mod 8`; `q=p q_0` is recorded but has no separate mod-8 requirement.",
            "- The fixed exact modulus ratio satisfies `q_0=61 == 5 mod 8`, and every selected `hat_q` satisfies `hat_q == 5 mod 8`.",
            "",
            "## Retained rows",
            "",
            "| N | OOM KiB | repeat bound | n | ell | p | q | p bits | hat q | hat q bits | hat n | hat k | phi a | phi s | phi m | phi b |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {N} | {oom_kb:.6f} | {repeat_bound:.6f} | {n} | {ell} | {p} | {q} | {p_bits} | "
            "{hat_q} | {hat_q_bits} | {hat_n} | {hat_k} | {phi_a} | {phi_s} | {phi_m} | {phi_b} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Every retained row has `repeat_bound <= {REPEAT_TARGET:g}`.",
            "- Detailed MLWR, MLWE, and MSIS instances are recorded in `data/final_oom_security.json`.",
            "- The security JSON is generated by `scripts/river_oom_math_checks.sage.py`.",
            "- The auxiliary `MSIS_{q,n+1,ell+n+1,beta_sis_2}` check is included in `data/final_oom_security.json`.",
            "- The real LWE estimator and selector A-MSIS estimator calls are re-run by `scripts/run_final_oom_estimators.sage.py`.",
            "- Selector A-MSIS acceptance is by root-Hermite factor `delta <= 1.004690`; cost bits are diagnostic only.",
            "- The A-MSIS estimator rerun uses BKZ blocksize step size 1.",
            "- The table contains only the selected OOM rows and no separate backend measurements.",
            "- The script records final rows only; it is not a broad-grid optimality certificate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def check_rows(rows: list[dict[str, object]]) -> None:
    seen = set()
    for row in rows:
        n_value = int(row["N"])
        if n_value in seen:
            raise SystemExit(f"duplicate N={n_value}")
        seen.add(n_value)
        repeat = float(row["repeat_bound"])
        if not math.isfinite(repeat) or repeat > REPEAT_TARGET:
            raise SystemExit(f"N={n_value}: repeat_bound={repeat} exceeds target {REPEAT_TARGET}")
        if float(row["oom_kb"]) <= 0:
            raise SystemExit(f"N={n_value}: non-positive OOM size")
        if int(row["phi_b"]) != 2:
            raise SystemExit(f"N={n_value}: unexpected phi_b")
        product = (
            GLOBAL_CONSTANTS["challenge_weight"]
            * GLOBAL_CONSTANTS["challenge_gamma"]
            * int(row["hat_n"])
            * GLOBAL_CONSTANTS["outer_ring_degree"]
        )
        k_a = (
            GLOBAL_CONSTANTS["K_b"]
            + (product - 1).bit_length()
            + GLOBAL_CONSTANTS["s_c"]
        )
        if int(row["K_a"]) != k_a:
            raise SystemExit(f"N={n_value}: K_a={row['K_a']} but BoundGen derives {k_a}")
        if int(row["q"]) != int(row["p"]) * 61:
            raise SystemExit(f"N={n_value}: q does not equal p*q0")
    if seen != {8, 16, 64, 128, 256}:
        raise SystemExit(f"unexpected N set: {sorted(seen)}")


def check_package_shape() -> None:
    """Check that the curated release contains the files needed to reproduce the tables.

    This deliberately does not fail on additional optional files.  Reviewers
    can add local logs while investigating the package; the release check only
    asserts that the reproducibility-critical files are present and that obvious
    cache/metadata files or generated Sage preparse files are absent.
    """
    required = {
        Path("README.md"),
        Path("THIRD_PARTY.md"),
        Path(".gitignore"),
        Path("Makefile"),
        Path("data/final_oom_parameters.tsv"),
        Path("data/final_oom_parameters.json"),
        Path("data/final_oom_security.json"),
        Path("data/final_oom_estimator_rerun.json"),
        Path("data/final_oom_all_parameters.tsv"),
        Path("data/oom_search_minimality_diagnostic.json"),
        Path("data/product_tau_validation.csv"),
        Path("data/product_tau_validation_metadata.json"),
        Path("report/final_oom_parameters.md"),
        Path("report/final_oom_all_parameters.md"),
        Path("scripts/river_oom_math_checks.sage.py"),
        Path("scripts/run_final_oom_estimators.sage.py"),
        Path("scripts/verify_oom_search_minimality.sage.py"),
        Path("scripts/reproduce_final_table.py"),
        Path("scripts/make_all_parameters_table.py"),
        Path("scripts/validate_product_tau_inputs.py"),
        Path("scripts/run_all_checks.sh"),
        Path("external/lattice-estimator/LANES.sage"),
        Path("external/lattice-estimator/UPSTREAM.txt"),
        Path("external/lattice-estimator/COPYING.LESSER-3.0.txt"),
        Path("optional_challenge_invertibility/README.md"),
        Path("optional_challenge_invertibility/d256_current_q_result.txt"),
    }
    required_external = {
        Path("external/lattice-estimator/estimator/__init__.py"),
        Path("external/lattice-estimator/requirements.txt"),
        Path("external/msis-security/MSIS_security.py"),
        Path("external/msis-security/model_BKZ.py"),
        Path("external/msis-security/proba_util.py"),
    }
    actual = {path.relative_to(ROOT) for path in ROOT.rglob("*") if path.is_file()}
    forbidden = [
        path
        for path in actual
        if (
            path.name == ".DS_Store"
            or path.suffix == ".pyc"
            or "__pycache__" in path.parts
            or (path.name.endswith(".sage.py") and path.parts[0] != "scripts")
            or path in {Path("CHANGE_SUMMARY.md"), Path("MANIFEST.tsv"), Path("SHA256SUMS")}
        )
    ]
    missing = (required | required_external) - actual
    if missing or forbidden:
        missing = sorted(str(path) for path in missing)
        forbidden = sorted(str(path) for path in forbidden)
        raise SystemExit(f"unexpected package shape; missing={missing}; forbidden={forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="check rows and package shape")
    parser.add_argument("--no-write", action="store_true", help="do not emit generated artifacts")
    args = parser.parse_args()

    rows = rows_with_constants()
    if not args.no_write:
        write_tsv(rows)
        write_json(rows)
        write_report(rows)
    check_rows(rows)
    if args.check:
        check_package_shape()
    print(json.dumps({"rows": len(rows), "repeat_target": REPEAT_TARGET, "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
