#!/usr/bin/env python3
"""Build flat all-parameter tables from the verified RiVeR OOM JSON outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"
SECURITY_PATH = DATA_DIR / "final_oom_security.json"
ESTIMATOR_PATH = DATA_DIR / "final_oom_estimator_rerun.json"
TSV_PATH = DATA_DIR / "final_oom_all_parameters.tsv"
MD_PATH = REPORT_DIR / "final_oom_all_parameters.md"


COLUMNS = [
    "N",
    "d",
    "w",
    "gamma",
    "q0",
    "q0_mod_8",
    "B_e",
    "beta",
    "embedded_key_rank",
    "tail_factor",
    "rhf_target",
    "attempt_target",
    "n",
    "ell",
    "p",
    "p_bits",
    "p_mod_8",
    "q",
    "log2_q",
    "hat_n",
    "hat_k",
    "hat_q",
    "hat_q_bits",
    "hat_q_mod_8",
    "hat_q_lower_bound",
    "K_b",
    "K_a",
    "s_c",
    "tau_rej",
    "q_tilde",
    "profile",
    "phi_a",
    "phi_s",
    "phi_m",
    "phi_m_constraint_lhs",
    "phi_m_constraint_rhs",
    "phi_b",
    "tau_g0",
    "tau_g1",
    "epsilon_g_upper",
    "epsilon_cmp_modelled",
    "compression_joint_pass_residues",
    "B_a",
    "mathcal_B",
    "B_s",
    "eta_m",
    "B_response",
    "sigma_s",
    "sigma_m",
    "beta_sis_1",
    "beta_sis",
    "beta_sis_bound_1_4wgamma_beta_sis_1",
    "beta_sis_bound_2_beta_sis_1_plus_2B_response",
    "beta_sis_active_bound",
    "beta_sis_2",
    "beta_sis_2_q_requirement",
    "twelve_sigma_s",
    "twelve_sigma_m",
    "outer_msis_mr09_lhs",
    "outer_msis_mr09_rhs",
    "outer_auxiliary_msis2_delta_required",
    "outer_auxiliary_msis2_mr09_lhs",
    "outer_auxiliary_msis2_mr09_rhs",
    "B_g_0",
    "B_g_1",
    "beta_sel_inf",
    "selector_six_widths",
    "selector_six_raw_bounds",
    "selector_six_inf_bounds",
    "selector_merged_widths",
    "selector_merged_raw_bounds",
    "selector_merged_inf_bounds",
    "mu_a",
    "mu_b",
    "mu_s",
    "mu_m",
    "epsilon_a_tail",
    "epsilon_b_tail",
    "epsilon_s_tail",
    "epsilon_m_tail",
    "epsilon_2",
    "epsilon_2_log2",
    "epsilon_2_dimension",
    "joint_response_success",
    "success_denominator",
    "repeat_bound",
    "oom_kb",
    "oom_total_bits",
    "oom_challenge_bits",
    "oom_B_bits",
    "oom_z_bits",
    "oom_zb_bits",
    "oom_zs_bits",
    "oom_zm_bits",
    "oom_f_bits",
    "outer_mlwr_delta",
    "outer_mlwr_blocksize",
    "outer_mlwr_arora_gb_bits",
    "outer_hiding_mlwe_delta",
    "outer_hiding_mlwe_blocksize",
    "selector_hiding_mlwe_delta",
    "selector_hiding_mlwe_blocksize",
    "selector_binding_asis_delta",
    "selector_binding_asis_blocksize",
    "selector_binding_asis_cost_bits_diagnostic",
    "all_checks_pass",
]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value == 0:
            return "0"
        if abs(value) >= 1e12 or abs(value) < 1e-4:
            return f"{value:.12e}"
        return f"{value:.12f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return "|".join(fmt(item) for item in value)
    return str(value)


def checked(row: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_rows() -> list[dict[str, Any]]:
    security = json.loads(SECURITY_PATH.read_text(encoding="utf-8"))
    estimator = json.loads(ESTIMATOR_PATH.read_text(encoding="utf-8"))
    constants = security["global_constants"]
    estimator_by_n = {int(row["N"]): row for row in estimator["rows"]}
    out_rows: list[dict[str, Any]] = []

    for row in security["rows"]:
        n_value = int(row["N"])
        est_row = estimator_by_n[n_value]
        est_results = est_row["results"]
        aux = row["checked_instances"]["outer_auxiliary_msis2"]
        sel = row["selector_asis_bounds"]
        repeat = row["repeat_accounting"]
        compression = repeat["compression_stability_model"]
        size = row["oom_size_breakdown_bits"]

        beta_sis_1 = row["beta_sis_1"]
        beta_sis_bound_1 = row["beta_sis_bound_1"]
        beta_sis_bound_2 = row["beta_sis_bound_2"]

        flat = {
            "N": n_value,
            "d": constants["d"],
            "w": constants["w"],
            "gamma": constants["gamma"],
            "q0": constants["q0"],
            "q0_mod_8": row["q0_mod_8"],
            "B_e": constants["B_e"],
            "beta": constants["beta"],
            "embedded_key_rank": constants["embedded_key_rank"],
            "tail_factor": constants["tail_factor"],
            "rhf_target": constants["rhf_target"],
            "attempt_target": 10,
            "n": row["n"],
            "ell": row["ell"],
            "p": row["p"],
            "p_bits": row["p_bits"],
            "p_mod_8": row["p_mod_8"],
            "q": row["q"],
            "log2_q": row["log2_q"],
            "hat_n": row["hat_n"],
            "hat_k": row["hat_k"],
            "hat_q": row["hat_q"],
            "hat_q_bits": row["hat_q_bits"],
            "hat_q_mod_8": row["hat_q_mod_8"],
            "hat_q_lower_bound": row["hat_q_lower_bound"],
            "K_b": constants["K_b"],
            "K_a": constants["K_a"],
            "s_c": constants["s_c"],
            "tau_rej": constants["tau_rej"],
            "q_tilde": constants["q_tilde"],
            "profile": row["profile"],
            "phi_a": row["phi_a"],
            "phi_s": row["phi_s"],
            "phi_m": row.get("phi_m", constants["phi_m"]),
            "phi_m_constraint_lhs": row.get("phi_m_constraint_lhs"),
            "phi_m_constraint_rhs": row.get("phi_m_constraint_rhs"),
            "phi_b": row["phi_b"],
            "tau_g0": row["tau_g0"],
            "tau_g1": row["tau_g1"],
            "epsilon_g_upper": row["epsilon_g_upper"],
            "epsilon_cmp_modelled": row["epsilon_cmp_modelled"],
            "compression_joint_pass_residues": compression["joint_numerator"],
            "B_a": constants["gamma"] * math.sqrt(2 * constants["w"]),
            "mathcal_B": checked(row, "selector_asis_bounds.six_raw_bounds", [None])[0] / (6 * row["phi_b"]),
            "B_s": row.get("B_s", aux["sigma_s"] / row["phi_s"]),
            "eta_m": row.get("eta_m", aux["sigma_m"] / row.get("phi_m", constants["phi_m"])),
            "B_response": row["B_response"],
            "sigma_s": row.get("sigma_s", aux["sigma_s"]),
            "sigma_m": row.get("sigma_m", aux["sigma_m"]),
            "beta_sis_1": beta_sis_1,
            "beta_sis": row["beta_sis"],
            "beta_sis_bound_1_4wgamma_beta_sis_1": beta_sis_bound_1,
            "beta_sis_bound_2_beta_sis_1_plus_2B_response": beta_sis_bound_2,
            "beta_sis_active_bound": "4w_gamma_beta_sis_1",
            "beta_sis_2": row["outer_auxiliary_msis2_beta_sis_2"],
            "beta_sis_2_q_requirement": row["outer_auxiliary_msis2_q_requirement"],
            "twelve_sigma_s": aux["twelve_sigma_s"],
            "twelve_sigma_m": aux["twelve_sigma_m"],
            "outer_msis_mr09_lhs": row["outer_msis_mr09_lhs"],
            "outer_msis_mr09_rhs": row["outer_msis_mr09_rhs"],
            "outer_auxiliary_msis2_delta_required": row["outer_auxiliary_msis2_delta_required"],
            "outer_auxiliary_msis2_mr09_lhs": row["outer_auxiliary_msis2_mr09_lhs"],
            "outer_auxiliary_msis2_mr09_rhs": row["outer_auxiliary_msis2_mr09_rhs"],
            "B_g_0": sel["B_g_0"],
            "B_g_1": sel["B_g_1"],
            "beta_sel_inf": sel["beta_sel_inf"],
            "selector_six_widths": sel["six_widths"],
            "selector_six_raw_bounds": sel["six_raw_bounds"],
            "selector_six_inf_bounds": sel["six_inf_bounds"],
            "selector_merged_widths": sel["merged_widths"],
            "selector_merged_raw_bounds": sel["merged_raw_bounds"],
            "selector_merged_inf_bounds": sel["merged_inf_bounds"],
            "mu_a": repeat["mu_a"],
            "mu_b": repeat["mu_b"],
            "mu_s": repeat["mu_s"],
            "mu_m": repeat["mu_m"],
            "epsilon_a_tail": repeat["epsilon_a_tail"],
            "epsilon_b_tail": repeat["epsilon_b_tail"],
            "epsilon_s_tail": repeat["epsilon_s_tail"],
            "epsilon_m_tail": repeat["epsilon_m_tail"],
            "epsilon_2": repeat["epsilon_2"],
            "epsilon_2_log2": repeat["epsilon_2_log2"],
            "epsilon_2_dimension": repeat["epsilon_2_dimension"],
            "joint_response_success": repeat["joint_response_success"],
            "success_denominator": repeat["success_denominator"],
            "repeat_bound": row["repeat_bound"],
            "oom_kb": row["oom_kb"],
            "oom_total_bits": size["total_bits"],
            "oom_challenge_bits": size["challenge_bits"],
            "oom_B_bits": size["B_bits"],
            "oom_z_bits": size["z_bits"],
            "oom_zb_bits": size["zb_bits"],
            "oom_zs_bits": size["zs_bits"],
            "oom_zm_bits": size["zm_bits"],
            "oom_f_bits": size["f_bits"],
            "outer_mlwr_delta": checked(est_results, "outer_mlwr_as_mlwe.delta"),
            "outer_mlwr_blocksize": checked(est_results, "outer_mlwr_as_mlwe.blocksize"),
            "outer_mlwr_arora_gb_bits": checked(est_results, "outer_mlwr_as_mlwe.arora_gb_bits"),
            "outer_hiding_mlwe_delta": checked(est_results, "outer_hiding_mlwe.delta"),
            "outer_hiding_mlwe_blocksize": checked(est_results, "outer_hiding_mlwe.blocksize"),
            "selector_hiding_mlwe_delta": checked(est_results, "selector_hiding_mlwe.delta"),
            "selector_hiding_mlwe_blocksize": checked(est_results, "selector_hiding_mlwe.blocksize"),
            "selector_binding_asis_delta": checked(est_results, "selector_binding_asis_msis.delta"),
            "selector_binding_asis_blocksize": checked(est_results, "selector_binding_asis_msis.blocksize"),
            "selector_binding_asis_cost_bits_diagnostic": checked(est_results, "selector_binding_asis_msis.cost_bits"),
            "all_checks_pass": all(bool(value) for value in row["checks"].values()) and not est_row["failures"],
        }
        out_rows.append(flat)

    return out_rows


def write_tsv(rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in COLUMNS})


def write_markdown(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overview_cols = [
        "N", "oom_kb", "repeat_bound", "n", "ell", "p_bits", "log2_q",
        "hat_q_bits", "hat_n", "hat_k", "phi_a", "phi_s", "phi_m", "phi_b",
        "tau_g0", "tau_g1", "outer_mlwr_delta", "outer_hiding_mlwe_delta",
        "selector_hiding_mlwe_delta", "selector_binding_asis_delta", "all_checks_pass",
    ]
    modulus_cols = ["N", "p", "q", "hat_q", "p_mod_8", "q0_mod_8", "hat_q_mod_8", "hat_q_lower_bound", "beta_sel_inf"]
    bound_cols = [
        "N", "B_s", "eta_m", "B_response", "sigma_s", "sigma_m", "beta_sis_1",
        "beta_sis", "beta_sis_2", "beta_sis_2_q_requirement", "B_g_0", "B_g_1",
    ]
    size_cols = [
        "N", "oom_total_bits", "oom_kb", "oom_challenge_bits", "oom_B_bits",
        "oom_z_bits", "oom_zb_bits", "oom_zs_bits", "oom_zm_bits", "oom_f_bits",
    ]
    repeat_cols = [
        "N", "mu_a", "mu_b", "mu_s", "mu_m", "epsilon_a_tail", "epsilon_b_tail",
        "epsilon_s_tail", "epsilon_m_tail", "epsilon_2", "epsilon_2_log2", "joint_response_success",
        "epsilon_g_upper", "epsilon_cmp_modelled", "success_denominator", "repeat_bound",
    ]

    def table(cols: list[str]) -> list[str]:
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
        return lines

    lines = [
        "# RiVeR OOM Complete Parameter Table",
        "",
        "This report flattens the verified JSON outputs into reviewer-readable tables.",
        "The corresponding machine-readable table is `data/final_oom_all_parameters.tsv`.",
        "Long selector vectors are kept in the TSV as `|`-separated entries.",
        "",
        "## Overview",
        "",
        *table(overview_cols),
        "",
        "## Moduli And Selector Limits",
        "",
        *table(modulus_cols),
        "",
        "## Main Bounds",
        "",
        *table(bound_cols),
        "",
        "## Size Breakdown",
        "",
        *table(size_cols),
        "",
        "## Repeat Accounting",
        "",
        *table(repeat_cols),
        "",
        "## Full TSV Columns",
        "",
        "The TSV additionally includes fixed constants, selector raw/inf-norm vectors,",
        "MR09 left/right sides, estimator block sizes, and diagnostic A-MSIS cost bits.",
        "",
        "```text",
        "\n".join(COLUMNS),
        "```",
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_tsv(rows)
    write_markdown(rows)
    print(json.dumps({"rows": len(rows), "tsv": str(TSV_PATH.relative_to(ROOT)), "markdown": str(MD_PATH.relative_to(ROOT)), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
