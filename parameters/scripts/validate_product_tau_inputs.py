#!/usr/bin/env python3
"""Validate the product-threshold experiment inputs used by repeat accounting.

The OOM math checker recomputes the final product thresholds B_g,0 and B_g,1
from the final phi_a values.  The empirical input is scale-free: each row fixes
tau_g0 and tau_g1 and records a fresh one-million-trial product-norm
validation run.  This script checks that the final parameter rows use exactly
those tau values and epsilon_g upper confidence bounds.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

try:
    from scipy.stats import beta as beta_distribution
except Exception as exc:  # pragma: no cover - dependency check.
    raise SystemExit("scipy is required; run this file with SageMath if system Python lacks scipy") from exc


cwd_root = Path.cwd().resolve()
if (cwd_root / "scripts" / "validate_product_tau_inputs.py").is_file():
    ROOT = cwd_root
else:
    ROOT = Path(__file__).resolve().parents[1]
SECURITY_PATH = ROOT / "data" / "final_oom_security.json"
TAU_PATH = ROOT / "data" / "product_tau_validation.csv"
TAU_METADATA_PATH = ROOT / "data" / "product_tau_validation_metadata.json"
EXPECTED_N = {8, 16, 64, 128, 256}


def clopper_pearson_upper(failures: int, trials: int, alpha: float) -> float:
    if failures == trials:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, failures + 1, trials - failures))


def close(a: float, b: float, tolerance: float = 5e-10) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def main() -> None:
    security = json.loads(SECURITY_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(TAU_METADATA_PATH.read_text(encoding="utf-8"))
    rows = {int(row["N"]): row for row in security["rows"]}
    tau_rows = {int(row["N"]): row for row in csv.DictReader(TAU_PATH.open(newline="", encoding="utf-8"))}

    if set(rows) != EXPECTED_N:
        raise SystemExit(f"unexpected security N set: {sorted(rows)}")
    if set(tau_rows) != EXPECTED_N:
        raise SystemExit(f"unexpected product-tau N set: {sorted(tau_rows)}")
    if int(metadata["validation_trials_per_row"]) != 1_000_000:
        raise SystemExit("unexpected product-tau validation trial count in metadata")
    if not close(float(metadata["alpha_cell"]), 0.01):
        raise SystemExit("unexpected product-tau alpha_cell in metadata")
    metadata_N = set()
    for run in metadata["validation_runs"]:
        metadata_N.update(int(value) for value in run["row_order"])
    if metadata_N != EXPECTED_N:
        raise SystemExit(f"unexpected product-tau metadata N set: {sorted(metadata_N)}")

    checked = []
    for n_value in sorted(EXPECTED_N):
        final = rows[n_value]
        tau = tau_rows[n_value]
        failures = int(tau["validation_failures"])
        trials = int(tau["validation_trials"])
        alpha_cell = float(tau["alpha_cell"])
        empirical = failures / trials
        upper = clopper_pearson_upper(failures, trials, alpha_cell)
        recorded_upper = float(tau["epsilon_g_validation_upper"])

        checks = {
            "tau_g0_matches": close(float(final["tau_g0"]), float(tau["tau_g0_fixed"])),
            "tau_g1_matches": close(float(final["tau_g1"]), float(tau["tau_g1_fixed"])),
            "empirical_rate_matches": close(empirical, float(tau["epsilon_g_validation_hat"])),
            "upper_bound_matches_csv": close(upper, recorded_upper),
            "upper_bound_matches_final": close(recorded_upper, float(final["epsilon_g_upper"])),
            "trial_count_is_one_million": trials == 1_000_000,
            "alpha_cell_is_0p01": close(alpha_cell, 0.01),
        }
        if not all(checks.values()):
            failed = [key for key, ok in checks.items() if not ok]
            raise SystemExit(f"N={n_value} failed product-tau validation checks: {failed}")

        checked.append(
            {
                "N": n_value,
                "failures": failures,
                "trials": trials,
                "epsilon_g_hat": empirical,
                "epsilon_g_upper": recorded_upper,
            }
        )

    print(json.dumps({"rows": len(checked), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
