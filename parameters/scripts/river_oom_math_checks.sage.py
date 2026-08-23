#!/usr/bin/env sage
"""Mathematical checks for the final RiVeR OOM parameter rows.

The script is self-contained on purpose: it does not import search logs. It
recomputes the deterministic quantities that define the selected
rows and verifies the pinned estimator outputs recorded in
data/final_oom_security.json.
"""

from __future__ import annotations

import json
import importlib.util
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

cwd_root = Path.cwd().resolve()
if (cwd_root / "scripts" / "reproduce_final_table.py").is_file():
    ROOT = cwd_root
else:
    ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
constants_spec = importlib.util.spec_from_file_location(
    "reproduce_final_table", SCRIPT_DIR / "reproduce_final_table.py"
)
constants_module = importlib.util.module_from_spec(constants_spec)
constants_spec.loader.exec_module(constants_module)
GLOBAL_CONSTANTS = constants_module.GLOBAL_CONSTANTS

try:
    from sage.all import RR, ZZ, Mod, RealField, binomial, ceil, exp, log, next_prime, previous_prime, sqrt
except Exception as exc:  # pragma: no cover - this is a Sage script.
    raise SystemExit("run this file with SageMath") from exc


SECURITY_PATH = ROOT / "data" / "final_oom_security.json"

RHF_TARGET = RR("1.004690")
W = ZZ(32)
GAMMA = ZZ(16)
D_OUT = ZZ(32)
Q0 = ZZ(61)
BETA = ZZ(1)
B_E = ZZ(30)
EMBEDDED_KEY_RANK = ZZ(1)
PHI_M = ZZ(32)
PHI_M_CONSTRAINT_BITS = ZZ(26)
Q_TILDE = ZZ(GLOBAL_CONSTANTS["q_tilde"])
TAU_REJ = ZZ(GLOBAL_CONSTANTS["tau_rej"])
K_B = ZZ(5)
K_A = ZZ(28)
S_C = ZZ(GLOBAL_CONSTANTS["s_c"])
TAIL_FACTOR = ZZ(6)
HP = RealField(256)
EUCLIDEAN_TAIL_RATIO = HP("1.19")


FINAL_ROWS = [
    {
        "N": 8,
        "oom_kb": 20.133209060562596,
        "repeat_bound": 8.34430735637057,
        "n": 44,
        "ell": 54,
        "p_bits": 44,
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
        "mlwe_q0_delta": 1.004647519882841,
        "mlwe_q0_blocksize": 319,
        "mlwe_q0_gb_bits": 3253.508200535606,
        "hiding_mlwe_delta": 1.0044695784127526,
        "hiding_mlwe_blocksize": 338,
        "selector_mlwe_delta": 1.0046870515700201,
        "selector_asis_delta": 1.0046671909922382,
        "tau_g0": 3.14,
        "tau_g1": 2.68,
        "epsilon_g_upper": 0.0079564543,
        "epsilon_cmp_modelled": 0.07876605629624323,
    },
    {
        "N": 16,
        "oom_kb": 21.409119640954838,
        "repeat_bound": 8.435196176201655,
        "n": 41,
        "ell": 59,
        "p_bits": 48,
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
        "mlwe_q0_delta": 1.0046183623349951,
        "mlwe_q0_blocksize": 322,
        "mlwe_q0_gb_bits": 3542.377770741416,
        "hiding_mlwe_delta": 1.0043648906208422,
        "hiding_mlwe_blocksize": 350,
        "selector_mlwe_delta": 1.0046087348754058,
        "selector_asis_delta": 1.004677097419558,
        "tau_g0": 3.09,
        "tau_g1": 3.08,
        "epsilon_g_upper": 0.007793341355707927,
        "epsilon_cmp_modelled": 0.08056203688975627,
    },
    {
        "N": 64,
        "oom_kb": 25.535994066823953,
        "repeat_bound": 8.626610904273274,
        "n": 44,
        "ell": 54,
        "p_bits": 44,
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
        "mlwe_q0_delta": 1.004647519882841,
        "mlwe_q0_blocksize": 319,
        "mlwe_q0_gb_bits": 3253.508200535606,
        "hiding_mlwe_delta": 1.0044695784127526,
        "hiding_mlwe_blocksize": 338,
        "selector_mlwe_delta": 1.0046573319311118,
        "selector_asis_delta": 1.0046280354292965,
        "tau_g0": 3.05,
        "tau_g1": 3.33,
        "epsilon_g_upper": 0.0090602037,
        "epsilon_cmp_modelled": 0.09304766034013878,
    },
    {
        "N": 128,
        "oom_kb": 28.95220911190496,
        "repeat_bound": 8.599820475055308,
        "n": 45,
        "ell": 54,
        "p_bits": 44,
        "hat_q_bits": 48,
        "hat_n": 49,
        "hat_k": 51,
        "profile": 'm3_pa24_ps34_pb2',
        "phi_a": 24,
        "phi_s": 34,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 589695.9861080962,
        "beta_sis": 8131983704064.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
        "mlwe_q0_delta": 1.004647519882841,
        "mlwe_q0_blocksize": 319,
        "mlwe_q0_gb_bits": 3253.508200535606,
        "hiding_mlwe_delta": 1.004487582246496,
        "hiding_mlwe_blocksize": 336,
        "selector_mlwe_delta": 1.0046573319311118,
        "selector_asis_delta": 1.0046870515700201,
        "tau_g0": 3.09,
        "tau_g1": 3.58,
        "epsilon_g_upper": 0.0078500789,
        "epsilon_cmp_modelled": 0.09127437216317247,
    },
    {
        "N": 256,
        "oom_kb": 36.04099899348253,
        "repeat_bound": 8.526923940593298,
        "n": 42,
        "ell": 59,
        "p_bits": 48,
        "hat_q_bits": 49,
        "hat_n": 48,
        "hat_k": 52,
        "profile": 'm3_pa22_ps40_pb2',
        "phi_a": 22,
        "phi_s": 40,
        "phi_m": 32,
        "phi_b": 2,
        "B_response": 570205.2766083457,
        "beta_sis": 9760313180160.0,
        "beta_sis_active_bound": '4w_gamma_beta_sis_1',
        "mlwe_q0_delta": 1.0046183623349951,
        "mlwe_q0_blocksize": 322,
        "mlwe_q0_gb_bits": 3542.377770741416,
        "hiding_mlwe_delta": 1.004381952541052,
        "hiding_mlwe_blocksize": 348,
        "selector_mlwe_delta": 1.0046671909922382,
        "selector_asis_delta": 1.0046573319311118,
        "tau_g0": 3.06,
        "tau_g1": 3.84,
        "epsilon_g_upper": 0.0085985639,
        "epsilon_cmp_modelled": 0.08949753540374028,
    },
]


def log2_rr(x):
    return RR(x).log() / RR(2).log()


def split_factor_count(d, p):
    order = Mod(ZZ(p), ZZ(2) * ZZ(d)).multiplicative_order()
    return ZZ(d) // ZZ(order)


def selected_p(d, bits):
    p = previous_prime(ZZ(2) ** ZZ(bits))
    lower = ZZ(2) ** (ZZ(bits) - 1)
    while p > lower:
        if p % ZZ(8) == ZZ(5) and split_factor_count(d, p) == 2:
            return ZZ(p)
        p = previous_prime(p)
    raise ValueError(f"no split prime satisfying p == 5 mod 8 found for bits={bits}")


def selected_hat_q(bits, lower_bound):
    # `next_prime` is strict, so starting at floor(lower_bound) includes a
    # prime at ceil(lower_bound) when the bound is non-integral.  Keeping the
    # floor in Sage integers avoids a binary64-dependent prime-selection edge.
    start = max(ZZ(2) ** (ZZ(bits) - 1), ZZ(RR(lower_bound).floor()))
    q = ZZ(next_prime(start))
    while q % ZZ(8) != ZZ(5):
        q = ZZ(next_prime(q))
    return q


def centered_uniform_sigma(B):
    B = RR(B)
    return sqrt(B * (B + RR(1)) / RR(3))


def h_gaussian(sigma):
    return log2_rr(RR("4.13") * RR(sigma))


def high_bits_width(q, dropped_bits):
    scale = ZZ(2) ** ZZ(dropped_bits)
    alphabet_size = (ZZ(q) + scale - ZZ(1)) // scale
    return ZZ(ceil(log2_rr(alphabet_size)))


def challenge_bits_direct():
    return log2_rr(binomial(int(D_OUT), int(W))) + RR(W) * log2_rr(RR(2) * RR(GAMMA))


def phi_m_constraint_holds(phi):
    """Decide `q_tilde > 24*phi*eta_m` using exact integer squares."""
    phi = ZZ(phi)
    if phi <= 0:
        return False
    lhs_scale = ZZ(24) * phi * W * GAMMA * B_E
    return lhs_scale ** 2 * D_OUT < Q_TILDE ** 2


def max_phi_m():
    """Largest positive integer satisfying the exact `q_tilde` condition."""
    scale_sq = (ZZ(24) * W * GAMMA * B_E) ** 2 * D_OUT
    phi = ZZ(math.isqrt(int((Q_TILDE ** 2 - ZZ(1)) // scale_sq)))
    if phi <= 0:
        raise ValueError("no positive phi_m satisfies q_tilde > 24*phi_m*eta_m")
    return phi


def response_bounds(row):
    n = RR(row["n"])
    ell = RR(row["ell"])
    phi_s = RR(row["phi_s"])
    B_s = RR(W) * RR(GAMMA) * RR(B_E) * sqrt(RR(D_OUT) * (ell + n))
    eta_m = RR(W) * RR(GAMMA) * RR(B_E) * sqrt(RR(D_OUT))
    phi_m = RR(row.get("phi_m", max_phi_m()))
    B_response = RR(W) * RR(GAMMA) * sqrt(
        RR(D_OUT) * (ell * RR(BETA) ** 2 + (n + RR(EMBEDDED_KEY_RANK)) * RR(B_E) ** 2)
    )
    sigma_s = phi_s * B_s
    sigma_m = phi_m * eta_m
    beta_sis_1 = RR("2.4") * sqrt(sigma_s ** 2 * (ell + n) * RR(D_OUT))
    beta_sis_2 = RR("2.4") * sqrt(RR(D_OUT) * (ell + n) * sigma_s ** 2 + RR(D_OUT) * sigma_m ** 2)
    beta_sis_bound_1 = RR(4) * RR(W) * RR(GAMMA) * beta_sis_1
    beta_sis_bound_2 = beta_sis_1 + RR(2) * B_response
    beta_sis_2_q_requirement = max(beta_sis_2, RR(12) * sigma_s, RR(12) * sigma_m)
    return {
        "B_a": RR(GAMMA) * sqrt(RR(2) * RR(W)),
        "mathcal_B": RR(GAMMA) * RR(W) * sqrt(RR(D_OUT) * RR(row["hat_k"])),
        "B_s": B_s,
        "eta_m": eta_m,
        "phi_m": phi_m,
        "phi_m_constraint_lhs": RR(24) * phi_m * eta_m,
        "phi_m_constraint_rhs": Q_TILDE,
        "B_response": B_response,
        "sigma_s": sigma_s,
        "sigma_m": sigma_m,
        "beta_sis_1": beta_sis_1,
        "beta_sis_bound_1": beta_sis_bound_1,
        "beta_sis_bound_2": beta_sis_bound_2,
        "beta_sis": max(beta_sis_bound_1, beta_sis_bound_2),
        "beta_sis_2": beta_sis_2,
        "beta_sis_2_q_requirement": beta_sis_2_q_requirement,
    }

def selector_asis_bounds(row):
    bounds = response_bounds(row)
    N = ZZ(row["N"])
    hat_n = ZZ(row["hat_n"])
    hat_k = ZZ(row["hat_k"])
    phi_a = RR(row["phi_a"])
    phi_b = RR(row["phi_b"])
    scale = RR(4) * RR(W) * RR(GAMMA)
    sigma_a = phi_a * bounds["B_a"]
    B_g_1 = RR(row["tau_g1"]) * sigma_a ** 2 * RR(D_OUT) / RR(2)
    B_g_0 = RR(row["tau_g0"]) * sigma_a ** 2 * RR(D_OUT) * RR(N - ZZ(1)) / RR(3)
    six_widths = [hat_k, ZZ(1), N - ZZ(1), ZZ(1), N - ZZ(1), hat_n]
    six_labels = ["z_b", "f_0", "f_1", "g_0", "g_1", "e_cmp"]
    six_raw = [
        RR(TAIL_FACTOR) * phi_b * bounds["mathcal_B"],
        RR(GAMMA) + RR(TAIL_FACTOR) * RR(N - ZZ(1)) * phi_a * bounds["B_a"],
        RR(TAIL_FACTOR) * phi_a * bounds["B_a"],
        B_g_0,
        B_g_1,
        RR(ZZ(2) ** K_A),
    ]
    six_inf = [scale * value for value in six_raw]
    merged_widths = [hat_k, N, ZZ(1), N - ZZ(1), hat_n]
    merged_labels = ["z_b", "f_merged", "g_0", "g_1", "e_cmp"]
    merged_raw = [six_raw[0], max(six_raw[1], six_raw[2]), six_raw[3], six_raw[4], six_raw[5]]
    merged_inf = [scale * value for value in merged_raw]
    return {
        "six_labels": six_labels,
        "six_widths": [int(x) for x in six_widths],
        "six_raw_bounds": [float(x) for x in six_raw],
        "six_inf_bounds": [float(x) for x in six_inf],
        "merged_labels": merged_labels,
        "merged_widths": [int(x) for x in merged_widths],
        "merged_raw_bounds": [float(x) for x in merged_raw],
        "merged_inf_bounds": [float(x) for x in merged_inf],
        "beta_sel_inf": float(max(six_inf)),
        "B_g_0": float(B_g_0),
        "B_g_1": float(B_g_1),
    }


def oom_size_bits(row, hat_q):
    bounds = response_bounds(row)
    N = RR(row["N"])
    ell = RR(row["ell"])
    n = RR(row["n"])
    hat_n = RR(row["hat_n"])
    hat_k = RR(row["hat_k"])
    b_B = high_bits_width(hat_q, K_B)
    B_bits = hat_n * RR(D_OUT) * RR(b_B)
    f_bits = (N - RR(1)) * RR(D_OUT) * h_gaussian(RR(row["phi_a"]) * bounds["B_a"])
    zb_bits = hat_k * RR(D_OUT) * h_gaussian(RR(row["phi_b"]) * bounds["mathcal_B"])
    # Current protocol sends (z_s, z_key) at sigma_s and only z_eval at
    # sigma_m.  This matches the response split used in the current manuscript
    # and implementation.
    zs_bits = (ell + n) * RR(D_OUT) * h_gaussian(bounds["sigma_s"])
    zm_bits = RR(EMBEDDED_KEY_RANK) * RR(D_OUT) * h_gaussian(bounds["sigma_m"])
    x_bits = challenge_bits_direct()
    total = B_bits + x_bits + f_bits + zb_bits + zs_bits + zm_bits
    return {
        "B_bits": float(B_bits),
        "challenge_bits": float(x_bits),
        "f_bits": float(f_bits),
        "zb_bits": float(zb_bits),
        "zs_bits": float(zs_bits),
        "zm_bits": float(zm_bits),
        "z_bits": float(zs_bits + zm_bits),
        "total_bits": float(total),
        "oom_kb": float(total / RR(8192)),
    }


def compression_stability_model(row):
    hat_q = selected_hat_q(row["hat_q_bits"], hat_q_lower(row))
    threshold = ZZ(2) ** (ZZ(K_A) - ZZ(1)) - ZZ(W) * ZZ(GAMMA) * (ZZ(2) ** (ZZ(K_B) - ZZ(1)))
    q_threshold = (ZZ(hat_q) - ZZ(1)) // ZZ(2) - ZZ(W) * ZZ(GAMMA) * (ZZ(2) ** (ZZ(K_B) - ZZ(1)))
    exponent = ZZ(row["hat_n"]) * D_OUT
    low_denominator = ZZ(2) ** ZZ(K_A)
    low_numerator = ZZ(2) * threshold - ZZ(1)
    q_numerator = ZZ(2) * q_threshold - ZZ(1)
    if low_numerator <= 0 or q_numerator <= 0:
        low_pass = RR(0)
        mod_pass = RR(0)
        combined_pass = RR(0)
        joint_numerator = ZZ(0)
    else:
        low_pass = exp(exponent * log(RR(low_numerator) / RR(low_denominator)))
        mod_pass = exp(exponent * log(RR(q_numerator) / RR(hat_q)))

        # Both checks inspect the same centred residue. Count their
        # intersection exactly over Z_hat_q instead of multiplying the two
        # marginal probabilities.
        def accepted_prefix(n):
            periods, remainder = divmod(ZZ(n), low_denominator)
            return (
                periods * low_numerator
                + min(remainder, threshold)
                + max(ZZ(0), remainder - (low_denominator - threshold + ZZ(1)))
            )

        lower = -(q_threshold - ZZ(1))
        upper = q_threshold - ZZ(1)
        joint_numerator = accepted_prefix(upper + ZZ(1)) - accepted_prefix(lower)
        combined_pass = exp(exponent * log(RR(joint_numerator) / RR(hat_q)))
    epsilon_cmp = max(RR(0), RR(1) - combined_pass)
    return {
        "hat_q": int(hat_q),
        "coefficient_count": int(row["hat_n"] * D_OUT),
        "low_threshold": int(threshold),
        "mod_threshold": int(q_threshold),
        "low_numerator": int(low_numerator),
        "low_denominator": int(low_denominator),
        "q_numerator": int(q_numerator),
        "q_denominator": int(hat_q),
        "joint_numerator": int(joint_numerator),
        "joint_denominator": int(hat_q),
        "low_pass": float(low_pass),
        "mod_pass": float(mod_pass),
        "combined_pass": float(combined_pass),
        "epsilon_cmp_modelled": float(epsilon_cmp),
        "model": "(jointly accepted residues / hat_q)^(hat_n*d)",
        "model_assumption": "the hat_n*d coefficients are independent and uniform in Z_hat_q",
    }


def repeat_accounting(row):
    N = RR(row["N"])
    ell = RR(row["ell"])
    n = RR(row["n"])
    hat_k = RR(row["hat_k"])
    phi_a = RR(row["phi_a"])
    phi_s = RR(row["phi_s"])
    phi_b = RR(row["phi_b"])
    phi_m = RR(response_bounds(row)["phi_m"])

    mu_a = exp(RR(TAU_REJ) / phi_a + RR(1) / (RR(2) * phi_a ** 2))
    mu_s = exp(RR(TAU_REJ) / phi_s + RR(1) / (RR(2) * phi_s ** 2))
    mu_m = exp(RR(TAU_REJ) / phi_m + RR(1) / (RR(2) * phi_m ** 2))
    mu_b = RR(2) * exp(RR(1) / (RR(2) * phi_b ** 2))

    epsilon_a_tail = RR(2) * RR(D_OUT) * (N - RR(1)) * exp(RR(-18))
    epsilon_b_tail = RR(2) * RR(D_OUT) * hat_k * exp(RR(-18))
    epsilon_s_tail = RR(2) * RR(D_OUT) * (n + ell) * exp(RR(-18))
    epsilon_m_tail = RR(2) * RR(D_OUT) * exp(RR(-18))
    euclidean_dimension = HP(D_OUT) * HP(row["ell"] + row["n"] + int(EMBEDDED_KEY_RANK))
    epsilon_2_log = euclidean_dimension * (
        EUCLIDEAN_TAIL_RATIO.log()
        + (HP(1) - EUCLIDEAN_TAIL_RATIO ** 2) / HP(2)
    )
    epsilon_2 = epsilon_2_log.exp()
    epsilon_g = RR(row["epsilon_g_upper"])
    compression_model = compression_stability_model(row)
    epsilon_cmp = RR(compression_model["epsilon_cmp_modelled"])
    # The paper applies these checks sequentially and defines each failure
    # probability after all preceding checks have succeeded. Their
    # conditional success probabilities therefore multiply; no independence
    # assumption is needed.
    joint_response_success = (
        (HP(1) - HP(epsilon_s_tail)) * (HP(1) - HP(epsilon_m_tail)) - epsilon_2
    )
    denominator = (
        (HP(1) - HP(epsilon_a_tail))
        * (HP(1) - HP(epsilon_b_tail))
        * joint_response_success
        * (HP(1) - HP(epsilon_g))
        * (HP(1) - HP(epsilon_cmp))
    )
    mu_oom = HP(mu_a) * HP(mu_b) * HP(mu_s) * HP(mu_m) / denominator
    return {
        "mu_a": float(mu_a),
        "mu_b": float(mu_b),
        "mu_s": float(mu_s),
        "mu_m": float(mu_m),
        "epsilon_a_tail": float(epsilon_a_tail),
        "epsilon_b_tail": float(epsilon_b_tail),
        "epsilon_s_tail": float(epsilon_s_tail),
        "epsilon_m_tail": float(epsilon_m_tail),
        "epsilon_2": float(epsilon_2),
        "epsilon_2_log2": float(epsilon_2_log / HP(2).log()),
        "epsilon_2_dimension": int(euclidean_dimension),
        "epsilon_2_tail_ratio": float(EUCLIDEAN_TAIL_RATIO),
        "joint_response_success": float(joint_response_success),
        "epsilon_g_upper": float(epsilon_g),
        "epsilon_cmp_modelled": float(epsilon_cmp),
        "epsilon_cmp_table_value": float(RR(row["epsilon_cmp_modelled"])),
        "compression_stability_model": compression_model,
        "success_denominator": float(denominator),
        "mu_oom": float(mu_oom),
    }


def hat_q_lower(row):
    B_a = RR(GAMMA) * sqrt(RR(2) * RR(W))
    term_rejection = RR(2) * (RR(2) * RR(GAMMA) + RR(12) * RR(row["phi_a"]) * B_a) ** 2
    term_ring = RR(2) * RR(row["N"]) ** 2
    term_compression = RR(2) ** (ZZ(K_A) + ZZ(1))
    return max(term_rejection, term_ring, term_compression)


def k_a_boundgen(row):
    product = W * GAMMA * ZZ(row["hat_n"]) * D_OUT
    ceil_log2 = ZZ(product - ZZ(1)).nbits()
    return K_B + ceil_log2 + S_C


def msis_mr09_l2(q, rank_rows, gamma_l2):
    exponent = RR(2) * sqrt(RR(rank_rows) * RR(D_OUT) * log2_rr(q) * log2_rr(RHF_TARGET))
    lhs = min(RR(q), RR(2) ** exponent)
    rhs = RR(2) * RR(gamma_l2)
    return lhs, rhs


def msis_mr09_required_delta(q, rank_rows, gamma_l2):
    rhs = RR(2) * RR(gamma_l2)
    if rhs >= RR(q):
        return oo
    return RR(2) ** ((log2_rr(rhs) ** RR(2)) / (RR(4) * RR(rank_rows) * RR(D_OUT) * log2_rr(q)))


def binary64_equal(a, b, max_ulps=8):
    """Compare full-precision pinned values, never rounded display values."""
    left = float(a)
    right = float(b)
    if not math.isfinite(left) or not math.isfinite(right):
        return left == right
    return abs(left - right) <= max_ulps * max(math.ulp(left), math.ulp(right))


def checked_instances(row, p, q, hat_q):
    lwe_rounding_sigma = centered_uniform_sigma(B_E)
    selector_sigma = RR(ZZ(2) ** ZZ(K_B)) / sqrt(RR(12))
    bounds = response_bounds(row)
    return {
        "outer_mlwr_as_mlwe": {
            "label": "MLWE_{ell,q,q0}",
            "expanded_lwe_n": int(D_OUT * ZZ(row["ell"])),
            "expanded_lwe_m": "unlimited for lattice attack; d*(n+2^128) for Arora-GB",
            "q": int(q),
            "secret_distribution": "Uniform[-1,1]",
            "error_distribution": "Uniform[-30,30]",
            "error_sigma": float(lwe_rounding_sigma),
            "delta": row["mlwe_q0_delta"],
            "blocksize": row["mlwe_q0_blocksize"],
            "arora_gb_bits": row["mlwe_q0_gb_bits"],
        },
        "outer_hiding_mlwe": {
            "label": "MLWE_{q,ell+embedded_key_rank,n,U_Be,U_beta}",
            "expanded_lwe_n": int(D_OUT * (ZZ(row["ell"]) + EMBEDDED_KEY_RANK)),
            "expanded_lwe_m": int(D_OUT * ZZ(row["n"])),
            "q": int(q),
            "secret_distribution": "Uniform[-1,1]",
            "paper_secret_distribution": "U_beta^(ell+embedded_key_rank)",
            "error_distribution": "Uniform[-30,30] approximated by DiscreteGaussian(error_sigma) in the dual estimator",
            "error_sigma": float(centered_uniform_sigma(B_E)),
            "delta": row["hiding_mlwe_delta"],
            "blocksize": row["hiding_mlwe_blocksize"],
        },
        "selector_hiding_mlwe": {
            "label": "MLWE_{hat_q,hat_k,hat_n,U_beta,D_{2^K_b/sqrt(12)}}",
            "expanded_lwe_n": int(D_OUT * ZZ(row["hat_k"])),
            "expanded_lwe_m": int(D_OUT * ZZ(row["hat_n"])),
            "q": int(hat_q),
            "secret_distribution": "Uniform[-1,1]",
            "error_sigma": float(selector_sigma),
            "delta": row["selector_mlwe_delta"],
        },
        "outer_systematic_msis": {
            "label": "MSIS_{q,n,n+ell+embedded_key_rank,beta_sis}",
            "rank_rows": int(row["n"]),
            "width_cols": int(ZZ(row["n"]) + ZZ(row["ell"]) + EMBEDDED_KEY_RANK),
            "q": int(q),
            "beta_sis": row["beta_sis"],
            "delta_test": "MR09 l2 condition with target root-Hermite factor 1.004690",
        },
        "outer_auxiliary_msis2": {
            "label": "MSIS_{q,n+embedded_key_rank,ell+n+embedded_key_rank,beta_sis_2}",
            "rank_rows": int(ZZ(row["n"]) + EMBEDDED_KEY_RANK),
            "width_cols": int(ZZ(row["ell"]) + ZZ(row["n"]) + EMBEDDED_KEY_RANK),
            "q": int(q),
            "beta_sis_2_formula": "2.4*sqrt(d*(ell+n)*sigma_s^2 + d*sigma_m^2)",
            "beta_sis_2": float(bounds["beta_sis_2"]),
            "sigma_s": float(bounds["sigma_s"]),
            "sigma_m": float(bounds["sigma_m"]),
            "twelve_sigma_s": float(RR(12) * bounds["sigma_s"]),
            "twelve_sigma_m": float(RR(12) * bounds["sigma_m"]),
            "q_requirement": float(bounds["beta_sis_2_q_requirement"]),
            "q_requirement_formula": "q > max(beta_sis_2, 12*sigma_s, 12*sigma_m)",
            "delta_test": "MR09 l2 required root-Hermite factor at target 1.004690",
        },
        "selector_binding_asis": {
            "label": "A-MSIS-infinity over hat_q",
            "rank_rows": int(row["hat_n"]),
            "native_block_widths": [int(row["hat_k"]), 1, int(row["N"]) - 1, 1, int(row["N"]) - 1, int(row["hat_n"])],
            "merged_estimator_widths": [int(row["hat_k"]), int(row["N"]), 1, int(row["N"]) - 1, int(row["hat_n"])],
            "q": int(hat_q),
            "delta": row["selector_asis_delta"],
        },
    }


def main():
    out_rows = []
    for row in FINAL_ROWS:
        p = selected_p(D_OUT, row["p_bits"])
        q = p * Q0
        h_lower = hat_q_lower(row)
        hat_q = selected_hat_q(row["hat_q_bits"], h_lower)
        bounds = response_bounds(row)
        selector_bounds = selector_asis_bounds(row)
        size = oom_size_bits(row, hat_q)
        repeat = repeat_accounting(row)
        euclidean_threshold_lhs = RR("1.2") * sqrt(
            bounds["sigma_s"] ** 2 * RR(D_OUT) * (RR(row["ell"]) + RR(row["n"]))
            + bounds["sigma_m"] ** 2 * RR(D_OUT)
        )
        euclidean_threshold_rhs = RR("1.19") * bounds["sigma_s"] * sqrt(
            RR(D_OUT) * (RR(row["ell"]) + RR(row["n"]) + RR(EMBEDDED_KEY_RANK))
        )
        lhs, rhs = msis_mr09_l2(q, row["n"], bounds["beta_sis"])
        lhs2, rhs2 = msis_mr09_l2(q, ZZ(row["n"]) + EMBEDDED_KEY_RANK, bounds["beta_sis_2"])
        delta_req2 = msis_mr09_required_delta(q, ZZ(row["n"]) + EMBEDDED_KEY_RANK, bounds["beta_sis_2"])

        checks = {
            "p_split_factor_is_2": split_factor_count(D_OUT, p) == 2,
            "p_congruent_5_mod_8": p % ZZ(8) == ZZ(5),
            "q_equals_p_times_q0": q == p * Q0,
            "q0_congruent_5_mod_8": Q0 % ZZ(8) == ZZ(5),
            "hat_q_congruent_5_mod_8": hat_q % ZZ(8) == ZZ(5),
            "hat_q_exceeds_lower_bound": RR(hat_q) > h_lower,
            "B_response_matches_table": binary64_equal(bounds["B_response"], row["B_response"]),
            "beta_sis_matches_table": binary64_equal(bounds["beta_sis"], row["beta_sis"]),
            "q_exceeds_beta_sis": RR(q) > bounds["beta_sis"],
            "q_exceeds_beta_sis_2_requirement": RR(q) > bounds["beta_sis_2_q_requirement"],
            "hat_q_exceeds_beta_sel_inf": RR(hat_q) > RR(selector_bounds["beta_sel_inf"]),
            "outer_msis_mr09_pass": lhs > rhs,
            "outer_auxiliary_msis2_mr09_pass": lhs2 > rhs2,
            "outer_auxiliary_msis2_delta_at_or_below_target": RR(delta_req2) <= RHF_TARGET,
            "oom_kb_matches_table": binary64_equal(size["oom_kb"], row["oom_kb"]),
            "repeat_bound_matches_formula": binary64_equal(repeat["mu_oom"], row["repeat_bound"]),
            "epsilon_cmp_model_matches_table": binary64_equal(repeat["epsilon_cmp_modelled"], row["epsilon_cmp_modelled"]),
            "euclidean_sigma_m_at_most_sigma_s": bounds["sigma_m"] <= bounds["sigma_s"],
            "euclidean_tail_threshold_condition": euclidean_threshold_lhs >= euclidean_threshold_rhs,
            "phi_m_is_max_for_eta": ZZ(row["phi_m"]) == max_phi_m(),
            "phi_m_constraint_strict": phi_m_constraint_holds(row["phi_m"]),
            "K_a_matches_boundgen": K_A == k_a_boundgen(row),
            "deltas_at_or_below_target": all(
                RR(row[key]) <= RHF_TARGET
                for key in ("mlwe_q0_delta", "hiding_mlwe_delta", "selector_mlwe_delta", "selector_asis_delta")
            ),
            "repeat_bound_at_or_below_10": RR(row["repeat_bound"]) <= RR(10),
            "product_bound_below_0p01": RR(row["epsilon_g_upper"]) <= RR("0.01"),
        }
        if not all(checks.values()):
            failed = [key for key, ok in checks.items() if not ok]
            raise SystemExit(f"N={row['N']} failed checks: {failed}")

        out = dict(row)
        out.update(
            {
                "p": int(p),
                "p_mod_8": int(p % ZZ(8)),
                "q": int(q),
                "log2_q": float(log2_rr(q)),
                "q0_mod_8": int(Q0 % ZZ(8)),
                "hat_q": int(hat_q),
                "hat_q_mod_8": int(hat_q % ZZ(8)),
                "hat_q_lower_bound": float(h_lower),
                "B_s": float(bounds["B_s"]),
                "eta_m": float(bounds["eta_m"]),
                "sigma_s": float(bounds["sigma_s"]),
                "sigma_m": float(bounds["sigma_m"]),
                "phi_m_constraint_lhs": float(bounds["phi_m_constraint_lhs"]),
                "phi_m_constraint_rhs": int(bounds["phi_m_constraint_rhs"]),
                "beta_sis_1": float(bounds["beta_sis_1"]),
                "beta_sis_bound_1": float(bounds["beta_sis_bound_1"]),
                "beta_sis_bound_2": float(bounds["beta_sis_bound_2"]),
                "outer_msis_mr09_lhs": float(lhs),
                "outer_msis_mr09_rhs": float(rhs),
                "outer_auxiliary_msis2_beta_sis_2": float(bounds["beta_sis_2"]),
                "outer_auxiliary_msis2_q_requirement": float(bounds["beta_sis_2_q_requirement"]),
                "outer_auxiliary_msis2_delta_required": float(delta_req2),
                "outer_auxiliary_msis2_mr09_lhs": float(lhs2),
                "outer_auxiliary_msis2_mr09_rhs": float(rhs2),
                "oom_size_breakdown_bits": size,
                "repeat_accounting": repeat,
                "euclidean_tail_check": {
                    "threshold_lhs": float(euclidean_threshold_lhs),
                    "threshold_rhs": float(euclidean_threshold_rhs),
                    "sigma_m_at_most_sigma_s": bool(bounds["sigma_m"] <= bounds["sigma_s"]),
                    "threshold_condition_pass": bool(euclidean_threshold_lhs >= euclidean_threshold_rhs),
                    "tail_bound_formula": "1.19^(d*(ell+n+1))*exp(d*(ell+n+1)*(1-1.19^2)/2)",
                },
                "selector_asis_bounds": selector_bounds,
                "checked_instances": checked_instances(row, p, q, hat_q),
                "checks": checks,
            }
        )
        out_rows.append(out)

    payload = {
        "global_constants": {
            "d": int(D_OUT),
            "q0": int(Q0),
            "B_e": int(B_E),
            "beta": int(BETA),
            "w": int(W),
            "gamma": int(GAMMA),
            "embedded_key_rank": int(EMBEDDED_KEY_RANK),
            "phi_m": int(PHI_M),
            "phi_m_constraint_bits": int(PHI_M_CONSTRAINT_BITS),
            "q_tilde": int(Q_TILDE),
            "phi_m_selection": "largest integer phi_m satisfying q_tilde > 24*phi_m*eta_m",
            "tau_rej": int(TAU_REJ),
            "K_b": int(K_B),
            "K_a": int(K_A),
            "s_c": int(S_C),
            "tail_factor": int(TAIL_FACTOR),
            "rhf_target": float(RHF_TARGET),
        },
        "bound_inventory": {
            "oom_response_bounds": {
                "B_s": "w*gamma*B_e*sqrt(d*(n+ell))",
                "eta_m": "w*gamma*B_e*sqrt(d)",
                "phi_m_selection": "largest integer phi_m satisfying q_tilde > 24*phi_m*eta_m",
                "sigma_s": "phi_s*B_s",
                "sigma_m": "phi_m*eta_m",
                "B_response": "w*gamma*sqrt(d*(ell*beta^2 + (n+embedded_key_rank)*B_e^2))",
            },
            "outer_systematic_msis": {
                "instance": "MSIS_{q,n,n+ell+embedded_key_rank,beta_sis}",
                "beta_sis_1": "2.4*sqrt(sigma_s^2*(ell+n)*d)",
                "beta_sis": "max(4*w*gamma*beta_sis_1, beta_sis_1 + 2*B_response)",
                "checks": ["q > beta_sis", "MR09 l2 at rhf_target"],
            },
            "outer_auxiliary_msis2": {
                "instance": "MSIS_{q,n+embedded_key_rank,ell+n+embedded_key_rank,beta_sis_2}",
                "beta_sis_2": "2.4*sqrt(d*(ell+n)*sigma_s^2 + d*sigma_m^2)",
                "checks": ["q > max(beta_sis_2, 12*sigma_s, 12*sigma_m)", "MR09 l2 required delta <= rhf_target"],
            },
            "selector_binding_asis": {
                "B_a": "gamma*sqrt(2*w)",
                "sigma_a": "phi_a*B_a",
                "mathcal_B": "gamma*w*sqrt(d*hat_k)",
                "B_g_0": "tau_g0*(d*(N-1)/3)*(phi_a*B_a)^2",
                "B_g_1": "tau_g1*(d/2)*(phi_a*B_a)^2",
                "native_order": ["z_b", "f_0", "f_1", "g_0", "g_1", "e_cmp"],
                "native_widths": ["hat_k", 1, "N-1", 1, "N-1", "hat_n"],
                "beta_sel_vector": "4*w*gamma*(6*phi_b*mathcal_B, gamma+6*(N-1)*phi_a*B_a, 6*phi_a*B_a, B_g_0, B_g_1, 2^K_a)",
                "inner_bounds": [
                    "6*phi_b*mathcal_B",
                    "gamma + 6*(N-1)*phi_a*B_a",
                    "6*phi_a*B_a",
                    "B_g_0",
                    "B_g_1",
                    "2^K_a",
                ],
                "merged_estimator_widths": ["hat_k", "N", 1, "N-1", "hat_n"],
            },
            "modulus_checks": {
                "q": "q = p*q0 with split_factor(d,p)=2 and p == 5 mod 8",
                "q0_congruence": "q0 == 5 mod 8",
                "hat_q_lower_bound": "hat_q > max(2*(2*gamma+12*phi_a*B_a)^2, 2*N^2, 2^(K_a+1))",
                "hat_q_selector_bound": "hat_q > beta_sel_inf",
                "hat_q_congruence": "hat_q == 5 mod 8",
                "K_a": "K_b + ceil(log2(w*gamma*hat_n*d)) + s_c",
            },
            "repeat_accounting": {
                "mu_a": "exp(tau_rej/phi_a + 1/(2*phi_a^2))",
                "mu_b": "2*exp(1/(2*phi_b^2))",
                "mu_s": "exp(tau_rej/phi_s + 1/(2*phi_s^2))",
                "mu_m": "exp(tau_rej/phi_m + 1/(2*phi_m^2))",
                "tail_epsilons": [
                    "epsilon_a_tail = 2*d*(N-1)*exp(-18)",
                    "epsilon_b_tail = 2*d*hat_k*exp(-18)",
                    "epsilon_s_tail = 2*d*(n+ell)*exp(-18)",
                    "epsilon_m_tail = 2*d*exp(-18)",
                ],
                "euclidean_tail": "epsilon_2 <= 1.19^(d*(ell+n+1))*exp(d*(ell+n+1)*(1-1.19^2)/2)",
                "joint_response_success": "(1-epsilon_s_tail)*(1-epsilon_m_tail) - epsilon_2",
                "mu_oom": "mu_a*mu_b*mu_s*mu_m / ((1-epsilon_a_tail)*(1-epsilon_b_tail)*((1-epsilon_s_tail)*(1-epsilon_m_tail)-epsilon_2)*(1-epsilon_g_upper)*(1-epsilon_cmp_modelled))",
                "scale_note": "The communication and tail bounds follow the current response split: (z_s,z_key) has d*(n+ell) released coefficients at sigma_s, and z_eval has d released coefficients at sigma_m. The infinity checks are sequential conditional events, so their success probabilities multiply without an independence assumption; the joint Euclidean-tail loss is then subtracted. The product-check thresholds B_g_0 and B_g_1 are recorded per row. epsilon_g_upper is checked from the included product-threshold validation CSV. Within the independent-uniform coefficient model, epsilon_cmp_modelled is recomputed by exactly counting residues satisfying both compression predicates.",
                "epsilon_g_source": "data/product_tau_validation.csv, checked by scripts/validate_product_tau_inputs.py",
                "epsilon_cmp_model": "exact per-coefficient residue-intersection count, raised to hat_n*d under an independent-uniform coefficient assumption",
            },
        },
        "auxiliary_msis2_summary": [
            {
                "N": int(row["N"]),
                "rank_rows": row["checked_instances"]["outer_auxiliary_msis2"]["rank_rows"],
                "width_cols": row["checked_instances"]["outer_auxiliary_msis2"]["width_cols"],
                "beta_sis_2": row["checked_instances"]["outer_auxiliary_msis2"]["beta_sis_2"],
                "twelve_sigma_s": row["checked_instances"]["outer_auxiliary_msis2"]["twelve_sigma_s"],
                "twelve_sigma_m": row["checked_instances"]["outer_auxiliary_msis2"]["twelve_sigma_m"],
                "q_requirement": row["checked_instances"]["outer_auxiliary_msis2"]["q_requirement"],
                "q_over_requirement": float(RR(row["q"]) / RR(row["checked_instances"]["outer_auxiliary_msis2"]["q_requirement"])),
                "required_delta": row["outer_auxiliary_msis2_delta_required"],
                "q_requirement_pass": row["checks"]["q_exceeds_beta_sis_2_requirement"],
                "delta_pass": row["checks"]["outer_auxiliary_msis2_delta_at_or_below_target"],
                "mr09_pass": row["checks"]["outer_auxiliary_msis2_mr09_pass"],
            }
            for row in out_rows
        ],
        "rows": out_rows,
    }
    SECURITY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(out_rows), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
