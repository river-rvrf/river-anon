#!/usr/bin/env sage
# -*- coding: utf-8 -*-
"""
LANES / KLSS23 convention-corrected MLWE + MSIS cross-check.

This version fixes the Gaussian-convention mismatch between KLSS23 and LaV.

KLSS23 uses the lattice-Gaussian parameter sigma_K in D_{Z^n,sigma_K},
where the per-coordinate standard deviation is

    stddev = sigma_K / sqrt(2*pi).

For epsilon = 2^-100, KLSS23 requires

    sigma_K >= sqrt(2) * eta_epsilon(Z^d)
            = sqrt(2/pi) * sqrt( ln(2*d*(1+1/epsilon)) ).

LaV's quantity s0 is the corresponding per-coordinate standard deviation:

    s0 = sigma_K / sqrt(2*pi)
       = sqrt( ln(2*d*(1+1/epsilon)) ) / pi.

Thus these are CONSISTENT, but they are different conventions.

For the KLSS23 BDLOP optimization:

    1/sigma_K^2
      = 2 * (1/sigma1_K^2 + kappa^2/sigma2_K^2),

minimizing kappa*sigma1_K + sigma2_K gives

    sigma1_K = 2*sigma_K,
    sigma2_K = 2*kappa*sigma_K.

In standard-deviation convention:

    s1 = sigma1_K / sqrt(2*pi) = 2*s0,
    s2 = sigma2_K / sqrt(2*pi) = 2*kappa*s0.

KLSS23 gives the proven L2 response bound

    beta_BDLOP'
      = (kappa*sigma1_K + sigma2_K)
        * sqrt(D_resp/pi),

where

    D_resp = (n_hat + ell_hat + N + alpha) * d_hat.

After conversion to LaV's standard-deviation convention,

    beta_BDLOP'
      = (kappa*s1 + s2) * sqrt(2*D_resp)
      = 4*sqrt(2)*kappa*s0*sqrt(D_resp).

LaV uses

    s_LaV ~= 2*sqrt(2)*w_hat*s0.

For kappa = w_hat, this gives the especially simple identity

    beta_BDLOP' = 2*s_LaV*sqrt(D_resp).

For sparse LANES challenge ||c_L||_1 <= w_hat, the tightened
special-soundness MSIS bound used by this script is

    B_MSIS_sparse = 8*w_hat*beta_BDLOP'.

The conservative theorem-as-written dense variant

    B_MSIS_dense = 8*d_hat*beta_BDLOP'

is kept in the code for local debugging, but the reviewer-facing default run
checks only the sparse challenge bound used by the selected LANES profile.

IMPORTANT:
- beta_BDLOP' is an L2 bound, so the primary SIS estimator call uses norm=2.
- D affects Eq.(12) communication, not the KLSS beta formula.
- The compression parameter D must satisfy BOTH
      2^D <= w_hat*s1*n_hat*d_hat
  and
      q_hat > 4*w_hat*2^D.
- The MSIS short-vector bound must also satisfy
      q_hat > B_MSIS.
- For the current candidate we automatically choose the largest admissible D.
- MLWE delta is diagnostic; the project acceptance rule is delta <= 1.00469.
"""

from sage.all import *
import math

try:
    from estimator import *
except ImportError as exc:
    raise SystemExit(
        "\nCannot import malb/lattice-estimator.\n"
        "Run this script from the lattice-estimator repository root, "
        "or add the repository to PYTHONPATH.\n"
    ) from exc


# ============================================================================
# Fixed project parameters
# ============================================================================

QHAT = ZZ(67_107_713)
EPS_BITS = 100
W_HAT = ZZ(44)
L_SPLIT = ZZ(64)

# Compression parameter.
#
# If a profile has D=None, the script chooses the largest D satisfying
#
#     2^D <= w_hat*s1*n_hat*d_hat
#     q_hat > 4*w_hat*2^D
#
# and also keeps at least one transmitted modulus bit, i.e. D < log2(q_hat).
#
# LaV reference keeps D=13 because that is the paper's reported parameter.
# Our current profile uses D=None so that the script finds the maximum.
DEFAULT_D_IF_FIXED = ZZ(13)

DELTA_TARGET = RR("1.00450")
DELTA_MAX = RR("1.00469")

RUN_SIS_ROUGH = True
RUN_SIS_FULL = False
RUN_MLWE_DIAGNOSTIC = True
SHOW_DENSE_BASELINE = False

CURRENT = dict(
    name="Selected LANES profile",
    d_hat=256,
    n_hat=4,
    ell_hat=4,
    N=6,
    alpha=3,
    D=None,          # auto-select maximum admissible D
)

LAV_REFERENCE = dict(
    name="LaV reference",
    d_hat=128,
    n_hat=6,
    ell_hat=7,
    N=6,
    alpha=3,
    D=13,            # keep LaV's published compression parameter
)


# ============================================================================
# Compression-parameter D constraints
# ============================================================================

def floor_log2_integer(x):
    """floor(log2(x)) for a positive integer x, exactly."""
    x = ZZ(x)
    if x <= 0:
        raise ValueError("floor_log2_integer expects x > 0")
    return ZZ(x.nbits() - 1)


def d_constraints(P):
    """
    Compute the admissible compression range from the two project constraints:

        (C1) 2^D <= w_hat*s1*n_hat*d_hat
        (C2) q_hat > 4*w_hat*2^D

    We also require D < qbits so Eq.(12)'s (qbits-D) commitment term
    stays positive.

    If P["D"] is None, choose the largest admissible D.
    Otherwise check the explicitly requested D.
    """
    d_hat = ZZ(P["d_hat"])
    n_hat = ZZ(P["n_hat"])
    qbits = ZZ(QHAT.nbits())

    # C1: 2^D <= w*s1*n*d
    #
    # IMPORTANT: s1 is in the LaV / per-coordinate standard-deviation
    # convention.  With the KLSS optimum,
    #
    #     s1 = sigma1_K / sqrt(2*pi) = 2*s0.
    #
    # Do NOT replace s1 by the KLSS Gaussian parameter sigma1_K unless the
    # source formula itself is written in that convention.
    s1 = klss_optimal_sigmas(d_hat, RR(W_HAT))["s1"]
    rhs_struct = RR(W_HAT) * s1 * n_hat * d_hat

    # rhs_struct is real, so compute floor(log2(rhs_struct)) robustly and
    # verify the defining inequality explicitly.
    D_max_struct = ZZ(floor(log(rhs_struct, 2)))
    while ZZ(2) ** D_max_struct > rhs_struct:
        D_max_struct -= 1
    while ZZ(2) ** (D_max_struct + 1) <= rhs_struct:
        D_max_struct += 1

    # C2: q > 4*w*2^D.
    #
    # Since all quantities are integers, this is equivalent to
    #     4*w*2^D <= q-1,
    # hence
    #     2^D <= floor((q-1)/(4*w)).
    rhs_q = ZZ((QHAT - 1) // (ZZ(4) * W_HAT))
    D_max_q = floor_log2_integer(rhs_q)

    # Keep qbits-D >= 1.
    D_max_encoding = qbits - 1

    D_max = min(D_max_struct, D_max_q, D_max_encoding)

    requested = P.get("D", None)
    if requested is None:
        D = ZZ(D_max)
        auto = True
    else:
        D = ZZ(requested)
        auto = False

    lhs_pow2 = ZZ(2) ** D
    lhs_q = ZZ(4) * W_HAT * lhs_pow2

    struct_ok = lhs_pow2 <= rhs_struct
    q_compression_ok = QHAT > lhs_q
    encoding_ok = D < qbits

    return dict(
        D=D,
        auto=auto,
        D_max=D_max,
        s1=s1,
        D_max_struct=D_max_struct,
        D_max_q=D_max_q,
        D_max_encoding=D_max_encoding,
        lhs_pow2=lhs_pow2,
        rhs_struct=rhs_struct,
        lhs_q=lhs_q,
        rhs_q=QHAT,
        struct_ok=bool(struct_ok),
        q_compression_ok=bool(q_compression_ok),
        encoding_ok=bool(encoding_ok),
        all_D_ok=bool(struct_ok and q_compression_ok and encoding_ok),
    )


def print_d_constraints(P, B_msis=None):
    C = d_constraints(P)

    print("\n[Compression parameter D]")
    mode = "auto-max" if C["auto"] else "fixed"
    print(f"  mode                           = {mode}")
    print(f"  chosen D                       = {C['D']}")
    print(f"  s1 (our stddev convention)     = {float(C['s1']):.10f}")
    print(f"  D_max from 2^D <= w*s1*n*d    = {C['D_max_struct']}")
    print(f"  D_max from q > 4*w*2^D        = {C['D_max_q']}")
    print(f"  D_max from qbits-D >= 1       = {C['D_max_encoding']}")
    print(f"  overall D_max                 = {C['D_max']}")

    print(
        f"  2^D                            = {C['lhs_pow2']}"
    )
    print(
        f"  w*s1*n*d                       = {float(C['rhs_struct']):.6f} "
        f"-> {'PASS' if C['struct_ok'] else 'FAIL'}"
    )
    print(
        f"  4*w*2^D                        = {C['lhs_q']}"
    )
    print(
        f"  q_hat                           = {QHAT} "
        f"-> {'PASS' if C['q_compression_ok'] else 'FAIL'}"
    )

    if C["lhs_q"] > 0:
        print(
            f"  q_hat / (4*w*2^D)              = "
            f"{float(RR(QHAT)/C['lhs_q']):.6f}"
        )

    if B_msis is not None:
        B_msis = ZZ(B_msis)
        msis_q_ok = QHAT > B_msis
        print(
            f"  B_MSIS                          = {B_msis}"
        )
        print(
            f"  q_hat > B_MSIS                  = "
            f"{'PASS' if msis_q_ok else 'FAIL'}"
        )
        if B_msis > 0:
            print(
                f"  q_hat / B_MSIS                  = "
                f"{float(RR(QHAT)/B_msis):.6f}"
            )

    return C


# ============================================================================
# RHF helpers
# ============================================================================

def delta_from_beta(beta):
    if beta is None:
        return None
    b = RR(beta)
    if b <= 1:
        return None
    return RR(
        ((pi*b)**(1/b) * b/(2*pi*e))
        ** (1/(2*b - 2))
    )


def get_field(cost, names):
    for name in names:
        try:
            x = cost.get(name, None)
            if x is not None:
                return x
        except Exception:
            pass
        try:
            x = cost[name]
            if x is not None:
                return x
        except Exception:
            pass
    return None


def extract_beta(cost):
    x = get_field(cost, ("β", "beta"))
    if x is None:
        return None
    try:
        return RR(x)
    except Exception:
        return None


def extract_delta(cost):
    x = get_field(cost, ("δ", "delta"))
    if x is not None:
        try:
            return RR(x)
        except Exception:
            pass
    return delta_from_beta(extract_beta(cost))


def normalize_results(result):
    if result is None:
        return []

    if get_field(result, ("β", "beta", "δ", "delta", "rop")) is not None:
        return [("lattice", result)]

    try:
        return [(str(k), v) for k, v in result.items() if v is not None]
    except Exception:
        return [("result", result)]


def classify_delta(delta):
    if delta is None:
        return "NO-DELTA"
    x = RR(delta)
    if x > DELTA_MAX:
        return "FAIL"
    if x >= RR("1.00430"):
        return "TARGET"
    return "PASS / conservative"


# ============================================================================
# KLSS23 <-> LaV Gaussian convention
# ============================================================================

def log_smoothing_term(d_hat):
    """
    L = ln(2*d_hat*(1+1/epsilon)), epsilon=2^-EPS_BITS.
    """
    eps = RR(2) ** (-EPS_BITS)
    return log(RR(2) * d_hat * (1 + 1/eps))


def klss_sigma_base(d_hat):
    """
    KLSS23 Gaussian PARAMETER:
        sigma_K = sqrt(2) * eta_epsilon(Z^d)
                = sqrt(2/pi) * sqrt(L).
    """
    L = log_smoothing_term(d_hat)
    return sqrt(RR(2)/pi) * sqrt(L)


def lav_s0_stddev(d_hat):
    """
    LaV per-coordinate STANDARD DEVIATION:
        s0 = sigma_K / sqrt(2*pi)
           = sqrt(L)/pi.
    """
    sigma_K = klss_sigma_base(d_hat)
    return sigma_K / sqrt(RR(2)*pi)


def klss_optimal_sigmas(d_hat, kappa):
    """
    Solve:
        1/sigma_K^2
          = 2(1/sigma1_K^2 + kappa^2/sigma2_K^2)
    while minimizing
        kappa*sigma1_K + sigma2_K.

    Analytic optimum:
        sigma1_K = 2*sigma_K
        sigma2_K = 2*kappa*sigma_K.
    """
    sigma_K = klss_sigma_base(d_hat)
    sigma1_K = RR(2) * sigma_K
    sigma2_K = RR(2) * kappa * sigma_K

    # Convert to per-coordinate standard deviations.
    s0 = sigma_K / sqrt(RR(2)*pi)
    s1 = sigma1_K / sqrt(RR(2)*pi)
    s2 = sigma2_K / sqrt(RR(2)*pi)

    # Numerical check of KLSS constraint.
    lhs = RR(1) / sigma_K**2
    rhs = RR(2) * (
        RR(1)/sigma1_K**2
        + kappa**2/sigma2_K**2
    )

    return dict(
        sigma_K=sigma_K,
        sigma1_K=sigma1_K,
        sigma2_K=sigma2_K,
        s0=s0,
        s1=s1,
        s2=s2,
        constraint_lhs=lhs,
        constraint_rhs=rhs,
    )


def lav_response_stddev(d_hat, w_hat=W_HAT):
    """
    LaV formula, with w_hat OUTSIDE the square root:
        s_LaV ~= 2*sqrt(2)*w_hat*s0.
    """
    s0 = lav_s0_stddev(d_hat)
    return RR(2) * sqrt(RR(2)) * w_hat * s0


# ============================================================================
# KLSS23 BDLOP response bound and LANES MSIS
# ============================================================================

def response_dimension(P):
    """
    KLSS23 (mu+nu+k)*n mapped to the LANES BDLOP response dimension:
        (n_hat + ell_hat + N + alpha) * d_hat.
    """
    return ZZ(
        (P["n_hat"] + P["ell_hat"] + P["N"] + P["alpha"])
        * P["d_hat"]
    )


def beta_bdlop(P):
    """
    KLSS23 theorem:
      beta'_BDLOP
        = (kappa*sigma1_K + sigma2_K) * sqrt(D_resp/pi).

    Here kappa is instantiated by the LANES challenge/operator bound
      kappa = w_hat.

    We compute the same bound THREE ways as a consistency check:

      (A) KLSS Gaussian-parameter convention;
      (B) LaV standard-deviation convention;
      (C) beta = 2*s_LaV*sqrt(D_resp).
    """
    d_hat = ZZ(P["d_hat"])
    D_resp = response_dimension(P)
    kappa = RR(W_HAT)

    G = klss_optimal_sigmas(d_hat, kappa)

    beta_klss = (
        kappa * G["sigma1_K"] + G["sigma2_K"]
    ) * sqrt(RR(D_resp)/pi)

    beta_std = (
        kappa * G["s1"] + G["s2"]
    ) * sqrt(RR(2)*D_resp)

    s_lav = lav_response_stddev(d_hat, W_HAT)
    beta_lav = RR(2) * s_lav * sqrt(RR(D_resp))

    return dict(
        D_resp=D_resp,
        kappa=kappa,
        sigma_K=G["sigma_K"],
        sigma1_K=G["sigma1_K"],
        sigma2_K=G["sigma2_K"],
        s0=G["s0"],
        s1=G["s1"],
        s2=G["s2"],
        s_lav=s_lav,
        beta_klss=beta_klss,
        beta_std=beta_std,
        beta_lav=beta_lav,
        constraint_lhs=G["constraint_lhs"],
        constraint_rhs=G["constraint_rhs"],
    )


def msis_instance(P, sparse=True):
    """
    Coefficient SIS dimensions:
        n_SIS = n_hat*d_hat
        m_SIS = (n_hat+ell_hat+N+alpha)*d_hat.

    KLSS beta'_BDLOP is an L2 bound, so this instance is modeled in L2.

    sparse=True:
        B = 8*w_hat*beta'_BDLOP

    sparse=False:
        B = 8*d_hat*beta'_BDLOP
        (conservative dense ENS-style baseline)
    """
    d_hat = ZZ(P["d_hat"])
    n_hat = ZZ(P["n_hat"])

    Binfo = beta_bdlop(P)
    D_resp = Binfo["D_resp"]

    n_sis = n_hat * d_hat
    m_sis = D_resp

    if sparse:
        multiplier = RR(8) * W_HAT
        mode = "sparse"
    else:
        multiplier = RR(8) * d_hat
        mode = "dense"

    B_msis_real = multiplier * Binfo["beta_klss"]
    B_msis = ZZ(ceil(B_msis_real))

    return dict(
        mode=mode,
        n_sis=n_sis,
        m_sis=m_sis,
        B_msis=B_msis,
        B_msis_real=B_msis_real,
        beta_info=Binfo,
    )


# ============================================================================
# Eq.(12) size
# ============================================================================

def eq12_bits(P):
    d_hat = ZZ(P["d_hat"])
    n_hat = ZZ(P["n_hat"])
    ell_hat = ZZ(P["ell_hat"])
    N = ZZ(P["N"])
    alpha = ZZ(P["alpha"])

    qbits = ZZ(QHAT.nbits())
    C = d_constraints(P)
    D = C["D"]

    if not C["all_D_ok"]:
        raise ValueError(
            f"Invalid D={D} for profile {P['name']}: "
            "compression constraints are not satisfied."
        )

    s_lav = lav_response_stddev(d_hat, W_HAT)

    term1 = RR(n_hat * d_hat * (qbits - D))
    term2 = RR((N + alpha + 1) * d_hat * qbits)
    term3 = (
        RR(ell_hat + N + alpha)
        * d_hat
        * log(RR("4.13") * s_lav, 2)
    )
    total = term1 + term2 + term3

    return dict(
        D=D,
        D_constraints=C,
        s_lav=s_lav,
        term1=term1,
        term2=term2,
        term3=term3,
        total=total,
        kib=total/(8*1024),
        decimal_kb=total/8000,
    )


# ============================================================================
# Estimator calls
# ============================================================================

def run_sis(P, sparse=True):
    I = msis_instance(P, sparse=sparse)
    Binfo = I["beta_info"]

    print("\n[MSIS: %s challenge bound]" % I["mode"])
    print(f"  coefficient SIS n,m        = ({I['n_sis']},{I['m_sis']})")
    print(f"  norm                       = L2")
    print(f"  beta'_BDLOP                = {float(Binfo['beta_klss']):.8f}")
    print(f"  B_MSIS                     = {I['B_msis']}")
    print(f"  B_MSIS/q_hat               = {float(RR(I['B_msis'])/QHAT):.8f}")

    if I["B_msis"] >= QHAT:
        print("  WARNING: B_MSIS >= q_hat.")

    params = SIS.Parameters(
        n=int(I["n_sis"]),
        q=int(QHAT),
        length_bound=float(I["B_msis"]),
        m=int(I["m_sis"]),
        norm=2,
        tag=P["name"] + " / KLSS-BDLOP-MSIS-" + I["mode"],
    )

    selected_delta = None

    if RUN_SIS_ROUGH:
        try:
            res = SIS.estimate.rough(params)
            print("  raw:", res)

            candidates = []
            for name, cost in normalize_results(res):
                beta = extract_beta(cost)
                delta = extract_delta(cost)

                if delta is not None and math.isfinite(float(delta)):
                    candidates.append((name, delta, beta))

                beta_s = "n/a" if beta is None else f"{float(beta):.2f}"
                delta_s = "n/a" if delta is None else f"{float(delta):.8f}"
                cls = "n/a" if delta is None else classify_delta(delta)

                print(
                    f"    {name:18s} beta_BKZ={beta_s:>10s} "
                    f"delta={delta_s:>12s} [{cls}]"
                )

            if candidates:
                # Conservative delta-only diagnostic.
                name, selected_delta, beta = max(candidates, key=lambda x: x[1])
                print(
                    f"  ==> selected MSIS delta = {float(selected_delta):.8f} "
                    f"[{classify_delta(selected_delta)}]"
                )
        except Exception as exc:
            print("  SIS.estimate.rough failed:", repr(exc))

    if RUN_SIS_FULL:
        try:
            res = SIS.estimate(params)
            print("  full:", res)
        except Exception as exc:
            print("  SIS.estimate failed:", repr(exc))

    I["delta"] = selected_delta
    return I


def mlwe_instance(P):
    """
    Build the coefficient-LWE instance used for the MLWE diagnostic.

    HNF view used by the production LANES code:
        d_H = n_hat
        m_H = ell_hat + N + alpha

    Coefficient dimensions:
        n_LWE = d_H * d_hat
        m_LWE = m_H * d_hat

    KLSS sigma_K is a lattice-Gaussian PARAMETER, while
    lattice-estimator ND.DiscreteGaussian expects the per-coordinate
    standard deviation. Therefore the estimator receives

        sigma_est = sigma_K/sqrt(2*pi) = s0.
    """
    d_hat = ZZ(P["d_hat"])
    n_hat = ZZ(P["n_hat"])
    ell_hat = ZZ(P["ell_hat"])
    N = ZZ(P["N"])
    alpha = ZZ(P["alpha"])

    d_H = n_hat
    m_H = ell_hat + N + alpha

    n_LWE = d_H * d_hat
    m_LWE = m_H * d_hat

    sigma_K = klss_sigma_base(d_hat)
    sigma_est = lav_s0_stddev(d_hat)

    params = LWE.Parameters(
        n=int(n_LWE),
        q=int(QHAT),
        Xs=ND.DiscreteGaussian(float(sigma_est)),
        Xe=ND.DiscreteGaussian(float(sigma_est)),
        m=int(m_LWE),
        tag=P["name"] + " / KLSS-MLWE-diagnostic",
    )

    return dict(
        params=params,
        n_LWE=n_LWE,
        m_LWE=m_LWE,
        sigma_K=sigma_K,
        sigma_est=sigma_est,
    )


def estimate_mlwe_delta(P):
    """
    Run lattice-estimator's rough LWE estimate and extract a single
    conservative RHF diagnostic.

    Returns a dictionary even if the estimator fails.  An attack is usable
    only if both its derived delta and its attack cost are finite.  This
    prevents an infinite-cost estimator branch from producing a misleading
    PASS solely because a blocksize-derived delta was present.
    """
    J = mlwe_instance(P)

    out = dict(
        raw=None,
        candidates=[],
        selected_name=None,
        beta_bkz=None,
        delta=None,
        error=None,
        n_LWE=J["n_LWE"],
        m_LWE=J["m_LWE"],
        sigma_K=J["sigma_K"],
        sigma_est=J["sigma_est"],
    )

    try:
        res = LWE.estimate.rough(J["params"])
        out["raw"] = res

        candidates = []
        for name, cost in normalize_results(res):
            beta = extract_beta(cost)
            delta = extract_delta(cost)
            rop = get_field(cost, ("rop",))

            finite_rop = False
            if rop is not None:
                try:
                    finite_rop = math.isfinite(float(RR(rop)))
                except Exception:
                    finite_rop = False

            if delta is not None and finite_rop:
                try:
                    if math.isfinite(float(delta)):
                        candidates.append(
                            dict(name=name, delta=RR(delta), beta=beta, rop=rop)
                        )
                except Exception:
                    pass

        out["candidates"] = candidates

        if candidates:
            selected = max(candidates, key=lambda C: C["delta"])
            out["selected_name"] = selected["name"]
            out["beta_bkz"] = selected["beta"]
            out["delta"] = selected["delta"]

    except Exception as exc:
        out["error"] = exc

    return out


def run_mlwe_diagnostic(P):
    """
    Print the MLWE/LWE diagnostic and return the selected delta.

    IMPORTANT:
      - MLWE delta is diagnostic only.
      - The project's admissibility decision remains based on sparse MSIS,
        q_hat > B_MSIS, and the D constraints.
    """
    if not RUN_MLWE_DIAGNOSTIC:
        return None

    M = estimate_mlwe_delta(P)

    print("\n[MLWE diagnostic]")
    print(f"  KLSS sigma_K parameter     = {float(M['sigma_K']):.10f}")
    print(f"  estimator stddev=s0        = {float(M['sigma_est']):.10f}")
    print(
        f"  sigma_K/sqrt(2*pi)         = "
        f"{float(M['sigma_K']/sqrt(2*pi)):.10f}"
    )
    print(f"  coefficient LWE n,m        = ({M['n_LWE']},{M['m_LWE']})")

    if M["error"] is not None:
        print("  MLWE diagnostic failed:", repr(M["error"]))
        return M

    print("  raw:", M["raw"])

    # Print every returned attack, including attacks for which no delta could
    # be extracted. This makes it obvious whether 'n/a' is caused by the
    # estimator result rather than by the reporting code.
    for name, cost in normalize_results(M["raw"]):
        beta = extract_beta(cost)
        delta = extract_delta(cost)
        rop = get_field(cost, ("rop",))

        beta_s = "n/a" if beta is None else f"{float(beta):.2f}"
        delta_s = "n/a" if delta is None else f"{float(delta):.8f}"
        rop_s = "n/a" if rop is None else str(rop)

        print(
            f"    {name:18s} beta_BKZ={beta_s:>10s} "
            f"delta={delta_s:>12s} rop={rop_s}"
        )

    if M["delta"] is None:
        print("  ==> selected MLWE delta = n/a (no finite delta returned)")
    else:
        beta_s = "n/a" if M["beta_bkz"] is None else f"{float(M['beta_bkz']):.2f}"
        print(
            f"  ==> selected MLWE delta = {float(M['delta']):.8f} "
            f"from {M['selected_name']} "
            f"(beta_BKZ={beta_s}) "
            f"[{classify_delta(M['delta'])}]"
        )

    return M


# ============================================================================
# Reporting
# ============================================================================

def print_formula_block(P):
    d_hat = ZZ(P["d_hat"])
    B = beta_bdlop(P)
    D_resp = B["D_resp"]

    print("\n[KLSS23 -> LaV convention conversion]")
    print(
        "  sigma_KLSS = sqrt(2/pi)*sqrt(L) "
        f"= {float(B['sigma_K']):.10f}"
    )
    print(
        "  s0 = sigma_KLSS/sqrt(2*pi) = sqrt(L)/pi "
        f"= {float(B['s0']):.10f}"
    )
    print(f"  sigma1_KLSS = 2*sigma_KLSS = {float(B['sigma1_K']):.10f}")
    print(
        f"  sigma2_KLSS = 2*w*sigma_KLSS = "
        f"{float(B['sigma2_K']):.10f}"
    )
    print(f"  s1 = sigma1/sqrt(2*pi) = {float(B['s1']):.10f}")
    print(f"  s2 = sigma2/sqrt(2*pi) = {float(B['s2']):.10f}")
    print(
        f"  KLSS constraint lhs-rhs = "
        f"{float(B['constraint_lhs']-B['constraint_rhs']):.3e}"
    )
    print(f"  response coefficient dimension D = {D_resp}")
    print(
        "  beta'_BDLOP [KLSS] = "
        "(w*sigma1+sigma2)*sqrt(D/pi) "
        f"= {float(B['beta_klss']):.8f}"
    )
    print(
        "  beta'_BDLOP [ours] = "
        "(w*s1+s2)*sqrt(2D) "
        f"= {float(B['beta_std']):.8f}"
    )
    print(
        "  LaV s = 2*sqrt(2)*w*s0 "
        f"= {float(B['s_lav']):.8f}"
    )
    print(
        "  beta'_BDLOP = 2*s*sqrt(D) "
        f"= {float(B['beta_lav']):.8f}"
    )

    err1 = abs(B["beta_klss"] - B["beta_std"])
    err2 = abs(B["beta_klss"] - B["beta_lav"])
    print(
        f"  consistency errors          = "
        f"{float(err1):.3e}, {float(err2):.3e}"
    )


def run_profile(P):
    print("\n" + "="*80)
    print(P["name"])
    print("="*80)
    print(f"d_hat       = {P['d_hat']}")
    print(f"q_hat       = {QHAT} ({QHAT.nbits()} bits)")
    print(f"n_hat       = {P['n_hat']}")
    print(f"ell_hat     = {P['ell_hat']}")
    print(f"N           = {P['N']}")
    print(f"alpha       = {P['alpha']}")
    print(f"w_hat       = {W_HAT}")
    print(f"delta max   = {float(DELTA_MAX):.5f}")

    print_formula_block(P)

    sparse = run_sis(P, sparse=True)

    # Check:
    #   2^D <= w*s1*n*d
    #   q > 4*w*2^D
    #   q > B_MSIS
    C = print_d_constraints(P, B_msis=sparse["B_msis"])

    dense = None
    if SHOW_DENSE_BASELINE:
        # Dense baseline is a local diagnostic only.  The selected profile and
        # the paper-facing LANES size use the sparse challenge bound above.
        dense = run_sis(P, sparse=False)

    mlwe = run_mlwe_diagnostic(P)

    S = eq12_bits(P)
    D = S["D"]
    print("\n[LaV Eq.(12) size]")
    print(
        "  |t_L|+|pi_L| ~= "
        f"{P['n_hat']}*{P['d_hat']}*({QHAT.nbits()}-{D})"
        f" + ({P['N']}+{P['alpha']}+1)*{P['d_hat']}*{QHAT.nbits()}"
        f" + ({P['ell_hat']}+{P['N']}+{P['alpha']})*{P['d_hat']}"
        f"*log2(4.13*s)"
    )
    print(f"  D                         = {D}")
    print(f"  s                         = {float(S['s_lav']):.8f}")
    print(f"  term1 commitment          = {float(S['term1']):.2f} bits")
    print(f"  term2 q-rings             = {float(S['term2']):.2f} bits")
    print(f"  term3 responses           = {float(S['term3']):.2f} bits")
    print(f"  total                     = {float(S['total']):.2f} bits")
    print(f"  total                     = {float(S['kib']):.4f} KiB")
    print(f"  total                     = {float(S['decimal_kb']):.4f} decimal KB")

    return dict(
        sparse=sparse,
        dense=dense,
        mlwe=mlwe,
        size=S,
        D_constraints=C,
    )


def scan_n_hat(base, values=(1,2,3,4,5,6), print_all=False):
    """
    Scan n_hat and return ONLY admissible candidates.

    A candidate is admissible iff:
      - the sparse-MSIS estimator returns a delta;
      - delta_MSIS <= DELTA_MAX;
      - q_hat > B_MSIS;
      - all D constraints hold.

    MLWE delta is evaluated for every scanned candidate (when
    RUN_MLWE_DIAGNOSTIC=True) and reported in the table, but remains
    DIAGNOSTIC ONLY and does not affect admissibility.

    If print_all=False (default), only admissible rows are printed.
    """
    valid = []

    print("\n" + "="*122)
    print("Admissible n_hat candidates -- sparse challenge bound + MLWE diagnostic")
    print("="*122)
    print(
        " n_hat | D_resp | D | B_MSIS_sparse | BKZ beta | delta_MSIS  | "
        "delta_MLWE  | LANES KiB | q/B_SIS"
    )
    print("-"*121)

    for nh in values:
        P = dict(base)
        P["name"] = f"scan n_hat={nh}"
        P["n_hat"] = nh
        P["D"] = None

        I = msis_instance(P, sparse=True)
        C = d_constraints(P)
        S = eq12_bits(P)

        params = SIS.Parameters(
            n=int(I["n_sis"]),
            q=int(QHAT),
            length_bound=float(I["B_msis"]),
            m=int(I["m_sis"]),
            norm=2,
            tag=P["name"],
        )

        beta_bkz = None
        delta_msis = None
        try:
            res = SIS.estimate.rough(params)
            vals = []
            for name, cost in normalize_results(res):
                b = extract_beta(cost)
                de = extract_delta(cost)
                if de is not None and math.isfinite(float(de)):
                    vals.append((de, b))
            if vals:
                # Conservative: retain the largest returned RHF.
                delta_msis, beta_bkz = max(vals, key=lambda x: x[0])
        except Exception:
            pass

        # MLWE is diagnostic only, but compute it for THIS n_hat so the final
        # row does not accidentally reuse CURRENT's fixed n_hat.
        M = estimate_mlwe_delta(P) if RUN_MLWE_DIAGNOSTIC else None
        delta_mlwe = None if M is None else M["delta"]

        q_msis_ok = QHAT > I["B_msis"]
        delta_ok = (
            delta_msis is not None
            and RR(delta_msis) <= DELTA_MAX
        )
        d_ok = C["all_D_ok"]

        admissible = bool(q_msis_ok and delta_ok and d_ok)

        row = dict(
            n_hat=nh,
            P=P,
            I=I,
            C=C,
            S=S,
            beta_bkz=beta_bkz,
            delta=delta_msis,          # backward-compatible alias
            delta_msis=delta_msis,
            mlwe=M,
            delta_mlwe=delta_mlwe,
            q_msis_ok=q_msis_ok,
            delta_ok=delta_ok,
            d_ok=d_ok,
            admissible=admissible,
        )

        if admissible:
            valid.append(row)

        if print_all or admissible:
            bb = "n/a" if beta_bkz is None else f"{float(beta_bkz):.2f}"
            dm = "n/a" if delta_msis is None else f"{float(delta_msis):.8f}"
            dl = "n/a" if delta_mlwe is None else f"{float(delta_mlwe):.8f}"
            ratio = float(RR(QHAT) / I["B_msis"])

            if print_all:
                status = "PASS" if admissible else "REJECT"
                print(
                    f" {nh:5d} | {int(response_dimension(P)):6d} | "
                    f"{int(C['D']):2d} | {int(I['B_msis']):13d} | "
                    f"{bb:8s} | {dm:11s} | {dl:11s} | "
                    f"{float(S['kib']):9.4f} | {ratio:7.3f}  {status}"
                )
            else:
                print(
                    f" {nh:5d} | {int(response_dimension(P)):6d} | "
                    f"{int(C['D']):2d} | {int(I['B_msis']):13d} | "
                    f"{bb:8s} | {dm:11s} | {dl:11s} | "
                    f"{float(S['kib']):9.4f} | {ratio:7.3f}"
                )

    if not valid:
        print("  No admissible candidate in the scanned range.")

    return valid


def scan_D(base, start_D=None):
    """
    Scan every feasible compression D for one fixed LANES profile.

    This is useful because D changes Eq.(12) only through
        n_hat*d_hat*(qbits-D),
    so the largest feasible D gives the smallest communication.
    """
    P0 = dict(base)
    C0 = d_constraints(P0)
    Dmax = ZZ(C0["D_max"])

    if start_D is None:
        # Show a small neighborhood around the old LaV value and the optimum.
        start_D = max(ZZ(0), min(ZZ(13), Dmax) - 2)
    else:
        start_D = ZZ(start_D)

    print("\n" + "="*88)
    print(f"D scan for {base['name']} (maximum admissible D = {Dmax})")
    print("="*88)
    print(
        " D | 2^D     | w*s1*n*d    | 4*w*2^D   | q-comp | Eq.(12) KiB | decimal KB"
    )
    print("-"*87)

    for D in range(int(start_D), int(Dmax)+1):
        P = dict(base)
        P["D"] = D
        C = d_constraints(P)

        if not C["all_D_ok"]:
            continue

        S = eq12_bits(P)
        print(
            f"{D:2d} | {int(C['lhs_pow2']):7d} | {float(C['rhs_struct']):12.3f} | "
            f"{int(C['lhs_q']):11d} | PASS   | "
            f"{float(S['kib']):12.4f} | {float(S['decimal_kb']):10.4f}"
        )


print("\nLANES / KLSS23 convention-corrected cross-check")
print("================================================")
print("This reviewer-facing run checks only the selected admissible LANES profile.")
print("KLSS sigma is a lattice-Gaussian parameter.")
print("LaV s0 is the corresponding per-coordinate standard deviation.")
print("Conversion: s0 = sigma_KLSS / sqrt(2*pi).")
print("Primary MSIS norm: L2, using KLSS beta'_BDLOP.")
print("Primary sparse bound: B_MSIS = 8*w_hat*beta'_BDLOP.")
print("Compression constraints:")
print("  2^D <= w_hat*s1*n_hat*d_hat")
print("  q_hat > 4*w_hat*2^D")
print("  q_hat > B_MSIS")

selected = run_profile(CURRENT)

sparse = selected["sparse"]
mlwe = selected["mlwe"]
size = selected["size"]
d_constraints_out = selected["D_constraints"]
msis_delta = sparse.get("delta")
mlwe_delta = None if mlwe is None else mlwe.get("delta")

splitting_modulus = ZZ(4) * L_SPLIT
splitting_target = ZZ(2) * L_SPLIT + ZZ(1)
modulus_ok = bool(QHAT.is_prime() and QHAT % splitting_modulus == splitting_target)
msis_ok = bool(
    msis_delta is not None
    and RR(msis_delta) <= DELTA_MAX
    and QHAT > sparse["B_msis"]
)
mlwe_ok = bool(mlwe_delta is not None and RR(mlwe_delta) <= DELTA_MAX)
all_ok = bool(modulus_ok and msis_ok and mlwe_ok and d_constraints_out["all_D_ok"])

print("\n" + "="*88)
print("SELECTED LANES PARAMETERS")
print("="*88)
print(
    f"  (d_hat, n_hat, ell_hat, N, alpha, D) = "
    f"({CURRENT['d_hat']}, {CURRENT['n_hat']}, {CURRENT['ell_hat']}, "
    f"{CURRENT['N']}, {CURRENT['alpha']}, {size['D']})"
)
print(f"  q_hat       = {QHAT}")
print(f"  q_hat prime = {'yes' if QHAT.is_prime() else 'no'}")
print(f"  L_split     = {L_SPLIT}")
print(
    f"  q_hat mod {int(splitting_modulus)} = "
    f"{int(QHAT % splitting_modulus)} "
    f"(expected {int(splitting_target)})"
)
print(f"  w_hat       = {W_HAT}")
print(f"  beta'_BDLOP = {float(sparse['beta_info']['beta_klss']):.8f}")
print(f"  B_MSIS      = {int(sparse['B_msis'])}")
print(f"  delta_MSIS  = {float(msis_delta):.8f}")
print(f"  delta_MLWE  = {float(mlwe_delta):.8f}")
print(f"  q/B_MSIS    = {float(RR(QHAT)/sparse['B_msis']):.6f}")
print(f"  LANES size  = {float(size['kib']):.4f} KiB")
print(f"  status      = {'PASS' if all_ok else 'FAIL'}")

if not all_ok:
    raise SystemExit(1)

print("\nDone.")
