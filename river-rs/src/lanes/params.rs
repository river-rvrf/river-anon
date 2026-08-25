//! LANES parameters and samplers — port of `river-py/lanes_params.py`.
//!
//! **The parameters are the paper's.** The Gaussian widths follow from a
//! closed form with no free constant:
//!
//! ```text
//! (ceil(log2 q~), n~, l~, D) = (26, 4, 4, 17),  q~ = 67107713
//! d~ = 256,  l = 64,  N_ex = 6,  alpha = 3,  w_hat = 44
//!
//! eps = 2^-100
//! s_0 = sqrt(ln(2 d~ (1 + 1/eps))) / pi   ~ 2.7668
//! s_1 = 2 s_0                              ~ 5.5336    commitment randomness
//! s_2 = 2 w_hat s_0                        ~ 243.4775  proof mask
//! s   = 2 sqrt(2) w_hat s_0                ~ 344.3291  response
//! ```
//!
//! and every published output follows: `beta' = 2 s sqrt(4352) = 45430.6`,
//! `B_MSIS = 8 w_hat beta' = 15991562`, `q~/B_MSIS = 4.2`,
//! `delta_MSIS = 1.0037`, and `D = 17` as the largest exponent with
//! `2^D <= w_hat s_1 n~ d~` and `q~ > 4 w_hat 2^D`.
//!
//! Two derived identities are load-bearing:
//! `s^2 = s_2^2 + w_hat^2 s_1^2` (so `s` is already the worst-case-`l1`
//! response width), and `sigma_MLWE = s_0` exactly on the unrounded widths —
//! the widths are chosen so the `[KLSS23]` hint reduction lands back on the
//! smoothing parameter for `eps = 2^-100`, which is why
//! `s_2 = w_hat s_1`. The stored rational widths are independently rounded
//! to multiples of `2^-20`, so the derived rational `sigma_MLWE` differs
//! from `s_0` only by that documented rounding.
//!
//! The paper's `s` is the standard deviation and its `sigma` the `[KLSS23]`
//! Gaussian parameter, `sigma = s sqrt(2 pi)`. This module works in standard
//! deviations throughout, as `crate::sample` does.
//!
//! The standard-deviation convention gives `delta_MLWE = 1.003996`,
//! reproducing the paper's printed `1.0040`.  The tested backend is exposed
//! as `lanes-experimental` because its concrete compression/recovery and
//! wire-format completion is implementation-defined and this artifact does
//! not supply a reduction for that exact composition.  See
//! `crate::exact::lanes_unavailable_reason`.
//!
//! Both widths are independently rounded to exact rationals with
//! denominator `2^20`, exactly as the finite-precision sampler consumes
//! them.
//!
//! ## The verifier's bounds on `z`
//!
//! Derived from the distribution of the *response* `z = y + c r`, not of
//! the mask alone.  The Euclidean one is a quadratic-form bound rather than
//! a chi-square: conditioned on `c`, negacyclic multiplication correlates
//! the coefficients of `c r`, so `||z||^2` is a quadratic form whose tail
//! sees the whole spectrum.  See `river-py/lanes_params.py` for the
//! derivation.

use super::ring::{self as lr, DTILDE, LSPLIT, QTILDE};
use crate::exact::ExactParams;
use crate::sample::{gaussian_int_ctx, uniform_int, GaussCtx, Xof, GAUSSIAN_TAILCUT};

// ---- dimensions ----------------------------------------------------------
//
// Read from `ExactParams` so the wrapper and prover share one source of truth.

/// Rows of `B_0`, the length of `t_0`, and the MLWE secret rank.
pub const N_TILDE: usize = ExactParams::N_TILDE;
/// Width of the shared random tail.
pub const ELL_TILDE: usize = ExactParams::ELL_TILDE;
/// Message ring elements.
pub const N_EX: usize = ExactParams::N_EX;
/// `g`, and the two product-proof commitments; the paper's `alpha`.
pub const AUX: usize = ExactParams::AUX_SLOTS;
/// The structural role of each rank, from the one place that mapping
/// lives — [`crate::exact::rank_roles`].
///
/// Every rank constant below is a field of this, and none of them is
/// written out again.  Both spellings of the response rank agree at
/// `(n~, l~) = (4, 4)`, which is exactly why this file carried the wrong
/// one — `KAPPA - N_TILDE` — for as long as it did, and why a numeric
/// test could not tell.  Deriving removes the choice rather than
/// asserting it away.
const ROLES: crate::exact::RankRoles = crate::exact::rank_roles(N_TILDE, ELL_TILDE, N_EX, AUX);

/// Randomness rank, `n~ + l~ + N_ex + alpha`.
pub const KAPPA: usize = ROLES.kappa;
/// `B_0`'s identity rank, and so the rows of `t_0`.
pub const IDENTITY_RANK: usize = ROLES.identity_rank;
/// The shared random tail every `b_i` draws from.
pub const TAIL_RANK: usize = ROLES.tail_rank;
/// Masked response rank after Bai--Galbraith compression.
///
/// The commitment remains rank [`KAPPA`]; the response omits `B_0`'s
/// **identity** block, which is `l~` wide — never `n~`.  The two letters
/// are easy to read the wrong way round here.
pub const RESPONSE_RANK: usize = ROLES.response_rank;

/// Challenge weight: `||c||_1 = w_hat`.
pub const W_HAT: usize = ExactParams::W_HAT;
/// Partition stride — the residue classes the support is spread over.
pub const DELTA: usize = lr::SUBDEG;
/// Nonzero coefficients per residue class.
pub const W_TILDE: usize = W_HAT / DELTA;

/// Commitment compression.
pub const D_DROP: u32 = ExactParams::D_DROP;
/// Maximum degree in the exact relation.
pub const ALPHA: usize = AUX;

// ---- commitment / response recovery ------------------------------------

/// Scale of the transmitted high part of `t_0`.
pub const T0_SCALE: u64 = 1 << D_DROP;
/// Largest absolute centred low part.
pub const T0_LOW_BOUND: u64 = T0_SCALE / 2;

const fn t0_round_raw(value: u64) -> (u64, i64) {
    let mut low = (value % T0_SCALE) as i64;
    if low > T0_LOW_BOUND as i64 {
        low -= T0_SCALE as i64;
    }
    (
        ((value as i128 - low as i128) / T0_SCALE as i128) as u64,
        low,
    )
}

/// `value = high * 2^D + low`, with `low` centred.
pub fn t0_power2round(value: u64) -> Option<(u64, i64)> {
    (value < QTILDE).then(|| t0_round_raw(value))
}

/// Exclusive bound for a compressed `t_0` coefficient.
pub const T0_HIGH_MODULUS: u64 = t0_round_raw(QTILDE - 1).0 + 1;

// ---- Gaussian widths -----------------------------------------------------

/// `-log2(eps)`: the smoothing-parameter target.  **Paper**.
pub const SMOOTHING_EPS_EXP: u32 = 100;

/// Denominator of the pinned rationals; the same one
/// [`crate::sample::rational_sigma`] uses.
pub const SIGMA_DEN: u64 = 1 << 20;

/// `round(s_1 · 2^20)`, before reduction, where `s_1 = 2 s_0`.
///
/// Public so the tests can state where [`SIGMA_R`] comes from rather than
/// asserting a pair of magic numbers.  The `s_0` behind it needs a
/// logarithm and a square root at 60 digits, which is a
/// parameter-selection computation rather than a protocol one; it lives in
/// `river-py/lanes_params.py` and is pinned here and by the KAT.
/// `the_widths_reproduce_the_papers_printed_digits` checks the result
/// against what the paper prints, which is the property that matters.
pub const SIGMA_R_NUM_UNREDUCED: u64 = 5_802_378;

/// `round(s_2 · 2^20)`, before reduction, where `s_2 = 2 w_hat s_0`.
pub const SIGMA_Y_NUM_UNREDUCED: u64 = 255_304_631;

/// `sigma_r` as `(num, den)`, **in lowest terms**.
///
/// `5802378 / 2^20` reduced by 2.  The reference's `Fraction` normalises,
/// so matching it here keeps the two sides carrying literally the same pair
/// rather than two spellings of one rational.  The sampler does not care —
/// `bound = tailcut·num/den` and the acceptance exponent `z²den²/(2num²)`
/// are both invariant under reduction — but a constant that reads
/// differently in two implementations is one more thing to have to check.
pub const SIGMA_R: (u64, u64) = (2_901_189, 524_288);

/// `sigma_y`, independently rounded at denominator `2^20`, in lowest terms.
/// `255304631` is odd, so this one does not reduce.
pub const SIGMA_Y: (u64, u64) = (255_304_631, 1_048_576);

/// Actual infinity support of one commitment-randomness coefficient.
pub const R_INF_SUPPORT: u64 = GAUSSIAN_TAILCUT * SIGMA_R.0 / SIGMA_R.1;

/// Worst coefficient of `c * (t_0,low - r_identity)`.
pub const RECOVERY_ERROR_BOUND: u64 = W_HAT as u64 * (T0_LOW_BOUND + R_INF_SUPPORT);

const fn recovery_buckets() -> u64 {
    let mut buckets = 1;
    while RECOVERY_ERROR_BOUND < QTILDE / (2 * buckets) {
        buckets *= 2;
    }
    buckets
}

/// Equal torus intervals used for recovery of the first transcript message.
/// The largest power of two whose smallest interval exceeds
/// [`RECOVERY_ERROR_BOUND`], so the cyclic carry is always ternary.
pub const RECOVERY_BUCKETS: u64 = recovery_buckets();
pub const RECOVERY_BITS: u32 = RECOVERY_BUCKETS.trailing_zeros();

const _: () = {
    assert!(RECOVERY_ERROR_BOUND < QTILDE / RECOVERY_BUCKETS);
    assert!(RECOVERY_ERROR_BOUND >= QTILDE / (2 * RECOVERY_BUCKETS));
};

// ---- the verifier's bounds on the response -------------------------------

/// Coefficients in one transmitted response: `RESPONSE_RANK · d~`.
pub const N_Z: usize = RESPONSE_RANK * DTILDE;

/// `||z||_2^2` bound: the paper's `beta' = 2 s sqrt(N_z)` rule, at the
/// transmitted rank.
///
/// the paper supplies the Euclidean bound, and it is a flat "two
/// standard deviations per coordinate" rule.  It is the bound the security
/// claim rests on: `B_MSIS = 8 w_hat beta'` is what the extractor gets
/// from two accepting forks, so a verifier enforcing anything looser would
/// not support the published `B_MSIS`.
///
/// One dimension mismatch, named rather than absorbed.  The paper bounds
/// the *full* rank-`kappa` opening, `N_z = kappa d~ = 4352`.  This
/// implementation applies Bai–Galbraith compression and transmits only the
/// `kappa - l~ = 13` non-identity elements, so the verifier has
/// `RESPONSE_RANK · d~ = 3328` coefficients in front of it and the
/// per-coordinate rule is applied to the coordinates that exist.  That is
/// **stricter** than the paper's own bound, so the published `B_MSIS`
/// remains an upper bound on what an extractor obtains here.
///
/// **Derived, not copied.**  It is evaluated on the *rounded* widths
/// through the identity `s^2 = s_2^2 + w_hat^2 s_1^2`, which makes it an
/// exact rational computation both implementations can do — the reference
/// does the same, deliberately, so that this integer (a verifier decision)
/// is bit-identical rather than agreeing to within a rounding.  It used to
/// be a shared literal for exactly the opposite reason: the old
/// Laurent–Massart route squared a `10^30` numerator on the way to a
/// square root and did not fit in `u128`.
pub const Z_NORM2_BOUND: i128 = z_norm2_bound();

/// `ceil(4 (sigma_y^2 + w_hat^2 sigma_r^2) · N_Z)`, in exact integers.
const fn z_norm2_bound() -> i128 {
    let (yn, yd) = (SIGMA_Y.0 as u128, SIGMA_Y.1 as u128);
    let (rn, rd) = (SIGMA_R.0 as u128, SIGMA_R.1 as u128);
    // `s^2 = sigma_y^2 + w_hat^2 sigma_r^2` over the common denominator.
    let den = yd * yd * rd * rd;
    let num = yn * yn * rd * rd + (W_HAT as u128) * (W_HAT as u128) * rn * rn * yd * yd;
    // times `(2)^2 N_Z`, then ceil.
    let scaled = 4 * (N_Z as u128) * num;
    (scaled.div_ceil(den)) as i128
}

/// `statistical_tailcut(N_Z)`: the union bound over all `N_Z` released
/// coefficients at `2^-128`, in standard deviations of a `z` coefficient —
/// not of a `y` coefficient, and not the `6 sigma_y` that was here before.
///
/// The tail arithmetic that produces this lives in `river-py/dgs.py`, in
/// exact `Decimal`; it is a parameter-selection quantity rather than a
/// protocol one, so it is carried here and pinned by the KAT.
pub const Z_TAILCUT: i64 = 14;

/// `ceil(t · sqrt(Var[z]))`, **derived** rather than copied.
///
/// `Var[z] = sigma_y^2 + w_hat sigma_r^2` is an exact rational, so the
/// bound is the least integer `B` with `B^2 >= t^2 Var[z]` — computed in
/// `u128` at compile time, with no floating point anywhere.
///
/// The reference used to compute `t · isqrt(floor(Var))`, which floors the
/// square root *before* multiplying and so loses up to `t` units rather
/// than one: 8988 where 14 standard deviations is 8994.5995, i.e. a bound
/// of **13.9897 sd** under a comment claiming 14.  Deriving it here rather
/// than copying the constant is what the paper asks for —
/// "derived from this variance", not "maintained as independent hard-coded
/// constants" — and it is why the error surfaced.
pub const Z_INF_BOUND: i64 = ceil_sqrt_var(Z_TAILCUT as u128);

/// `Var[z]` as an exact rational `(num, den)`.
///
/// `sigma_y^2 + w_hat sigma_r^2` over the common denominator
/// `(den_y · den_r)^2`, which both fit in `u128` at these widths.
const fn var_z_rational() -> (u128, u128) {
    let (yn, yd) = (SIGMA_Y.0 as u128, SIGMA_Y.1 as u128);
    let (rn, rd) = (SIGMA_R.0 as u128, SIGMA_R.1 as u128);
    // sigma_y^2 = yn^2 / yd^2, sigma_r^2 = rn^2 / rd^2
    let den = yd * yd * rd * rd;
    let num = yn * yn * rd * rd + (W_HAT as u128) * rn * rn * yd * yd;
    (num, den)
}

/// Least integer `B` with `B^2 · den >= t^2 · num`.
///
/// Divide before bisecting: `cand^2 · den` with a bisection starting at
/// `2^32` leaves `u128`, while `cand^2` alone does not.  So take the
/// integer square root of `target / den` first — which can only undershoot,
/// since integer division rounds down — and then walk up on the exact
/// comparison.  At these widths the walk is a single step or none.
const fn ceil_sqrt_var(t: u128) -> i64 {
    let (num, den) = var_z_rational();
    let target = t * t * num;
    let floor_q = target / den;

    let mut b: u128 = 0;
    let mut step: u128 = 1 << 32;
    while step > 0 {
        let cand = b + step;
        if cand <= u128::MAX / cand && cand * cand <= floor_q {
            b = cand;
        }
        step /= 2;
    }
    while b * b * den < target {
        b += 1;
    }
    b as i64
}

/// Total scalars the commitment carries: `N_ex · l`.
pub const fn message_slot_count() -> usize {
    N_EX * LSPLIT
}

// ---- samplers ------------------------------------------------------------

pub fn sample_uniform_poly(xof: &mut Xof) -> lr::CoeffPoly {
    let v: Vec<u64> = (0..DTILDE).map(|_| uniform_int(xof, QTILDE)).collect();
    lr::CoeffPoly::new(&v).expect("uniform_int returns canonical residues")
}

/// One Gaussian polynomial at `sigma`.
///
/// The width context is built **once** and reused across the `d~` draws.
/// [`crate::sample::gaussian_int`] constructs a [`GaussCtx`] per call —
/// including a bit-by-bit reciprocal — so a `KAPPA` vector rebuilt it 2944
/// times.  Same draws, same XOF consumption, same bytes: `gaussian_int` is
/// exactly `gaussian_int_ctx` with the setup inlined.
pub fn sample_gaussian_poly(xof: &mut Xof, sigma: (u64, u64)) -> lr::CoeffPoly {
    let ctx = GaussCtx::new(sigma.0, sigma.1, GAUSSIAN_TAILCUT);
    sample_gaussian_poly_ctx(xof, &ctx)
}

/// [`sample_gaussian_poly`] against a prepared width.
pub fn sample_gaussian_poly_ctx(xof: &mut Xof, ctx: &GaussCtx) -> lr::CoeffPoly {
    let v: Vec<i64> = (0..DTILDE).map(|_| gaussian_int_ctx(xof, ctx)).collect();
    lr::CoeffPoly::from_centered(&v).expect("d~ coefficients")
}

/// `len` Gaussian polynomials, on one width context.
pub fn sample_gaussian_vec(xof: &mut Xof, sigma: (u64, u64), len: usize) -> Vec<lr::CoeffPoly> {
    let ctx = GaussCtx::new(sigma.0, sigma.1, GAUSSIAN_TAILCUT);
    (0..len)
        .map(|_| sample_gaussian_poly_ctx(xof, &ctx))
        .collect()
}

/// LANES challenge: low-weight ternary with **partitioned support**.
///
/// The challenge space of `[ENS20]`.  For each of the `DELTA = 4` residue
/// classes mod 4, a partial Fisher–Yates places exactly `W_TILDE = 11`
/// coefficients in `{-1, +1}`, giving total weight `DELTA · W_TILDE = 44`.
///
/// The partition is not cosmetic: spreading the weight evenly across
/// residue classes controls the challenge's behaviour in each NTT block,
/// which a plain weight-44 ternary polynomial would not.  Section 2.3
/// labels this the *OOM* challenge space, but the OOM layer actually uses
/// `C^d_{w,gamma}`; it belongs here.
pub fn sample_challenge(xof: &mut Xof) -> lr::CoeffPoly {
    let mut poly = vec![0u64; DTILDE];
    for i in 0..DELTA {
        for j in (LSPLIT - W_TILDE)..LSPLIT {
            let x = uniform_int(xof, j as u64 + 1) as usize; // x in [0, j]
            poly[j * DELTA + i] = poly[x * DELTA + i];
            poly[x * DELTA + i] = if xof.bit() != 0 { 1 } else { QTILDE - 1 };
        }
    }
    lr::CoeffPoly::new(&poly).expect("ternary coefficients are canonical")
}

/// `||c||_1` on the centred representation.
pub fn challenge_l1_norm(poly: &lr::CoeffPoly) -> i64 {
    poly.centered().into_iter().map(|c| c.abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{Part, Xof};

    /// Not "these equal 7 and 6", but "these *are* the exact layer's".
    ///
    /// A test on the numeric values passes just as happily when both
    /// modules carry the same wrong literal, which is the failure mode
    /// deriving them is meant to remove.
    #[test]
    fn every_shared_dimension_is_the_exact_layers() {
        // A test on the numeric values passes just as happily when both
        // modules carry the same wrong literal, which is the failure mode
        // deriving them is meant to remove.
        assert_eq!(N_TILDE, ExactParams::N_TILDE);
        assert_eq!(ELL_TILDE, ExactParams::ELL_TILDE);
        assert_eq!(N_EX, ExactParams::N_EX);
        assert_eq!(AUX, ExactParams::AUX_SLOTS);
        assert_eq!(W_HAT, ExactParams::W_HAT);
        assert_eq!(D_DROP, ExactParams::D_DROP);
        assert_eq!(DTILDE, ExactParams::D_TILDE);
        assert_eq!(LSPLIT, ExactParams::L_SPLIT);
        assert_eq!(QTILDE, ExactParams::Q_TILDE);

        // and what follows from them
        assert_eq!(KAPPA, N_TILDE + ELL_TILDE + N_EX + AUX);
        assert_eq!(DELTA * W_TILDE, W_HAT);
        assert_eq!(N_Z, RESPONSE_RANK * DTILDE);
        assert_eq!(ALPHA, AUX);
    }

    /// The transmitted response rank drops `B_0`'s **identity** block.
    ///
    /// `RESPONSE_RANK` is written `KAPPA - N_TILDE` here, which is the
    /// wrong formula and gives the right answer: `n~ = l~ = 4` at this
    /// profile, so the two coincide at 13.  The labels are easy to reverse
    /// for exactly this reason, so the assertion is against
    /// `exact::rank_roles`, which is the single place the mapping lives
    /// and which a test can drive with unequal ranks.
    #[test]
    fn the_response_rank_is_kappa_minus_the_identity_rank() {
        // `RESPONSE_RANK` is now `ROLES.response_rank`, so asserting it
        // equals `roles.response_rank` proves nothing.  What is worth
        // pinning is that the *source* is the right one, and that the two
        // spellings genuinely differ off this profile — otherwise the
        // derivation is a no-op dressed up as a fix.
        assert_eq!(RESPONSE_RANK, ROLES.response_rank);
        assert_eq!(ROLES.identity_rank, ELL_TILDE);
        assert_eq!(ROLES.tail_rank, N_TILDE);
        assert_eq!(IDENTITY_RANK, ELL_TILDE);
        assert_eq!(TAIL_RANK, N_TILDE);

        // When the two ranks differ the readings give 16 and 17, and
        // `kappa - n~` — what this file used to compute — was the wrong
        // one.  Driving `rank_roles` with unequal ranks is the only way
        // to see that, because `n~ = l~ = 4` here.
        let unequal = crate::exact::rank_roles(7, 8, N_EX, AUX);
        assert_eq!(unequal.kappa, 24);
        assert_eq!(unequal.response_rank, 16, "kappa - l~");
        assert_ne!(unequal.response_rank, unequal.kappa - 7, "not kappa - n~");
        // and the paper's own MLWE dimensions read the same way round
        assert_eq!(unequal.lwe_secret_rank, 7);
        assert_eq!(unequal.lwe_sample_rank, 8 + N_EX + AUX);
    }

    /// The widths reproduce the digits the paper prints, and the
    /// closed form behind them.
    ///
    /// `s_0 = sqrt(ln(2 d~ (1 + 1/eps))) / pi` needs a log and a square
    /// root at 60 digits, which is a parameter-selection computation and
    /// lives in `river-py`.  What is checkable here is the result: `f64`
    /// reproduces `s_0` to about `10^-15`, far inside the four printed
    /// places, so the pinned rationals can be checked against the paper
    /// directly rather than against the reference's arithmetic.
    #[test]
    fn the_widths_reproduce_the_papers_printed_digits() {
        // Both are reductions of their independently rounded `2^20`
        // numerators.
        assert_eq!(SIGMA_R.0 * (SIGMA_DEN / SIGMA_R.1), SIGMA_R_NUM_UNREDUCED);
        assert_eq!(SIGMA_Y.0 * (SIGMA_DEN / SIGMA_Y.1), SIGMA_Y_NUM_UNREDUCED);

        let eps_inv = (2f64).powi(SMOOTHING_EPS_EXP as i32);
        let s0 = (2.0 * DTILDE as f64 * (1.0 + eps_inv)).ln().sqrt() / std::f64::consts::PI;
        let s1 = 2.0 * s0;
        let s2 = 2.0 * W_HAT as f64 * s0;

        // the paper's printed values
        assert!((s0 - 2.7668).abs() < 5e-5, "s_0 = {s0}");
        assert!((s1 - 5.5336).abs() < 5e-5, "s_1 = {s1}");
        assert!((s2 - 243.4775).abs() < 5e-5, "s_2 = {s2}");

        // ...and the pinned rationals are those, rounded once at 2^-20
        let sr = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        let sy = SIGMA_Y.0 as f64 / SIGMA_Y.1 as f64;
        assert!((sr - s1).abs() < 1e-6, "sigma_r {sr} vs s_1 {s1}");
        assert!((sy - s2).abs() < 1e-6, "sigma_y {sy} vs s_2 {s2}");

        // The relation that makes `sigma_MLWE = s_0`: `s_2 = w_hat s_1`.
        // The previous candidate deliberately broke it (`556 != 44·13`);
        // the paper's widths restore it, and it is why they are these.
        assert!((sy - W_HAT as f64 * sr).abs() < 1e-4);

        // The `[KLSS23]` reduction returns the smoothing parameter.
        let sigma_mlwe =
            (1.0 / (2.0 * (1.0 / (sr * sr) + (W_HAT * W_HAT) as f64 / (sy * sy)))).sqrt();
        assert!((sigma_mlwe - s0).abs() < 1e-6, "sigma_MLWE = {sigma_mlwe}");
    }

    /// The published security chain, re-derived from the widths.
    #[test]
    fn the_published_security_chain_reproduces() {
        let eps_inv = (2f64).powi(SMOOTHING_EPS_EXP as i32);
        let s0 = (2.0 * DTILDE as f64 * (1.0 + eps_inv)).ln().sqrt() / std::f64::consts::PI;
        let s = 2.0 * (2f64).sqrt() * W_HAT as f64 * s0;
        assert!((s - 344.3291).abs() < 5e-5, "s = {s}");

        // `N_z` is the *full* rank-kappa opening, not the transmitted one.
        let n_z_paper = KAPPA * DTILDE;
        assert_eq!(n_z_paper, 4352);
        let beta_prime = 2.0 * s * (n_z_paper as f64).sqrt();
        assert!((beta_prime - 45430.6).abs() < 0.05, "beta' = {beta_prime}");

        let b_msis = 8.0 * W_HAT as f64 * beta_prime;
        assert!((b_msis - 15_991_562.0).abs() < 1.0, "B_MSIS = {b_msis}");
        assert!((QTILDE as f64 / b_msis - 4.2).abs() < 0.05);

        // delta_MSIS, closed form at n = n~ d~
        let log2 = |x: f64| x.log2();
        let delta = (2f64)
            .powf(log2(b_msis).powi(2) / (4.0 * (N_TILDE * DTILDE) as f64 * log2(QTILDE as f64)));
        assert!((delta - 1.0037).abs() < 5e-5, "delta_MSIS = {delta}");

        // `D = 17`: the largest exponent satisfying both inequalities.
        let sr = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        let limit = W_HAT as f64 * sr * (N_TILDE * DTILDE) as f64;
        assert!((1u64 << D_DROP) as f64 <= limit);
        assert!(QTILDE > 4 * W_HAT as u64 * (1u64 << D_DROP));
        assert!(
            (1u64 << (D_DROP + 1)) as f64 > limit,
            "D = {D_DROP} is maximal"
        );
    }

    #[test]
    fn the_challenge_is_ternary_with_partitioned_support() {
        let mut x = Xof::new(b"lanes-test", &[Part::Bytes(b"challenge")]);
        for _ in 0..50 {
            let c = sample_challenge(&mut x);
            let cent = c.centered();
            assert!(cent.iter().all(|&v| (-1..=1).contains(&v)), "ternary");
            assert_eq!(cent.iter().filter(|&&v| v != 0).count(), W_HAT, "weight");
            for i in 0..DELTA {
                let per_class = (0..LSPLIT).filter(|&j| cent[j * DELTA + i] != 0).count();
                assert_eq!(per_class, W_TILDE, "class {i}");
            }
            assert_eq!(challenge_l1_norm(&c), W_HAT as i64);
        }
    }

    #[test]
    fn the_gaussian_widths_sample_at_the_right_scale() {
        let mut x = Xof::new(b"lanes-test", &[Part::Bytes(b"gauss")]);
        let ctx = GaussCtx::new(SIGMA_R.0, SIGMA_R.1, GAUSSIAN_TAILCUT);
        let samples: Vec<i64> = (0..4000).map(|_| gaussian_int_ctx(&mut x, &ctx)).collect();
        let rms = (samples.iter().map(|&s| (s * s) as f64).sum::<f64>() / 4000.0).sqrt();
        let want = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        assert!((rms - want).abs() / want < 0.08, "{rms} vs {want}");
    }

    /// Both response bounds re-derive from the widths, independently.
    ///
    /// The reference reaches them through `Decimal`; this reaches them
    /// through `f64` from `sigma_r` and `sigma_y`.  Two routes to one
    /// number is a check — one route and a copy is not, which is how the
    /// infinity bound stayed wrong in both for as long as it did.
    #[test]
    fn the_euclidean_bound_re_derives() {
        let sr = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        let sy = SIGMA_Y.0 as f64 / SIGMA_Y.1 as f64;
        let (sr2, sy2) = (sr * sr, sy * sy);
        let w = W_HAT as f64;
        let n = N_Z as f64;

        // Sigma is kappa blocks of sigma_y^2 I + sigma_r^2 M M^T, bounded
        // over every challenge by |c-hat|^2 <= ||c||_1^2 = w^2 and
        // sum |c-hat|^4 <= max |c-hat|^2 · sum |c-hat|^2.
        let trace = n * (sy2 + w * sr2);
        let frob_sq = n * (sy2 * sy2 + 2.0 * w * sy2 * sr2 + w * w * w * sr2 * sr2);
        let op = sy2 + w * w * sr2;

        let t = 128.0 * std::f64::consts::LN_2;
        let bound = trace + 2.0 * (frob_sq * t).sqrt() + 2.0 * op * t;

        // the *enforced* bound is the paper's
        // `(2 s)^2 N_Z` rule, and this Laurent–Massart figure is the
        // honest-response **requirement** it has to clear: the smallest
        // bound an honest `z` can be held to at `2^-128`.  The margin is
        // measured, not assumed, because `lanes::proof::prove` aborts on
        // this bound and an inverted inequality would mean an honest
        // prover that never terminates.
        assert_eq!(N_Z, 3328, "N_Z follows the paper's ranks");
        assert!(
            (bound.ceil() as i128) < Z_NORM2_BOUND,
            "honest requirement {bound} exceeds the enforced {Z_NORM2_BOUND}"
        );
        let margin = Z_NORM2_BOUND as f64 / bound;
        assert!((5.0..6.0).contains(&margin), "margin {margin}");

        // ...and the enforced bound really is the paper's rule, evaluated
        // on the rounded widths through `s^2 = s_2^2 + w_hat^2 s_1^2`.
        let s_sq = sy2 + w * w * sr2;
        // `f64` against an exact `u128` ceiling: the shipped integer is
        // the ceiling of this, so it is at most one unit above and the
        // `f64` route carries its own ~1e-16 relative error on a 1.6e9
        // quantity.  A unit out of 1.6e9 never moves an accept; what this
        // checks is that the *rule* is the paper's, and the exact form is
        // what both implementations agree on bit for bit.
        let rule = 4.0 * s_sq * N_Z as f64;
        assert!(
            (rule - Z_NORM2_BOUND as f64).abs() < 2.0,
            "{rule} vs {Z_NORM2_BOUND}"
        );
        assert!(
            Z_NORM2_BOUND as f64 >= rule,
            "the shipped value is the ceiling"
        );

        // The *shape* of the derivation is still checkable, and it is the
        // half that was wrong twice: the quadratic-form bound, not
        // the chi-square form used by the paper's response model.
        //
        // Which of the two is larger is **not** a fixed fact.  The
        // quadratic-form correction grows like `sqrt(n)` while the trace
        // grows like `n`, so the relative correction shrinks as `n` does
        // the opposite: at a smaller `N_Z` the bound sat 6.5%
        // *above* the chi-square figure, and at `3328` it sits 2.6%
        // below.  Asserting "above" would be pinning an accident of the
        // old dimension; what is invariant is that the two agree to a few
        // percent and that the correction terms are the `sqrt(n)` and
        // constant ones.
        let iid = 1.4759352630939924 * trace;
        let correction = 2.0 * (frob_sq * t).sqrt() + 2.0 * op * t;
        assert!(correction > 0.0 && correction < 0.6 * trace);
        assert!(
            (bound / iid - 1.0).abs() < 0.1,
            "quadratic-form {bound} against chi-square {iid}"
        );
    }

    /// `Z_TAILCUT` is the least integer the union bound admits.
    ///
    /// The requirement is `statistical_tailcut`'s:
    /// `N_Z · Pr[|X| > t] <= 2^-128`.  Settling it needs **two** bounds
    /// pointing opposite ways, which is the correction to the first
    /// version of this test.
    ///
    /// *Sufficiency at 14* needs an **upper** bound on the tail.
    /// `Pr[|X| > t] <= 2 e^{-t^2/2}` is loose against the Mills-ratio
    /// continued fraction the reference evaluates in `Decimal`, and still
    /// clears the budget — so 14 works however the tail is computed.
    ///
    /// *Minimality* needs a **lower** bound at 13.  The first version
    /// reused the upper bound here and asserted it failed, which shows
    /// only that the bound is inconclusive at 13, not that 13 is
    /// inadmissible — the true tail sits below it.  The standard Mills
    /// lower bound `Pr[X > t] > phi(t) · t/(t^2 + 1)` settles it: at 13
    /// it already exceeds the budget by three orders of magnitude, so no
    /// sharper tail estimate can rescue 13.
    #[test]
    fn the_tail_cut_is_the_least_integer_the_union_bound_admits() {
        let budget = 2f64.powi(-128);
        let n = N_Z as f64;
        assert!(budget > 0.0 && budget.is_normal(), "budget underflowed");

        // sufficiency: the loose upper bound already clears it at t
        let t = Z_TAILCUT as f64;
        let upper = 2.0 * (-0.5 * t * t).exp();
        assert!(
            n * upper <= budget,
            "t = {t} does not clear the budget: {} > {budget}",
            n * upper
        );

        // minimality: the Mills lower bound exceeds it at t - 1
        let s = t - 1.0;
        let phi = (-0.5 * s * s).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let lower = 2.0 * phi * s / (s * s + 1.0);
        assert!(lower.is_normal(), "the lower bound underflowed to {lower}");
        assert!(
            n * lower > budget,
            "t = {s} is not ruled out: {} <= {budget}",
            n * lower
        );
    }

    /// The infinity bound is `ceil(t sqrt(Var))`, derived here.
    ///
    /// It was `t · isqrt(floor(Var))` in the reference, which floors the
    /// root before multiplying: 8988, i.e. 13.9897 sd under a comment
    /// claiming 14.  Deriving it is what surfaced that, and is what the
    /// paper asks for.
    #[test]
    fn the_infinity_bound_is_derived_and_is_really_t_sigma() {
        assert_eq!(Z_INF_BOUND, 3448);

        // `B^2 >= t^2 Var` and `(B-1)^2 < t^2 Var`, exactly
        let (num, den) = var_z_rational();
        let target = (Z_TAILCUT as u128) * (Z_TAILCUT as u128) * num;
        let b = Z_INF_BOUND as u128;
        assert!(b * b * den >= target, "not an upper bound");
        assert!((b - 1) * (b - 1) * den < target, "not the least one");

        // and the old form really was short
        let sr = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        let sy = SIGMA_Y.0 as f64 / SIGMA_Y.1 as f64;
        let sd = (sy * sy + W_HAT as f64 * sr * sr).sqrt();
        // `ceil(14 sd)` is 3448 where `14 sd` is 3447.20, so the shipped
        // bound is 14.0032 sd.  Above 14, never below — which is the
        // direction that matters, and the direction the reference's old
        // `t · isqrt(floor(Var))` got wrong (13.9897 sd under a comment
        // claiming 14).
        let sd_ratio = Z_INF_BOUND as f64 / sd;
        assert!((14.0..14.01).contains(&sd_ratio), "{sd_ratio}");
    }

    /// `Var[z]` as a rational agrees with the widths it is built from.
    #[test]
    fn var_z_rational_is_sigma_y_squared_plus_w_hat_sigma_r_squared() {
        let (num, den) = var_z_rational();
        let sr = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        let sy = SIGMA_Y.0 as f64 / SIGMA_Y.1 as f64;
        let want = sy * sy + W_HAT as f64 * sr * sr;
        let got = num as f64 / den as f64;
        assert!((got - want).abs() / want < 1e-12, "{got} vs {want}");
        assert!((got - 60628.57991315564).abs() < 1e-6, "{got}");
    }

    /// The Euclidean bound must exceed the expected norm, or every honest
    /// proof fails.
    ///
    /// The response bounds leave real margin at the paper's widths.
    ///
    /// Both bounds are checked: the Euclidean one against the expected
    /// `||z||^2`, and the infinity one as a per-coefficient statement that
    /// does not depend on `N_Z`.
    #[test]
    fn the_response_bounds_leave_margin_at_the_papers_widths() {
        let sr = SIGMA_R.0 as f64 / SIGMA_R.1 as f64;
        let sy = SIGMA_Y.0 as f64 / SIGMA_Y.1 as f64;
        let var_z = sy * sy + W_HAT as f64 * sr * sr;
        let expected = N_Z as f64 * var_z;

        let slack = Z_NORM2_BOUND as f64 / expected;
        assert!((7.0..9.0).contains(&slack), "live slack {slack}");

        assert!(Z_INF_BOUND as f64 > 6.0 * var_z.sqrt(), "not 6 sigma");
        assert_eq!(Z_TAILCUT, 14);
        assert!(Z_INF_BOUND as f64 > 13.99 * var_z.sqrt());
    }

    /// The commitment-recovery accounting at the selected `D = 17`.
    #[test]
    fn recovery_carries_cover_the_combined_perturbation() {
        assert_eq!(D_DROP, 17, "D is the paper's");
        assert_eq!(T0_SCALE, 1 << D_DROP);
        // `t0_power2round` rounds to the *centred* low part, so the top
        // high value can be one past `floor((q~-1)/2^D)` — which is the
        // whole reason `T0_HIGH_MODULUS` is computed from the function
        // rather than written as a quotient.
        assert_eq!(
            T0_HIGH_MODULUS,
            t0_power2round(QTILDE - 1).unwrap().0 + 1,
            "the compressed domain must come from the rounding itself"
        );
        // The centred low part can push the top high value one past the
        // plain quotient, which is exactly why this is `>`.
        const {
            assert!(T0_HIGH_MODULUS > (QTILDE - 1) / T0_SCALE);
        }
        // `RECOVERY_BUCKETS` is a power of two by construction, whatever
        // the error bound turns out to be.
        assert!(RECOVERY_BUCKETS.is_power_of_two());
        assert_eq!(1u64 << RECOVERY_BITS, RECOVERY_BUCKETS);

        for value in [
            0,
            T0_LOW_BOUND - 1,
            T0_LOW_BOUND,
            T0_LOW_BOUND + 1,
            QTILDE - 1,
        ] {
            let (high, low) = t0_power2round(value).unwrap();
            assert_eq!(high as i128 * T0_SCALE as i128 + low as i128, value as i128);
            assert!(low > -(T0_LOW_BOUND as i64) && low <= T0_LOW_BOUND as i64);
        }
        assert_eq!(t0_power2round(T0_LOW_BOUND), Some((0, T0_LOW_BOUND as i64)));
        assert_eq!(
            t0_power2round(T0_LOW_BOUND + 1),
            Some((1, 1 - T0_LOW_BOUND as i64))
        );
        assert_eq!(t0_power2round(QTILDE), None);
    }
}
