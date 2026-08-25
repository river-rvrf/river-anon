//! The exact layer `Pi_ex` — port of `river-py/exact.py`.
//!
//! `Pi_ex` is a two-stage commit-and-prove system for
//!
//! ```text
//! R^_ex = { ((W, z_eval, x), (e_eval, y_eval)) :
//!             W  = Com_{ck_ex}(e_eval, y_eval)
//!           & z_eval = x * e_eval + y_eval        (over Z)
//!           & coeffs(e_eval + B_e) in [0, q_0 - 1]^d }
//! ```
//!
//! The two-stage split matters: `Com` runs *before* the statement exists,
//! so `W` can be folded into the OOM Fiat–Shamir context and is fixed
//! before the challenge `x`.
//!
//! ## What is modelled here
//!
//! The paper treats `Pi_ex` as a black box and instantiates it with LANES
//! (ENS20 / LANES+ / Hint-MLWE).  It publishes the LANES dimensions,
//! sampler widths, response-norm model, compression exponent, and a fixed
//! `|pi_ex| = 13.5` KB entropy estimate.  This module implements those
//! parameters together with a concrete codec and recovery-hint completion:
//!
//! * the exact-backend parameters
//!   `(n~, l~, d~, w_hat, D, N_ex, q~) = (4, 4, 256, 44, 17, 6, 67107713)`
//!   with splitting factor `l = 64`;
//! * the adjusted radix-3 reconstruction vector `(1, 3, 9, 17)` with
//!   digits in `{0,1,2}`, which covers `[0, 60]` exactly
//!   (max `2(1+3+9+17) = 60`);
//! * the six semantic exact messages, in the paper's order
//!   `(y_eval, e_eval, d_0, d_1, d_2, d_3)`, one per LANES message block:
//!   each block carries the element's `d = 32` coefficients in its first
//!   32 slots and is zero-padded to `l = 64`.  The old `6 d == N_ex l`
//!   identity is gone — `192 != 384`, and that is intentional;
//! * a BDLOP-style commitment over `R_q~` with those ranks
//!   (`kappa = n~ + l~ + N_ex + alpha = 17`, transmitted response rank 13).
//!
//! and then plugs in a **proof backend** behind a small interface.  The
//! backend shipped here, [`OpeningBackend`], enforces every clause of
//! `R^_ex` — including the integer equality — but `sigma_ex` *is* the
//! opening, so it is not zero knowledge.
//!
//! `OpeningBackend` is a mock, not a stand-in for LANES.  Substituting a
//! real prover is not confined to `prove`/`verify`: it changes the
//! commitment randomness distribution, the transcript, the encoding and
//! the size.  Read `|pi_ex|` from this module as the cost of *this*
//! opening, not as a concrete encoding of the paper's entropy estimate.
//!
//! [`crate::lanes::backend::LanesBackend`] is the candidate LANES
//! instantiation. It runs under the `lanes-experimental` name; the
//! production alias is reserved because this artifact does not supply a
//! reduction for its concrete composition — see [`lanes_unavailable_reason`].
//!
//! ## The modulus condition and centred representation
//!
//! LANES has a single modulus, so it can only check the link modulo `q~`.
//! That pins an integer only when no accepted response can wrap.  The
//! difference of two accepted error responses is at most
//! `12 sigma_m = 12 phi_m eta_m`, so
//!
//! ```text
//! q~ > 24 phi_m eta_m
//! ```
//!
//! makes `z_eval - x e_eval` have a unique centred lift.  With
//! `eta_m = w gamma B_e sqrt(d)` this is one number for all five profiles:
//! `66730968.02` against the selected `q~ = 67107713`, a margin of
//! `376744.98` — about 0.56%.
//!
//! The construction explicitly translates the canonical rounding error in
//! `[0, q_0-1] = [0, 60]` to its centred representation in `[-B_e,B_e]`
//! before applying norm bounds.  This implementation follows that
//! translation in [`crate::ring::to_centered_error`].  The distinction is
//! arithmetically load-bearing: using 60 as a magnitude bound would double
//! the requirement to `133461936.03`, which the selected modulus would not
//! satisfy.
//!
//! Because 0.56% is inside what a float `sqrt` and a multiplication chain
//! can move, [`ExactParams::q_tilde_clears`] decides the condition over
//! the integers as `q~^2 > (24 phi_m w gamma B_e)^2 d` rather than in
//! floating point.
//!
//! [`check_relation`] still verifies the link over `Z`.  Under the new
//! modulus that is no longer load-bearing for this backend — it is
//! redundant with the commitment — but it is the clause `R^_ex` actually
//! states, and a backend that proves it only modulo `q~` should have to
//! say so.

use crate::codec::{Coder, Field, FieldValue, Layout, Result as CodecResult};
use crate::lanes::ring::{self as lanes_ring, NttPoly as LanesNtt};
use crate::params::{is_prime, RiVeRParams};
use crate::ring::{Poly, PolyMat, PolyVec, Ring};
use crate::sample::{sam_mat, uniform_beta_vec, Xof};

/// Reconstruction weights for the rounding-error range.  Digits in
/// `{0,1,2}` give exactly `[0, 60] = [0, q_0 - 1]`.
pub const RADIX_WEIGHTS: [i64; 4] = [1, 3, 9, 17];

/// Digit alphabet `{0, 1, 2}`.
pub const RADIX_DIGITS: u64 = 3;

/// `sum_j g_j`: the shift between `{0,1,2}` and `{-1,0,1}` digits.
pub const WEIGHT_SUM: i64 = 30;

/// LANES-side parameters, fixed for every RiVeR profile.
///
/// paper, Appendix "Detailed Parameter Setting":
/// `(n~, l~, d~, w_hat, D) = (4, 4, 256, 44, 17)` with splitting factor
/// `l = 64`, `N_ex = 6` and `q~ = 67107713`.  All **Paper**.
///
/// **Opaque.**  Every field was public while `rt`, the commitment key and
/// both layouts are cached from them, so a dimension or the modulus could
/// be moved after the fact and leave the cache describing a different
/// scheme.  `check()` could not catch that either, because it validated
/// relations *between* the fields rather than the fields against the
/// constants they are supposed to be.  Read access is unrestricted; there
/// is nothing secret in a parameter set.
#[derive(Clone)]
pub struct ExactParams {
    par: RiVeRParams,
    d_tilde: usize,
    l_split: usize,
    q_tilde: u64,
    n_tilde: usize,
    ell_tilde: usize,
    d_drop: u32,
    w_hat: usize,
    n_ex: usize,
    d: usize,
    slots: usize,
    t0_rows: usize,
    aux_slots: usize,
    kappa: usize,
    rt: Ring,
}

impl ExactParams {
    /// The outer profile.
    pub fn par(&self) -> &RiVeRParams {
        &self.par
    }
    /// Internal LANES ring dimension.
    pub fn d_tilde(&self) -> usize {
        self.d_tilde
    }
    /// Splitting factor, so `l` NTT slots.
    pub fn l_split(&self) -> usize {
        self.l_split
    }
    /// 26-bit prime, `= 129 mod 256`.  **Paper.**
    pub fn q_tilde(&self) -> u64 {
        self.q_tilde
    }
    /// `n~`: the shared random tail every `b_i` draws from.
    ///
    /// The two ranks carry the roles the *structure* gives them, not the
    /// ones the letters suggest — see [`rank_roles`], which found the
    /// labels reversed once already.  Both are 4 at this profile,
    /// so no expression evaluates differently and no byte moves, which is
    /// precisely why the labels drifted back.
    pub fn n_tilde(&self) -> usize {
        self.n_tilde
    }
    /// `l~`: `B_0`'s identity rank, and so the rows of `t_0`.
    pub fn ell_tilde(&self) -> usize {
        self.ell_tilde
    }
    /// Commitment compression — recorded, not applied here.
    pub fn d_drop(&self) -> u32 {
        self.d_drop
    }
    /// LANES challenge weight.
    pub fn w_hat(&self) -> usize {
        self.w_hat
    }
    /// Message ring elements.
    pub fn n_ex(&self) -> usize {
        self.n_ex
    }
    /// Outer ring dimension, `= par.d`.
    pub fn d(&self) -> usize {
        self.d
    }
    /// `d~ / l`: the coefficient index step between adjacent slots.
    pub fn slot_stride(&self) -> usize {
        self.slots
    }
    /// Rows of `t_0`, i.e. `B_0`'s identity rank — which is `l~`.
    ///
    /// Derived, not restated: [`rank_roles`] is the only place the
    /// mapping lives, so driving *it* with unequal ranks tests this too.
    pub fn t0_rows(&self) -> usize {
        self.t0_rows
    }
    /// `n~` again, under the name that says what it is.
    pub fn tail_rank(&self) -> usize {
        self.n_tilde
    }
    /// `kappa - l~ = 13`: the part of the opening actually transmitted.
    ///
    /// The Bai--Galbraith compression masks and sends only the response
    /// to `B_0`'s non-identity columns, so the *identity* rank comes off
    /// — never `n~`.  When the two ranks differ the formulas give 16
    /// and 17; here they agree at 13, which is why this is derived from
    /// [`rank_roles`] rather than written out.
    pub fn response_rank(&self) -> usize {
        self.kappa - self.ell_tilde
    }
    /// Slots in one message block: `l`.
    pub fn block_slots(&self) -> usize {
        self.l_split
    }
    /// Slots of a block that carry coefficients: `d`.
    pub fn block_used(&self) -> usize {
        self.d
    }
    /// Slots of a block that are explicit zero padding: `l - d`.
    pub fn block_pad(&self) -> usize {
        self.l_split - self.d
    }
    /// `g` and the two product-proof commitments; the paper's `alpha`.
    pub fn aux_slots(&self) -> usize {
        self.aux_slots
    }
    /// Randomness rank, `n~ + l~ + N_ex + alpha = 17`.
    pub fn kappa(&self) -> usize {
        self.kappa
    }
    /// The commitment ring `R_q~`.
    pub fn rt(&self) -> &Ring {
        &self.rt
    }
}

/// Which of `n~` and `l~` plays which structural role, for any pair.
///
/// **The single place this mapping exists.**  [`ExactParams`] and
/// `lanes::params` both derive from it rather than restating it, so a
/// test can drive it with *unequal* ranks and see the mapping the
/// production code actually implements.  A numeric check at the published
/// parameters cannot distinguish the role names because both ranks are 4.
///
/// The paper's own MLWE dimensions fix the assignment.  The
/// coefficient-embedded instance has *secret* dimension `n~ d~` and
/// `(l~ + N_ex + alpha) d~` samples; the secret is the shared randomness
/// tail, and the samples are the rows that touch it — `B_0`'s rows plus
/// the `N_ex + alpha` commitment rows.  Hence:
///
/// * `l~` is the identity rank: rows of `t_0`, width of `B_0`'s `I` block;
/// * `n~` is the shared tail: columns each `b_i` draws randomness from.
///
/// Returned as a named struct rather than a tuple so callers use structural
/// role names instead of relying on positional conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankRoles {
    pub identity_rank: usize,
    pub tail_rank: usize,
    pub kappa: usize,
    pub response_rank: usize,
    pub lwe_secret_rank: usize,
    pub lwe_sample_rank: usize,
}

/// See [`RankRoles`].
pub const fn rank_roles(
    n_tilde: usize,
    ell_tilde: usize,
    n_ex: usize,
    aux_slots: usize,
) -> RankRoles {
    let kappa = n_tilde + ell_tilde + n_ex + aux_slots;
    RankRoles {
        identity_rank: ell_tilde,
        tail_rank: n_tilde,
        kappa,
        // The Bai--Galbraith response drops `B_0`'s identity block, so it
        // is `kappa` minus the *identity* rank — never minus `n~`.
        response_rank: kappa - ell_tilde,
        // The two dimensions the paper prints, which is what decides it.
        lwe_secret_rank: n_tilde,
        lwe_sample_rank: ell_tilde + n_ex + aux_slots,
    }
}

impl ExactParams {
    /// Internal LANES ring dimension.  **Paper.**
    pub const D_TILDE: usize = 256;
    /// Splitting factor, so `l = 64` maximal NTT slots.  **Paper.**
    pub const L_SPLIT: usize = 64;
    /// **Paper**; 26 bits, `129 mod 256`.
    pub const Q_TILDE: u64 = 67_107_713;
    /// Shared random tail width.  **Paper.**
    pub const N_TILDE: usize = 4;
    /// `B_0` identity rank == rows of `t_0`.  **Paper.**
    pub const ELL_TILDE: usize = 4;
    /// Commitment compression.  **Paper.**
    pub const D_DROP: u32 = 17;
    /// LANES challenge weight.  **Paper.**
    pub const W_HAT: usize = 44;
    /// Message ring elements.  **Paper.**
    pub const N_EX: usize = 6;
    /// `g` and the two product-proof commitments.
    pub const AUX_SLOTS: usize = 3;

    /// `Err` names every condition [`ExactParams::check`] rejected.
    ///
    /// Fallible rather than a diagnostic nobody calls: a stale, composite
    /// or independently edited exact parameter used to survive
    /// construction and surface only as a wrong proof.  The reference
    /// raises from `__init__` for the same reason.
    pub fn new(par: &RiVeRParams) -> Result<Self, String> {
        let ex = Self::unchecked(*par);
        let bad = ex.check();
        if bad.is_empty() {
            Ok(ex)
        } else {
            Err(format!("exact parameters rejected: {}", bad.join("; ")))
        }
    }

    /// Build without validating — for tests that need to see [`check`]
    /// reject something.
    ///
    /// [`check`]: ExactParams::check
    pub fn unchecked(par: RiVeRParams) -> Self {
        let roles = rank_roles(Self::N_TILDE, Self::ELL_TILDE, Self::N_EX, Self::AUX_SLOTS);
        Self {
            d: par.d,
            par,
            d_tilde: Self::D_TILDE,
            l_split: Self::L_SPLIT,
            q_tilde: Self::Q_TILDE,
            n_tilde: Self::N_TILDE,
            ell_tilde: Self::ELL_TILDE,
            d_drop: Self::D_DROP,
            w_hat: Self::W_HAT,
            n_ex: Self::N_EX,
            slots: Self::D_TILDE / Self::L_SPLIT,
            t0_rows: roles.identity_rank,
            aux_slots: Self::AUX_SLOTS,
            kappa: roles.kappa,
            rt: Ring::with_backend(Self::Q_TILDE, Self::D_TILDE, roles.kappa),
        }
    }

    /// The structural role of each rank; see [`rank_roles`].
    pub fn roles(&self) -> RankRoles {
        rank_roles(self.n_tilde, self.ell_tilde, self.n_ex, self.aux_slots)
    }

    /// `24 phi_m eta_m`: the no-wrap bound `q~` has to clear.
    ///
    /// Two accepted responses each satisfy `||z_m||_inf <= 6 sigma_m`, so
    /// their difference is at most `12 sigma_m = 12 phi_m eta_m`, and a
    /// unique centred lift of `z_eval - x e_eval` needs `q~` above twice
    /// that.  `eta_m = w gamma B_e sqrt(d)` is profile-independent, so
    /// this is one number for all five: 66730968.02, against
    /// `q~ = 67107713`.
    ///
    /// The margin is 376744.98, about 0.56%, and it exists only because
    /// `B_e = 30` — see [`Self::q_tilde_clears`], which is the form that
    /// actually decides it.
    ///
    /// The old reconstruction-side floor `w gamma 3^4 = 41472` is kept in
    /// the max only because it costs nothing; it is three orders of
    /// magnitude below the response-side term.
    pub fn q_tilde_need(&self) -> f64 {
        let response = 24.0 * self.par.phi_m as f64 * self.par.eta_m();
        let reconstruction = (self.par.w as f64) * (self.par.gamma as f64) * 81.0;
        response.max(reconstruction)
    }

    /// Exact `q~ > 24 phi_m w gamma B_e sqrt(d)`, with no float in it.
    ///
    /// The margin is 0.56%, which is inside what a float `sqrt` plus a
    /// multiplication chain can move, so the condition the whole exact
    /// backend rests on is decided over the integers:
    /// `q~^2 > (24 phi_m w gamma B_e)^2 d`.
    ///
    /// `b_e` is a parameter so a test can confirm that the specified
    /// centred bound is essential to the selected modulus inequality.
    pub fn q_tilde_clears(&self, b_e: u64) -> bool {
        let par = &self.par;
        let k = 24u128 * par.phi_m as u128 * par.w as u128 * par.gamma as u128 * b_e as u128;
        let q = self.q_tilde as u128;
        q * q > k * k * par.d as u128
    }

    /// Every condition the exact parameters have to meet, as a list of
    /// what failed.  Empty means supported.
    pub fn check(&self) -> Vec<String> {
        let mut e = Vec::new();
        // Each field against the constant it is supposed to be.  Without
        // this, `check` validated relations *between* the fields, so a
        // moved dimension that happened to keep those relations — or that
        // broke none of them — still "validated" while `rt` and the
        // commitment key had already been built from the old value.
        for (label, got, want) in [
            ("d~", self.d_tilde, Self::D_TILDE),
            ("l", self.l_split, Self::L_SPLIT),
            ("n~", self.n_tilde, Self::N_TILDE),
            ("l~", self.ell_tilde, Self::ELL_TILDE),
            ("N_ex", self.n_ex, Self::N_EX),
            ("alpha", self.aux_slots, Self::AUX_SLOTS),
            ("w_hat", self.w_hat, Self::W_HAT),
        ] {
            if got != want {
                e.push(format!("{label} = {got} != {want}"));
            }
        }
        if self.q_tilde != Self::Q_TILDE {
            e.push(format!("q~ = {} != {}", self.q_tilde, Self::Q_TILDE));
        }
        if self.d_drop != Self::D_DROP {
            e.push(format!("D = {} != {}", self.d_drop, Self::D_DROP));
        }
        // Structural invariants the *derived* LANES constants assume.
        // Pinning each field to its constant above is not the same thing:
        // it says the values have not moved, not that a future edit which
        // moves them coherently stays well formed.  `lanes::ring` computes
        // `SUBDEG = D_TILDE / L_SPLIT` and `LEVELS = L_SPLIT
        // .trailing_zeros()`, and `lanes::params` computes `W_TILDE =
        // W_HAT / DELTA` — each of which silently truncates, or stops
        // being a logarithm, if these do not hold.
        if self.l_split == 0 {
            e.push("l = 0".into());
        } else {
            if !self.d_tilde.is_multiple_of(self.l_split) {
                e.push(format!(
                    "l = {} does not divide d~ = {}, so SUBDEG truncates",
                    self.l_split, self.d_tilde
                ));
            }
            if !self.l_split.is_power_of_two() {
                e.push(format!(
                    "l = {} is not a power of two, so LEVELS = trailing_zeros(l) is not log2(l)",
                    self.l_split
                ));
            }
            // `DELTA` is the slot stride `d~ / l`; `W_TILDE = w_hat /
            // DELTA` has to be the exact per-residue-class weight, or the
            // challenge sampler places the wrong total weight.
            let delta = self.d_tilde / self.l_split;
            if delta == 0 {
                e.push("d~ < l, so the slot stride is 0".into());
            } else if !self.w_hat.is_multiple_of(delta) {
                e.push(format!(
                    "DELTA = {delta} does not divide w_hat = {}, so W_TILDE truncates",
                    self.w_hat
                ));
            }
        }
        if self.rt.q != self.q_tilde || self.rt.d != self.d_tilde {
            e.push("the commitment ring does not match the parameters".into());
        }
        if self.d != self.par.d {
            e.push("d does not match the outer profile".into());
        }
        let roles = self.roles();
        if self.slots != self.d_tilde / self.l_split
            || self.t0_rows != roles.identity_rank
            || self.kappa != roles.kappa
        {
            e.push("a derived dimension does not follow from the others".into());
        }
        // Primality was assumed rather than tested, which left a composite
        // `q~` — where `R_q~` is not even a domain — indistinguishable
        // from a good one until a proof failed.
        if !is_prime(self.q_tilde) {
            e.push(format!("q~ = {} is not prime", self.q_tilde));
        }
        if self.q_tilde % (4 * self.l_split as u64) != 2 * self.l_split as u64 + 1 {
            e.push("q~ != 2l+1 mod 4l (fully-splitting condition)".into());
        }
        if !((1u64 << 25)..(1u64 << 26)).contains(&self.q_tilde) {
            e.push("ceil(log2 q~) is not 26 as reported".into());
        }
        // The six exact messages are `(y_eval, e_eval, d_0, d_1, d_2,
        // d_3)`, one per block.  Each block carries `d` coefficients in
        // its first `d` slots and is zero-padded to `l`.  The old
        // `6 d == N_ex l` identity is gone: `192 != 384` is intentional.
        if 1 + 1 + RADIX_WEIGHTS.len() != self.n_ex {
            e.push("exact message count != N_ex".into());
        }
        if self.l_split < self.d {
            e.push("l < d, so a message block cannot hold d slots".into());
        }
        if 2 * RADIX_WEIGHTS.iter().sum::<i64>() != self.par.q0 as i64 - 1 {
            e.push("radix weights do not cover [0, q_0-1] exactly".into());
        }
        // Decided exactly; `q_tilde_need` is the float form, for
        // reporting.  With the *unshifted* bound `q_0 - 1 = 60` the
        // requirement doubles and this modulus fails outright, which is
        // why `ring::to_centered_error` is not presentational.
        if !self.q_tilde_clears(self.par.B_e()) {
            e.push(format!(
                "q~ <= {:.7} (internal modulus condition)",
                self.q_tilde_need()
            ));
        }
        e
    }
}

// ---- radix-3 range encoding ---------------------------------------------

/// Digits `(a_0, a_1, a_2, a_3)` in `{0,1,2}` with `sum g_j a_j == value`.
///
/// Greedy from the largest weight.  The encoding is not injective
/// (`17 = (0,0,0,1) = (2,2,1,0)`); that is harmless, soundness only needs
/// the reachable set to be exactly `[0, 60]`.
///
/// `None` outside `[0, 60]` — including for every negative value, which is
/// where this module and the paper's reconstruction equation part
/// company.  The relation is stated over the *canonical* error `[0, 60]`
/// and the equation `e_eval + 30 = sum_j g_j d_j` over the *centred*
/// `[-30, 30]`; this follows the relation.
pub fn radix_decompose(value: i64) -> Option<[i64; 4]> {
    if !(0..=2 * WEIGHT_SUM).contains(&value) {
        return None;
    }
    let mut digits = [0i64; 4];
    let mut remainder = value;
    for idx in (0..RADIX_WEIGHTS.len()).rev() {
        let weight = RADIX_WEIGHTS[idx];
        let digit = (RADIX_DIGITS as i64 - 1).min(remainder / weight);
        digits[idx] = digit;
        remainder -= digit * weight;
    }
    (remainder == 0).then_some(digits)
}

pub fn radix_recompose(digits: &[i64]) -> i64 {
    RADIX_WEIGHTS
        .iter()
        .zip(digits.iter())
        .map(|(w, a)| w * a)
        .sum()
}

/// Decompose each coefficient; returns 4 rows of `d` digits.
pub fn decompose_poly(coeffs: &[i64]) -> Option<Vec<Vec<i64>>> {
    let mut out = vec![vec![0i64; coeffs.len()]; RADIX_WEIGHTS.len()];
    for (i, &c) in coeffs.iter().enumerate() {
        let digits = radix_decompose(c)?;
        for j in 0..RADIX_WEIGHTS.len() {
            out[j][i] = digits[j];
        }
    }
    Some(out)
}

// ---- witness packing -----------------------------------------------------

/// Lay the six exact messages into `N_ex` elements of `R_q~`.
///
/// The paper gives **each of the six outer ring elements
/// its own LANES message block**: a block holds `l = 64` slots, the
/// element's `d = 32` coefficients occupy the first 32, and the remaining
/// 32 are explicit zero padding.  So `6 d = 192` scalars sit in
/// `N_ex l = 384` slots and the old `6 d == N_ex l` identity is gone —
/// `192 != 384` is intentional, not a shortfall.
///
/// Scalar `j` of message element `i` goes to coefficient
/// `j * slot_stride`, which mirrors the NTT-slot layout a real LANES
/// backend uses: `[ENS20]` commits one scalar per NTT block, at index
/// `j * delta` of the transformed array.
///
/// The element order is the paper's, `(y_eval, e_eval, d_0, ..., d_3)`.
///
/// `None` on a witness of the wrong shape — reachable from `Verify`
/// through a `pp` whose `N_ex` does not match the profile, where a panic
/// would be the wrong shape of failure: a verifier returns a bit.
pub fn pack_witness(
    ex: &ExactParams,
    e_eval: &[i64],
    y_eval: &[i64],
    digits: &[Vec<i64>],
) -> Option<PolyVec> {
    let mut elements: Vec<&[i64]> = Vec::with_capacity(ex.n_ex);
    elements.push(y_eval);
    elements.push(e_eval);
    for row in digits {
        elements.push(row);
    }
    if elements.len() != ex.n_ex {
        return None;
    }
    if elements.iter().any(|e| e.len() != ex.block_used()) {
        return None;
    }
    let q = ex.q_tilde as i128;
    Some(
        elements
            .into_iter()
            .map(|element| {
                let mut poly = vec![0u64; ex.d_tilde];
                for (j, &v) in element.iter().enumerate() {
                    poly[j * ex.slots] = (v as i128).rem_euclid(q) as u64;
                }
                // Slots `block_used .. block_slots-1` stay zero: that is
                // the padding, and `padding_is_zero` is what makes it a
                // checked property rather than a convention.
                poly
            })
            .collect(),
    )
}

/// Inverse of [`pack_witness`]: the `N_ex * d` carried scalars mod `q~`.
///
/// The padding slots are *not* returned.  Use [`padding_is_zero`] to
/// check them; a verifier that silently ignored them would accept a
/// witness carrying data the relation does not constrain.
pub fn unpack_witness(ex: &ExactParams, message: &[Poly]) -> Vec<u64> {
    let mut out = Vec::with_capacity(ex.n_ex * ex.block_used());
    for poly in message {
        for j in 0..ex.block_used() {
            out.push(poly[j * ex.slots]);
        }
    }
    out
}

/// Every slot past `block_used` in every block is zero mod `q~`.
///
/// The paper makes the padding part of the committed message, so it is
/// part of what the exact relation has to pin. The LANES backend includes
/// it in the proved linear system; here it is also enforced at the
/// boundary so the two cannot disagree about which messages are well
/// formed.
pub fn padding_is_zero(ex: &ExactParams, message: &[Poly]) -> bool {
    if message.len() != ex.n_ex {
        return false;
    }
    message.iter().all(|poly| {
        poly.len() == ex.d_tilde
            && (ex.block_used()..ex.block_slots()).all(|j| poly[j * ex.slots] % ex.q_tilde == 0)
    })
}

// ---- commitment ----------------------------------------------------------

/// BDLOP commitment key `(A_1, A_2)` over `R_q~`.
///
/// `A_1` is `t0_rows x kappa` and `A_2` is `N_ex x kappa`, mirroring `B_0`
/// and `(b_j)` of the `[BDLOP18]` commitment as `[ENS20]` Figure 3 uses it.
///
/// One deliberate divergence: LANES samples the commitment randomness from
/// a discrete Gaussian, whereas this key uses ternary randomness, the
/// BDLOP convention.  The paper gives no LANES-internal Gaussian width for
/// *this* commitment, and guessing one would be worse than stating the
/// difference.
pub struct ExactCommitmentKey {
    a1: PolyMat,
    a2: PolyMat,
    /// The same two matrices, pre-transformed.
    ///
    /// `R_q~` is `Z_q~[X]/(X^256+1)` with `q~ = 129 mod 256`, so it has a
    /// native incomplete NTT with 64 degree-4 slots — that is the whole
    /// reason the paper selects this modulus, and [`crate::lanes::ring`]
    /// implements it.  A commitment is `10 x 17` ring products; through
    /// schoolbook that is 11.1 million coefficient multiplies, and through
    /// the transform about 200 thousand.
    ///
    /// The matrices are fixed for the lifetime of the key, so they are
    /// transformed once here rather than on every `commit` — the same
    /// trade [`crate::aux_ntt`] makes for `G'` one layer up.  `None` if
    /// the exact ring is not this one, in which case `commit` falls back
    /// to schoolbook and agrees exactly; the arithmetic is over `Z_q~`
    /// either way, so no byte moves.
    ntt: Option<NttKey>,
    n_ex: usize,
}

/// `A_1` and `A_2` in the NTT domain.
struct NttKey {
    a1: Vec<Vec<LanesNtt>>,
    a2: Vec<Vec<LanesNtt>>,
}

fn transform_matrix(m: &PolyMat) -> Option<Vec<Vec<LanesNtt>>> {
    m.iter()
        .map(|row| {
            row.iter()
                .map(|p| lanes_ring::CoeffPoly::new(p).map(|c| lanes_ring::ntt(&c)))
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

impl ExactCommitmentKey {
    pub fn new(ex: &ExactParams, seed: &[u8]) -> Self {
        let a1 = sam_mat(
            seed, ex.q_tilde, ex.t0_rows, ex.kappa, ex.d_tilde, "Pi_ex.A1",
        );
        let a2 = sam_mat(seed, ex.q_tilde, ex.n_ex, ex.kappa, ex.d_tilde, "Pi_ex.A2");
        // Only when the exact ring *is* `lanes::ring`'s.  `ExactParams`
        // pins both to the same constants and `check` enforces it, so this
        // holds for every profile; the test is here because a future
        // parameter change should lose the fast path rather than silently
        // transform in the wrong ring.
        let ntt = (ex.q_tilde == lanes_ring::QTILDE && ex.d_tilde == lanes_ring::DTILDE)
            .then(|| {
                Some(NttKey {
                    a1: transform_matrix(&a1)?,
                    a2: transform_matrix(&a2)?,
                })
            })
            .flatten();
        Self {
            a1,
            a2,
            ntt,
            n_ex: ex.n_ex,
        }
    }

    /// `W = (A_1 r, A_2 r + m)`.
    pub fn commit(&self, rt: &Ring, message: &[Poly], randomness: &[Poly]) -> ExactCommitment {
        let (t0, mut t1) = match self.ntt_product(rt, randomness) {
            Some(pair) => pair,
            None => (
                rt.mat_vec(&self.a1, randomness),
                rt.mat_vec(&self.a2, randomness),
            ),
        };
        for i in 0..self.n_ex {
            t1[i] = rt.add(&t1[i], &message[i]);
        }
        ExactCommitment { t0, t1 }
    }

    /// `(A_1 r, A_2 r)` through the native transform, or `None` to fall
    /// back — a ring that is not `R_q~`, or a `randomness` that is not
    /// `kappa` canonical elements of it.
    ///
    /// The fallback is not decorative: `randomness` reaches `commit` from
    /// a peer's proof on the verifying side, so a wrong shape has to be a
    /// slower answer or a rejection, never a panic.
    fn ntt_product(&self, rt: &Ring, randomness: &[Poly]) -> Option<(PolyVec, PolyVec)> {
        let key = self.ntt.as_ref()?;
        if rt.q != lanes_ring::QTILDE || rt.d != lanes_ring::DTILDE {
            return None;
        }
        if randomness.len() != key.a1.first()?.len() {
            return None;
        }
        let r_ntt: Vec<LanesNtt> = randomness
            .iter()
            .map(|p| lanes_ring::CoeffPoly::new(p).map(|c| lanes_ring::ntt(&c)))
            .collect::<Option<_>>()?;
        let apply = |rows: &[Vec<LanesNtt>]| -> Option<PolyVec> {
            rows.iter()
                .map(|row| {
                    lanes_ring::inner_ntt(row, &r_ntt).map(|acc| lanes_ring::intt(&acc).to_vec())
                })
                .collect()
        };
        Some((apply(&key.a1)?, apply(&key.a2)?))
    }
}

/// `W = (t_0, t_1)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactCommitment {
    pub t0: PolyVec,
    pub t1: PolyVec,
}

// ---- statement, witness and proof ---------------------------------------

/// `(W, z_eval, x)` — everything `Pi_ex.Ver` is told.
pub struct ExactStatement<'a> {
    pub w: &'a ExactCommitment,
    /// `z_eval`, centred integers.
    pub z_eval: &'a [i64],
    /// `x`, centred integers.
    pub x: &'a [i64],
}

/// `w_ex = (e_eval, y_eval)`.  A backend never sees the outer secret key.
#[derive(Clone, Debug)]
pub struct ExactWitness {
    /// Centred, in `[-B_e, B_e]`; adding `B_e` gives the canonical error.
    pub e_eval: Vec<i64>,
    /// Centred integers — a coordinate of the OOM mask.
    pub y_eval: Vec<i64>,
}

/// `sigma_ex` for [`OpeningBackend`]: the opening itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningProof {
    pub e_eval: Vec<i64>,
    pub y_eval: Vec<i64>,
    pub digits: Vec<Vec<i64>>,
    /// Ternary, carried as residues mod `q~`.
    pub randomness: PolyVec,
}

/// Prover state carried from `Com` to `Prove`.
pub struct OpeningState {
    pub digits: Vec<Vec<i64>>,
    pub randomness: PolyVec,
}

/// Decide `R^_ex` directly; returns the violated clauses.
///
/// Used by the backend verifier and, independently, by the tests — the
/// relation is small enough to state once and check literally.
pub fn check_relation(
    ex: &ExactParams,
    statement: &ExactStatement<'_>,
    witness: &ExactWitness,
    digits: &[Vec<i64>],
) -> Vec<String> {
    let mut errors = Vec::new();
    let (q0, d) = (ex.par.q0 as i64, ex.d);

    // Shapes first, and **return** on any of them.  The reference is total
    // here because Python indexing is checked; this indexes `p[i]` and
    // `e_eval[i]` below, so a short row that only *recorded* an error and
    // carried on panicked instead of returning a violated clause.
    let mut shaped = witness.e_eval.len() == d;
    if !shaped {
        errors.push("e_eval has the wrong length".into());
    }
    if witness.y_eval.len() != d || statement.z_eval.len() != d || statement.x.len() != d {
        errors.push("witness or statement has the wrong length".into());
        shaped = false;
    }
    if digits.len() != RADIX_WEIGHTS.len() {
        errors.push("wrong number of digit polynomials".into());
        shaped = false;
    }
    for (j, poly) in digits.iter().enumerate() {
        if poly.len() != d {
            errors.push(format!("digit polynomial {j} has the wrong length"));
            shaped = false;
        }
    }
    if !shaped {
        return errors;
    }

    let b_e = q0 / 2;
    if witness.e_eval.iter().any(|&c| !(-b_e..=b_e).contains(&c)) {
        errors.push("e_eval outside [-B_e, B_e]^d".into());
    }
    for (j, poly) in digits.iter().enumerate() {
        if poly.iter().any(|&a| !(0..=2).contains(&a)) {
            errors.push(format!("digit polynomial {j} is not ternary in {{0,1,2}}"));
        }
    }
    for i in 0..d {
        let row: Vec<i64> = digits.iter().map(|p| p[i]).collect();
        if radix_recompose(&row) != witness.e_eval[i] + b_e {
            errors.push(format!("digit reconstruction fails at coefficient {i}"));
            break;
        }
    }

    // `z_eval = x e_eval + y_eval`, as an equality over Z.
    //
    // This must not be checked modulo `q~` (or any other protocol
    // modulus).  Doing so accepts `y_eval + k q~` for any `k`: the
    // commitment reduces the witness mod `q~`, so those lifts are
    // indistinguishable to it.  The bound rules those lifts out anyway,
    // but the relation says `over Z` and this checks `over Z`.
    let x: Vec<i128> = statement.x.iter().map(|&c| c as i128).collect();
    let e: Vec<i128> = witness.e_eval.iter().map(|&c| c as i128).collect();
    let product = Ring::mul_int(&x, &e);
    let matches =
        (0..d).all(|i| product[i] + witness.y_eval[i] as i128 == statement.z_eval[i] as i128);
    if !matches {
        errors.push("z_eval != x * e_eval + y_eval over Z".into());
    }
    errors
}

/// An exact non-negative rational, as `(numerator, denominator)`.
///
/// Two of the gated constants are rationals, so comparing them by their
/// numerator alone would let a different denominator through.
pub type Rational = (u128, u128);

/// The wire-visible LANES constants the gate covers:
/// `(name, what it is, the value the paper's closed form implies)`.
///
/// The gate requires a source audit reporting zero *active* constants that
/// have drifted from that closed form.  Listing them with their values
/// makes the audit executable: a constant that is renamed, deleted **or
/// quietly changed** shows up, where a list of names alone would only
/// catch the first two.
///
/// The dimensions are deliberately absent — `d~`, `l`, `q~`, `(n~, l~)` and
/// `D` were already current before the widths were.
pub const GATED_LANES_CONSTANTS: [(&str, &str, Rational); 6] = [
    (
        "SIGMA_R",
        "commitment randomness width, s_1 rounded to 2^-20",
        (2_901_189, 524_288),
    ),
    (
        "SIGMA_Y",
        "proof mask width, s_2 rounded to 2^-20",
        (255_304_631, 1_048_576),
    ),
    ("Z_INF_BOUND", "response infinity bound", (3_448, 1)),
    (
        "Z_NORM2_BOUND",
        "response Euclidean bound, the paper's (2s)^2 rule",
        (1_578_304_756, 1),
    ),
    ("RECOVERY_ERROR_BOUND", "hint bound", (2_886_972, 1)),
    ("RECOVERY_BUCKETS", "fixed-hint bucket count", (16, 1)),
];

/// The inputs a manifest has to account for.  In the paper these are
/// the two widths; the rest are derivations that move with them.
pub const LANES_MANIFEST_INPUTS: [&str; 2] = ["SIGMA_R", "SIGMA_Y"];

/// What each gated constant is *actually* set to in `lanes::params`.
///
/// Read from the module rather than from the table above, so the two can
/// be compared: `(name, what, live value)`.
pub fn live_lanes_constants() -> Vec<(&'static str, &'static str, Rational)> {
    use crate::lanes::params as lp;
    GATED_LANES_CONSTANTS
        .iter()
        .map(|&(name, what, _)| {
            let live: Rational = match name {
                "SIGMA_R" => (lp::SIGMA_R.0 as u128, lp::SIGMA_R.1 as u128),
                "SIGMA_Y" => (lp::SIGMA_Y.0 as u128, lp::SIGMA_Y.1 as u128),
                "Z_INF_BOUND" => (lp::Z_INF_BOUND as u128, 1),
                "Z_NORM2_BOUND" => (lp::Z_NORM2_BOUND as u128, 1),
                "RECOVERY_ERROR_BOUND" => (lp::RECOVERY_ERROR_BOUND as u128, 1),
                "RECOVERY_BUCKETS" => (lp::RECOVERY_BUCKETS as u128, 1),
                _ => unreachable!("GATED_LANES_CONSTANTS names only these"),
            };
            (name, what, live)
        })
        .collect()
}

fn rational_eq(a: Rational, b: Rational) -> bool {
    a.0 * b.1 == b.0 * a.1
}

fn show(r: Rational) -> String {
    if r.1 == 1 {
        r.0.to_string()
    } else {
        format!("{}/{}", r.0, r.1)
    }
}

/// One constant the manifest selects: its value and its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestConstant {
    pub name: &'static str,
    /// The value the manifest selects — compared against what the code
    /// consumes, because a **Paper** label on a retained value does not
    /// make the paper have chosen it.
    pub value: Rational,
    pub provenance: &'static str,
}

/// `n~` / `l~` roles, and the ranks that follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankRoleSpec {
    pub identity_rank: usize,
    pub tail_rank: usize,
    pub kappa: usize,
    pub response_rank: usize,
}

/// Sampler widths as exact rationals, with their tail parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerSpec {
    pub sigma_r: Rational,
    pub sigma_y: Rational,
    /// `-log2(eps)`, the smoothing-parameter target the widths follow
    /// from.  The paper gives the widths in closed form, so the manifest carries the
    /// form's input, not a pair of searched integers.
    pub epsilon_exponent: u32,
    /// `"standard deviation"` or `"gaussian parameter"` — the two differ
    /// by `sqrt(2 pi)` and the paper prints both, so a manifest that did
    /// not say which it meant would be 18 bits ambiguous.
    pub convention: &'static str,
    pub tail_cut_r: u64,
    pub tail_cut_y: u64,
    pub prob_bits: u32,
}

/// Exact integer response bounds and how they are compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseBoundSpec {
    pub inf: u128,
    pub l2: u128,
    /// `"<"` or `"<="`.
    pub comparison: &'static str,
    /// The population each failure probability is union-bounded over.
    pub population: &'static str,
}

/// The `D` compression and commitment-recovery algorithm, in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoverySpec {
    pub d_drop: u32,
    pub rounding: &'static str,
    pub ties: &'static str,
    /// Response ring elements Bai–Galbraith omits: `kappa - response_rank`.
    pub omitted_response_rows: usize,
    /// Those, in coefficients.
    pub omitted_response_coefficients: usize,
    /// Low bits of `t_0` dropped by `power2round`: `l~ d~ D`.
    pub omitted_t0_low_bits: usize,
    /// Ternary carries transmitted in their place: `l~ d~`.
    pub recovery_carries: usize,
    pub hint_alphabet: &'static str,
    pub limit: u128,
    pub failure_rule: &'static str,
    pub verification_rule: &'static str,
    pub encoding: &'static str,
}

/// One Fiat–Shamir transcript field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptField {
    pub name: &'static str,
    pub domain_separator: &'static str,
    /// Whether the hashed value is transmitted or recovered.
    pub hashed_form: &'static str,
}

/// One Fiat–Shamir round: a challenge and what is hashed before it.
///
/// The three rounds are the protocol's shape, and flattening them to a
/// field list loses which challenge each group precedes — which is what a
/// port needs, since absorbing the same fields in the same order but
/// drawing the challenges at different points is a different protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptRound {
    /// `"alpha"`, `"gamma"` or `"c"`.
    pub challenge: &'static str,
    /// The domain-separator suffix this challenge is drawn under.
    pub separator: &'static str,
    /// Absorbed before it, in order.  A `|`-joined name is one `absorb`
    /// argument: the parts are concatenated before hashing.
    pub absorbs: &'static [&'static str],
}

/// One serialized field, with everything a port needs to encode it.
///
/// A coder's *name* is not enough: `Uniform` needs its modulus and the
/// width that follows, `Signed` its bound, and `Rice` its `k` and its cap
/// — and `k` is wire-visible, so two implementations that chose it
/// differently would produce different bytes with no other symptom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireField {
    pub name: &'static str,
    /// Ring elements in the field.  One is recorded as `1`, not absent.
    pub rows: usize,
    /// Coefficients per element.
    pub cols: usize,
    pub coder: &'static str,
    /// `None` for a variable-length field — a fact about the format, not a
    /// gap.  Recording it as `0` made a Rice-coded field indistinguishable
    /// from an empty one.
    pub bits: Option<u64>,
    /// `Uniform`'s modulus.
    pub modulus: Option<u64>,
    /// `Signed`'s or `Rice`'s bound.
    pub bound: Option<u64>,
    /// Fixed width, for the fixed-width coders.
    pub width_bits: Option<u32>,
    /// `Rice`'s parameter.
    pub rice_k: Option<u32>,
}

/// The wire layout, and the size it sums to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireSpec {
    pub fields: &'static [WireField],
    /// `None` when the layout has no fixed total — the Rice-coded `z`
    /// makes it sample-dependent.  `discrepancy` must then say so.
    pub total_bits: Option<u64>,
    /// What the fixed-width fields contribute, which *is* fixed.
    pub fixed_bits: u64,
    pub kb_convention: &'static str,
    /// Set when `total_bits` does not reproduce the stated size, saying
    /// so.  The acceptance criteria allow either — reproducing 13.5 KB
    /// field by field, or reporting the gap — but not silence.
    pub discrepancy: Option<&'static str>,
}

/// The exact-layer dimensions the table freezes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionSpec {
    pub d_tilde: usize,
    pub l_split: usize,
    pub sub_degree: usize,
    pub q_tilde: u64,
    pub q_tilde_bits: u32,
    pub n_tilde: usize,
    pub ell_tilde: usize,
    pub n_ex: usize,
    pub alpha: usize,
    pub d_drop: u32,
    pub w_hat: usize,
    pub w_tilde: usize,
    pub delta_stride: usize,
    pub n_lwe: usize,
    pub m_lwe: usize,
    pub block_slots: usize,
    pub block_payload: usize,
    pub message_blocks: usize,
}

/// The estimator run that approved the widths and bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EstimatorSpec {
    pub hint_mlwe_inputs: &'static str,
    pub hint_mlwe_outputs: &'static str,
    pub msis_inputs: &'static str,
    pub msis_outputs: &'static str,
    /// The challenge section: what reproduces the paper's own figures.
    ///
    /// The parameter section gives the LANES challenge-difference
    /// noninvertibility probability as `2^-90.5` and the outer RVRF figure
    /// as `2^-91.5`.  These are the published quantities that separate
    /// these LANES parameters from any other
    /// ones, so a manifest that does not reproduce them has not shown it
    /// describes the re-optimized set.  Neither reaches 128 bits.
    pub challenge: &'static str,
}

/// A frozen LANES **parameter** manifest.
///
/// Every section carries data rather than a flag: representing them as
/// booleans would let eight `true`s satisfy "complete manifest" without
/// containing any of the information a port actually needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanesManifest {
    /// SHA-256 of the canonical JSON of `river-py/lanes_manifest.json`.
    ///
    /// The projection into Rust is lossy by construction — it is a typed
    /// view of a table with prose in it — so "the two agree" cannot mean
    /// "the two are equal".  This is what makes projection drift
    /// *detectable*: regenerate, and if the source moved without the
    /// generated view moving, the digest says so.
    pub source_sha256: &'static str,
    pub dimensions: DimensionSpec,
    pub rank_roles: RankRoleSpec,
    pub sampler: SamplerSpec,
    pub response_bounds: ResponseBoundSpec,
    pub recovery: RecoverySpec,
    pub transcript: &'static [TranscriptField],
    pub rounds: &'static [TranscriptRound],
    pub wire: WireSpec,
    pub estimator: EstimatorSpec,
    pub constants: &'static [ManifestConstant],
}

/// The exact-proof entropy estimate the paper reports, **in bits**.
///
/// `13.5 KB` at the size model's `1 KB = 8192 bits`, so `13.5 * 8192 =
/// 110592`.  It was `13_824` — which is `13.5 * 1024`, the figure in
/// *bytes* — while being compared against `LanesManifest::wire.total_bits`,
/// a bit count.  Nothing depended on it while the total was recorded as
/// absent, but the comparison it exists for was off by a factor of eight.
pub const LANES_STATED_BITS: u64 = 110_592;

/// The same figure in bytes, for callers rendering it as `13.5 KB` under
/// the `1 KB = 1024 B` convention.
pub const LANES_STATED_BYTES: u64 = 13_824;

/// Bits per KB under the convention the size model uses.
pub const BITS_PER_KB: u64 = 8_192;

const _: () = {
    assert!(LANES_STATED_BITS == LANES_STATED_BYTES * 8);
    assert!(LANES_STATED_BITS == 27 * BITS_PER_KB / 2); // 13.5 KB
};

/// The validated LANES parameter manifest, or `None`.
///
/// Having one is *not* by itself permission to run the backend — see
/// [`LANES_BACKEND_READY`].
pub const LANES_PARAMETER_MANIFEST: Option<&LanesManifest> =
    Some(&crate::lanes_manifest::LANES_MANIFEST);

/// Whether the candidate composition is exposed under the production
/// backend alias.
///
/// The paper-derived M-SIS and MLWE root-Hermite factors reproduce.  This
/// remains `false` as an artifact-scope decision: the concrete recovery,
/// compression, and wire-format completion is implementation-defined, and
/// the artifact does not supply a reduction for that exact composition.
/// The fully tested code is available under `lanes-experimental`.
pub const LANES_SECURITY_MEETS_TARGET: bool = false;

/// Whether the LANES *implementation* has passed its gates.
///
/// Deliberately a second, separate flag.  With one state, obtaining a
/// manifest would enable [`crate::lanes::backend::LanesBackend`]
/// immediately — while the sampler, bounds, hint code, transcript and
/// layout still consumed nothing from it.  Possession of a parameter table
/// is not by itself a reason to lift the runtime gate.
///
/// **True** since the port landed.  What the gate asks for, and where each of
/// them now is:
///
/// * proof and hint rules ported from the manifest — `lanes::params`
///   derives every published figure, and `lanes_manifest.rs` is generated
///   from the same table `river-py` is gated on;
/// * every Python LANES KAT field matched — the `lanes_ring`,
///   `lanes_params` and `lanes_proof` blocks, all three active;
/// * serializer and verifier green;
/// * negative and totality tests green;
/// * the two LANES vector cases restored — shipped as
///   `lanes-experimental` and re-derived from their seeds here.
///
/// Leaving it `false` after all of that was a *stale* flag rather than a
/// gate: [`LANES_SECURITY_MEETS_TARGET`] is checked first, so it hid
/// behind a condition that is genuinely outstanding.  A gate closed for a
/// reason that is no longer true cannot be told from one closed for a
/// reason that is.
pub const LANES_BACKEND_READY: bool = true;

/// Short, stable tokens for *why* the gate is closed.
///
/// The prose reason names each language's own API — `exact::LANES_*` here,
/// `exact.LANES_*` in `river-py` — so the two cannot be compared byte for
/// byte.  These can, which is what lets a generated artifact record the
/// cause and a consumer in the other language check it has not drifted.
/// The vocabulary is shared with `river-py` so the two are comparable,
/// but this side cannot produce `audit-drift`: [`live_lanes_constants`]
/// reads `lanes::params` by name, so deleting or renaming a gated constant
/// fails to compile here rather than shrinking the audit at run time.
pub const LANES_GATE_CAUSES: [&str; 8] = [
    "audit-drift",           // a gated constant was renamed or deleted
    "constant-changed",      // ...or given a different value, unrecorded
    "no-parameter-manifest", // no frozen table yet
    "manifest-invalid",      // it landed and does not validate
    "manifest-experimental", // it validates and does not claim to be final
    "no-security-evidence",  // no recorded estimator run yet
    "production-alias-reserved",
    "backend-not-ready", // everything else; the implementation gate
];

/// Which of [`LANES_GATE_CAUSES`] applies, or `None` if the gate is open.
///
/// The same decision [`lanes_unavailable_reason`] makes, reported as a
/// token rather than as prose.
pub fn lanes_gate_cause() -> Option<&'static str> {
    lanes_gate_cause_for(LANES_PARAMETER_MANIFEST, LANES_BACKEND_READY)
}

/// [`lanes_gate_cause`] against a supplied state.
pub fn lanes_gate_cause_for(
    manifest: Option<&LanesManifest>,
    backend_ready: bool,
) -> Option<&'static str> {
    let live = live_lanes_constants();
    if live
        .iter()
        .zip(GATED_LANES_CONSTANTS.iter())
        .any(|((_, _, now), (_, _, pinned))| !rational_eq(*now, *pinned))
    {
        return Some("constant-changed");
    }
    let Some(m) = manifest else {
        return Some("no-parameter-manifest");
    };
    if !validate_lanes_manifest(m, &live).is_empty() {
        return Some("manifest-invalid");
    }
    if !LANES_SECURITY_MEETS_TARGET {
        return Some("production-alias-reserved");
    }
    if !backend_ready {
        return Some("backend-not-ready");
    }
    None
}

/// Why the LANES backend cannot run, or `None` if it can.
///
/// A **readiness** test, not a dimension diff, with four independent
/// conditions, evaluated in this order by [`lanes_gate_cause_for`]:
///
/// 1. every live LANES constant still matches the paper's closed form;
/// 2. a frozen parameter manifest exists, carries real data in every
///    section, and *selects a value* for every gated constant, matching
///    what the code
///    consumes;
/// 3. the candidate-composition scope permits the production alias;
/// 4. the implementation has passed its own gates and says so.
///
/// All are required.  A manifest alone would mean the backend ran on
/// a concrete format whose scope was not recorded.
///
/// That dimensions do not enter is the point, and the
/// widths do not either: `d~`, `l`, `q~`, `(n~, l~)`, `D` *and* `s_1`,
/// `s_2`, `beta'`, `B_MSIS`, `delta_MSIS`, and `delta_MLWE` are all the
/// paper's, and
/// [`crate::lanes::ring`] and [`crate::lanes::params`] are current and
/// cross-checked against `river-py`.  The implementation is
/// current as well — `lanes::{mp, proof, backend}` run the proof end to
/// end and both `lanes-experimental` vector cases are re-derived byte for
/// byte, so condition 4 holds.  The only live blocker is condition 3, the
/// artifact-scope gate — see [`LANES_SECURITY_MEETS_TARGET`].
pub fn lanes_unavailable_reason() -> Option<String> {
    lanes_readiness(LANES_PARAMETER_MANIFEST, LANES_BACKEND_READY)
}

/// [`lanes_unavailable_reason`] against a supplied state, so a test can
/// drive the acceptance rule before a real manifest exists.
pub fn lanes_readiness(manifest: Option<&LanesManifest>, backend_ready: bool) -> Option<String> {
    let live = live_lanes_constants();

    let drifted: Vec<String> = live
        .iter()
        .zip(GATED_LANES_CONSTANTS.iter())
        .filter(|((_, _, now), (_, _, pinned))| !rational_eq(*now, *pinned))
        .map(|((name, _, now), (_, _, pinned))| {
            format!("{name} = {} (pinned {})", show(*now), show(*pinned))
        })
        .collect();
    if !drifted.is_empty() {
        return Some(format!(
            "a LANES constant does not match the paper's closed form: {}. \
             The paper derives every one of these from \
             s_0 = sqrt(ln(2 d~ (1 + 1/eps)))/pi with no free constant, so \
             a mismatch is a port defect or a reintroduced selection, not \
             a choice",
            drifted.join(", ")
        ));
    }

    let Some(m) = manifest else {
        let detail = live
            .iter()
            .map(|(n, w, _)| format!("{n} ({w})"))
            .collect::<Vec<_>>()
            .join(", ");
        return Some(format!(
            "no frozen LANES parameter manifest. The paper supplies the \
             widths and entropy size estimate, while the concrete recovery \
             and encoding are implementation-level choices, so the \
             wire-visible values — {detail} — have to be frozen with their \
             provenance before the production name opens. Supply \
             exact::LANES_PARAMETER_MANIFEST (see LanesManifest), then \
             set LANES_BACKEND_READY once the implementation gate passes. \
             Use \
             BackendKind::Opening or BackendKind::LanesExperimental."
        ));
    };

    let bad = validate_lanes_manifest(m, &live);
    if !bad.is_empty() {
        return Some(format!(
            "the LANES manifest is not usable: {}",
            bad.join("; ")
        ));
    }

    if !LANES_SECURITY_MEETS_TARGET {
        return Some(
            "the production LANES alias is reserved: the paper-derived \
             parameters and root-Hermite factors reproduce, but the concrete \
             compression/recovery and wire-format completion is \
             implementation-defined, and this artifact does not supply a \
             reduction for that exact composition. Use \
             BackendKind::LanesExperimental for the tested candidate"
                .into(),
        );
    }

    if !backend_ready {
        return Some(
            "the LANES parameter manifest is present and valid, but the \
             implementation has not passed its own gate: the sampler, \
             response bounds, hint rules, transcript and wire layout must \
             be built *from* it, with every reference LANES KAT field \
             matched and the serializer, verifier, negative tests and both \
             LANES vector cases green. Set exact::LANES_BACKEND_READY only \
             then."
                .to_string(),
        );
    }
    None
}

/// Every way a manifest can fail to be one.  Empty means usable.
///
/// Checks *data*, not headings.  The typed sections make most emptiness
/// unrepresentable; what remains — empty slices, zero counts, blank
/// strings, and above all a `constants` entry whose value is not the one
/// the code consumes — is checked here.
pub fn validate_lanes_manifest(
    m: &LanesManifest,
    live: &[(&'static str, &'static str, Rational)],
) -> Vec<String> {
    let mut e = Vec::new();

    // Sections whose emptiness the type system cannot rule out.
    if m.rank_roles.kappa == 0 || m.rank_roles.response_rank == 0 {
        e.push("rank_roles carries no ranks".into());
    }
    if m.sampler.sigma_r.0 == 0 || m.sampler.sigma_y.0 == 0 {
        e.push("sampler carries no widths".into());
    }
    if m.sampler.epsilon_exponent == 0 || m.sampler.prob_bits == 0 {
        e.push("sampler carries no smoothing target or probability precision".into());
    }
    if m.sampler.convention.is_empty() {
        e.push("sampler does not say which Gaussian convention it is in".into());
    }
    if m.sampler.tail_cut_r == 0 || m.sampler.tail_cut_y == 0 {
        e.push("sampler carries no tail cuts".into());
    }
    if m.response_bounds.inf == 0 || m.response_bounds.l2 == 0 {
        e.push("response_bounds carries no bounds".into());
    }
    if !matches!(m.response_bounds.comparison, "<" | "<=") {
        e.push("response_bounds does not say `<` or `<=`".into());
    }
    if m.response_bounds.population.is_empty() {
        e.push("response_bounds names no union-bound population".into());
    }
    if m.recovery.d_drop == 0 || m.recovery.limit == 0 {
        e.push("recovery carries no compression or limit".into());
    }
    // The two omissions, and the carries that replace them.  Counted
    // rather than described: `omitted_coordinates` was a single prose cell
    // reading "none", while the transmitted rank was 13 against
    // `kappa = 17` — four ring elements, 1024 coefficients, omitted.
    if m.recovery.omitted_response_coefficients
        != m.recovery.omitted_response_rows * m.dimensions.d_tilde
    {
        e.push("recovery: omitted response coefficients are not rows x d~".into());
    }
    if m.recovery.recovery_carries != m.rank_roles.identity_rank * m.dimensions.d_tilde {
        e.push("recovery: the carry count is not l~ d~".into());
    }
    if m.recovery.omitted_t0_low_bits
        != m.rank_roles.identity_rank * m.dimensions.d_tilde * m.recovery.d_drop as usize
    {
        e.push("recovery: the dropped t_0 low bits are not l~ d~ D".into());
    }
    for (label, text) in [
        ("recovery.rounding", m.recovery.rounding),
        ("recovery.ties", m.recovery.ties),
        ("recovery.hint_alphabet", m.recovery.hint_alphabet),
        ("recovery.failure_rule", m.recovery.failure_rule),
        ("recovery.verification_rule", m.recovery.verification_rule),
        ("recovery.encoding", m.recovery.encoding),
        ("wire.kb_convention", m.wire.kb_convention),
        ("estimator.hint_mlwe_inputs", m.estimator.hint_mlwe_inputs),
        ("estimator.hint_mlwe_outputs", m.estimator.hint_mlwe_outputs),
        ("estimator.msis_inputs", m.estimator.msis_inputs),
        ("estimator.msis_outputs", m.estimator.msis_outputs),
        ("estimator.challenge", m.estimator.challenge),
    ] {
        if text.is_empty() {
            e.push(format!("{label} is empty"));
        }
    }
    if m.transcript.is_empty() {
        e.push("transcript lists no fields".into());
    }
    if m.wire.fields.is_empty() {
        e.push("wire lists no fields".into());
    }
    // `total_bits == 0` means *no fixed total*, which is a fact about a
    // Rice-coded layout rather than a gap — the `z` field's size is
    // sample-dependent.  It has to be *recorded*, though, or a manifest
    // with no size accounting at all reads the same way; that is what
    // `discrepancy` is for.  Conflating the two was why the generated
    // table read as "wire lists no fields".
    if m.wire.fixed_bits == 0 {
        e.push("wire.fixed_bits is zero: no field has a fixed width".into());
    }
    if m.recovery.omitted_response_rows == 0 && m.rank_roles.response_rank != m.rank_roles.kappa {
        e.push(format!(
            "recovery omits no response rows but the transmitted rank is {} \
             against kappa = {}",
            m.rank_roles.response_rank, m.rank_roles.kappa
        ));
    }
    if let Some(total) = m.wire.total_bits {
        let summed: u64 = m.wire.fields.iter().filter_map(|f| f.bits).sum();
        if summed != total {
            e.push(format!(
                "wire.total_bits is {total} but its fields sum to {summed}"
            ));
        }
        if total != LANES_STATED_BITS && m.wire.discrepancy.is_none() {
            e.push(format!(
                "wire.total_bits is {:.4} KB against the stated {:.1} KB, and \
                 wire.discrepancy does not record it",
                total as f64 / BITS_PER_KB as f64,
                LANES_STATED_BITS as f64 / BITS_PER_KB as f64
            ));
        }
    } else if m.wire.discrepancy.is_none() {
        e.push(
            "wire.total_bits is absent and wire.discrepancy does not say \
             why — a manifest with no size accounting must say so"
                .into(),
        );
    }
    // The fixed-width half must sum, whether or not the whole does.
    let fixed: u64 = m.wire.fields.iter().filter_map(|f| f.bits).sum();
    if fixed != m.wire.fixed_bits {
        e.push(format!(
            "wire.fixed_bits is {} but its fixed fields sum to {fixed}",
            m.wire.fixed_bits
        ));
    }
    if m.rounds.is_empty() {
        e.push("transcript records no rounds".into());
    }
    if m.source_sha256.len() != 64 || !m.source_sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        e.push("source_sha256 is not a SHA-256 digest".into());
    }

    // The one that matters most: a label is not a selection.
    for (name, _, consumed) in live {
        match m.constants.iter().find(|c| c.name == *name) {
            None => e.push(format!(
                "{name} is still live and the manifest does not select it"
            )),
            Some(c) => {
                if !matches!(c.provenance, "Paper" | "Derived" | "Repair") {
                    e.push(format!("{name} carries no Paper/Derived/Repair provenance"));
                }
                if c.value.1 == 0 {
                    e.push(format!("{name}: manifest value has a zero denominator"));
                } else if !rational_eq(c.value, *consumed) {
                    e.push(format!(
                        "{name}: the manifest selects {} but the code consumes {}",
                        show(c.value),
                        show(*consumed)
                    ));
                }
            }
        }
    }
    e
}

/// The reason at the current state, for a `#[test]` that must skip.
///
/// Printing and returning is deliberate: a silently skipped test set is
/// how coverage shrinks unnoticed, and `cargo test` has no first-class
/// skip.  `lanes::ring`'s tests do **not** call it — the ring is current,
/// and its KAT block is active rather than withheld.
pub fn lanes_skip_reason() -> Option<String> {
    lanes_unavailable_reason()
}

// ---- the opening backend -------------------------------------------------

/// Honest-opening backend: complete and binding, **not** zero knowledge.
///
/// `Com` is a real BDLOP commitment over `R_q~` with the paper's ranks,
/// and `verify` re-derives the commitment and checks every clause of
/// `R^_ex`.  What it does not do is hide the witness: `sigma_ex` *is* the
/// opening, so `e_eval` leaks and with it `<G(m), s>` — the secret key
/// falls out after about `ell` distinct messages.  It is here to bracket
/// the problem, not to be used.
pub struct OpeningBackend {
    ex: ExactParams,
    ck: ExactCommitmentKey,
    /// Bound on `y_eval` in the wire format.
    ///
    /// **Not** the verifier's `6 sigma_m`.  That bounds `z_eval`, and
    /// `y_eval = z_eval - x e_eval`, so an accepted transcript reaches
    /// `6 sigma_m + ||x||_1 ||e_eval||_inf = 6 sigma_m + w gamma B_e`.
    /// Using `6 sigma_m` alone left the honest prover able to produce a
    /// proof its own serializer refused — reachable with probability about
    /// `6.3e-8` per proof, which is why it was never observed.  Widening
    /// costs nothing on the wire: a Rice codeword depends only on `k`, so
    /// no encodable value moves and only [`proof_bytes`] grows.
    ///
    /// `z_eval` is the last coordinate of the `z_m` block, so its width is
    /// `sigma_m` and its verifier bound is `zm_inf_bound`.
    ///
    /// [`proof_bytes`]: OpeningBackend::proof_bytes
    pub bound_y: i64,
    w_layout: Layout,
    proof_layout: Layout,
}

impl OpeningBackend {
    pub const NAME: &'static str = "opening";

    /// The exact-layer parameters this backend was built for.
    pub fn ex(&self) -> &ExactParams {
        &self.ex
    }

    /// `Err` if the exact parameters do not validate at this profile.
    pub fn new(par: RiVeRParams, seed: &[u8]) -> Result<Self, String> {
        let ex = ExactParams::new(&par)?;
        let ck = ExactCommitmentKey::new(&ex, seed);
        let bound_y = par.zm_inf_bound_sq().floor_sqrt() as i64
            + (par.w as i64) * (par.gamma as i64) * par.B_e() as i64;
        let qt = Coder::uniform(ex.q_tilde);
        // The Rice parameter comes from the frozen manifest when the
        // profile has one; `manifest`'s own tests re-derive it.
        let y_coder = match crate::manifest::for_params(&par) {
            Some(m) => Coder::rice_with_k(m.exact.y_eval.rice_k, bound_y),
            None => Coder::rice_sigma(par.sigma_m(), bound_y),
        };

        let w_layout = Layout::new(vec![
            Field::rows("t0", qt, ex.d_tilde, ex.t0_rows),
            Field::rows("t1", qt, ex.d_tilde, ex.n_ex),
        ]);
        // Each field gets the coder its distribution asks for.  `y_eval` is
        // a Gaussian coordinate of the OOM mask, so Rice; the digits and
        // the randomness are tiny and uniform, so 2 bits each rather than
        // the byte a fixed-width field would round up to.
        let proof_layout = Layout::new(vec![
            Field::rows("t0", qt, ex.d_tilde, ex.t0_rows),
            Field::rows("t1", qt, ex.d_tilde, ex.n_ex),
            Field::flat("e_eval", Coder::signed(par.B_e() as i64), ex.d),
            Field::flat("y_eval", y_coder, ex.d),
            Field::rows(
                "digits",
                Coder::uniform(RADIX_DIGITS),
                ex.d,
                RADIX_WEIGHTS.len(),
            ),
            Field::ring_rows(
                "randomness",
                Coder::signed(1),
                ex.d_tilde,
                ex.kappa,
                ex.q_tilde,
            ),
        ]);

        Ok(Self {
            ex,
            ck,
            bound_y,
            w_layout,
            proof_layout,
        })
    }

    /// `(W, st) <- Pi_ex.Com(w_ex)`; the statement is not known yet.
    ///
    /// `None` when `e_eval` is outside `[-30, 30]`, which is a witness the
    /// relation does not admit rather than a failure of the commitment.
    pub fn com(
        &self,
        witness: &ExactWitness,
        xof: &mut Xof,
    ) -> Option<(ExactCommitment, OpeningState)> {
        let ex = &self.ex;
        let canonical: Vec<i64> = witness
            .e_eval
            .iter()
            .map(|&c| c + ex.par.q0 as i64 / 2)
            .collect();
        let digits = decompose_poly(&canonical)?;
        let message = pack_witness(ex, &witness.e_eval, &witness.y_eval, &digits)?;
        let randomness = uniform_beta_vec(xof, 1, ex.d_tilde, ex.kappa, ex.q_tilde);
        let w = self.ck.commit(&ex.rt, &message, &randomness);
        Some((w, OpeningState { digits, randomness }))
    }

    /// Reveal the opening.  A LANES backend proves it in zero knowledge.
    pub fn prove(&self, witness: &ExactWitness, state: &OpeningState) -> OpeningProof {
        OpeningProof {
            e_eval: witness.e_eval.clone(),
            y_eval: witness.y_eval.clone(),
            digits: state.digits.clone(),
            randomness: state.randomness.clone(),
        }
    }

    /// `Pi_ex.Ver`.  Total on `proof`: every shape is checked before use.
    pub fn verify(&self, statement: &ExactStatement<'_>, proof: &OpeningProof) -> bool {
        let ex = &self.ex;
        if proof.randomness.len() != ex.kappa
            || proof.randomness.iter().any(|p| p.len() != ex.d_tilde)
            || proof
                .randomness
                .iter()
                .any(|p| p.iter().any(|&c| c >= ex.q_tilde))
        {
            return false;
        }
        // Canonicality **before** centring, and not only for tidiness:
        // `Ring::centered` assumes its input is already in `[0, q~)`, so
        // `q~` centres to `0` and `q~ + 1` to `1` — both of which would pass
        // the ternary test below.  Adding that test is what made this check
        // load-bearing; dropping it at the same time was a regression.
        //
        // The randomness must be **ternary**, not merely canonical.
        //
        // BDLOP binding is a statement about *short* openings: with an
        // unbounded `r` the commitment binds nothing, since `A_1 r` can be
        // steered anywhere.  The wire format enforces this — the layout
        // declares `randomness` as `Coder::signed(1)` over the ring, so a
        // larger value cannot be decoded — but a proof handed in as a value
        // never passed a decoder.  Checking it here is what makes this
        // function's binding claim true of its own argument rather than of
        // the bytes some caller might have had.
        if proof
            .randomness
            .iter()
            .any(|p| ex.rt.centered(p).iter().any(|&c| !(-1..=1).contains(&c)))
        {
            return false;
        }
        let witness = ExactWitness {
            e_eval: proof.e_eval.clone(),
            y_eval: proof.y_eval.clone(),
        };
        if !check_relation(ex, statement, &witness, &proof.digits).is_empty() {
            return false;
        }
        let Some(message) = pack_witness(ex, &proof.e_eval, &proof.y_eval, &proof.digits) else {
            return false;
        };
        &self.ck.commit(&ex.rt, &message, &proof.randomness) == statement.w
    }

    // -- encoding ----------------------------------------------------------

    /// `t_0` and `t_1` are uniform mod `q~` and the reference declares them
    /// without a ring, so they encode as plain integers under a `Uniform`
    /// coder — which range-checks them exactly as a ring field's
    /// canonicality check would.  Only `randomness` is a ring field, because
    /// it is centred in transit.
    fn w_fields(w: &ExactCommitment) -> Vec<FieldValue> {
        vec![as_ints(&w.t0), as_ints(&w.t1)]
    }

    pub fn w_encode(&self, w: &ExactCommitment) -> CodecResult<Vec<u8>> {
        self.w_layout.encode(&Self::w_fields(w))
    }

    pub fn w_decode(&self, data: &[u8]) -> CodecResult<ExactCommitment> {
        let mut f = self.w_layout.decode(data)?.into_iter();
        let t0 = as_residues(f.next().unwrap());
        let t1 = as_residues(f.next().unwrap());
        Ok(ExactCommitment { t0, t1 })
    }

    /// `W` is all uniform, so this is exact rather than a bound.
    pub fn w_bytes(&self) -> usize {
        self.w_layout.max_bytes()
    }

    /// `pi_ex = (W, sigma_ex)`.
    pub fn proof_encode(&self, w: &ExactCommitment, sigma: &OpeningProof) -> CodecResult<Vec<u8>> {
        self.proof_layout.encode(&[
            as_ints(&w.t0),
            as_ints(&w.t1),
            FieldValue::flat(sigma.e_eval.clone()),
            FieldValue::flat(sigma.y_eval.clone()),
            FieldValue::Ints(sigma.digits.clone()),
            FieldValue::Residues(sigma.randomness.clone()),
        ])
    }

    pub fn proof_decode(&self, data: &[u8]) -> CodecResult<(ExactCommitment, OpeningProof)> {
        let mut f = self.proof_layout.decode(data)?.into_iter();
        let t0 = as_residues(f.next().unwrap());
        let t1 = as_residues(f.next().unwrap());
        let e_eval = flat_ints(f.next().unwrap());
        let y_eval = flat_ints(f.next().unwrap());
        let digits = ints(f.next().unwrap());
        let randomness = residues(f.next().unwrap());
        Ok((
            ExactCommitment { t0, t1 },
            OpeningProof {
                e_eval,
                y_eval,
                digits,
                randomness,
            },
        ))
    }

    /// The layout `pi_ex` is framed against, for [`crate::codec::proof_unframe`].
    pub fn proof_layout(&self) -> &Layout {
        &self.proof_layout
    }

    /// Worst-case `|pi_ex|`.
    ///
    /// Rice-coding `y_eval` makes the real length sample-dependent, so
    /// this is an upper bound; measure with [`OpeningBackend::proof_encode`].
    pub fn proof_bytes(&self) -> usize {
        self.proof_layout.max_bytes()
    }
}

fn residues(v: FieldValue) -> PolyVec {
    match v {
        FieldValue::Residues(r) => r,
        FieldValue::Ints(_) => unreachable!("layout field is a ring field"),
    }
}

/// Residues as the plain integers a non-ring `Uniform` field wants.
fn as_ints(v: &[Poly]) -> FieldValue {
    FieldValue::Ints(
        v.iter()
            .map(|p| p.iter().map(|&c| c as i64).collect())
            .collect(),
    )
}

fn as_residues(v: FieldValue) -> PolyVec {
    ints(v)
        .into_iter()
        .map(|row| row.into_iter().map(|c| c as u64).collect())
        .collect()
}

fn ints(v: FieldValue) -> Vec<Vec<i64>> {
    match v {
        FieldValue::Ints(r) => r,
        FieldValue::Residues(_) => unreachable!("layout field is an integer field"),
    }
}

fn flat_ints(v: FieldValue) -> Vec<i64> {
    ints(v).into_iter().next().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{RIVER_N8, RIVER_TOY};
    use crate::sample::{
        challenge_from_hash, gaussian_int, rational_sigma, uniform_int, Part, DS_EXACT,
    };

    /// A witness the relation admits, drawn the way `Eval` would draw it.
    fn witness_for(par: &RiVeRParams, label: &[u8]) -> (ExactWitness, Vec<i64>, Vec<i64>) {
        let mut x = Xof::new(DS_EXACT, &[Part::Bytes(label)]);
        let e_eval: Vec<i64> = (0..par.d)
            .map(|_| uniform_int(&mut x, par.q0) as i64 - par.B_e() as i64)
            .collect();
        let (num, den) = rational_sigma(par.sigma_m());
        let y_eval: Vec<i64> = (0..par.d)
            .map(|_| gaussian_int(&mut x, num, den, crate::sample::GAUSSIAN_TAILCUT))
            .collect();
        let x_hat = challenge_from_hash(par.d, par.w, par.gamma, par.q_hat, &[Part::Bytes(label)]);
        let half = par.q_hat / 2;
        let x_c: Vec<i64> = x_hat
            .iter()
            .map(|&c| {
                if c > half {
                    c as i64 - par.q_hat as i64
                } else {
                    c as i64
                }
            })
            .collect();
        let prod = Ring::mul_int(
            &x_c.iter().map(|&c| c as i128).collect::<Vec<_>>(),
            &e_eval.iter().map(|&c| c as i128).collect::<Vec<_>>(),
        );
        let z_eval: Vec<i64> = (0..par.d)
            .map(|i| (prod[i] + y_eval[i] as i128) as i64)
            .collect();
        (ExactWitness { e_eval, y_eval }, x_c, z_eval)
    }

    /// The radix encoding covers `[0, 60]` and nothing else.
    #[test]
    fn radix_covers_exactly_the_range() {
        let mut reachable = std::collections::BTreeSet::new();
        for a in 0..3i64 {
            for b in 0..3i64 {
                for c in 0..3i64 {
                    for e in 0..3i64 {
                        reachable.insert(radix_recompose(&[a, b, c, e]));
                    }
                }
            }
        }
        assert_eq!(reachable, (0..=60).collect());
        assert_eq!(2 * RADIX_WEIGHTS.iter().sum::<i64>(), 60);
        for v in 0..=60i64 {
            let d = radix_decompose(v).unwrap();
            assert!(d.iter().all(|&a| (0..=2).contains(&a)));
            assert_eq!(radix_recompose(&d), v);
        }
        for bad in [-1i64, 61, 1000, i64::MIN] {
            assert!(radix_decompose(bad).is_none(), "accepted {bad}");
        }
        // not injective, which is harmless and worth pinning
        assert_eq!(
            radix_recompose(&[0, 0, 0, 1]),
            radix_recompose(&[2, 2, 1, 0])
        );
    }

    /// The final figures use the centred relation
    /// `e_eval + B_e = sum_j g_j d_j`.
    ///
    /// The main relation paragraph still says `e_eval in [0,60]`; this test
    /// pins the executable figures' coherent reading.
    #[test]
    fn the_reconstruction_equation_is_the_centred_one() {
        let reachable: std::collections::BTreeSet<i64> = (0..3i64)
            .flat_map(|a| {
                (0..3i64).flat_map(move |b| {
                    (0..3i64)
                        .flat_map(move |c| (0..3i64).map(move |e| radix_recompose(&[a, b, c, e])))
                })
            })
            .collect();
        let centered: std::collections::BTreeSet<i64> =
            reachable.iter().map(|&v| v - WEIGHT_SUM).collect();
        assert_eq!(centered, (-30..=30).collect());
        for e in centered {
            assert_eq!(
                radix_recompose(&radix_decompose(e + WEIGHT_SUM).unwrap()),
                e + WEIGHT_SUM
            );
        }
    }

    // ---- the provenance audit ----------------------------------------

    /// No *active* constant may have drifted from the paper's closed form
    /// while the backend is available.
    ///
    /// The audit is executable rather than narrated: the live values are
    /// the paper's, and what it enforces is that none has drifted away
    /// from them.  The gate stays closed either way — on security
    /// evidence now rather than on parameters.
    #[test]
    fn the_gate_cannot_lift_while_a_constant_has_drifted() {
        let live = live_lanes_constants();
        assert_eq!(
            live.len(),
            GATED_LANES_CONSTANTS.len(),
            "the live LANES constants and gated-constant list differ"
        );
        assert!(lanes_unavailable_reason().is_some(), "the gate is closed");

        // Every live value is the paper's closed form.  Only a value
        // comparison can tell: several of these coincide across sets that
        // differ elsewhere, so a name check alone proves nothing.
        for (name, _, value) in &live {
            let want = GATED_LANES_CONSTANTS
                .iter()
                .find(|(n, _, _)| n == name)
                .unwrap()
                .2;
            assert!(rational_eq(*value, want), "{name}: {value:?} vs {want:?}");
        }
    }

    /// Moving a dimension must not make the LANES backend available.
    ///
    /// The predecessor of this gate compared [`ExactParams`]'s dimensions
    /// with a fixed profile's, which could only ever answer
    /// "gated" — and the one way to make it answer "available" was to move
    /// the exact parameters *backwards*, which is the opposite of
    /// readiness.  So the property worth pinning is that dimensions do not
    /// enter the decision at all.
    #[test]
    fn the_gate_is_readiness_not_a_dimension_diff() {
        let baseline = lanes_unavailable_reason().expect("closed");

        type Move = fn(&mut ExactParams);
        let moves: [(&str, Move); 6] = [
            ("d~", |e| e.d_tilde = 128),
            ("l", |e| e.l_split = 32),
            ("q~", |e| e.q_tilde = 427_634_113),
            ("n~", |e| e.n_tilde = 7),
            ("l~", |e| e.ell_tilde = 8),
            ("D", |e| e.d_drop = 13),
        ];
        for (label, mv) in moves {
            let mut ex = ExactParams::unchecked(RIVER_TOY);
            mv(&mut ex);
            assert_eq!(
                lanes_unavailable_reason().as_deref(),
                Some(baseline.as_str()),
                "moving {label} changed the gate's verdict"
            );
        }
    }

    /// A manifest carrying real data in every section.
    ///
    /// Built here rather than shipped: there is no LANES manifest, and the
    /// point is that the acceptance rule is executable before one arrives.
    /// The values are placeholders in the sense that nobody selected them
    /// — but they are *values*, and `constants` selects exactly what the
    /// code consumes, which is what the gate compares.
    /// A manifest that passes every check: **the shipped one**.
    ///
    /// It used to be hand-written, which meant it drifted from the shape
    /// the validator actually meets the moment a section was added — and
    /// it was the fixture, so nothing caught that.  Now that
    /// `LANES_PARAMETER_MANIFEST` is generated from
    /// `river-py/lanes_manifest.json`, the fixture is that table and the
    /// hollowing tests below mutate copies of it.
    ///
    /// `LanesManifest` is `Copy`, so a mutation here cannot reach the
    /// shipped const.
    fn valid_manifest() -> LanesManifest {
        *LANES_PARAMETER_MANIFEST.expect("the generated manifest is present")
    }

    /// A provenance label is not a parameter selection.
    ///
    /// Marking a wrong width as **Paper** does not make the paper
    /// have printed it.  What the gate compares is the manifest's value
    /// against the value the code actually consumes, so a manifest
    /// selecting a *different* value reports that the code has not been
    /// updated — which is the useful failure, not a silent pass.
    #[test]
    fn re_selection_compares_values_not_just_labels() {
        let live = live_lanes_constants();

        // A wrong value, relabelled Paper: the exact
        // failure mode this check exists for.
        static MOVED: [ManifestConstant; 1] = [ManifestConstant {
            name: "Z_INF_BOUND",
            value: (6_691, 1),
            provenance: "Paper",
        }];
        let mut m = valid_manifest();
        m.constants = &MOVED;
        let bad = validate_lanes_manifest(&m, &live);
        assert!(
            bad.iter()
                .any(|b| b
                    .contains("Z_INF_BOUND: the manifest selects 6691 but the code consumes 3448")),
            "{bad:?}"
        );
        // and the five it no longer mentions are reported too
        assert_eq!(
            bad.iter().filter(|b| b.contains("does not select")).count(),
            5
        );

        // a denominator that differs is a different value
        static WRONG_DEN: [ManifestConstant; 1] = [ManifestConstant {
            name: "SIGMA_R",
            value: (2_901_189, 524_289),
            provenance: "Derived",
        }];
        let mut m = valid_manifest();
        m.constants = &WRONG_DEN;
        assert!(validate_lanes_manifest(&m, &live)
            .iter()
            .any(|b| b.contains("SIGMA_R: the manifest selects 2901189/524289")));

        // a value with no provenance label
        static UNLABELLED: [ManifestConstant; 1] = [ManifestConstant {
            name: "Z_INF_BOUND",
            value: (3_448, 1),
            provenance: "",
        }];
        let mut m = valid_manifest();
        m.constants = &UNLABELLED;
        assert!(validate_lanes_manifest(&m, &live)
            .iter()
            .any(|b| b.contains("Z_INF_BOUND carries no Paper/Derived/Repair")));
    }

    /// `wire.total_bits` has to sum from its fields, and either reproduce
    /// the stated size or say it cannot.
    ///
    /// The shipped layout has *no* fixed total — `z` is Rice-coded, so its
    /// size is sample-dependent — which is recorded as `None` plus a
    /// `discrepancy`.  It used to be recorded as `0`, which the validator
    /// could not tell from "no fields at all".
    #[test]
    fn the_wire_total_must_reproduce_or_record_the_stated_size() {
        let live = live_lanes_constants();

        // As shipped: no total, and a discrepancy that says why.
        let shipped = valid_manifest();
        assert!(shipped.wire.total_bits.is_none());
        assert!(shipped.wire.discrepancy.is_some());
        assert_eq!(
            validate_lanes_manifest(&shipped, &live),
            Vec::<String>::new()
        );

        // No total and no explanation is not allowed.
        let mut silent = valid_manifest();
        silent.wire.discrepancy = None;
        assert!(validate_lanes_manifest(&silent, &live)
            .iter()
            .any(|b| b.contains("wire.discrepancy does not say why")));

        // A total that does not sum from its fields.
        let mut mismatched = valid_manifest();
        mismatched.wire.total_bits = Some(99_999);
        assert!(validate_lanes_manifest(&mismatched, &live)
            .iter()
            .any(|b| b.contains("but its fields sum to")));

        // `fixed_bits` has to sum too, whether or not the whole does.
        let mut wrong_fixed = valid_manifest();
        wrong_fixed.wire.fixed_bits += 1;
        assert!(validate_lanes_manifest(&wrong_fixed, &live)
            .iter()
            .any(|b| b.contains("wire.fixed_bits is")));

        // The exact figure, summed from its fields, needs no discrepancy.
        static EXACT: [WireField; 1] = [WireField {
            name: "all",
            rows: 1,
            cols: 1,
            coder: "Uniform",
            bits: Some(LANES_STATED_BITS),
            modulus: Some(2),
            bound: None,
            width_bits: Some(1),
            rice_k: None,
        }];
        let mut good = valid_manifest();
        good.wire = WireSpec {
            fields: &EXACT,
            total_bits: Some(LANES_STATED_BITS),
            fixed_bits: LANES_STATED_BITS,
            kb_convention: "1 KB = 8192 bits",
            discrepancy: None,
        };
        assert_eq!(validate_lanes_manifest(&good, &live), Vec::<String>::new());
    }

    /// The stated size is in **bits**, and is 13.5 KB.
    ///
    /// It was `13_824` — 13.5 KB in *bytes* — while being compared against
    /// `wire.total_bits`, a bit count.  Nothing depended on it while the
    /// total was recorded as absent, but the comparison it exists for was
    /// off by a factor of eight.
    #[test]
    fn the_stated_size_is_in_bits() {
        assert_eq!(LANES_STATED_BITS, 110_592);
        assert_eq!(LANES_STATED_BITS, LANES_STATED_BYTES * 8);
        assert_eq!(LANES_STATED_BITS as f64 / BITS_PER_KB as f64, 13.5);
        assert_eq!(LANES_STATED_BYTES as f64 / 1024.0, 13.5);
    }

    /// The projection carries what a port needs, and says what it is a
    /// projection *of*.
    #[test]
    fn the_generated_projection_is_complete_enough_to_encode_from() {
        let m = valid_manifest();

        // A coder name is not enough: every field carries its parameters.
        for f in m.wire.fields {
            match f.coder {
                "Uniform" => {
                    assert!(f.modulus.is_some(), "{}: no modulus", f.name);
                    assert!(f.width_bits.is_some(), "{}: no width", f.name);
                    assert_eq!(
                        f.bits,
                        Some((f.rows * f.cols) as u64 * f.width_bits.unwrap() as u64),
                        "{}",
                        f.name
                    );
                }
                "Signed" => {
                    assert!(f.bound.is_some(), "{}: no bound", f.name);
                    assert!(f.width_bits.is_some(), "{}: no width", f.name);
                }
                "Rice" => {
                    assert!(f.rice_k.is_some(), "{}: no k", f.name);
                    assert!(f.bound.is_some(), "{}: no cap", f.name);
                    assert!(f.bits.is_none(), "{}: Rice is variable", f.name);
                }
                other => panic!("unhandled coder {other} on {}", f.name),
            }
            assert!(f.rows >= 1, "{}: rows recorded as 0", f.name);
        }

        // The Rice cap on `z` *is* the response infinity bound, not a
        // second bound that happens to be close to it.  The two cells come
        // from different bindings on the Python side -- the cap from the
        // serializer's coder, `inf` from `lanes_params` -- so nothing but
        // this compares them, and a divergence would be a manifest that
        // contradicts itself while each half matched its own source.  It
        // is also the wire statement of the same defect: a coefficient the coder
        // accepts but the verifier rejects serializes and does not verify.
        let z = m
            .wire
            .fields
            .iter()
            .find(|f| f.name == "z")
            .expect("no `z` field in the layout");
        assert_eq!(z.coder, "Rice");
        assert_eq!(
            z.bound.map(u128::from),
            Some(m.response_bounds.inf),
            "the Rice cap on `z` and response_bounds.inf disagree"
        );
        assert_eq!(
            m.response_bounds.inf,
            crate::lanes::params::Z_INF_BOUND as u128
        );

        // The dimensions are here, not only the ranks derived from them.
        assert_eq!(m.dimensions.d_tilde, crate::lanes::ring::DTILDE);
        assert_eq!(m.dimensions.l_split, crate::lanes::ring::LSPLIT);
        assert_eq!(m.dimensions.q_tilde, crate::lanes::ring::QTILDE);
        assert_eq!(m.dimensions.d_drop, crate::lanes::params::D_DROP);
        assert_eq!(m.dimensions.w_hat, crate::lanes::params::W_HAT);

        // The three rounds, with which challenge each precedes.
        assert_eq!(m.rounds.len(), 3);
        let names: Vec<&str> = m.rounds.iter().map(|r| r.challenge).collect();
        assert_eq!(names, ["alpha", "gamma", "c"]);
        for r in m.rounds {
            assert!(!r.separator.is_empty(), "{}: no separator", r.challenge);
            assert!(!r.absorbs.is_empty(), "{}: absorbs nothing", r.challenge);
        }
        // ...and flattening them gives the field list, so the two views
        // cannot disagree.
        let flat: Vec<&str> = m
            .rounds
            .iter()
            .flat_map(|r| r.absorbs.iter().copied())
            .collect();
        let listed: Vec<&str> = m.transcript.iter().map(|f| f.name).collect();
        assert_eq!(flat, listed);

        // And it says what it is a projection of.
        assert_eq!(m.source_sha256.len(), 64);
    }

    /// The parameter manifest and production-alias policy are separate.
    #[test]
    fn a_manifest_alone_does_not_enable_the_backend() {
        let m = valid_manifest();

        // A valid manifest does not by itself enable the reserved alias.
        let reason = lanes_readiness(Some(&m), false).expect("still gated");
        assert!(
            reason.contains("production LANES alias is reserved"),
            "{reason}"
        );
        assert!(
            lanes_readiness(Some(&m), true).is_some(),
            "security outranks ready"
        );

        assert!(
            lanes_readiness(None, true).is_some(),
            "ready without a manifest"
        );

        // The generated manifest is present and valid in the shipped tree.
        assert!(LANES_PARAMETER_MANIFEST.is_some());
        let bad =
            validate_lanes_manifest(LANES_PARAMETER_MANIFEST.unwrap(), &live_lanes_constants());
        assert!(
            bad.is_empty(),
            "the generated manifest must validate: {bad:?}"
        );
        assert_eq!(lanes_gate_cause(), Some("production-alias-reserved"));
        // deliberately tests and not `const` assertions: these flags are
        // meant to move, and when they do these should go red rather than
        // refuse to compile.
        //
        // Implementation readiness and the production-alias policy remain
        // separate conditions.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(LANES_BACKEND_READY, "implementation gates pass");
            assert!(!LANES_SECURITY_MEETS_TARGET, "production alias is reserved");
        }
    }

    /// Every token the gate can reach, and the order it reaches them in.
    ///
    /// The order matters: a later blocker must not hide an earlier one.
    /// A tree whose constants have been retuned should say so, not report
    /// the missing manifest and let the retune through with it.
    #[test]
    fn the_gate_names_the_first_blocker_not_the_last() {
        let m = valid_manifest();
        let live = live_lanes_constants();

        assert_eq!(
            lanes_gate_cause_for(None, false),
            Some("no-parameter-manifest")
        );
        assert_eq!(
            lanes_gate_cause_for(None, true),
            Some("no-parameter-manifest")
        );
        // The security verdict sits between the manifest and readiness, so
        // with it down a valid manifest reports *it*, not `backend-not-ready`.
        assert_eq!(
            lanes_gate_cause_for(Some(&m), false),
            Some("production-alias-reserved")
        );
        assert_eq!(
            lanes_gate_cause_for(Some(&m), true),
            Some("production-alias-reserved")
        );

        let mut bad = valid_manifest();
        bad.rank_roles.kappa = 0;
        assert_eq!(
            lanes_gate_cause_for(Some(&bad), true),
            Some("manifest-invalid")
        );

        // and every token is one this side actually uses, bar the ones it
        // documents it cannot reach
        for cause in [
            lanes_gate_cause_for(None, false),
            lanes_gate_cause_for(Some(&m), false),
            lanes_gate_cause_for(Some(&bad), true),
        ]
        .into_iter()
        .flatten()
        {
            assert!(LANES_GATE_CAUSES.contains(&cause), "{cause} is not a token");
        }
        assert!(live
            .iter()
            .zip(GATED_LANES_CONSTANTS.iter())
            .all(|((_, _, now), (_, _, pinned))| rational_eq(*now, *pinned)));
    }

    #[test]
    fn parameters_match_the_paper() {
        for par in [RIVER_TOY, RIVER_N8] {
            let ex = ExactParams::new(&par).expect("shipped profile");
            assert_eq!(
                (
                    ex.d_tilde(),
                    ex.l_split(),
                    ex.n_tilde(),
                    ex.ell_tilde(),
                    ex.d_drop(),
                    ex.n_ex(),
                    ex.w_hat()
                ),
                (256, 64, 4, 4, 17, 6, 44)
            );
            assert_eq!(ex.q_tilde(), 67_107_713);
            assert_eq!(ex.q_tilde() % 256, 129, "q~ = 2l+1 mod 4l");
            assert_eq!(ex.kappa(), 17);
            assert_eq!(ex.response_rank(), 13);
            // `t_0` has `l~` rows, not `n~` — see `rank_roles`.
            // Both are 4 here, so the *assignment* is what this pins.
            assert_eq!(ex.t0_rows(), ex.roles().identity_rank);
            assert_eq!(ex.roles().identity_rank, ex.ell_tilde());
            assert_eq!(ex.roles().tail_rank, ex.n_tilde());
            // Six 64-slot message blocks, 32 carried and 32 padding.  The
            // old `6 d == N_ex l` identity is gone: 192 != 384.
            assert_eq!((ex.block_slots(), ex.block_used()), (64, 32));
            assert_ne!(ex.n_ex() * ex.d(), ex.n_ex() * ex.l_split());
            assert!((1u64 << 25..1u64 << 26).contains(&ex.q_tilde()));
            assert!(ex.check().is_empty());
            // The no-wrap condition, with 0.56% to spare — and only
            // because `B_e = 30`.  With the *unshifted* range bound 60 the
            // requirement doubles and this modulus fails outright, which
            // is what makes `ring::to_centered_error` load-bearing.
            assert!(ex.q_tilde_clears(par.B_e()), "q~ fails at B_e");
            assert!(
                !ex.q_tilde_clears(par.q0 - 1),
                "q~ still clears at the unshifted bound — the centred \
                 range shift would no longer be load-bearing"
            );
            assert!(ex.q_tilde() as f64 > ex.q_tilde_need());
        }
    }

    /// `check` is fail-closed, and each clause is reachable.
    #[test]
    fn check_rejects_a_bad_modulus() {
        // `q~ - 512`: still `129 mod 256`, still 26 bits, still above the
        // no-wrap bound — and divisible by 3, which only the primality
        // test catches.
        let mut ex = ExactParams::unchecked(RIVER_TOY);
        ex.q_tilde = 67_107_713 - 512;
        assert_eq!(ex.q_tilde % 256, 129);
        assert_eq!(ex.q_tilde % 3, 0);
        let bad = ex.check();
        assert!(bad.iter().any(|m| m.contains("not prime")), "{bad:?}");
        // and two more now: the field no longer equals its constant, and
        // `rt` was built from the old value — which is the point of
        // pinning fields to constants rather than only to each other
        assert!(
            bad.iter().any(|m| m.contains("q~ = 67107201 !=")),
            "{bad:?}"
        );
        assert!(
            bad.iter()
                .any(|m| m.contains("commitment ring does not match")),
            "{bad:?}"
        );

        // The margin is 0.56%, so a `q~` one splitting-step below the
        // selected one is still prime-shaped and still 26 bits and still
        // fails the no-wrap condition.  33553921 = 2^25 + 129 mod 256.
        let mut thin = ExactParams::unchecked(crate::params::RIVER_N256);
        thin.q_tilde = 33_554_561; // 2^25 + 641: 26 bits shy, 129 mod 256
        let bad = thin.check();
        assert!(
            bad.iter().any(|m| m.contains("internal modulus condition")),
            "{bad:?}"
        );

        let mut composite = ExactParams::unchecked(RIVER_TOY);
        composite.q_tilde = 1 << 25; // not prime, wrong residue
        let bad = composite.check();
        assert!(bad.iter().any(|m| m.contains("not prime")));
        assert!(bad.iter().any(|m| m.contains("2l+1 mod 4l")));

        // and the paper's own constant clears everything.
        let printed = ExactParams::unchecked(crate::params::RIVER_N256);
        assert!(printed.check().is_empty(), "{:?}", printed.check());
    }

    /// A field moved after construction is caught, because `check`
    /// compares fields to their constants and not only to each other.
    #[test]
    fn a_field_moved_after_construction_is_caught() {
        let good = ExactParams::new(&RIVER_TOY).expect("shipped profile");
        assert!(good.check().is_empty());

        type Mutate = fn(&mut ExactParams);
        let mutations: [(&str, Mutate); 7] = [
            ("d~", |e| e.d_tilde = 128),
            ("l", |e| e.l_split = 32),
            ("n~", |e| e.n_tilde = 6),
            ("l~", |e| e.ell_tilde = 7),
            ("N_ex", |e| e.n_ex = 5),
            ("alpha", |e| e.aux_slots = 4),
            ("D", |e| e.d_drop = 13),
        ];
        for (label, mutate) in mutations {
            let mut ex = good.clone();
            mutate(&mut ex);
            assert!(!ex.check().is_empty(), "{label} moved and still validated");
        }
    }

    /// The structural invariants the derived LANES constants rest on.
    ///
    /// Pinning each field to its constant catches a value that moved; it
    /// does not catch a *coherent* future edit that leaves the derived
    /// constants ill formed.  Each mutation here keeps the fields
    /// self-consistent in the sense the old `check` tested and still
    /// breaks a division or a logarithm downstream, so each is caught
    /// only by the invariant it targets.
    #[test]
    fn check_enforces_what_the_derived_constants_assume() {
        type Mutate = fn(&mut ExactParams);
        let cases: [(&str, Mutate, &str); 3] = [
            // l does not divide d~ => SUBDEG = d~/l truncates
            (
                "l does not divide d~",
                |e| e.l_split = 48,
                "does not divide",
            ),
            // l not a power of two => trailing_zeros is not log2
            (
                "l is not a power of two",
                |e| e.l_split = 24,
                "power of two",
            ),
            // DELTA does not divide w_hat => W_TILDE truncates
            (
                "DELTA does not divide w_hat",
                |e| e.w_hat = 43,
                "W_TILDE truncates",
            ),
        ];
        for (label, mutate, expect) in cases {
            let mut ex = ExactParams::unchecked(RIVER_TOY);
            mutate(&mut ex);
            let errs = ex.check().join("; ");
            assert!(
                errs.contains(expect),
                "{label}: expected an error mentioning {expect:?}, got {errs:?}"
            );
        }
    }

    #[test]
    fn construction_refuses_an_unsupported_profile() {
        let mut par = RIVER_TOY;
        par.q0 = 59; // radix weights no longer cover [0, q_0-1]
        let err = match ExactParams::new(&par) {
            Err(e) => e,
            Ok(_) => panic!("q_0 = 59 was accepted"),
        };
        assert!(err.contains("exact parameters rejected"), "{err}");
        assert!(err.contains("radix weights"), "{err}");
    }

    #[test]
    fn honest_proof_verifies_and_round_trips() {
        for par in [RIVER_TOY, RIVER_N8] {
            let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
            let (witness, x, z_eval) = witness_for(&par, b"unit");
            let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"unit.com")]);
            let (w, state) = backend.com(&witness, &mut xof).unwrap();
            let statement = ExactStatement {
                w: &w,
                z_eval: &z_eval,
                x: &x,
            };
            let sigma = backend.prove(&witness, &state);
            assert!(backend.verify(&statement, &sigma), "{}", par.name);

            let blob = backend.proof_encode(&w, &sigma).unwrap();
            assert!(blob.len() <= backend.proof_bytes());
            let (w2, sigma2) = backend.proof_decode(&blob).unwrap();
            assert_eq!((w2, sigma2), (w.clone(), sigma.clone()));
            assert_eq!(backend.w_encode(&w).unwrap().len(), backend.w_bytes());
            assert_eq!(backend.w_decode(&backend.w_encode(&w).unwrap()).unwrap(), w);
        }
    }

    /// The commitment is deterministic in its XOF and binds the message.
    #[test]
    fn commitment_is_deterministic_and_binding() {
        let par = RIVER_TOY;
        let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
        let (witness, _, _) = witness_for(&par, b"bind");

        let mut a = Xof::new(DS_EXACT, &[Part::Bytes(b"c")]);
        let mut b = Xof::new(DS_EXACT, &[Part::Bytes(b"c")]);
        let (w1, _) = backend.com(&witness, &mut a).unwrap();
        let (w2, _) = backend.com(&witness, &mut b).unwrap();
        assert_eq!(w1, w2, "same XOF, same commitment");

        let mut c = Xof::new(DS_EXACT, &[Part::Bytes(b"d")]);
        let (w3, _) = backend.com(&witness, &mut c).unwrap();
        assert_ne!(w1, w3, "different XOF, different commitment");

        let mut moved = witness.clone();
        moved.e_eval[0] = if moved.e_eval[0] < par.B_e() as i64 {
            moved.e_eval[0] + 1
        } else {
            moved.e_eval[0] - 1
        };
        let mut e = Xof::new(DS_EXACT, &[Part::Bytes(b"c")]);
        let (w4, _) = backend.com(&moved, &mut e).unwrap();
        assert_ne!(w1, w4, "a different message must commit differently");
    }

    /// The link is checked over `Z`, not modulo `q~` — which is what makes
    /// `y_eval + k q~` a different witness rather than the same one.
    #[test]
    fn the_link_is_checked_over_the_integers() {
        let par = RIVER_TOY;
        let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
        let ex = &backend.ex;
        let (witness, x, z_eval) = witness_for(&par, b"lift");
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"lift.com")]);
        let (w, state) = backend.com(&witness, &mut xof).unwrap();
        let statement = ExactStatement {
            w: &w,
            z_eval: &z_eval,
            x: &x,
        };
        assert!(backend.verify(&statement, &backend.prove(&witness, &state)));

        // the same residues, a different integer
        let mut lifted = witness.clone();
        lifted.y_eval[0] += ex.q_tilde as i64;
        let canonical: Vec<i64> = lifted
            .e_eval
            .iter()
            .map(|&c| c + par.B_e() as i64)
            .collect();
        let digits = decompose_poly(&canonical).unwrap();
        let errs = check_relation(ex, &statement, &lifted, &digits);
        assert!(
            errs.iter().any(|m| m.contains("over Z")),
            "a lift by q~ must break the link: {errs:?}"
        );

        // and the commitment cannot see it, which is the point
        let m1 = pack_witness(ex, &witness.e_eval, &witness.y_eval, &digits).unwrap();
        let m2 = pack_witness(ex, &lifted.e_eval, &lifted.y_eval, &digits).unwrap();
        assert_eq!(m1, m2, "the commitment reduces mod q~");
    }

    /// The commitment binds only against a **short** opening.
    ///
    /// With an unbounded `r`, `A_1 r` can be steered anywhere, so BDLOP
    /// binding says nothing.  The wire format enforces ternary randomness
    /// through `Coder::signed(1)`, but a proof handed in as a value never
    /// passed a decoder — so `verify` has to enforce it itself, or its own
    /// binding claim is about bytes it never saw.
    #[test]
    fn a_long_opening_is_refused_even_though_it_reopens_the_commitment() {
        let par = RIVER_TOY;
        let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
        let ex = &backend.ex;
        let (witness, x, z_eval) = witness_for(&par, b"short");
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"short.com")]);
        let (w, state) = backend.com(&witness, &mut xof).unwrap();
        let statement = ExactStatement {
            w: &w,
            z_eval: &z_eval,
            x: &x,
        };
        let sigma = backend.prove(&witness, &state);
        assert!(backend.verify(&statement, &sigma));
        // the honest randomness really is ternary
        assert!(sigma.randomness.iter().all(|p| ex
            .rt
            .centered(p)
            .iter()
            .all(|&c| (-1..=1).contains(&c))));

        // A commitment that still *reopens* — same `W` — but with a
        // randomness coefficient outside `{-1,0,1}`.  Take the honest
        // opening and move one coefficient of `r` by `q~`: the residue is
        // unchanged, so `A_1 r` and `A_2 r` are unchanged and the
        // commitment recomputes identically, but the opening is no longer
        // short.  Before this check that verified.
        // and a non-canonical one, which `centered` would fold back into
        // the ternary range: `q~ -> 0`, `q~ + 1 -> 1`
        for noncanon in [ex.q_tilde, ex.q_tilde + 1, u64::MAX] {
            let mut bad = sigma.clone();
            bad.randomness[0][0] = noncanon;
            assert!(
                !backend.verify(&statement, &bad),
                "a non-canonical randomness coefficient must not centre into range"
            );
        }

        let mut long = sigma.clone();
        long.randomness[0][0] = 2;
        // recompute what `W` would be, so the *only* thing wrong is the norm
        let canonical: Vec<i64> = witness
            .e_eval
            .iter()
            .map(|&c| c + par.B_e() as i64)
            .collect();
        let digits = decompose_poly(&canonical).unwrap();
        let message = pack_witness(ex, &witness.e_eval, &witness.y_eval, &digits).unwrap();
        let w_long =
            ExactCommitmentKey::new(ex, &[0x11; 32]).commit(&ex.rt, &message, &long.randomness);
        let long_statement = ExactStatement {
            w: &w_long,
            z_eval: &z_eval,
            x: &x,
        };
        assert!(
            !backend.verify(&long_statement, &long),
            "a non-ternary opening must be refused even when it reopens"
        );

        // and the wire format refuses it too, which is the other half
        assert!(backend.proof_encode(&w_long, &long).is_err());
    }

    /// Malformed proofs are `false`, never a panic.
    #[test]
    fn malformed_proofs_are_rejected_without_panicking() {
        let par = RIVER_TOY;
        let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
        let ex = &backend.ex;
        let (witness, x, z_eval) = witness_for(&par, b"bad");
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"bad.com")]);
        let (w, _) = backend.com(&witness, &mut xof).unwrap();
        let statement = ExactStatement {
            w: &w,
            z_eval: &z_eval,
            x: &x,
        };

        let empty = OpeningProof {
            e_eval: vec![],
            y_eval: vec![],
            digits: vec![],
            randomness: vec![],
        };
        assert!(!backend.verify(&statement, &empty));

        let ragged = OpeningProof {
            e_eval: vec![0; par.d],
            y_eval: vec![0; par.d],
            digits: vec![vec![0; par.d]; 4],
            randomness: vec![vec![0u64; ex.d_tilde - 1]; ex.kappa],
        };
        assert!(!backend.verify(&statement, &ragged));

        let mut noncanon = ragged;
        noncanon.randomness = vec![vec![0u64; ex.d_tilde]; ex.kappa];
        noncanon.randomness[0][0] = ex.q_tilde;
        assert!(!backend.verify(&statement, &noncanon));

        // an out-of-range e_eval never reaches the commitment
        let mut out_of_range = witness.clone();
        out_of_range.e_eval[0] = par.B_e() as i64 + 1;
        let mut x2 = Xof::new(DS_EXACT, &[Part::Bytes(b"oor")]);
        assert!(backend.com(&out_of_range, &mut x2).is_none());

        // `check_relation` indexes `p[i]` and `e_eval[i]`, so a short row
        // has to *return* rather than record an error and carry on
        let canonical: Vec<i64> = witness
            .e_eval
            .iter()
            .map(|&c| c + par.B_e() as i64)
            .collect();
        let good = decompose_poly(&canonical).unwrap();
        for bad_digits in [
            vec![],
            good[..3].to_vec(),
            {
                let mut d = good.clone();
                d[0].pop();
                d
            },
            {
                let mut d = good.clone();
                d[2].clear();
                d
            },
        ] {
            let errs = check_relation(ex, &statement, &witness, &bad_digits);
            assert!(!errs.is_empty(), "a malformed digit set must be rejected");
        }
        let mut short_e = witness.clone();
        short_e.e_eval.pop();
        assert!(!check_relation(ex, &statement, &short_e, &good).is_empty());
        let mut short_y = witness.clone();
        short_y.y_eval.clear();
        assert!(!check_relation(ex, &statement, &short_y, &good).is_empty());
    }

    /// `y_eval` reaches `6 sigma_rs + w gamma B_e`, not `6 sigma_rs`.
    ///
    /// An accepted transcript bounds `z_eval`, and `y_eval = z_eval - x
    /// e_eval`, so the serializer has to admit the wider range or an honest
    /// prover can produce a proof it refuses to encode.
    #[test]
    fn the_serializer_admits_every_y_eval_an_accepted_transcript_can_hold() {
        for par in [RIVER_TOY, RIVER_N8] {
            let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
            let reachable = par.zm_inf_bound_sq().floor_sqrt() as i64
                + (par.w as i64) * (par.gamma as i64) * par.B_e() as i64;
            assert_eq!(backend.bound_y, reachable, "{}", par.name);
            assert!(
                backend.bound_y > par.zm_inf_bound_sq().floor_sqrt() as i64,
                "{}: the verifier's bound is not the encoder's",
                par.name
            );

            // the extreme value encodes rather than raising
            let (witness, x, z_eval) = witness_for(&par, b"edge");
            let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"edge.com")]);
            let (w, state) = backend.com(&witness, &mut xof).unwrap();
            let _ = (&x, &z_eval);
            let mut sigma = backend.prove(&witness, &state);
            sigma.y_eval[0] = reachable;
            assert!(
                backend.proof_encode(&w, &sigma).is_ok(),
                "{}: the widest admissible y_eval must encode",
                par.name
            );
        }
    }
}
