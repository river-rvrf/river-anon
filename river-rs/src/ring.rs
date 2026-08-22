//! `R_q = Z_q[X]/(X^d + 1)`, plus rounding and bit dropping — port of
//! `river-py/ring.py`.
//!
//! A polynomial is a `Vec<u64>` of length `d` holding *unsigned
//! canonical* coefficients in `[0, q)`.  Centred representatives in
//! `(-q/2, q/2]` are used for every norm and for the exact-link
//! equation, and are produced explicitly by [`Ring::centered`].
//!
//! **Multiplication.**  [`Ring::mul`] is schoolbook.  That is the
//! considered choice, not a placeholder: at `d = 32` an isolated
//! CRT-NTT multiply costs six 32-point transforms against 1024
//! multiply-accumulates, and loses.  The transform earns its keep only
//! when the transforms amortise across a matrix product with a fixed
//! matrix, which is what [`Ring::mat_to_ntt`] and [`Ring::mat_vec_ntt`]
//! are for — `G'` and `A` are derived from `rho` and never change.

use crate::aux_ntt::{CrtBackend, CrtNttMat};

// ---- Barrett reduction ---------------------------------------------------

/// `v mod q` for a fixed `q`, without a division.
///
/// Two reasons, and they point the same way.
///
/// *Timing.*  `%` and `rem_euclid` on a 128-bit value compile to a
/// software division routine whose running time depends on the operands.
/// Every coefficient of a masked response and of a secret key goes
/// through one, so that is a division on secret data in the hot path.
/// Barrett is a fixed sequence of multiplies and one masked subtraction:
/// no branch and no divide on any value derived from a secret.
///
/// *Speed.*  Secondary here, and worth stating so nobody quotes it as a
/// win it is not: `mul_schoolbook` is bound by its 1024 widening
/// multiplies, so replacing its 32 trailing divisions moves wall clock by
/// less than the measurement noise.  The reduction itself is roughly six
/// times faster than the divide; it simply is not where the time goes.
///
/// The estimate is `floor(v · floor(2^128 / q) / 2^128)`; see
/// [`Barrett::reduce`] for how far it undershoots and why two masked
/// subtractions close it.
#[derive(Clone, Copy, Debug)]
pub struct Barrett {
    q: u64,
    /// `floor(2^128 / q)`.
    mu: u128,
}

impl Barrett {
    pub fn new(q: u64) -> Self {
        assert!(q >= 2, "modulus must be at least 2");
        assert!(q < 1 << 62, "modulus too large for the accumulator bound");
        // floor(2^128 / q) = floor((2^128 - 1) / q) whenever q is not a
        // power of two, and one less otherwise; compute it exactly.
        let mu = (u128::MAX / q as u128) + u128::from(u128::MAX % q as u128 == q as u128 - 1);
        Self { q, mu }
    }

    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// High 128 bits of a 256-bit product, from four 64-bit multiplies.
    #[inline(always)]
    fn mul_hi(a: u128, b: u128) -> u128 {
        let (a0, a1) = (a as u64 as u128, a >> 64);
        let (b0, b1) = (b as u64 as u128, b >> 64);
        let m0 = a0 * b0;
        let m1 = a1 * b0;
        let m2 = a0 * b1;
        let m3 = a1 * b1;
        // carry out of the low 128 bits
        let mid = (m0 >> 64) + (m1 as u64 as u128) + (m2 as u64 as u128);
        m3 + (m1 >> 64) + (m2 >> 64) + (mid >> 64)
    }

    /// `v mod q`, for any `v: u128`.  Branchless.
    ///
    /// `mu = floor(2^128 / q)` undershoots, so the quotient estimate is
    /// short by at most two over the whole `u128` range; two masked
    /// subtractions finish it.  Both run unconditionally — the point is
    /// that nothing here depends on the value.
    #[inline(always)]
    pub fn reduce(&self, v: u128) -> u64 {
        let quo = Self::mul_hi(v, self.mu);
        let mut r = (v - quo * self.q as u128) as u64;
        for _ in 0..2 {
            let mask = 0u64.wrapping_sub((r >= self.q) as u64);
            r = r.wrapping_sub(self.q & mask);
        }
        debug_assert!(r < self.q, "barrett: {v} mod {} left {r}", self.q);
        r
    }
}

pub type Poly = Vec<u64>;
pub type PolyVec = Vec<Poly>;
pub type PolyMat = Vec<PolyVec>;

/// Smallest multiple of `q` at or above `d·(q-1)^2`, or `None` if that
/// does not fit a **positive** `i128`.
///
/// `q < 2^62` is not the condition, which is what the parameter check used
/// to test: at `d = 32` that admits `d·(q-1)^2 ≈ 2^129`, which wraps `u128`
/// and, one step earlier, becomes negative on the cast to `i128`.  The
/// condition is this computation succeeding.
pub fn checked_wrap_bias(q: u64, d: usize) -> Option<i128> {
    // `q == 1` divides by zero below and `q == 0` underflows; `d == 0`
    // is a degenerate ring that would otherwise return `Some(0)`.
    if q < 2 || d == 0 {
        return None;
    }
    let qm1 = (q as u128).checked_sub(1)?;
    let worst = (d as u128).checked_mul(qm1)?.checked_mul(qm1)?;
    let lifted = worst.div_ceil(q as u128).checked_mul(q as u128)?;
    // The bias fitting is not the condition: `mul_schoolbook` evaluates
    // `v + bias`, and the accumulator itself reaches `±worst`, so the
    // *sum* is what has to fit.  At `q = 2^60 + 1`, `d = 64` the bias
    // alone clears `2^127` while `worst + bias` exceeds `i128::MAX` —
    // a debug panic, and in release a wrapped, wrong residue.
    let top = lifted.checked_add(worst)?;
    (top <= i128::MAX as u128).then_some(lifted as i128)
}

fn wrap_bias(q: u64, d: usize) -> i128 {
    checked_wrap_bias(q, d)
        .expect("d·(q-1)^2 must fit a positive i128; RiVeRParams::check tests this")
}

/// Arithmetic context for `R_q`.
#[derive(Clone)]
pub struct Ring {
    pub q: u64,
    pub d: usize,
    pub half_q: u64,
    bar: Barrett,
    /// Smallest multiple of `q` at or above `d·(q-1)^2` — the negacyclic
    /// wrap-around makes the schoolbook accumulator negative, and this
    /// lifts it without changing the residue.
    wrap_bias: i128,
    backend: Option<CrtBackend>,
}

impl Ring {
    /// A ring with no transform backend — every product is schoolbook.
    pub fn new(q: u64, d: usize) -> Self {
        Self {
            q,
            d,
            half_q: q / 2,
            bar: Barrett::new(q),
            wrap_bias: wrap_bias(q, d),
            backend: None,
        }
    }

    /// A ring whose matrix paths use the CRT-NTT backend, sized for
    /// accumulations of up to `max_terms` products.  Falls back to
    /// schoolbook if the reconstruction bound does not hold.
    pub fn with_backend(q: u64, d: usize, max_terms: usize) -> Self {
        Self {
            q,
            d,
            half_q: q / 2,
            bar: Barrett::new(q),
            wrap_bias: wrap_bias(q, d),
            backend: CrtBackend::new(q, d, max_terms),
        }
    }

    pub fn has_backend(&self) -> bool {
        self.backend.is_some()
    }

    // ---- element creation ------------------------------------------------

    pub fn zero(&self) -> Poly {
        vec![0u64; self.d]
    }

    pub fn one(&self) -> Poly {
        let mut p = self.zero();
        p[0] = 1;
        p
    }

    /// The constant polynomial `c`.
    ///
    /// The one `rem_euclid` here is on the caller's *scalar*, once per
    /// call rather than per coefficient, and no caller in the scheme
    /// passes a secret — nothing in `river`, `oom` or `exact` calls this
    /// at all today.  Said explicitly because the crate's timing rule is
    /// about where a divide is reachable from, not about auditing each
    /// one as it appears.
    pub fn const_poly(&self, c: i64) -> Poly {
        let mut p = self.zero();
        p[0] = c.rem_euclid(self.q as i64) as u64;
        p
    }

    /// `1 + X + ... + X^{d-1}`, the constant shift a centred-error
    /// repair would need.
    pub fn all_ones(&self) -> Poly {
        vec![1u64; self.d]
    }

    // ---- representative conversions --------------------------------------

    /// The Barrett context for this ring's modulus.
    pub fn barrett(&self) -> &Barrett {
        &self.bar
    }

    pub fn reduce(&self, a: &[u64]) -> Poly {
        a.iter().map(|&c| self.bar.reduce(c as u128)).collect()
    }

    /// Unsigned `[0, q)` → centred `(-q/2, q/2]`.
    ///
    /// Masked, not branched.  This runs on the secret key, on every
    /// Gaussian mask and on every response — `flat_centered` is what the
    /// rejection samplers take — so the `if c > half_q` it replaces was a
    /// source-level branch on a secret coefficient at the busiest site in
    /// the crate.  Same instruction sequence either way.
    pub fn centered(&self, a: &[u64]) -> Vec<i64> {
        let q = self.q as i64;
        a.iter()
            .map(|&c| {
                // all-ones iff `c` is above the half-way point
                let mask = 0i64.wrapping_sub((c > self.half_q) as i64);
                c as i64 - (q & mask)
            })
            .collect()
    }

    /// Centred `(-q/2, q/2]` → unsigned `[0, q)`.
    ///
    /// One masked add for the in-range case, which is every caller in the
    /// scheme; anything wider falls back to a full reduction.  No
    /// division on a value that came from a secret.
    ///
    /// The fallback *is* a branch, but not on a secret: a centred value
    /// lands in `[0, q)` after the masked add for **every** input the
    /// domain admits, so the test's outcome is the same whatever the
    /// coefficient is.  It is there to keep the function total on a
    /// caller-supplied `i64`, not to handle a case the scheme reaches.
    pub fn from_centered(&self, a: &[i64]) -> Poly {
        let q = self.q as i64;
        a.iter()
            .map(|&c| {
                // arithmetic shift gives q when c < 0 and 0 otherwise
                let r = c.wrapping_add((c >> 63) & q);
                if (r as u64) < self.q && r >= 0 {
                    r as u64
                } else {
                    self.bar.reduce(c.rem_euclid(q) as u128)
                }
            })
            .collect()
    }

    pub fn vec_centered(&self, v: &[Poly]) -> Vec<Vec<i64>> {
        v.iter().map(|a| self.centered(a)).collect()
    }

    /// Flatten a vector of polynomials to one centred coefficient
    /// slice — the form the rejection samplers take.
    pub fn flat_centered(&self, v: &[Poly]) -> Vec<i64> {
        v.iter().flat_map(|a| self.centered(a)).collect()
    }

    // ---- element-wise arithmetic -----------------------------------------

    #[inline]
    fn add_coeff(&self, a: u64, b: u64) -> u64 {
        let s = a + b;
        let mask = 0u64.wrapping_sub((s >= self.q) as u64);
        s.wrapping_sub(self.q & mask)
    }

    #[inline]
    fn sub_coeff(&self, a: u64, b: u64) -> u64 {
        let d = a.wrapping_sub(b);
        let mask = 0u64.wrapping_sub((a < b) as u64);
        d.wrapping_add(self.q & mask)
    }

    pub fn add(&self, a: &[u64], b: &[u64]) -> Poly {
        (0..self.d).map(|i| self.add_coeff(a[i], b[i])).collect()
    }

    pub fn sub(&self, a: &[u64], b: &[u64]) -> Poly {
        (0..self.d).map(|i| self.sub_coeff(a[i], b[i])).collect()
    }

    /// `-a mod q`.
    ///
    /// Through the masked `sub_coeff`.  The `if c == 0`
    /// this replaces existed because `q - 0` is `q` rather than `0`, and
    /// it was a branch on whether a secret coefficient is zero — the same
    /// shape as the zero-coefficient skip removed from `mul_schoolbook`,
    /// and on data that really is a third zeros when it is ternary.
    pub fn neg(&self, a: &[u64]) -> Poly {
        a.iter().map(|&c| self.sub_coeff(0, c)).collect()
    }

    /// Integer scalar times polynomial.
    ///
    /// `c.rem_euclid(q)` is a divide, and it stays: `c` is the *public*
    /// `q_0` at every call site in the scheme (`OomStatement::c_i` and
    /// `combine_c` scale the derived vectors by it), it is reduced once
    /// per call rather than once per coefficient, and the per-coefficient
    /// work below goes through Barrett.  A caller that passes a secret
    /// scalar would be outside what this rule covers, and there is none.
    pub fn scale(&self, c: i64, a: &[u64]) -> Poly {
        let cm = c.rem_euclid(self.q as i64) as u128;
        a.iter()
            .map(|&ai| self.bar.reduce(cm * ai as u128))
            .collect()
    }

    #[inline]
    pub fn add_assign(&self, a: &mut [u64], b: &[u64]) {
        for i in 0..self.d {
            a[i] = self.add_coeff(a[i], b[i]);
        }
    }

    // ---- multiplication --------------------------------------------------

    /// Negacyclic convolution.  See the module docs for why this stays
    /// schoolbook while the matrix paths transform.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Poly {
        self.mul_schoolbook(a, b)
    }

    /// Negacyclic convolution, finished with Barrett rather than a
    /// 128-bit division.
    ///
    /// The wrap-around term is negative, so the accumulator is signed;
    /// adding a multiple of `q` that covers the worst case makes it
    /// non-negative without changing the residue, and one Barrett
    /// reduction finishes each coefficient.  No `rem_euclid` on a value
    /// derived from a secret.
    ///
    /// This is a *timing* change, not a throughput one: the inner loop is
    /// 1024 widening multiplies and dominates, so wall clock is within
    /// noise of the divide-based version it replaced.  What changed is
    /// that no coefficient reaches a variable-time divide.
    ///
    /// **No zero-coefficient skip.**  There used to be one, documented as
    /// seeing only public challenge polynomials.  That was wrong twice
    /// over.  `OomStatement::combine_c` passes the mask `a_i` as its first
    /// operand and `Oom::com` squares every `a_i`, so it saw secret data;
    /// and the leak is per-coefficient, not the measure-zero event that a
    /// whole polynomial is zero.  Since `f_i = a_i + x b_i` is published,
    /// an observer who learns where `a_i` has zeros can test each `i`
    /// against `f_i` and `f_i - x`, so "`a_i` is independent of `j*`" does
    /// not settle it.
    ///
    /// It also bought nothing: `w == d == 32` at every shipped profile, so
    /// a challenge polynomial has **no** zero coefficients and the branch
    /// never fired where it was supposed to help.  It fired on ternary
    /// secrets, where a third of the coefficients are zero.
    pub fn mul_schoolbook(&self, a: &[u64], b: &[u64]) -> Poly {
        let d = self.d;
        let bias = self.wrap_bias;
        let mut c = vec![0i128; d];
        for i in 0..d {
            let ai = a[i] as i128;
            for j in 0..d {
                let prod = ai * b[j] as i128;
                if i + j < d {
                    c[i + j] += prod;
                } else {
                    c[i + j - d] -= prod;
                }
            }
        }
        c.into_iter()
            .map(|v| self.bar.reduce((v + bias) as u128))
            .collect()
    }

    /// The negacyclic product over the *integers* — no modular
    /// reduction anywhere.
    ///
    /// The exact relation requires `z_eval = x·e_eval + y_eval` as an
    /// equality over `Z`, not merely modulo a protocol modulus.  Every
    /// modular multiply would silently satisfy the weaker statement.
    /// No zero skip here either, for the reason [`Ring::mul_schoolbook`]
    /// gives — its first operand is the public challenge, but uniformity
    /// beats an argument about callers, and at `w == d` it never fired.
    pub fn mul_int(a: &[i128], b: &[i128]) -> Vec<i128> {
        let d = a.len();
        let mut out = vec![0i128; d];
        for i in 0..d {
            for j in 0..d {
                let prod = a[i] * b[j];
                if i + j < d {
                    out[i + j] += prod;
                } else {
                    out[i + j - d] -= prod;
                }
            }
        }
        out
    }

    // ---- norms (on the centred representation) ---------------------------

    pub fn inf_norm(&self, a: &[u64]) -> u64 {
        a.iter()
            .map(|&c| if c > self.half_q { self.q - c } else { c })
            .max()
            .unwrap_or(0)
    }

    pub fn l2_norm_sq(&self, a: &[u64]) -> u128 {
        a.iter()
            .map(|&c| {
                let v = if c > self.half_q {
                    (self.q - c) as u128
                } else {
                    c as u128
                };
                v * v
            })
            .sum()
    }

    pub fn l1_norm(&self, a: &[u64]) -> u128 {
        a.iter()
            .map(|&c| {
                if c > self.half_q {
                    (self.q - c) as u128
                } else {
                    c as u128
                }
            })
            .sum()
    }

    // ---- vector operations -----------------------------------------------

    pub fn vec_zero(&self, n: usize) -> PolyVec {
        (0..n).map(|_| self.zero()).collect()
    }

    pub fn vec_add(&self, u: &[Poly], v: &[Poly]) -> PolyVec {
        u.iter()
            .zip(v.iter())
            .map(|(a, b)| self.add(a, b))
            .collect()
    }

    pub fn vec_sub(&self, u: &[Poly], v: &[Poly]) -> PolyVec {
        u.iter()
            .zip(v.iter())
            .map(|(a, b)| self.sub(a, b))
            .collect()
    }

    pub fn vec_neg(&self, v: &[Poly]) -> PolyVec {
        v.iter().map(|a| self.neg(a)).collect()
    }

    /// `c · v` element-wise.  Uses the transform when the ring has a
    /// backend: `c` is forward-transformed once and reused.
    pub fn vec_scale(&self, c: &[u64], v: &[Poly]) -> PolyVec {
        if let Some(bk) = self.backend.as_ref() {
            let c_ntt = bk.to_ntt(c);
            return v.iter().map(|p| bk.mul_with_lhs_ntt(&c_ntt, p)).collect();
        }
        v.iter().map(|p| self.mul(c, p)).collect()
    }

    pub fn vec_scale_int(&self, c: i64, v: &[Poly]) -> PolyVec {
        v.iter().map(|p| self.scale(c, p)).collect()
    }

    pub fn inner(&self, u: &[Poly], v: &[Poly]) -> Poly {
        let mut acc = self.zero();
        for i in 0..u.len() {
            let prod = self.mul(&u[i], &v[i]);
            self.add_assign(&mut acc, &prod);
        }
        acc
    }

    /// `<vec(u), vec(v)>` over `Z`, on centred coefficients — the inner
    /// product the rejection samplers use.
    pub fn vec_inner_int(&self, u: &[Poly], v: &[Poly]) -> i128 {
        u.iter()
            .zip(v.iter())
            .map(|(a, b)| {
                self.centered(a)
                    .into_iter()
                    .zip(self.centered(b))
                    .map(|(x, y)| x as i128 * y as i128)
                    .sum::<i128>()
            })
            .sum()
    }

    pub fn vec_inf_norm(&self, v: &[Poly]) -> u64 {
        v.iter().map(|a| self.inf_norm(a)).max().unwrap_or(0)
    }

    pub fn vec_l2_norm_sq(&self, v: &[Poly]) -> u128 {
        v.iter().map(|a| self.l2_norm_sq(a)).sum()
    }

    pub fn vec_concat(&self, parts: &[&[Poly]]) -> PolyVec {
        let mut out = Vec::new();
        for p in parts {
            out.extend_from_slice(p);
        }
        out
    }

    // ---- matrix operations -----------------------------------------------

    /// `m · v`.  Uses the CRT backend when the ring has one *and* the
    /// product is inside the accumulation its `P` was sized for;
    /// otherwise schoolbook, which is always exact.  Falling back rather
    /// than refusing is right here — the two paths agree by construction,
    /// so the only cost is speed.
    pub fn mat_vec(&self, m: &[PolyVec], v: &[Poly]) -> PolyVec {
        if let Some(bk) = self.backend.as_ref() {
            if let Some(out) = bk.mat_to_ntt(m).and_then(|m_ntt| bk.mat_vec_ntt(&m_ntt, v)) {
                return out;
            }
        }
        m.iter().map(|row| self.inner(row, v)).collect()
    }

    /// Pre-transform a matrix for repeated matrix-vector products.
    ///
    /// `None` when the ring has no backend, and also when the matrix is
    /// ragged or wider than the accumulation the backend's `P` was sized
    /// for — see [`CrtBackend::mat_to_ntt`].  The caller falls back to
    /// [`Ring::mat_vec`], which is schoolbook and always exact.
    pub fn mat_to_ntt(&self, m: &[PolyVec]) -> Option<CrtNttMat> {
        self.backend.as_ref().and_then(|bk| bk.mat_to_ntt(m))
    }

    /// Matrix-vector product with a pre-transformed matrix.  `None` when
    /// the matrix does not belong to this ring's backend.
    pub fn mat_vec_ntt(&self, m: &CrtNttMat, v: &[Poly]) -> Option<PolyVec> {
        self.backend.as_ref().and_then(|bk| bk.mat_vec_ntt(m, v))
    }
}

// ---- rounding (Fact 1) ---------------------------------------------------

/// Division by a small **public** divisor, without a `div` instruction.
///
/// [`round_p`] divides every coefficient of `A s` and of `<h_m, s>` by
/// `q_0`.  The divisor is a public per-profile constant, but the
/// *dividend* is the secret key's image, and `div` on both x86-64 and
/// AArch64 has operand-dependent latency.  So the reciprocal is formed
/// once, from the public divisor, and each coefficient costs a widening
/// multiply and a shift.
///
/// Granlund–Montgomery: with `m = floor(2^S / d) + 1`,
/// `floor(n · m / 2^S)` is `floor(n / d)` for every `n < 2^N` provided
/// `m·d - 2^S <= 2^(S-N)`.  Since `m·d - 2^S = d - (2^S mod d) <= d`,
/// `S = 70` and `N = 62` admit every `d <= 256` — three orders of
/// magnitude above the only `q_0` the scheme uses, 61.
///
/// [`Reciprocal::new`] returns `None` outside that, and [`round_p`] then
/// falls back to `/`.  That fallback branches on the *profile's*
/// divisor, which is public, so it leaks nothing about a coefficient.
pub struct Reciprocal {
    m: u128,
    d: u64,
}

impl Reciprocal {
    const S: u32 = 70;
    const N: u32 = 62;

    /// `None` for a divisor outside the range the bound above covers.
    pub fn new(d: u64) -> Option<Self> {
        if d == 0 || d > (1u64 << (Self::S - Self::N)) {
            return None;
        }
        Some(Self {
            m: (1u128 << Self::S) / d as u128 + 1,
            d,
        })
    }

    /// `n / d`, for `n < 2^62`.
    #[inline(always)]
    pub fn div(&self, n: u64) -> u64 {
        debug_assert!(n < 1u64 << Self::N, "dividend outside the proved range");
        ((n as u128 * self.m) >> Self::S) as u64
    }

    pub fn divisor(&self) -> u64 {
        self.d
    }
}

/// `floor(a)_p`: canonical coefficients integer-divided by `q_0 = q/p`.
pub fn round_p(a: &[u64], q0: u64) -> Poly {
    match Reciprocal::new(q0) {
        Some(r) => a.iter().map(|&c| r.div(c)).collect(),
        None => a.iter().map(|&c| c / q0).collect(),
    }
}

pub fn round_p_vec(v: &[Poly], q0: u64) -> PolyVec {
    // One reciprocal for the whole matrix rather than one per row: the
    // setup division is on the public divisor, and doing it `n` times is
    // just waste.
    match Reciprocal::new(q0) {
        Some(r) => v
            .iter()
            .map(|a| a.iter().map(|&c| r.div(c)).collect())
            .collect(),
        None => v.iter().map(|a| round_p(a, q0)).collect(),
    }
}

/// `a - q_0 · rounded mod q`, the canonical rounding error in
/// `[0, q_0 - 1]`.
///
/// Together with [`round_p`] this is Fact 1: `v = floor(u)_p` iff there
/// is an `e` with coefficients in `[0, q/p - 1]` and
/// `e = u - (q/p)·v mod q`.
pub fn rounding_error(ring: &Ring, a: &[u64], rounded: &[u64], q0: u64) -> Poly {
    let q = ring.q as i128;
    (0..ring.d)
        .map(|i| {
            let v = a[i] as i128 - q0 as i128 * rounded[i] as i128;
            // Both terms are below `q` — `a[i]` because it is a ring
            // coefficient, `q_0 · rounded[i]` because `rounded[i] < p` —
            // so the difference lands in `(-q, q)` and one masked add
            // finishes it.  That matters: this is the *secret key's*
            // rounding error, and `rem_euclid` is a divide.
            if v > -q && v < q {
                let v = v as i64;
                let mask = (v >> 63) as u64;
                (v as u64).wrapping_add(mask & ring.q)
            } else {
                // Unreachable for any `rounded` this crate produces or
                // any public key `validate_ring` admits, so the branch's
                // outcome is the same whatever the coefficients are.  It
                // keeps the function total on a hand-built argument.
                v.rem_euclid(q) as u64
            }
        })
        .collect()
}

// ---- the centred range shift --------------------------------------------
//
// REPAIR.  The rounding relations
// are written throughout the paper with errors in `[0, q_0-1]`, and the
// `Eval` figure builds the OOM targets as `c_i = (q_0 t_i, q_0 v)` with
// no offset.  The parameter derivation nevertheless uses
// `B_e = floor(q_0/2) = 30`, saying only that "the range proved by the
// underlying proof system can be translated so that it is centered at
// zero".  The algorithms never define that translation, and without it
// 30 is not a valid norm bound on a coefficient ranging over `[0, 60]`.
//
// It is not presentational.  The selected LANES modulus clears
// `24 phi_m eta_m` by 0.56% with `B_e = 30`; with the literal 60 the
// requirement doubles and the modulus fails outright
// (`exact::ExactParams::q_tilde_clears`).
//
// So the shift is carried explicitly, and only here: the OOM witness is
// `e^c = e - B_e in [-B_e, B_e]`, every public target gains `+B_e`, and
// the exact relation proves `e^c + B_e = d_0 + 3d_1 + 9d_2 + 17d_3` with
// digits in `{0,1,2}`.  Keeping it behind these two names means a later
// clarification from the authors changes one boundary rather than the
// scheme.

/// `[0, q_0-1]` rounding error -> the centred OOM witness `[-B_e, B_e]`.
///
/// `Err` names the coefficient that fell outside, because reaching it
/// means the caller's key pair or rounding is inconsistent rather than
/// that the protocol aborted.
pub fn to_centered_error(coeffs: &[u64], b_e: u64) -> Result<Vec<i64>, String> {
    let b = b_e as i64;
    let mut out = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let v = c as i64 - b;
        if !(-b..=b).contains(&v) {
            return Err(format!("centred error {v} outside [-{b_e}, {b_e}]"));
        }
        out.push(v);
    }
    Ok(out)
}

/// The inverse: centred witness -> the `[0, q_0-1]` range the relation
/// states, which is what the radix-3 decomposition consumes.
pub fn from_centered_error(coeffs: &[i64], b_e: u64) -> Result<Vec<u64>, String> {
    let top = 2 * b_e as i64;
    let mut out = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let v = c + b_e as i64;
        if !(0..=top).contains(&v) {
            return Err(format!("canonical error {v} outside [0, {top}]"));
        }
        out.push(v as u64);
    }
    Ok(out)
}

// ---- bit dropping --------------------------------------------------------
//
// `[[u]]_K` (high bits) and `u mod^pm 2^K` (centred low bits).  The
// paper's Preliminaries define both, on the *centred* representative:
//
//   a mod^pm 2^K := \bar a - 2^K floor((\bar a + 2^{K-1} - 1) / 2^K)
//   [[a]]_K      := (\bar a - (a mod^pm 2^K)) / 2^K
//
// with the low part in `(-2^{K-1}, 2^{K-1}]` — closed at the top, which
// is the tie [`mod_pm`] has always used.  (The form was the other
// way round; that difference is gone.)  `mod_pm` is
// representative-independent, so the only thing the definition fixes is
// which representative goes in.
//
// Closed and aligned: `oom::Oom::high_low` centres before calling
// [`power2round`], so about half the high parts are negative and the
// transmitted `B` field is signed.

/// Centred representative of `value` modulo `2^k`, in
/// `(-2^{k-1}, 2^{k-1}]`.
///
/// Both steps are masked.  `rem_euclid` by a power of two is a mask in
/// two's complement, and the tie correction is a conditional subtract —
/// the `if low > power / 2` it replaces was a branch on a coefficient of
/// `u_A` / `u_B`, which are products of the secret masks.  `k` is public
/// (`K_a`, `K_b`), so the shift amount carries nothing.
pub fn mod_pm(value: i128, k: u32) -> i128 {
    let power = 1i128 << k;
    // `value & (2^k - 1)` is `value.rem_euclid(2^k)` on two's complement,
    // for negative `value` as well.
    let low = value & (power - 1);
    // all-ones exactly when `low > power/2`
    let over = ((power >> 1) - low) >> 127;
    low - (power & over)
}

/// `(high, low)` with `value = high · 2^k + low` and `|low| <= 2^{k-1}`.
pub fn power2round(value: i128, k: u32) -> (i128, i128) {
    let low = mod_pm(value, k);
    ((value - low) >> k, low)
}

/// `[[a]]_K` coefficient-wise on the *centred* representation.
pub fn high_bits(ring: &Ring, a: &[u64], k: u32) -> Vec<i128> {
    ring.centered(a)
        .into_iter()
        .map(|c| power2round(c as i128, k).0)
        .collect()
}

/// `a mod^pm 2^K` coefficient-wise on the centred representation.
pub fn low_bits(ring: &Ring, a: &[u64], k: u32) -> Vec<i128> {
    ring.centered(a)
        .into_iter()
        .map(|c| power2round(c as i128, k).1)
        .collect()
}

/// `max |[[a]]_K|` over `a in Z_qhat`, taken on the centred rep.
///
/// `\bar a` runs over `(-qhat/2, qhat/2]` and
/// `[[a]]_K = floor((\bar a + 2^{K-1} - 1) / 2^K)` is monotone in
/// `\bar a`, so the extremes are at the ends of that interval.
/// Computed rather than estimated, because it is the codec's field
/// bound: one too small refuses an honest proof, one too large costs a
/// bit per coefficient.
pub fn high_bits_bound(q_hat: u64, k: u32) -> i64 {
    let top = (q_hat / 2) as i128;
    let bottom = -top + i128::from(q_hat.is_multiple_of(2));
    let hi_top = power2round(top, k).0;
    let hi_bottom = power2round(bottom, k).0;
    hi_top.abs().max(hi_bottom.abs()) as i64
}

/// `||.||_inf` of a slice of already-centred integer coefficient lists.
///
/// `saturating_abs`, because `i128::MIN.abs()` panics.  Nothing in the
/// crate calls this today — it mirrors `river-py`'s helper and is public
/// surface — and unused public surface that can panic on a caller's value
/// is audit cost with no cover, which is the same argument `Cargo.toml`
/// makes about unused dependencies.
pub fn int_vec_inf_norm(v: &[Vec<i128>]) -> i128 {
    v.iter()
        .map(|a| a.iter().map(|c| c.saturating_abs()).max().unwrap_or(0))
        .max()
        .unwrap_or(0)
}

// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{RIVER_N256, RIVER_TOY};

    fn lcg(seed: u64) -> impl FnMut(u64) -> u64 {
        let mut s = seed;
        move |bound: u64| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 16) % bound
        }
    }

    #[test]
    fn barrett_agrees_with_the_division_it_replaces() {
        // Across every profile modulus and the extremes of the range the
        // schoolbook accumulator can reach: `d·(q-1)^2`.
        let mut r = lcg(0xBA88E77);
        for par in crate::params::PROFILES {
            for q in [par.q(), par.q_hat, par.p, 61] {
                let bar = Barrett::new(q);
                let top = 32u128 * (q as u128 - 1) * (q as u128 - 1);
                let mut cases = vec![0u128, 1, q as u128 - 1, q as u128, q as u128 + 1, top];
                for _ in 0..2000 {
                    let v = ((r(u64::MAX) as u128) << 64 | r(u64::MAX) as u128) % (top + 1);
                    cases.push(v);
                }
                for v in cases {
                    assert_eq!(bar.reduce(v), (v % q as u128) as u64, "q={q} v={v}");
                }
            }
        }
    }

    #[test]
    fn multiplication_is_commutative_and_has_an_identity() {
        let r = Ring::new(RIVER_TOY.q(), 32);
        let mut g = lcg(1);
        let a: Poly = (0..32).map(|_| g(r.q)).collect();
        let b: Poly = (0..32).map(|_| g(r.q)).collect();
        assert_eq!(r.mul(&a, &b), r.mul(&b, &a));
        assert_eq!(r.mul(&a, &r.one()), a);
    }

    #[test]
    fn x_to_the_d_is_minus_one() {
        let r = Ring::new(RIVER_TOY.q(), 32);
        let mut x = r.zero();
        x[1] = 1;
        let mut prod = r.one();
        for _ in 0..32 {
            prod = r.mul(&prod, &x);
        }
        let mut want = r.zero();
        want[0] = r.q - 1;
        assert_eq!(prod, want);
    }

    #[test]
    fn the_backend_path_agrees_with_schoolbook() {
        let par = RIVER_N256;
        let r = Ring::with_backend(par.q_hat, 32, par.gprime_cols());
        assert!(r.has_backend());
        let plain = Ring::new(par.q_hat, 32);
        let mut g = lcg(5);
        let mat: PolyMat = (0..3)
            .map(|_| (0..4).map(|_| (0..32).map(|_| g(r.q)).collect()).collect())
            .collect();
        let v: PolyVec = (0..4).map(|_| (0..32).map(|_| g(r.q)).collect()).collect();
        assert_eq!(r.mat_vec(&mat, &v), plain.mat_vec(&mat, &v));

        let m_ntt = r.mat_to_ntt(&mat).unwrap();
        assert_eq!(r.mat_vec_ntt(&m_ntt, &v).unwrap(), plain.mat_vec(&mat, &v));
    }

    #[test]
    fn vec_scale_agrees_across_backends() {
        let par = RIVER_TOY;
        let r = Ring::with_backend(par.q_hat, 32, 12);
        let plain = Ring::new(par.q_hat, 32);
        let mut g = lcg(6);
        let c: Poly = (0..32).map(|_| g(r.q)).collect();
        let v: PolyVec = (0..5).map(|_| (0..32).map(|_| g(r.q)).collect()).collect();
        assert_eq!(r.vec_scale(&c, &v), plain.vec_scale(&c, &v));
    }

    #[test]
    fn norms_use_the_centred_form() {
        let r = Ring::new(101, 4);
        let a = vec![0u64, 1, 100, 51];
        assert_eq!(r.centered(&a), vec![0i64, 1, -1, -50]);
        assert_eq!(r.inf_norm(&a), 50);
        assert_eq!(r.l1_norm(&a), 52);
        assert_eq!(r.l2_norm_sq(&a), 1 + 1 + 2500);
    }

    #[test]
    fn rounding_is_fact_one() {
        let par = RIVER_TOY;
        let r = Ring::new(par.q(), 32);
        let q0 = par.q0;
        let mut g = lcg(7);
        let a: Poly = (0..32).map(|_| g(r.q)).collect();
        let rounded = round_p(&a, q0);
        let e = rounding_error(&r, &a, &rounded, q0);
        assert!(e.iter().all(|&c| c < q0), "error outside [0, q0-1]");
        for i in 0..32 {
            assert_eq!((q0 * rounded[i] + e[i]) % r.q, a[i]);
        }
        assert!(rounded.iter().all(|&c| c < par.p));
    }

    #[test]
    fn power2round_reconstructs_and_ties_high() {
        for k in [4u32, 5, 12, 28] {
            let half = 1i128 << (k - 1);
            // the tie: low part exactly 2^{k-1} stays positive here,
            // which is the convention the OOM layer and the codec are
            // built around.
            let (hi, lo) = power2round(half, k);
            assert_eq!(lo, half);
            assert_eq!(hi, 0);
            for v in [-1000i128, -1, 0, 1, 1000, (1 << 40) + 7] {
                let (h, l) = power2round(v, k);
                assert_eq!(h * (1 << k) + l, v);
                assert!(l.abs() <= half);
            }
        }
    }

    #[test]
    fn low_bits_meet_the_correctness_proof_bound() {
        // ||e_B||_inf <= 2^{K_b - 1}
        let par = RIVER_TOY;
        let r = Ring::new(par.q_hat, 32);
        let mut g = lcg(9);
        for _ in 0..20 {
            let a: Poly = (0..32).map(|_| g(r.q)).collect();
            let lo = low_bits(&r, &a, par.K_b);
            assert!(lo.iter().all(|&c| c.abs() <= 1i128 << (par.K_b - 1)));
        }
    }

    #[test]
    fn mul_int_is_exact_over_the_integers() {
        let a: Vec<i128> = (0..32).map(|i| (i as i128) - 16).collect();
        let mut b = vec![0i128; 32];
        b[1] = 1; // multiply by X
        let out = Ring::mul_int(&a, &b);
        // X · sum a_i X^i  =  -a_{31} + sum_{i<31} a_i X^{i+1}
        assert_eq!(out[0], -a[31]);
        for i in 0..31 {
            assert_eq!(out[i + 1], a[i]);
        }
    }

    /// The bias fitting `i128` is not the condition — the *addition* is.
    ///
    /// Reported against the first version of this helper, which tested
    /// `lifted < 2^127`.  At `q = 2^60 + 1`, `d = 64` and both operands
    /// entirely `q - 1`, the accumulator reaches `+worst` and the sum
    /// `worst + bias` passes `i128::MAX` by 1152921504606846914.
    #[test]
    fn the_wrap_bias_condition_covers_the_addition_not_just_the_bias() {
        let q = (1u64 << 60) + 1;
        let d = 64usize;
        let qm1 = (q - 1) as u128;
        let worst = d as u128 * qm1 * qm1;
        let lifted = worst.div_ceil(q as u128) * q as u128;
        // the old condition would have accepted it
        assert!(lifted < 1u128 << 127, "premise: the bias alone fits");
        assert_eq!(
            (lifted + worst) - (i128::MAX as u128),
            1_152_921_504_606_846_914,
            "premise: the sum does not"
        );
        assert!(
            checked_wrap_bias(q, d).is_none(),
            "accepted a modulus whose accumulator overflows i128"
        );
    }

    /// The reciprocal agrees with `/` everywhere `round_p` can reach it.
    ///
    /// A wrong magic constant would not fail loudly: it would shift the
    /// VRF value by one at a handful of coefficients, which is a
    /// different function that still verifies against itself.  So the
    /// agreement is checked at both ends of the proved range, at every
    /// multiple-of-`q_0` boundary near them, and over a wide sample.
    #[test]
    fn the_reciprocal_agrees_with_division_over_the_proved_range() {
        for par in crate::params::PROFILES {
            let q0 = par.q0;
            let r = Reciprocal::new(q0).expect("q_0 is small");
            assert_eq!(r.divisor(), q0);
            let top = par.q() - 1;
            assert!(
                top < 1u64 << 62,
                "{} q is outside the proved range",
                par.name
            );

            let mut cases: Vec<u64> = vec![0, 1, q0 - 1, q0, q0 + 1, top];
            // every boundary within a few multiples of each end
            for k in 0..8u64 {
                cases.push(k * q0);
                cases.push(k * q0 + q0 - 1);
                cases.push(top - k);
                cases.push(top / q0 * q0 + k.min(q0 - 1));
            }
            let mut state = 0x243F_6A88_85A3_08D3u64 ^ par.q();
            for _ in 0..20_000 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                cases.push((state >> 2) % par.q());
            }
            for c in cases {
                assert_eq!(r.div(c), c / q0, "{}: {c} / {q0}", par.name);
            }
        }
        // and the guard: a divisor past the proved range is refused
        // rather than answered wrongly.
        assert!(Reciprocal::new(0).is_none());
        assert!(Reciprocal::new(256).is_some());
        assert!(Reciprocal::new(257).is_none());
        assert!(Reciprocal::new(u64::MAX).is_none());
    }

    /// `mod_pm` masks rather than dividing, and still ties upward.
    #[test]
    fn mod_pm_is_masked_and_keeps_the_tie_convention() {
        for k in 1..=40u32 {
            let power = 1i128 << k;
            let half = power >> 1;
            for v in [
                0,
                1,
                -1,
                half,
                half + 1,
                -half,
                -half - 1,
                power,
                power + half,
                -power - half,
                power * 3 + half,
            ] {
                // the definition: centred representative in
                // `(-2^{k-1}, 2^{k-1}]`, closed at the top
                let naive = {
                    let mut low = v.rem_euclid(power);
                    if low > power / 2 {
                        low -= power;
                    }
                    low
                };
                let got = mod_pm(v, k);
                assert_eq!(got, naive, "mod_pm({v}, {k})");
                assert!(got > -half && got <= half, "mod_pm({v}, {k}) = {got}");
                assert_eq!((v - got) % power, 0, "not congruent");
            }
        }
        // the tie itself: `+2^{k-1}` stays, `+2^{k-1}+1` wraps down
        assert_eq!(mod_pm(1 << 7, 8), 1 << 7);
        assert_eq!(mod_pm((1 << 7) + 1, 8), -((1 << 7) - 1));
    }

    #[test]
    fn the_wrap_bias_helper_rejects_degenerate_rings() {
        assert!(checked_wrap_bias(0, 32).is_none(), "q = 0");
        // `q = 1` used to divide by zero
        assert!(checked_wrap_bias(1, 32).is_none(), "q = 1");
        // `d = 0` used to return Some(0)
        assert!(checked_wrap_bias(7, 0).is_none(), "d = 0");
        // and the shipped moduli still pass
        assert!(checked_wrap_bias(crate::params::QHAT_49, 32).is_some());
    }
}
