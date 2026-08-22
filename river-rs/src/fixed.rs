//! Exact big-integer and fixed-point arithmetic — the acceptance thresholds.
//!
//! Every accept/reject decision in RiVeR (Gaussian sampling, `Rej_1`,
//! `Rej_2`) compares a uniform `PROB_BITS`-wide integer against
//! `floor(scale · exp(num/den))`.  That integer is **wire-visible**: a
//! threshold off by one flips a decision, which changes a mask, which
//! changes every byte downstream.  So it has to be computed exactly, and
//! identically to `river-py/sample.py`.
//!
//! `river-py` gets there with `decimal` at a pinned precision, relying on
//! `Decimal.exp()` being correctly rounded by specification.  There is no
//! equivalent in `std`, and a `f64::exp` comparison would fork a test
//! vector on the last ulp.  This module computes the *mathematically
//! exact* floor instead:
//!
//! 1. evaluate `e^{-X}` in fixed point at `f` fractional bits, carrying a
//!    rigorous error bound in ulps;
//! 2. bracket `floor(scale · e^{-X})` between the two endpoints;
//! 3. if the bracket spans more than one integer, redo at higher `f`.
//!
//! Step 3 terminates because `e^{-X}` is transcendental for rational
//! `X != 0` (Lindemann), so `scale · e^{-X}` is never exactly an integer
//! on the path that reaches it.  In practice `f = 128` settles it; the
//! escalation exists so that "close call" is never "wrong answer".
//!
//! Agreement with the Python reference is not proved here, it is
//! *measured*: `Decimal` at 80 significant digits leaves ~22 digits of
//! slack over a `2^192` fixed point, so its floor is the exact floor
//! unless the true value sits within ~`10^-21` of an integer.
//! `tests/kat.rs` pins the shared values.

use core::cmp::Ordering;

// =========================================================================
// Nat — minimal arbitrary-precision unsigned integer
// =========================================================================

/// Arbitrary-precision unsigned integer, little-endian `u64` limbs,
/// always normalized (no trailing zero limb).
///
/// Deliberately minimal: this exists to make one predicate exact, not to
/// be a general bignum library.  No division by a general divisor beyond
/// what [`Nat::div_rem`] needs, no signed type beyond [`Int`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Nat {
    limbs: Vec<u64>,
}

impl Nat {
    pub const fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    pub fn from_u64(v: u64) -> Self {
        let mut n = Self { limbs: vec![v] };
        n.normalize();
        n
    }

    pub fn from_u128(v: u128) -> Self {
        let mut n = Self {
            limbs: vec![v as u64, (v >> 64) as u64],
        };
        n.normalize();
        n
    }

    /// Little-endian byte interpretation, as `int.from_bytes(b, "little")`.
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut limbs = vec![0u64; bytes.len().div_ceil(8)];
        for (i, &b) in bytes.iter().enumerate() {
            limbs[i / 8] |= (b as u64) << (8 * (i % 8));
        }
        let mut n = Self { limbs };
        n.normalize();
        n
    }

    /// `2^k`.
    pub fn pow2(k: u32) -> Self {
        let mut limbs = vec![0u64; (k / 64 + 1) as usize];
        limbs[(k / 64) as usize] = 1u64 << (k % 64);
        Self { limbs }
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    /// Limb `i`, little-endian, or `0` past the end.
    ///
    /// The fixed-width sampler path needs to read a `Nat` back out
    /// without going through bytes; everything else in the crate treats
    /// a `Nat` as opaque.
    pub fn limb(&self, i: usize) -> u64 {
        self.limbs.get(i).copied().unwrap_or(0)
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Number of significant bits; `0` for zero.  Matches Python's
    /// `int.bit_length()`.
    pub fn bit_len(&self) -> u32 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => (self.limbs.len() as u32 - 1) * 64 + (64 - top.leading_zeros()),
        }
    }

    /// Low 128 bits, saturating (used only for diagnostics and tests).
    pub fn low_u128(&self) -> u128 {
        let lo = *self.limbs.first().unwrap_or(&0) as u128;
        let hi = *self.limbs.get(1).unwrap_or(&0) as u128;
        lo | (hi << 64)
    }

    pub fn add(&self, other: &Nat) -> Nat {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = 0u64;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0);
            let b = *other.limbs.get(i).unwrap_or(&0);
            let (s1, c1) = a.overflowing_add(b);
            let (s2, c2) = s1.overflowing_add(carry);
            out.push(s2);
            carry = (c1 as u64) + (c2 as u64);
        }
        if carry != 0 {
            out.push(carry);
        }
        let mut r = Nat { limbs: out };
        r.normalize();
        r
    }

    pub fn add_u64(&self, v: u64) -> Nat {
        self.add(&Nat::from_u64(v))
    }

    /// `self - other`.  Panics if `other > self`.
    ///
    /// The panic is a real `assert!`, not a `debug_assert!`: every
    /// recommended invocation here is `cargo test --release`, and under a
    /// `debug_assert!` an underflow returned a plausible-looking natural
    /// instead — a wrong acceptance threshold with no other symptom.  The
    /// check is on the final borrow rather than an up-front comparison,
    /// so it costs one branch rather than a second pass over the limbs.
    pub fn sub(&self, other: &Nat) -> Nat {
        // The borrow check below only sees limbs the loop visits, and the
        // loop runs to `self`'s length.  A longer `other` is normalized,
        // so its top limb is nonzero and it is strictly larger — an
        // underflow the final borrow would miss entirely.
        assert!(other.limbs.len() <= self.limbs.len(), "Nat::sub underflow");
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0u64;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i];
            let b = *other.limbs.get(i).unwrap_or(&0);
            let (d1, b1) = a.overflowing_sub(b);
            let (d2, b2) = d1.overflowing_sub(borrow);
            out.push(d2);
            borrow = (b1 as u64) + (b2 as u64);
        }
        assert_eq!(borrow, 0, "Nat::sub underflow");
        let mut r = Nat { limbs: out };
        r.normalize();
        r
    }

    /// `self - v`, saturating at zero.
    pub fn saturating_sub(&self, other: &Nat) -> Nat {
        if self <= other {
            Nat::zero()
        } else {
            self.sub(other)
        }
    }

    pub fn mul(&self, other: &Nat) -> Nat {
        if self.is_zero() || other.is_zero() {
            return Nat::zero();
        }
        let mut out = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            if a == 0 {
                continue;
            }
            let mut carry = 0u128;
            for (j, &b) in other.limbs.iter().enumerate() {
                let cur = out[i + j] as u128 + (a as u128) * (b as u128) + carry;
                out[i + j] = cur as u64;
                carry = cur >> 64;
            }
            let mut k = i + other.limbs.len();
            while carry != 0 {
                let cur = out[k] as u128 + carry;
                out[k] = cur as u64;
                carry = cur >> 64;
                k += 1;
            }
        }
        let mut r = Nat { limbs: out };
        r.normalize();
        r
    }

    pub fn mul_u64(&self, v: u64) -> Nat {
        self.mul(&Nat::from_u64(v))
    }

    pub fn shl(&self, bits: u32) -> Nat {
        if self.is_zero() {
            return Nat::zero();
        }
        let limb_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        let mut out = vec![0u64; limb_shift];
        if bit_shift == 0 {
            out.extend_from_slice(&self.limbs);
        } else {
            let mut carry = 0u64;
            for &l in &self.limbs {
                out.push((l << bit_shift) | carry);
                carry = l >> (64 - bit_shift);
            }
            if carry != 0 {
                out.push(carry);
            }
        }
        let mut r = Nat { limbs: out };
        r.normalize();
        r
    }

    pub fn shr(&self, bits: u32) -> Nat {
        let limb_shift = (bits / 64) as usize;
        if limb_shift >= self.limbs.len() {
            return Nat::zero();
        }
        let bit_shift = bits % 64;
        let src = &self.limbs[limb_shift..];
        let mut out = Vec::with_capacity(src.len());
        if bit_shift == 0 {
            out.extend_from_slice(src);
        } else {
            for i in 0..src.len() {
                let hi = *src.get(i + 1).unwrap_or(&0);
                out.push((src[i] >> bit_shift) | (hi << (64 - bit_shift)));
            }
        }
        let mut r = Nat { limbs: out };
        r.normalize();
        r
    }

    /// `floor(self / v)` and the remainder, for a single-limb divisor.
    pub fn div_rem_small(&self, v: u64) -> (Nat, u64) {
        assert!(v != 0, "division by zero");
        let mut out = vec![0u64; self.limbs.len()];
        let mut rem = 0u128;
        for i in (0..self.limbs.len()).rev() {
            let cur = (rem << 64) | self.limbs[i] as u128;
            out[i] = (cur / v as u128) as u64;
            rem = cur % v as u128;
        }
        let mut q = Nat { limbs: out };
        q.normalize();
        (q, rem as u64)
    }

    pub fn div_small(&self, v: u64) -> Nat {
        self.div_rem_small(v).0
    }

    /// `floor(self / d)` and `self mod d`, by shift-and-subtract.
    ///
    /// Not Knuth's algorithm D: this runs in `O(bits · limbs)` and is
    /// called once per exponential evaluation, not per limb.  Clarity
    /// wins here; if the sampler ever becomes the bottleneck the fix is
    /// to remove the division entirely (see [`ExpCtx`]), not to make it
    /// faster.
    pub fn div_rem(&self, d: &Nat) -> (Nat, Nat) {
        assert!(!d.is_zero(), "division by zero");
        if self < d {
            return (Nat::zero(), self.clone());
        }
        let shift = self.bit_len() - d.bit_len();
        let mut rem = self.clone();
        let mut cur = d.shl(shift);
        let mut quo = vec![0u64; (shift / 64 + 1) as usize];
        for i in (0..=shift).rev() {
            if cur <= rem {
                rem = rem.sub(&cur);
                quo[(i / 64) as usize] |= 1u64 << (i % 64);
            }
            if i > 0 {
                cur = cur.shr(1);
            }
        }
        let mut q = Nat { limbs: quo };
        q.normalize();
        (q, rem)
    }

    pub fn div(&self, d: &Nat) -> Nat {
        self.div_rem(d).0
    }

    /// `floor(self · other / 2^f)` — the fixed-point product.
    pub fn mul_shr(&self, other: &Nat, f: u32) -> Nat {
        self.mul(other).shr(f)
    }

    /// Parse a decimal string.  `None` on any non-digit.
    pub fn from_dec_str(s: &str) -> Option<Nat> {
        let mut acc = Nat::zero();
        for ch in s.chars() {
            let d = ch.to_digit(10)?;
            acc = acc.mul_u64(10).add_u64(d as u64);
        }
        Some(acc)
    }

    /// Parse a lowercase or uppercase hex string.  `None` on any
    /// non-hex-digit.
    pub fn from_hex_str(s: &str) -> Option<Nat> {
        let mut acc = Nat::zero();
        for ch in s.chars() {
            let d = ch.to_digit(16)?;
            acc = acc.mul_u64(16).add_u64(d as u64);
        }
        Some(acc)
    }

    /// Lowercase hex, no leading zeros, `"0"` for zero — the format
    /// Python's `format(value, "x")` produces.
    pub fn to_hex_string(&self) -> String {
        if self.is_zero() {
            return "0".into();
        }
        let mut s = String::new();
        for (i, limb) in self.limbs.iter().enumerate().rev() {
            if i == self.limbs.len() - 1 {
                s.push_str(&format!("{limb:x}"));
            } else {
                s.push_str(&format!("{limb:016x}"));
            }
        }
        s
    }
}

impl Ord for Nat {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {}
            o => return o,
        }
        for i in (0..self.limbs.len()).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => {}
                o => return o,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for Nat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Sign-magnitude integer.  Only the exponent numerator needs one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int {
    pub neg: bool,
    pub mag: Nat,
}

impl Int {
    pub fn from_i128(v: i128) -> Self {
        Self {
            neg: v < 0,
            mag: Nat::from_u128(v.unsigned_abs()),
        }
    }

    pub fn neg_mag(mag: Nat) -> Self {
        Self { neg: true, mag }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_zero()
    }
}

// =========================================================================
// fixed-point exponential
// =========================================================================

/// Fractional bits tried, in order, by [`exp_accept`].
///
/// The first rung settles the *comparison* essentially always without
/// pinning the threshold itself: at `f = 128` the bracket is `2^64`
/// wide against a `scale` of `2^192`, so a uniform `u` falls inside it
/// with probability about `2^-109`.  The rest exist so a near-integer
/// threshold escalates instead of being decided wrongly.
const ACCEPT_LADDER: [u32; 4] = [128, 448, 1024, 2048];

/// Fractional bits tried, in order, by [`exp_threshold`].
///
/// Starts above `PROB_BITS`: pinning the exact integer needs the
/// bracket narrower than 1, and `scale/2^f` is the bracket's scale
/// factor, so anything at or below 192 fractional bits can never
/// settle it.
const THRESHOLD_LADDER: [u32; 3] = [448, 1024, 2048];

/// `e^{-X}` in fixed point, where `x_fix = round(X · 2^f)` and `X >= 0`.
///
/// Returns `(E, err)` with `|E - e^{-X}·2^f| <= err`, both as ulps at
/// `2^-f`.  Argument reduction halves `X` down to `<= 1/2`, sums the
/// alternating Taylor series until the term vanishes, then squares back.
///
/// The bound is deliberately loose: it is multiplied into a bracket that
/// is checked for ambiguity, so slack costs a retry and never an answer.
fn exp_neg(x_fix: &Nat, f: u32) -> (Nat, Nat) {
    let one = Nat::pow2(f);
    if x_fix.is_zero() {
        return (one, Nat::zero());
    }

    // X < 2^(bit_len - f); halve until X/2^k <= 1/2.
    let xb = x_fix.bit_len();
    let k = if xb > f { xb - f + 1 } else { 0 };
    debug_assert!(k < 64, "exp_neg: argument reduction out of range");
    let r = x_fix.shr(k);

    // Alternating Taylor series for e^{-r}.  Partial sums stay in
    // (0, 1], so the subtraction never underflows.
    let mut term = one.clone();
    let mut acc = one;
    let mut n: u64 = 1;
    loop {
        term = term.mul_shr(&r, f).div_small(n);
        if term.is_zero() {
            break;
        }
        acc = if n % 2 == 1 {
            acc.sub(&term)
        } else {
            acc.add(&term)
        };
        n += 1;
        assert!(n < 100_000, "exp_neg: Taylor series failed to terminate");
    }

    // e^{-X} = (e^{-r})^{2^k}
    for _ in 0..k {
        acc = acc.mul_shr(&acc, f);
    }

    // Truncation: <= 1 ulp per Taylor term and per squaring, and the
    // squarings double whatever came before.  `(terms + 2k + 4) << k`
    // dominates all of it with room to spare.
    let err = Nat::from_u128((n as u128 + 2 * k as u128 + 4) << k);
    (acc, err)
}

/// Bracket `floor(scale · e^{-mag/den})` at `f` fractional bits.
fn threshold_bracket(mag: &Nat, den: &Nat, scale: &Nat, f: u32) -> (Nat, Nat) {
    // x_fix = floor(mag · 2^f / den), so the represented X is at most one
    // ulp below the true one; e^{-X} is correspondingly at most one ulp
    // high, which the +1 absorbs.
    let x_fix = mag.shl(f).div(den);
    let (e, err) = exp_neg(&x_fix, f);
    let err = err.add_u64(1);

    let lo_e = e.saturating_sub(&err);
    let hi_e = e.add(&err);
    (scale.mul(&lo_e).shr(f), scale.mul(&hi_e).shr(f))
}

/// `floor(scale · exp(num / den))`, exactly.
///
/// Mirrors `river-py/sample.py::exp_threshold`, including its two early
/// exits: a threshold that has floored to zero, and an exponent `>= 0`
/// clamped to `scale`.
pub fn exp_threshold(num: &Int, den: &Nat, scale: &Nat) -> Nat {
    assert!(!den.is_zero(), "exp_threshold: zero denominator");

    // exp(x) >= 1 for x >= 0; the callers all want that clamped.
    if !num.neg || num.is_zero() {
        return scale.clone();
    }

    // Below -(bit_length(scale) + 1) the product floors to zero.  Decided
    // in integers, exactly as the Python does, so the two agree on the
    // boundary rather than merely near it.
    let cutoff = den.mul_u64(scale.bit_len() as u64 + 1);
    if num.mag > cutoff {
        return Nat::zero();
    }

    for &f in &THRESHOLD_LADDER {
        let (lo, hi) = threshold_bracket(&num.mag, den, scale, f);
        if lo == hi {
            return lo;
        }
    }
    panic!("exp_threshold: bracket did not converge at 2048 fractional bits");
}

/// `u < floor(scale · exp(num / den))`, decided exactly.
///
/// Same predicate as comparing against [`exp_threshold`] — and, like the
/// Python reference's two-stage form, it usually settles before the
/// exact value is needed: `u` is uniform over `2^PROB_BITS` and the
/// `f = 128` bracket spans a vanishing fraction of that range.
pub fn exp_accept(u: &Nat, num: &Int, den: &Nat, scale: &Nat) -> bool {
    assert!(!den.is_zero(), "exp_accept: zero denominator");

    if !num.neg || num.is_zero() {
        return u < scale;
    }
    let cutoff = den.mul_u64(scale.bit_len() as u64 + 1);
    if num.mag > cutoff {
        return false; // threshold is 0: never accept
    }

    for &f in &ACCEPT_LADDER {
        let (lo, hi) = threshold_bracket(&num.mag, den, scale, f);
        if u < &lo {
            return true; //  u < lo <= T
        }
        if u >= &hi {
            return false; //  T <= hi <= u
        }
        if lo == hi {
            return u < &lo;
        }
    }
    panic!("exp_accept: bracket did not converge at 2048 fractional bits");
}

// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s >> 11
        }
    }

    #[test]
    fn nat_roundtrips_u128() {
        for v in [0u128, 1, u64::MAX as u128, u64::MAX as u128 + 1, u128::MAX] {
            assert_eq!(Nat::from_u128(v).low_u128(), v);
        }
    }

    #[test]
    fn nat_bit_len_matches_python_semantics() {
        assert_eq!(Nat::zero().bit_len(), 0);
        assert_eq!(Nat::from_u64(1).bit_len(), 1);
        assert_eq!(Nat::pow2(192).bit_len(), 193);
    }

    #[test]
    fn nat_sub_underflow_panics_in_release_too() {
        // Under a `debug_assert!` these returned a plausible natural in
        // the release builds every recommended target uses.  Both shapes
        // matter: a same-length underflow, caught by the final borrow,
        // and a longer subtrahend, whose extra limbs the loop never
        // visits.
        for (a, b) in [
            (Nat::from_u64(5), Nat::from_u64(6)),
            (Nat::from_u64(0), Nat::from_u64(1)),
            (Nat::from_u64(1), Nat::pow2(200)),
        ] {
            assert!(
                std::panic::catch_unwind(|| a.sub(&b)).is_err(),
                "sub did not panic on underflow"
            );
        }
        // and the boundary is not over-eager
        assert_eq!(Nat::from_u64(6).sub(&Nat::from_u64(6)), Nat::from_u64(0));
    }

    #[test]
    fn nat_arithmetic_agrees_with_u128() {
        let mut r = lcg(7);
        for _ in 0..2000 {
            let a = (r() as u128) << 8 | r() as u128 & 0xff;
            let b = r() as u128;
            let (na, nb) = (Nat::from_u128(a), Nat::from_u128(b));
            assert_eq!(na.add(&nb).low_u128(), a + b);
            if a >= b {
                assert_eq!(na.sub(&nb).low_u128(), a - b);
            }
            // A precondition of the case, not a division to be made
            // checked: `div_rem` panics on a zero divisor by contract.
            if let (Some(want_q), Some(want_r)) = (a.checked_div(b), a.checked_rem(b)) {
                let (q, rem) = na.div_rem(&nb);
                assert_eq!(q.low_u128(), want_q, "{a} / {b}");
                assert_eq!(rem.low_u128(), want_r);
            }
        }
    }

    #[test]
    fn nat_mul_agrees_with_u128() {
        let mut r = lcg(11);
        for _ in 0..2000 {
            let a = r() as u128;
            let b = r() as u128;
            assert_eq!(
                Nat::from_u128(a).mul(&Nat::from_u128(b)).low_u128(),
                a.wrapping_mul(b)
            );
        }
    }

    #[test]
    fn nat_shifts_are_inverse_when_lossless() {
        let mut r = lcg(13);
        for _ in 0..500 {
            let v = Nat::from_u128(r() as u128);
            for bits in [1u32, 7, 64, 65, 200] {
                assert_eq!(v.shl(bits).shr(bits), v);
            }
        }
    }

    #[test]
    fn nat_from_bytes_le_matches_manual() {
        let bytes = [0x01u8, 0x02, 0x03];
        assert_eq!(Nat::from_bytes_le(&bytes).low_u128(), 0x030201);
    }

    #[test]
    fn exp_of_zero_is_the_scale() {
        let scale = Nat::pow2(192);
        let t = exp_threshold(&Int::from_i128(0), &Nat::from_u64(1), &scale);
        assert_eq!(t, scale);
    }

    #[test]
    fn exp_of_a_large_negative_is_zero() {
        let scale = Nat::pow2(192);
        // -1000 is far below the -(193+1) cutoff
        let t = exp_threshold(&Int::from_i128(-1000), &Nat::from_u64(1), &scale);
        assert!(t.is_zero());
    }

    #[test]
    fn exp_threshold_is_monotone_in_the_exponent() {
        let scale = Nat::pow2(192);
        let den = Nat::from_u64(1000);
        let mut prev: Option<Nat> = None;
        for a in (0..40_000).step_by(997) {
            let t = exp_threshold(&Int::from_i128(-(a as i128)), &den, &scale);
            if let Some(p) = prev {
                assert!(t <= p, "not monotone at a={a}");
            }
            prev = Some(t);
        }
    }

    #[test]
    fn exp_threshold_brackets_a_known_value() {
        // e^-1 = 0.36787944117144232159552377016146...
        // floor(2^192 · e^-1) has 191 bits and starts 0x5E2D...
        let t = exp_threshold(&Int::from_i128(-1), &Nat::from_u64(1), &Nat::pow2(192));
        assert_eq!(t.bit_len(), 191);
        // cross-check against a second precision: recomputing at 1024
        // fractional bits must land on the same integer
        let (lo, hi) =
            threshold_bracket(&Nat::from_u64(1), &Nat::from_u64(1), &Nat::pow2(192), 1024);
        assert_eq!(lo, hi);
        assert_eq!(lo, t);
    }

    #[test]
    fn exp_accept_agrees_with_the_exact_threshold() {
        // The Rust analogue of
        // `test_kat.py::test_exp_accept_agrees_with_the_exact_threshold`:
        // the bracket may change the cost, never the decision.
        let scale = Nat::pow2(192);
        let mut r = lcg(20_260_801);
        for _ in 0..200 {
            let den_v = 1 + r() % 1_000_000;
            let num_v = r() % (130 * den_v);
            let den = Nat::from_u64(den_v);
            let num = Int::neg_mag(Nat::from_u64(num_v));
            let exact = exp_threshold(&num, &den, &scale);
            let probes = [
                Nat::zero(),
                exact.saturating_sub(&Nat::from_u64(1)),
                exact.clone(),
                exact.add_u64(1),
                Nat::from_u128(r() as u128).mul(&Nat::from_u128(r() as u128)),
            ];
            for u in probes {
                assert_eq!(
                    exp_accept(&u, &num, &den, &scale),
                    u < exact,
                    "num=-{num_v} den={den_v}"
                );
            }
        }
    }

    #[test]
    fn every_threshold_rung_agrees() {
        let scale = Nat::pow2(192);
        let mut r = lcg(99);
        for _ in 0..40 {
            let den = Nat::from_u64(1 + r() % 100_000);
            let mag = Nat::from_u64(r() % 90_000);
            let mut seen: Option<Nat> = None;
            for &f in &THRESHOLD_LADDER {
                let (lo, hi) = threshold_bracket(&mag, &den, &scale, f);
                assert_eq!(lo, hi, "ambiguous at f={f}");
                if let Some(ref s) = seen {
                    assert_eq!(*s, lo, "precision {f} disagrees");
                }
                seen = Some(lo);
            }
        }
    }

    #[test]
    fn the_accept_rung_cannot_pin_the_threshold_but_still_decides() {
        // f = 128 leaves a bracket of order 2^64 against a 2^192 scale,
        // which is why `exp_threshold` starts higher — and why
        // `exp_accept` can still use it: the bracket is a vanishing
        // fraction of `u`'s range.
        let scale = Nat::pow2(192);
        let den = Nat::from_u64(7);
        let mag = Nat::from_u64(22);
        let (lo, hi) = threshold_bracket(&mag, &den, &scale, 128);
        assert!(lo < hi, "expected an ambiguous bracket at f = 128");
        let width = hi.sub(&lo);
        assert!(width.bit_len() < 100, "bracket wider than expected");
        let exact = exp_threshold(&Int::neg_mag(mag.clone()), &den, &scale);
        assert!(lo <= exact && exact <= hi);
    }
}
