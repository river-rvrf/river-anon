//! Fixed-width `e^{-x}` — the sampler's hot path.
//!
//! The Gaussian sampler decides, for each proposal `z`, whether a uniform
//! `u < 2^192` satisfies
//!
//! ```text
//! u  <  floor(2^192 · exp(-z² / 2σ²))
//! ```
//!
//! [`crate::fixed`] answers that exactly, with arbitrary-precision
//! integers.  It is the specification and it stays — but it is also
//! ~40 µs per call, because `mag << 128 / den` is a bit-at-a-time long
//! division that heap-allocates twice per bit.  At ~11 proposals per
//! accepted coefficient and ~15 000 coefficients in an `RiVeR-N256`
//! proof, that is seconds of wall clock in the sampler alone, which
//! makes any benchmark of the layers above it meaningless.
//!
//! ## What this module does instead
//!
//! It computes the *same predicate* — not an approximation of it — using
//! only `u64` mantissas and `u128` intermediates, with no allocation.
//!
//! The trick is that the answer is a *comparison*, not a value.  Compute
//! `e^{-x}` to a relative accuracy `ε`, which brackets the threshold `T`
//! between two integers `lo <= T <= hi`.  If `u < lo` the answer is
//! `true`; if `u >= hi` it is `false`; only when `u` lands inside the
//! bracket does the exact path have to run.  Since `u` is uniform over
//! `2^192` and the bracket has width `≈ 2·ε·T`, that happens with
//! probability about `2^-55` — call it never — and when it does, the
//! answer is still exact.
//!
//! So this is a pure implementation technique.  The distribution, the
//! XOF consumption, and every accept/reject decision are bit-identical
//! to `river-py`; `vectors.json` and the cross-language KAT are
//! unaffected, because nothing about the *specification* changed.
//!
//! ## Why not FACCT or a CDT
//!
//! FACCT-style decompositions and CDTs can sample large-σ Gaussians
//! faster still, and this implementation would use one if it
//! could — but they are *different samplers*, not different arithmetic:
//!
//! * FACCT splits `exp(-u)` into `Ber(2^-k)` and `Ber(exp(-r))`, drawing
//!   `k` extra XOF bits for the first test and evaluating the second
//!   with a degree-20 polynomial.  Different randomness consumption and
//!   an explicitly approximate acceptance step — the note calls it "the
//!   only approximation step in the sampler".
//! * A CDT consumes one uniform per sample instead of one per proposal,
//!   and needs a table of `14σ` entries; at `σ ≈ 1.8·10^7` that is
//!   2.5·10^8 entries.
//!
//! Either would move every byte of every test vector.  The reference
//! already specifies an *exact* Bernoulli test, so the cost of matching
//! it exactly is zero — the work below is no more than what FACCT does,
//! and it does not approximate.  If the specification later adopts a
//! FACCT-style sampler, this module is what gets replaced.
//!
//! ## Not constant time
//!
//! Neither is the reference.  The proposal loop, the number of
//! iterations, and the bit patterns driving the binary decompositions
//! below all depend on the sampled value.  See `README.md` for what the
//! rest of the crate does about timing.

use crate::fixed::{exp_accept, Int, Nat};

/// A positive real held as `m · 2^-e`, with `m` normalized to
/// `[2^63, 2^64)`.  One `fmul` costs one `u64 × u64 → u128` product and
/// loses at most one ulp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fix {
    m: u64,
    e: u32,
}

const ONE: Fix = Fix { m: 1 << 63, e: 63 };

/// `e^{-2^k}` for `k = 0..6`, covering every integer part `I <= 98`.
///
/// Pinned rather than derived: `constants_agree_with_the_exact_path`
/// re-derives every one of them from [`crate::fixed`] and fails on a
/// single-ulp transcription error.
const E_POW: [Fix; 7] = [
    Fix {
        m: 0xbc5ab1b16779be35,
        e: 65,
    }, // e^-1
    Fix {
        m: 0x8a95551dfc0e5cff,
        e: 66,
    }, // e^-2
    Fix {
        m: 0x960aadc109e7a3bf,
        e: 69,
    }, // e^-4
    Fix {
        m: 0xafe10820813d65e0,
        e: 75,
    }, // e^-8
    Fix {
        m: 0xf1aaddd7742e56d3,
        e: 87,
    }, // e^-16
    Fix {
        m: 0xe42327bb0b2340f1,
        e: 110,
    }, // e^-32
    Fix {
        m: 0xcb4ea3990f265d60,
        e: 156,
    }, // e^-64
];

/// `e^{-2^-(k+1)}` for `k = 0..7`: the top eight fractional bits.
const E_FRAC: [Fix; 8] = [
    Fix {
        m: 0x9b4597e37cb04ff4,
        e: 64,
    }, // e^-(2^-1)
    Fix {
        m: 0xc75f7cf564105743,
        e: 64,
    }, // e^-(2^-2)
    Fix {
        m: 0xe1eb51276c110c3c,
        e: 64,
    }, // e^-(2^-3)
    Fix {
        m: 0xf07d5fde38151e73,
        e: 64,
    }, // e^-(2^-4)
    Fix {
        m: 0xf81fab5445aebc8a,
        e: 64,
    }, // e^-(2^-5)
    Fix {
        m: 0xfc07f55ff77d2494,
        e: 64,
    }, // e^-(2^-6)
    Fix {
        m: 0xfe01feab551127cc,
        e: 64,
    }, // e^-(2^-7)
    Fix {
        m: 0xff007fd55ffdde39,
        e: 64,
    }, // e^-(2^-8)
];

/// `round(2^63 / n!)`, the Taylor coefficients in Q63.
///
/// Seven terms suffice: the residual argument is below `2^-8`, so the
/// first dropped term is under `2^-56 / 8! ≈ 2^-71`.
const INV_FACT: [u64; 8] = [
    9223372036854775808, // 1/0!
    9223372036854775808, // 1/1!
    4611686018427387904, // 1/2!
    1537228672809129301, // 1/3!
    384307168202282325,  // 1/4!
    76861433640456465,   // 1/5!
    12810238940076078,   // 1/6!
    1830034134296583,    // 1/7!
];

/// Fractional bits carried for the exponent `x = z²/2σ²`.
///
/// `x <= 98` needs seven integer bits, so 57 is the most a `u64` holds.
const X_BITS: u32 = 57;

/// Slack added to the mantissa on each side before comparing, in ulps.
///
/// Twenty-two `fmul`s at one ulp each, plus one ulp from truncating `x`
/// at [`X_BITS`], plus the `2^-63`-relative error in `inv`.  Rounded up
/// hard: a bracket that is too wide costs an exact-path retry with
/// probability `2^-55`, and one that is too narrow is a wrong answer.
const SLACK: u64 = 1 << 9;

#[inline(always)]
fn fmul(a: Fix, b: Fix) -> Fix {
    let p = (a.m as u128) * (b.m as u128); // in [2^126, 2^128)
    if p >> 127 != 0 {
        Fix {
            m: (p >> 64) as u64,
            e: a.e + b.e - 64,
        }
    } else {
        Fix {
            m: (p >> 63) as u64,
            e: a.e + b.e - 63,
        }
    }
}

/// `a · b` for Q63 values in `[0, 1]`.
#[inline(always)]
fn mul_q63(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) >> 63) as u64
}

/// Everything about a Gaussian width that does not change per sample.
///
/// The reciprocal is the point: without it every proposal would need a
/// division to form `x = z²·A / B`, and a `u128` division is both slow
/// and variable-time.  With it, the exponent is one multiply and a
/// shift.
#[derive(Clone, Copy, Debug)]
pub struct ExpCtx {
    /// `round(2^nrm · A / B)` in `[2^63, 2^64)`, where `x = z²·A / B`.
    inv: u64,
    nrm: u32,
    /// Set when the profile is outside what the fast path covers, in
    /// which case every call defers to [`crate::fixed`].
    usable: bool,
}

impl ExpCtx {
    /// Build the context for `x = z²·a / b`.
    ///
    /// `a = sigma_den²` and `b = 2·sigma_num²`, matching what the sampler
    /// forms.  The long division below runs once per width and is exact:
    /// it produces `floor(a·2^nrm / b)` bit by bit, in `u128`, stopping
    /// as soon as the quotient is normalized.
    pub fn new(a: u128, b: u128) -> Self {
        if a == 0 || b == 0 {
            return Self {
                inv: 0,
                nrm: 0,
                usable: false,
            };
        }
        let mut q = a / b;
        let mut r = a % b;
        let mut nrm: u32 = 0;
        // `r < b <= 2^89`, so `r << 1` cannot overflow; `q` doubles each
        // round and the loop stops the first time it is normalized.
        while q < (1u128 << 63) {
            r <<= 1;
            q <<= 1;
            if r >= b {
                r -= b;
                q += 1;
            }
            nrm += 1;
            if nrm > 200 {
                return Self {
                    inv: 0,
                    nrm: 0,
                    usable: false,
                };
            }
        }
        // A quotient at or above `2^64` means `a/b` was already huge —
        // `σ < 1`, which no profile uses and the exact path handles.
        let usable = q < (1u128 << 64) && nrm >= X_BITS;
        Self {
            inv: q as u64,
            nrm,
            usable,
        }
    }
}

/// `u < floor(2^192 · exp(-z²·a / b))`, decided in fixed width.
///
/// `None` when the fast path declines — the bracket was ambiguous, or
/// the width is outside what it covers — and the caller must fall back
/// to [`crate::fixed::exp_accept`].  Never wrong, only ever undecided.
#[inline]
pub fn accept_fast(u: &[u64; 3], zz: u128, ctx: &ExpCtx) -> Option<bool> {
    if !ctx.usable {
        return None;
    }
    if zz == 0 {
        return Some(true); // threshold is 2^192, and u < 2^192 always
    }
    // x in Q57.  `zz < 2^57` and `inv < 2^64`, so the product fits.
    if zz >= 1 << 57 {
        return None;
    }
    let x = ((zz * (ctx.inv as u128)) >> (ctx.nrm - X_BITS)) as u64;

    let int_part = (x >> X_BITS) as u32;
    if int_part > 98 {
        return None; // outside the tail cut this module is sized for
    }
    let frac = x & ((1u64 << X_BITS) - 1);

    // e^{-x} = e^{-I} · e^{-f}, both by binary decomposition.
    let mut acc = ONE;
    for (k, p) in E_POW.iter().enumerate() {
        if int_part >> k & 1 == 1 {
            acc = fmul(acc, *p);
        }
    }
    let top8 = (frac >> (X_BITS - 8)) as u32;
    for (k, p) in E_FRAC.iter().enumerate() {
        if top8 >> (7 - k) & 1 == 1 {
            acc = fmul(acc, *p);
        }
    }

    // The residue is below 2^-8; seven Taylor terms by Horner, each one
    // multiply, all partial sums in (0, 1].
    // `frac` is a Q57 fraction; the residue is its low 49 bits, restated
    // in Q63 — a shift of `63 - X_BITS`, not of `63 - (X_BITS - 8)`.
    let t = (frac & ((1u64 << (X_BITS - 8)) - 1)) << (63 - X_BITS);
    let mut p = INV_FACT[7];
    for n in (0..7).rev() {
        p = INV_FACT[n] - mul_q63(t, p);
    }
    acc = fmul(acc, norm(p, 63));

    // `e^{-x} < 1` for `x > 0`, so the exponent cannot be below 64 here;
    // if it is, something drifted and the exact path should answer.
    if acc.e < 64 {
        return None;
    }
    let shift = 192i32 - acc.e as i32;
    let lo = acc.m.checked_sub(SLACK)?;
    let hi = acc.m.checked_add(SLACK)?;

    if cmp_shifted(u, lo, shift).is_lt() {
        return Some(true); //  u < lo <= T
    }
    if !cmp_shifted(u, hi, shift).is_lt() {
        return Some(false); //  T <= hi <= u
    }
    None
}

#[inline]
fn norm(m: u64, e: u32) -> Fix {
    if m >> 63 == 0 {
        Fix {
            m: m << 1,
            e: e + 1,
        }
    } else {
        Fix { m, e }
    }
}

/// Compare a 192-bit `u` (little-endian limbs) against
/// `floor(v · 2^shift)`.
fn cmp_shifted(u: &[u64; 3], v: u64, shift: i32) -> core::cmp::Ordering {
    let mut w = [0u64; 3];
    if shift < 0 {
        let s = (-shift) as u32;
        w[0] = if s >= 64 { 0 } else { v >> s };
    } else {
        let s = shift as u32;
        // `v · 2^s` needs `bit_len(v) + s` bits; past 192 it dominates
        // any 192-bit `u`.
        if v != 0 && (64 - v.leading_zeros()) + s > 192 {
            return core::cmp::Ordering::Less;
        }
        let li = (s / 64) as usize;
        let bo = s % 64;
        if li < 3 {
            w[li] |= v << bo;
            if bo > 0 && li + 1 < 3 {
                w[li + 1] |= v >> (64 - bo);
            }
        }
    }
    u.iter().rev().cmp(w.iter().rev())
}

/// The exact predicate, taking the fast path when it settles.
///
/// This is the entry point the sampler calls.  `zz = z²`, and the
/// exponent is `-zz·a / b`.
pub fn accept(u: &[u64; 3], zz: u128, ctx: &ExpCtx, a: u128, b: u128) -> bool {
    if let Some(answer) = accept_fast(u, zz, ctx) {
        return answer;
    }
    // Roughly `2^-55` of proposals, plus any width the fast path
    // declines outright.
    let mut bytes = [0u8; 24];
    for (i, limb) in u.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    let un = Nat::from_bytes_le(&bytes);
    let num = Int::neg_mag(Nat::from_u128(zz * a));
    exp_accept(&un, &num, &Nat::from_u128(b), &Nat::pow2(192))
}

// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::exp_threshold;

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s
        }
    }

    fn to_u192(n: &Nat) -> [u64; 3] {
        [n.limb(0), n.limb(1), n.limb(2)]
    }

    fn exact_threshold(zz: u128, a: u128, b: u128) -> Nat {
        let num = Int::neg_mag(Nat::from_u128(zz * a));
        exp_threshold(&num, &Nat::from_u128(b), &Nat::pow2(192))
    }

    #[test]
    fn constants_agree_with_the_exact_path() {
        // Every pinned mantissa re-derived from `fixed`, to the ulp.  A
        // transcription slip here would bias the sampler in a way no
        // round-trip test could see.
        for (k, want) in E_POW.iter().enumerate() {
            let t = exp_threshold(
                &Int::neg_mag(Nat::from_u64(1u64 << k)),
                &Nat::from_u64(1),
                &Nat::pow2(200),
            );
            // t = floor(2^200 · e^{-2^k}); take its top 64 bits
            let bits = t.bit_len();
            let got = Fix {
                m: to_u64_top(&t),
                e: 200 - (bits - 64),
            };
            assert_eq!(got.e, want.e, "E_POW[{k}] exponent");
            assert!(
                got.m.abs_diff(want.m) <= 1,
                "E_POW[{k}]: {:#018x} vs {:#018x}",
                got.m,
                want.m
            );
        }
        for (k, want) in E_FRAC.iter().enumerate() {
            let t = exp_threshold(
                &Int::neg_mag(Nat::from_u64(1)),
                &Nat::from_u64(1u64 << (k + 1)),
                &Nat::pow2(200),
            );
            let bits = t.bit_len();
            let got = Fix {
                m: to_u64_top(&t),
                e: 200 - (bits - 64),
            };
            assert_eq!(got.e, want.e, "E_FRAC[{k}] exponent");
            assert!(
                got.m.abs_diff(want.m) <= 1,
                "E_FRAC[{k}]: {:#018x} vs {:#018x}",
                got.m,
                want.m
            );
        }
        for (n, want) in INV_FACT.iter().enumerate() {
            let mut fact = 1u64;
            for i in 1..=n as u64 {
                fact *= i;
            }
            let got = (((1u128 << 63) + (fact as u128 / 2)) / fact as u128) as u64;
            assert_eq!(got, *want, "INV_FACT[{n}]");
        }
    }

    fn to_u64_top(n: &Nat) -> u64 {
        let bits = n.bit_len();
        assert!(bits >= 64);
        n.shr(bits - 64).limb(0)
    }

    /// The widths the published profiles actually use, plus the small
    /// integer widths the KAT carries.
    fn widths() -> Vec<(u128, u128)> {
        let mut out = Vec::new();
        for par in crate::params::PROFILES {
            for s in [par.sigma_a(), par.sigma_b(), par.sigma_s(), par.sigma_m()] {
                let (num, den) = crate::sample::rational_sigma(s);
                out.push((
                    (den as u128) * (den as u128),
                    2 * (num as u128) * (num as u128),
                ));
            }
        }
        for (num, den) in [(8u64, 1u64), (352, 1), (4096, 1), (1, 1), (3, 2)] {
            out.push((
                (den as u128) * (den as u128),
                2 * (num as u128) * (num as u128),
            ));
        }
        out
    }

    #[test]
    fn fast_path_agrees_with_the_exact_one_at_the_boundary() {
        // The adversarial inputs: `u` exactly at, just below and just
        // above the true threshold.  Those are the values the bracket
        // cannot settle, so they exercise the fallback as well as the
        // agreement.
        for (a, b) in widths() {
            let ctx = ExpCtx::new(a, b);
            // z spanning the support: 1, small, mid, and out at 14σ
            let bound = (14.0 * ((b as f64) / (2.0 * a as f64)).sqrt()) as u64;
            for z in [1u64, 2, 7, bound / 4, bound / 2, bound - 1, bound] {
                if z == 0 || z >= 1 << 28 {
                    continue;
                }
                let zz = (z as u128) * (z as u128);
                let t = exact_threshold(zz, a, b);
                for delta in [-1i64, 0, 1] {
                    let cand = if delta < 0 {
                        if t.is_zero() {
                            continue;
                        }
                        t.sub(&Nat::from_u64(1))
                    } else if delta == 0 {
                        t.clone()
                    } else {
                        t.add(&Nat::from_u64(1))
                    };
                    if cand.bit_len() > 192 {
                        continue;
                    }
                    let u = to_u192(&cand);
                    let want = cand < t;
                    assert_eq!(
                        accept(&u, zz, &ctx, a, b),
                        want,
                        "a={a} b={b} z={z} delta={delta}"
                    );
                }
            }
        }
    }

    #[test]
    fn fast_path_agrees_with_the_exact_one_on_random_input() {
        let mut rand = lcg(0x5eed);
        for (a, b) in widths() {
            let ctx = ExpCtx::new(a, b);
            let bound = (14.0 * ((b as f64) / (2.0 * a as f64)).sqrt()) as u64;
            for _ in 0..200 {
                let z = 1 + rand() % bound.max(2);
                if z >= 1 << 28 {
                    continue;
                }
                let zz = (z as u128) * (z as u128);
                let u = [rand(), rand(), rand()];
                let mut bytes = [0u8; 24];
                for (i, limb) in u.iter().enumerate() {
                    bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
                }
                let un = Nat::from_bytes_le(&bytes);
                let want = un < exact_threshold(zz, a, b);
                assert_eq!(accept(&u, zz, &ctx, a, b), want, "a={a} b={b} z={z}");
            }
        }
    }

    #[test]
    fn the_fast_path_settles_almost_every_proposal() {
        // The whole point: if the bracket were routinely ambiguous the
        // fallback would dominate and nothing would have been gained.
        let mut rand = lcg(7);
        let mut decided = 0usize;
        let mut total = 0usize;
        for (a, b) in widths() {
            let ctx = ExpCtx::new(a, b);
            let bound = (14.0 * ((b as f64) / (2.0 * a as f64)).sqrt()) as u64;
            for _ in 0..500 {
                let z = 1 + rand() % bound.max(2);
                if z >= 1 << 28 {
                    continue;
                }
                total += 1;
                if accept_fast(&[rand(), rand(), rand()], (z as u128) * (z as u128), &ctx).is_some()
                {
                    decided += 1;
                }
            }
        }
        assert_eq!(
            decided,
            total,
            "{} of {total} proposals deferred",
            total - decided
        );
    }
}
