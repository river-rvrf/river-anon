//! `R_q~ = Z_q~[X]/(X^d~ + 1)` for the LANES exact proof — port of
//! `river-py/lanes_ring.py`.
//!
//! The incomplete-NTT ring of `[ENS20]` Section 2, at the parameters the
//! paper fixes for LANES.
//!
//! ## The ring
//!
//! `d~ = 256`, `q~ = 67107713` (26 bits).  `q~ - 1 = 2^7 · 7 · 74897` with
//! the cofactor odd, so there is a primitive 128th root of unity and no
//! 256th; equivalently `q~ mod 512 = 385`, whose multiplicative order mod
//! 512 is four.  Either way `X^256 + 1` factors into **64 irreducible
//! blocks of degree 4**:
//!
//! ```text
//! X^256 + 1 = prod_{i<64} (X^4 - psi^{e_i}),   e_i odd, psi^64 = -1
//! ```
//!
//! That is the `l = 64` splitting factor the paper quotes, and it is why
//! the exact witness gives each message element a 64-slot block: one
//! scalar per NTT slot.  The transform is therefore *incomplete* — six
//! butterfly levels, stopping at degree-4 residues, which are then
//! multiplied directly.
//!
//! ## Arithmetic
//!
//! `q~ = 2^26 - 1151` is pseudo-Mersenne, and that is worth exploiting:
//! `2^26 = 1151 (mod q~)`, so folding a value's high part down costs a
//! shift, a mask and one small multiply — see [`reduce_product`].  Two
//! folds and one masked subtraction reduce any canonical product, which
//! is cheaper than the generic 128-bit Barrett this module used to route
//! every multiplication through.  [`crate::ring::Barrett`] is kept beside
//! it and a test drives both over the whole range, so the specialisation
//! is checked rather than trusted; `river-bench --lanes` measures them
//! against each other.
//!
//! The reduction budget is stated, not assumed:
//!
//! * `(q~ - 1)^2` is just below `2^52`, so **exactly 4096** maximum
//!   canonical products fit a `u64` — which is the bound on how much may
//!   be accumulated before reducing, and it is a bound on a *sum of
//!   products*, not a licence to leave butterfly levels unreduced.  One
//!   unreduced twiddle product already sits at `2^52`; a second would be
//!   near `2^78`.  So every twiddle product is reduced, and only additions
//!   and subtractions are ever lazy.
//! * A degree-4 fold accumulates four canonical products, `< 2^54`, and
//!   [`reduce_product`] covers everything below `2^57`.
//!
//! ## Shape and domain are in the type
//!
//! A `Vec<u64>` cannot distinguish a coefficient-domain polynomial from an
//! NTT-domain one, a canonical residue from a wild `u64`, or 256
//! coefficients from 255.  That is survivable in a leaf module and not
//! survivable underneath a proof system: `inner_ntt` silently truncated
//! unequal vectors to the shorter, `scale_blocks` read a short scalar list
//! as zero blocks and ignored extras, `add_slots_inplace` accepted partial
//! input and panicked on overlong, and a short polynomial panicked inside a
//! transform.  Every one of those becomes a malformed-proof path once a
//! verifier sits on top.
//!
//! So the three shapes are three types — [`CoeffPoly`], [`NttPoly`] and
//! [`Slots`] — each a fixed-size array of canonical residues, constructible
//! only through a checked constructor.  Length and canonicality are
//! established once, at the boundary, and every operation below is total.
//!
//! Slot `j` is index `j · SUBDEG` of an [`NttPoly`], the constant term of
//! the residue modulo `X^4 - zeta_j`.  That is what makes slots independent
//! under multiplication, which `[ENS20]` relies on throughout; placing them
//! in the coefficient domain would silently commit to a different message.
//!
//! ## Integration
//!
//! `river-py/lanes_ring.py` implements the same ring, its KATs are active,
//! and [`crate::exact`] commits over exactly this ring.
//! The layers above it run at the paper's own widths; only the production
//! `lanes` *name* is withheld, on security evidence — see
//! [`crate::exact::lanes_unavailable_reason`].

use crate::exact::ExactParams;
use crate::ring::Barrett;

/// `q~`, from the one place it is defined.
pub const QTILDE: u64 = ExactParams::Q_TILDE;
/// Internal LANES ring dimension, from [`ExactParams`].
pub const DTILDE: usize = ExactParams::D_TILDE;
/// NTT blocks, and message slots, from [`ExactParams`].
pub const LSPLIT: usize = ExactParams::L_SPLIT;
/// Degree of each residue.
pub const SUBDEG: usize = DTILDE / LSPLIT;
/// `log2(LSPLIT)` — the number of butterfly levels.
pub const LEVELS: usize = LSPLIT.trailing_zeros() as usize;
/// Order of `psi`: `X^{d~} + 1` needs a primitive `2l`-th root, and no
/// `4l`-th one, which is exactly the `q~ = 2l + 1 mod 4l` condition.
const PSI_ORDER: usize = 2 * LSPLIT;

// ---- modular arithmetic --------------------------------------------------

/// `q~ = 2^SHIFT - C`.
const SHIFT: u32 = 26;
const MASK: u64 = (1 << SHIFT) - 1;
/// `2^26 mod q~`.
const C: u64 = (1 << SHIFT) - QTILDE;

/// One pseudo-Mersenne fold: `x -> (x mod 2^26) + 1151 (x div 2^26)`.
///
/// Congruent to `x` because `2^26 = 1151 (mod q~)`, and much smaller: for
/// `x` above `2^26` the result is about `x / 58305 + 2^26`.
#[inline(always)]
const fn fold(x: u64) -> u64 {
    (x & MASK) + (x >> SHIFT) * C
}

/// `x` reduced into `[0, q~)`, for `x < 2^57`.
///
/// Two folds bring any such `x` below `2 q~`, and one masked subtraction
/// finishes it.  The bound is not decoration: at `x = 2^54 - 1` — four
/// maximal canonical products — the two folds land at 72,409,218, which is
/// under `2 q~ = 134,215,426`; the construction survives to about
/// `2^57.6`, and 57 is the round number below that.
///
/// Branchless: the conditional subtraction is an arithmetic-shift mask, so
/// nothing here is a data-dependent jump or a divide.
#[inline(always)]
const fn reduce_narrow(x: u64) -> u64 {
    debug_assert!(x < 1 << 57, "reduce_narrow is only valid below 2^57");
    let t = fold(fold(x));
    let d = t.wrapping_sub(QTILDE);
    // all-ones iff `t < q~`, in which case add `q~` back
    let mask = ((d as i64) >> 63) as u64;
    d.wrapping_add(QTILDE & mask)
}

/// `x` reduced into `[0, q~)`, for any `u64`.
///
/// One more fold than [`reduce_narrow`]: `2^64` needs three.  Fixed work,
/// so an accumulator whose magnitude depends on a secret does not change
/// the instruction sequence.
#[inline(always)]
const fn reduce_u64(x: u64) -> u64 {
    let t = fold(fold(fold(x)));
    let d = t.wrapping_sub(QTILDE);
    let mask = ((d as i64) >> 63) as u64;
    d.wrapping_add(QTILDE & mask)
}

/// `x` reduced into `[0, q~)`, for any `u128`.
///
/// Five `u128` folds bring `2^128` below `2^49`, then [`reduce_u64`]
/// finishes.  Fixed work rather than "fold while it does not fit", which
/// would make the iteration count a function of the operand's magnitude.
/// Only the schoolbook reference and the public scalar helper reach this;
/// the transforms never do.
#[inline(always)]
const fn reduce_u128(x: u128) -> u64 {
    let mut v = x;
    let mut i = 0;
    while i < 5 {
        v = (v & MASK as u128) + (v >> SHIFT) * C as u128;
        i += 1;
    }
    reduce_u64(v as u64)
}

/// `a + b (mod q~)`, branchless.
#[inline(always)]
const fn addm(a: u64, b: u64) -> u64 {
    // Both are canonical, so the sum is below `2 q~` and cannot overflow.
    let d = (a + b).wrapping_sub(QTILDE);
    let mask = ((d as i64) >> 63) as u64;
    d.wrapping_add(QTILDE & mask)
}

/// `a - b (mod q~)`, branchless.
///
/// The source-level `if` this replaces was the one place in the module a
/// compiler was free to emit a branch on a secret comparison.  LLVM
/// usually picks a conditional move; "usually" is not the guarantee
/// `README.md` makes, and the masked form does not depend on it.
#[inline(always)]
const fn subm(a: u64, b: u64) -> u64 {
    let d = a.wrapping_sub(b);
    let mask = ((d as i64) >> 63) as u64; // all-ones iff a < b
    d.wrapping_add(QTILDE & mask)
}

/// `a · b (mod q~)` for canonical `a`, `b`.
///
/// The product is below `2^52`, so [`reduce_narrow`] applies.
#[inline(always)]
const fn mulm(a: u64, b: u64) -> u64 {
    debug_assert!(a < QTILDE && b < QTILDE);
    reduce_narrow(a * b)
}

/// The hot-path reduction: a canonical product, in `[0, q~)`.
///
/// This is what every multiplication in the transforms goes through,
/// and it is the one worth
/// comparing against [`barrett_reduce`] — the public [`reduce`] takes a
/// `u128` and pays for the width it may not need.
#[inline]
pub fn reduce_product(x: u64) -> u64 {
    reduce_narrow(x)
}

/// The generic Barrett reduction, kept for comparison.
///
/// Not used by anything on the hot path: it is what the specialisation
/// replaced, and `tests::the_specialised_reduction_agrees_with_barrett`
/// drives both over the boundaries and a wide sample so the replacement is
/// checked rather than argued.  `river-bench --lanes` times them.
#[inline]
pub fn barrett_reduce(v: u128) -> u64 {
    tables().bar.reduce(v)
}

#[inline]
fn redm(v: u128) -> u64 {
    reduce_u128(v)
}

/// Everything derived from `q~`: the twiddles, the tree, the leaf zetas.
///
/// Built once.  The reference computes these at import; a `OnceLock` is the
/// same thing with the initialisation order made explicit.
struct Tables {
    bar: Barrett,
    psi_pow: [u64; PSI_ORDER],
    /// Twiddle *exponents* per level, in butterfly order.
    levels: Vec<Vec<usize>>,
    leaf_exps: Vec<usize>,
    leaf_zeta: Vec<u64>,
    inv_split: u64,
    /// `psi^{-e}` for every exponent the inverse transform needs.
    psi_inv: [u64; PSI_ORDER],
}

fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut acc = 1u128;
    let mut b = base as u128 % q as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * b % q as u128;
        }
        b = b * b % q as u128;
        exp >>= 1;
    }
    base = acc as u64;
    base
}

/// Smallest primitive root modulo a prime.
fn primitive_root(q: u64) -> u64 {
    let n = q - 1;
    let mut factors = Vec::new();
    let mut x = n;
    let mut f = 2u64;
    while f * f <= x {
        if x.is_multiple_of(f) {
            factors.push(f);
            while x.is_multiple_of(f) {
                x /= f;
            }
        }
        f += 1;
    }
    if x > 1 {
        factors.push(x);
    }
    (2..1000)
        .find(|&g| factors.iter().all(|&p| pow_mod(g, n / p, q) != 1))
        .expect("no primitive root found")
}

/// Twiddle exponents per level, and the `LSPLIT` leaf exponents.
///
/// A block `X^{2m} - psi^e` splits into `X^m - psi^{e/2}` and
/// `X^m + psi^{e/2} = X^m - psi^{e/2 + LSPLIT}`, so the tree is generated
/// by halving exponents and offsetting by `LSPLIT`.  Starting from
/// `X^{d~} - psi^{LSPLIT}` (`psi^{LSPLIT} = -1`), `LEVELS` levels leave
/// `LSPLIT` **odd** exponents -- which is exactly the condition for
/// `prod_j (X^{SUBDEG} - psi^{e_j})` to be `X^{d~} + 1`.
///
/// The body below derives both constants from `LSPLIT`, and must keep
/// doing so.  The Python reference wrote them out as `32` and `64 / 2` --
/// the values they take at `d~ = 128` -- and at `d~ = 256` that produced
/// the exponents `0..63`, evens included: a transform whose leaves
/// multiply to something other than `X^256 + 1`.  It round-tripped, and
/// disagreed with schoolbook on every coefficient.  See
/// `river-py/test_lanes_ring.py`, which also records which algebraic laws
/// fail to detect it (most of them do not).
fn build_tree() -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut exps = vec![LSPLIT];
    let mut levels = Vec::with_capacity(LEVELS);
    for _ in 0..LEVELS {
        levels.push(exps.iter().map(|e| e / 2).collect());
        exps = exps.iter().flat_map(|&e| [e / 2, e / 2 + LSPLIT]).collect();
    }
    (levels, exps)
}

fn tables() -> &'static Tables {
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let g = primitive_root(QTILDE);
        let psi = pow_mod(g, (QTILDE - 1) / PSI_ORDER as u64, QTILDE);
        assert_eq!(
            pow_mod(psi, LSPLIT as u64, QTILDE),
            QTILDE - 1,
            "psi^l must be -1: q~ - 1 is divisible by 2l and not by 4l"
        );
        let mut psi_pow = [0u64; PSI_ORDER];
        let mut psi_inv = [0u64; PSI_ORDER];
        for (i, slot) in psi_pow.iter_mut().enumerate() {
            *slot = pow_mod(psi, i as u64, QTILDE);
        }
        for (i, slot) in psi_inv.iter_mut().enumerate() {
            *slot = pow_mod(psi_pow[i], QTILDE - 2, QTILDE);
        }
        let (levels, leaf_exps) = build_tree();
        let leaf_zeta = leaf_exps.iter().map(|&e| psi_pow[e]).collect();
        Tables {
            bar: Barrett::new(QTILDE),
            inv_split: pow_mod(LSPLIT as u64, QTILDE - 2, QTILDE),
            psi_pow,
            psi_inv,
            levels,
            leaf_exps,
            leaf_zeta,
        }
    })
}

/// The `l` leaf exponents `e_i`, all odd and distinct.
pub fn leaf_exps() -> &'static [usize] {
    &tables().leaf_exps
}

/// `zeta_i = psi^{e_i}`: the `l` blocks are `X^{d~/l} - zeta_i`.
pub fn leaf_zeta() -> &'static [u64] {
    &tables().leaf_zeta
}

// ---- the three shapes ----------------------------------------------------

macro_rules! poly_type {
    ($name:ident, $n:expr, $what:literal) => {
        #[doc = concat!("A canonical ", $what, ".")]
        ///
        /// Fixed length, every coefficient in `[0, q~)`.  Constructible only
        /// through a checked constructor, so every operation on it is total.
        #[derive(Clone, PartialEq, Eq, Debug)]
        pub struct $name([u64; $n]);

        impl $name {
            /// All-zero.
            pub fn zero() -> Self {
                Self([0u64; $n])
            }

            /// `None` on the wrong length or a non-canonical residue.
            pub fn new(values: &[u64]) -> Option<Self> {
                if values.len() != $n || values.iter().any(|&c| c >= QTILDE) {
                    return None;
                }
                let mut out = [0u64; $n];
                out.copy_from_slice(values);
                Some(Self(out))
            }

            /// Reduce rather than reject — for values this crate produced.
            ///
            /// Barrett, not `%`: the caller may be reducing a mask.
            pub fn from_reduced(values: &[u64]) -> Option<Self> {
                if values.len() != $n {
                    return None;
                }
                let mut out = [0u64; $n];
                for (o, &v) in out.iter_mut().zip(values.iter()) {
                    *o = reduce_u64(v);
                }
                Some(Self(out))
            }

            /// From centred integers in `(-q~, q~)`.
            ///
            /// **No division.**  Every caller here is on a secret path —
            /// this is where Gaussian samples enter the ring — and
            /// `rem_euclid` is a divide.  A centred value is already within
            /// one modulus of the canonical range, so the reduction is a
            /// masked conditional add: `v + (q~ & -[v < 0])`, branchless
            /// and constant-time.
            ///
            /// The domain restriction is not a limitation.  A "centred"
            /// value outside `(-q~, q~)` is not centred, and admitting one
            /// would mean reintroducing the divide for an input no honest
            /// caller has; `None` says so.
            pub fn from_centered(values: &[i64]) -> Option<Self> {
                if values.len() != $n {
                    return None;
                }
                let q = QTILDE as i64;
                if values.iter().any(|&v| v <= -q || v >= q) {
                    return None;
                }
                let mut out = [0u64; $n];
                for (o, &v) in out.iter_mut().zip(values.iter()) {
                    let mask = (v >> 63) as u64; // all-ones iff v < 0
                    *o = (v as u64).wrapping_add(QTILDE & mask);
                }
                Some(Self(out))
            }

            pub fn as_slice(&self) -> &[u64] {
                &self.0
            }

            pub fn to_vec(&self) -> Vec<u64> {
                self.0.to_vec()
            }

            /// The centred representatives, in `(-q~/2, q~/2]`.
            pub fn centered(&self) -> Vec<i64> {
                let h = QTILDE / 2;
                self.0
                    .iter()
                    .map(|&c| {
                        if c > h {
                            c as i64 - QTILDE as i64
                        } else {
                            c as i64
                        }
                    })
                    .collect()
            }

            pub fn inf_norm(&self) -> i64 {
                self.centered()
                    .into_iter()
                    .map(|c| c.abs())
                    .max()
                    .unwrap_or(0)
            }

            pub fn l2_norm_sq(&self) -> i128 {
                self.centered()
                    .into_iter()
                    .map(|c| (c as i128) * (c as i128))
                    .sum()
            }

            pub fn add(&self, other: &Self) -> Self {
                Self(std::array::from_fn(|i| addm(self.0[i], other.0[i])))
            }

            pub fn sub(&self, other: &Self) -> Self {
                Self(std::array::from_fn(|i| subm(self.0[i], other.0[i])))
            }

            pub fn neg(&self) -> Self {
                Self(std::array::from_fn(|i| subm(0, self.0[i])))
            }

            /// Barrett on the scalar too — it may be a challenge
            /// coefficient or a mask, and `%` is a divide either way.
            pub fn scale(&self, c: u64) -> Self {
                let c = reduce_u64(c);
                Self(std::array::from_fn(|i| mulm(c, self.0[i])))
            }
        }
    };
}

poly_type!(CoeffPoly, DTILDE, "coefficient-domain polynomial");
poly_type!(NttPoly, DTILDE, "NTT-domain polynomial");
poly_type!(Slots, LSPLIT, "slot vector");

/// Forward incomplete NTT: coefficients → `l` blocks of degree `d~/l`.
///
/// Every twiddle product is reduced, by construction: leaving one lazy
/// would put the next level's operand near `2^52`, and the level after
/// that near `2^78`.  Only the butterfly's add and subtract are cheap,
/// and they are already canonical-in, canonical-out.
pub fn ntt(a: &CoeffPoly) -> NttPoly {
    let t = tables();
    let mut a = a.0;
    let mut m = DTILDE;
    for level in t.levels.iter().take(LEVELS) {
        let half = m >> 1;
        for (blk, &exp) in level.iter().enumerate() {
            let z = t.psi_pow[exp];
            let base = blk * m;
            for j in base..base + half {
                let u = a[j];
                let v = mulm(a[j + half], z);
                a[j] = addm(u, v);
                a[j + half] = subm(u, v);
            }
        }
        m = half;
    }
    NttPoly(a)
}

/// Inverse of [`ntt`].
pub fn intt(a: &NttPoly) -> CoeffPoly {
    let t = tables();
    let mut a = a.0;
    let mut m = SUBDEG;
    for level in t.levels.iter().take(LEVELS).rev() {
        for (blk, &exp) in level.iter().enumerate() {
            let z_inv = t.psi_inv[exp];
            let base = blk * (m << 1);
            for j in base..base + m {
                let (u, v) = (a[j], a[j + m]);
                a[j] = addm(u, v);
                a[j + m] = mulm(subm(u, v), z_inv);
            }
        }
        m <<= 1;
    }
    CoeffPoly(std::array::from_fn(|i| mulm(a[i], t.inv_split)))
}

/// Blockwise product in the NTT domain: `l` multiplications mod
/// `X^{d~/l} - zeta`.
///
/// The degree-4 convolution accumulates four canonical products, so it
/// stays under `2^54` in `u64` — no `u128` needed, and one
/// [`reduce_product`] per output coefficient rather than one per product.
pub fn ntt_mul(a: &NttPoly, b: &NttPoly) -> NttPoly {
    let t = tables();
    let mut out = [0u64; DTILDE];
    for blk in 0..LSPLIT {
        let z = t.leaf_zeta[blk];
        let base = blk * SUBDEG;
        // At most `SUBDEG` canonical products per entry: `4 · 2^52 = 2^54`,
        // inside `u64` and inside `reduce_narrow`'s domain.
        let mut acc = [0u64; 2 * SUBDEG - 1];
        for i in 0..SUBDEG {
            let xi = a.0[base + i];
            for j in 0..SUBDEG {
                acc[i + j] += xi * b.0[base + j];
            }
        }
        for k in 0..SUBDEG {
            let lo = reduce_narrow(acc[k]);
            out[base + k] = if k + SUBDEG < acc.len() {
                // `X^{SUBDEG} = zeta`, so the overflow limbs fold back in.
                // Reduce first: `z · acc` would otherwise be `2^26 · 2^54`.
                addm(lo, mulm(z, reduce_narrow(acc[k + SUBDEG])))
            } else {
                lo
            };
        }
    }
    NttPoly(out)
}

/// Coefficient-domain product, via the NTT.
pub fn mul(a: &CoeffPoly, b: &CoeffPoly) -> CoeffPoly {
    intt(&ntt_mul(&ntt(a), &ntt(b)))
}

/// Smallest multiple of `q~` at or above `d~ · (q~-1)^2`.
///
/// The negacyclic wrap makes the schoolbook accumulator negative, and
/// Barrett takes a `u128`; adding this lifts it without changing the
/// residue.  Computed once, from public constants — the only `%` left in
/// this module, and it never sees an operand.
const WRAP_BIAS: i128 = {
    let hi = (DTILDE as i128) * (QTILDE as i128 - 1) * (QTILDE as i128 - 1);
    hi - hi % QTILDE as i128 + QTILDE as i128
};

/// Negacyclic convolution from the definition.  Correctness reference.
///
/// No zero-coefficient skip, for the reason
/// [`crate::ring::Ring::mul_schoolbook`] gives: it branches on data that is
/// secret here — commitment randomness is ternary, so a third of its
/// coefficients are zero — and buys nothing where it was justified.
pub fn mul_schoolbook(a: &CoeffPoly, b: &CoeffPoly) -> CoeffPoly {
    let mut out = [0i128; DTILDE];
    for i in 0..DTILDE {
        let ai = a.0[i] as i128;
        for j in 0..DTILDE {
            let prod = ai * b.0[j] as i128;
            if i + j < DTILDE {
                out[i + j] += prod;
            } else {
                out[i + j - DTILDE] -= prod;
            }
        }
    }
    CoeffPoly(std::array::from_fn(|i| redm((out[i] + WRAP_BIAS) as u128)))
}

/// `sum_i u_i v_i`, both operands already transformed.
///
/// `None` on unequal lengths.  It used to `zip`, which silently truncates
/// to the shorter — an inner product over a *prefix*, which is a different
/// commitment rather than an error.
pub fn inner_ntt(u: &[NttPoly], v: &[NttPoly]) -> Option<NttPoly> {
    if u.len() != v.len() {
        return None;
    }
    // Each `ntt_mul` returns canonical residues, so the accumulator grows
    // by under `2^26` per term: `u64` overflows only past `2^38` terms,
    // and `reduce_u64` covers the whole range regardless.
    let mut acc = [0u64; DTILDE];
    for (a, b) in u.iter().zip(v.iter()) {
        let prod = ntt_mul(a, b);
        for (slot, &c) in acc.iter_mut().zip(prod.0.iter()) {
            *slot += c;
        }
    }
    Some(NttPoly(std::array::from_fn(|i| reduce_u64(acc[i]))))
}

// ---- slot access ---------------------------------------------------------

/// NTT-domain element carrying `values[j]` in slot `j`, zero elsewhere.
pub fn slots_to_ntt(values: &Slots) -> NttPoly {
    let mut out = [0u64; DTILDE];
    for (j, &v) in values.0.iter().enumerate() {
        out[j * SUBDEG] = v;
    }
    NttPoly(out)
}

/// Read slot values out of an NTT-domain element.
pub fn ntt_to_slots(hat: &NttPoly) -> Slots {
    Slots(std::array::from_fn(|j| hat.0[j * SUBDEG]))
}

/// Add `values[j]` into slot `j`.
///
/// Total by construction: [`Slots`] is exactly `LSPLIT` canonical values,
/// so there is no partial input to read as zeros and no overlong one to
/// panic on.
pub fn add_slots_inplace(hat: &mut NttPoly, values: &Slots) {
    for (j, &v) in values.0.iter().enumerate() {
        hat.0[j * SUBDEG] = addm(hat.0[j * SUBDEG], v);
    }
}

/// Multiply NTT block `j` by `scalars[j]`.
///
/// In the NTT domain this is multiplication by a slot-diagonal element,
/// which is how the linear proof's coefficients `phi` are applied.
pub fn scale_blocks(hat: &NttPoly, scalars: &Slots) -> NttPoly {
    let mut out = [0u64; DTILDE];
    for (j, &s) in scalars.0.iter().enumerate() {
        let base = j * SUBDEG;
        for k in 0..SUBDEG {
            out[base + k] = mulm(s, hat.0[base + k]);
        }
    }
    NttPoly(out)
}

/// Coefficient-domain constant term of an NTT-domain element.
///
/// The linear proof forces this to zero.
pub fn constant_coefficient(hat: &NttPoly) -> u64 {
    intt(hat).0[0]
}

/// This ring's reduction, for scalar arithmetic outside the three shapes.
///
/// The compression challenge `gamma` contracts a public linear system into
/// `phi` before any of it is a polynomial, so that arithmetic happens on
/// plain integers.  Doing it with `%` would be the only divide on a path
/// that touches `gamma` — public, but the crate's rule is about where a
/// divide is *reachable* from, not about auditing each one — and it would
/// also be a second reduction convention one module away from this one.
pub fn reduce(value: u128) -> u64 {
    reduce_u128(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) % QTILDE
        }
    }

    fn rand_coeff(next: &mut impl FnMut() -> u64) -> CoeffPoly {
        CoeffPoly::new(&(0..DTILDE).map(|_| next()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn the_ring_splits_into_64_blocks_of_degree_4() {
        assert_eq!((LSPLIT, SUBDEG, DTILDE), (64, 4, 256));
        let exps = leaf_exps();
        assert_eq!(exps.len(), LSPLIT);
        assert!(
            exps.iter().all(|e| e % 2 == 1),
            "leaf exponents must be odd"
        );
        let unique: std::collections::BTreeSet<_> = exps.iter().collect();
        assert_eq!(unique.len(), LSPLIT, "leaf exponents must be distinct");
        assert!((QTILDE - 1).is_multiple_of(PSI_ORDER as u64));
        assert!(!(QTILDE - 1).is_multiple_of(2 * PSI_ORDER as u64));
        assert_eq!(PSI_ORDER, 2 * LSPLIT);
    }

    /// The dimensions are the exact layer's, not a second copy.
    ///
    /// Reading the dimensions from [`ExactParams`] is what keeps the
    /// wrapper and the prover working over the same ring, rather than over
    /// two that happen to agree today.
    #[test]
    fn the_dimensions_are_the_exact_layers() {
        assert_eq!(QTILDE, ExactParams::Q_TILDE);
        assert_eq!(DTILDE, ExactParams::D_TILDE);
        assert_eq!(LSPLIT, ExactParams::L_SPLIT);
        assert_eq!(DTILDE, LSPLIT * SUBDEG);
        assert_eq!(1usize << LEVELS, LSPLIT);
        assert_eq!((DTILDE, LSPLIT, SUBDEG), (256, 64, 4));
    }

    /// `q~ = 2^26 - 1151`, and the reduction budget that follows.
    #[test]
    fn the_modulus_is_pseudo_mersenne_with_the_stated_budget() {
        assert_eq!(QTILDE, (1u64 << 26) - 1151);
        assert_eq!(C, 1151);
        assert_eq!(MASK, (1 << 26) - 1);
        // `(q~-1)^2` is just below `2^52`, not `2^51`.
        let sq = (QTILDE as u128 - 1) * (QTILDE as u128 - 1);
        assert!(sq < 1u128 << 52 && sq > 1u128 << 51);
        // So exactly 4096 maximal canonical products fit a `u64`.
        assert_eq!((u64::MAX as u128) / sq, 4096);
        // And two folds leave any `x < 2^57` below `2 q~`.
        for x in [
            (1u64 << 54) - 1,
            (1u64 << 57) - 1,
            u64::from(u32::MAX),
            QTILDE * QTILDE.saturating_sub(1) % (1 << 52),
        ] {
            assert!(fold(fold(x)) < 2 * QTILDE, "{x} folds to {}", fold(fold(x)));
        }
    }

    /// The specialisation agrees with the generic Barrett everywhere it
    /// is used, including at every boundary.
    ///
    /// A reduction that is wrong only above some threshold produces a ring
    /// that is self-consistent and not this one, which is exactly the
    /// failure a round-trip test cannot see.
    #[test]
    fn the_specialised_reduction_agrees_with_barrett() {
        let mut cases: Vec<u64> = vec![
            0,
            1,
            QTILDE - 1,
            QTILDE,
            QTILDE + 1,
            2 * QTILDE - 1,
            2 * QTILDE,
            (1 << 26) - 1,
            1 << 26,
            (1 << 52) - 1,
            (1 << 54) - 1,
            (1 << 57) - 1,
            u64::MAX,
        ];
        cases.push((QTILDE - 1) * (QTILDE - 1));
        let mut next = lcg(0xB0FF);
        for _ in 0..2000 {
            let a = next();
            let b = next();
            cases.push(a * b);
        }
        for x in cases {
            let want = barrett_reduce(x as u128);
            assert_eq!(reduce_u64(x), want, "reduce_u64({x})");
            if x < 1 << 57 {
                assert_eq!(reduce_narrow(x), want, "reduce_narrow({x})");
            }
            assert_eq!(reduce_u128(x as u128), want, "reduce_u128({x})");
        }
        // and the wide path, where only `reduce_u128` applies
        for shift in [64u32, 96, 120, 127] {
            let x = (1u128 << shift) | 0x9E37_79B9;
            assert_eq!(reduce_u128(x), barrett_reduce(x), "reduce_u128(2^{shift}+)");
        }
    }

    /// `addm` and `subm` are masked, so neither branches on its operands.
    #[test]
    fn masked_add_and_sub_agree_with_the_definition() {
        let edge = [0u64, 1, 2, QTILDE / 2, QTILDE - 2, QTILDE - 1];
        for &a in &edge {
            for &b in &edge {
                assert_eq!(addm(a, b), (a + b) % QTILDE, "addm({a}, {b})");
                assert_eq!(subm(a, b), (a + QTILDE - b) % QTILDE, "subm({a}, {b})");
            }
        }
        let mut next = lcg(0x5EED);
        for _ in 0..5000 {
            let (a, b) = (next(), next());
            assert_eq!(addm(a, b), (a + b) % QTILDE);
            assert_eq!(subm(a, b), (a + QTILDE - b) % QTILDE);
            assert_eq!(mulm(a, b), (a as u128 * b as u128 % QTILDE as u128) as u64);
        }
    }

    #[test]
    fn ntt_round_trips_and_agrees_with_schoolbook() {
        let mut next = lcg(12345);
        for _ in 0..8 {
            let a = rand_coeff(&mut next);
            let b = rand_coeff(&mut next);
            assert_eq!(intt(&ntt(&a)), a, "NTT round trip");
            assert_eq!(mul(&a, &b), mul_schoolbook(&a, &b), "NTT vs schoolbook");
        }
    }

    #[test]
    fn the_ring_is_negacyclic() {
        let mut x = vec![0u64; DTILDE];
        x[1] = 1;
        let x = CoeffPoly::new(&x).unwrap();
        let mut prod = x.clone();
        for _ in 0..DTILDE - 1 {
            prod = mul(&prod, &x);
        }
        let mut expect = vec![0u64; DTILDE];
        expect[0] = QTILDE - 1;
        assert_eq!(prod, CoeffPoly::new(&expect).unwrap());
    }

    /// Every constructor refuses what the untyped slices used to accept.
    #[test]
    fn the_types_refuse_what_the_slices_used_to_accept() {
        let good = vec![1u64; DTILDE];
        assert!(CoeffPoly::new(&good).is_some());
        assert!(CoeffPoly::new(&good[..DTILDE - 1]).is_none(), "short");
        let mut long = good.clone();
        long.push(0);
        assert!(CoeffPoly::new(&long).is_none(), "long");
        let mut noncanon = good.clone();
        noncanon[0] = QTILDE;
        assert!(CoeffPoly::new(&noncanon).is_none(), "non-canonical");
        assert!(CoeffPoly::from_reduced(&noncanon).is_some(), "reduced");
        assert!(Slots::new(&vec![0u64; LSPLIT]).is_some());
        assert!(Slots::new(&vec![0u64; LSPLIT - 1]).is_none());
        assert!(Slots::new(&vec![0u64; LSPLIT + 1]).is_none());
        assert!(CoeffPoly::from_centered(&vec![0i64; DTILDE - 1]).is_none());

        // `inner_ntt` used to truncate to the shorter operand, which is an
        // inner product over a prefix rather than an error
        let mut next = lcg(7);
        let u: Vec<NttPoly> = (0..3).map(|_| ntt(&rand_coeff(&mut next))).collect();
        let v: Vec<NttPoly> = (0..3).map(|_| ntt(&rand_coeff(&mut next))).collect();
        assert!(inner_ntt(&u, &v).is_some());
        assert!(inner_ntt(&u[..2], &v).is_none(), "unequal lengths");
        assert!(inner_ntt(&u, &v[..1]).is_none());
        assert_eq!(inner_ntt(&[], &[]), Some(NttPoly::zero()));
    }

    #[test]
    fn inner_ntt_is_the_sum_of_the_products() {
        let mut next = lcg(31);
        let u: Vec<NttPoly> = (0..4).map(|_| ntt(&rand_coeff(&mut next))).collect();
        let v: Vec<NttPoly> = (0..4).map(|_| ntt(&rand_coeff(&mut next))).collect();
        let got = inner_ntt(&u, &v).unwrap();
        let mut want = NttPoly::zero();
        for (a, b) in u.iter().zip(v.iter()) {
            want = want.add(&ntt_mul(a, b));
        }
        assert_eq!(got, want);
        // and it is the transform of the coefficient-domain inner product
        let mut coeff = CoeffPoly::zero();
        for (a, b) in u.iter().zip(v.iter()) {
            coeff = coeff.add(&mul(&intt(a), &intt(b)));
        }
        assert_eq!(intt(&got), coeff);
    }

    #[test]
    fn slot_helpers_are_ntt_domain() {
        let vals = Slots::new(&(0..LSPLIT as u64).map(|j| j * 7 + 1).collect::<Vec<_>>()).unwrap();
        let hat = slots_to_ntt(&vals);
        assert_eq!(ntt_to_slots(&hat), vals);

        // slots are independent under multiplication, which is the point
        let other = Slots::new(&(0..LSPLIT as u64).map(|j| j + 3).collect::<Vec<_>>()).unwrap();
        let prod = ntt_mul(&hat, &slots_to_ntt(&other));
        let want = Slots::new(
            &vals
                .as_slice()
                .iter()
                .zip(other.as_slice().iter())
                .map(|(&a, &b)| a * b % QTILDE)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(ntt_to_slots(&prod), want);
    }

    #[test]
    fn add_slots_lands_in_the_slots_and_nowhere_else() {
        let mut next = lcg(99);
        let hat = ntt(&rand_coeff(&mut next));
        let vals = Slots::new(&(0..LSPLIT as u64).map(|j| j * 13 + 2).collect::<Vec<_>>()).unwrap();
        let mut moved = hat.clone();
        add_slots_inplace(&mut moved, &vals);
        for i in 0..DTILDE {
            if i.is_multiple_of(SUBDEG) {
                assert_eq!(
                    moved.as_slice()[i],
                    addm(hat.as_slice()[i], vals.as_slice()[i / SUBDEG])
                );
            } else {
                assert_eq!(moved.as_slice()[i], hat.as_slice()[i], "index {i} moved");
            }
        }
        // adding zero is the identity
        let mut same = hat.clone();
        add_slots_inplace(&mut same, &Slots::zero());
        assert_eq!(same, hat);
    }

    #[test]
    fn constant_coefficient_is_the_slot_mean() {
        let inv_l = pow_mod(LSPLIT as u64, QTILDE - 2, QTILDE);
        let vals = Slots::new(&(0..LSPLIT as u64).map(|j| j * 11 + 5).collect::<Vec<_>>()).unwrap();
        let hat = slots_to_ntt(&vals);
        let want = mulm(vals.as_slice().iter().fold(0u64, |a, &b| addm(a, b)), inv_l);
        assert_eq!(constant_coefficient(&hat), want);
    }

    #[test]
    fn scale_blocks_is_slot_diagonal() {
        let mut next = lcg(99);
        let hat = ntt(&rand_coeff(&mut next));
        let scal = Slots::new(&(0..LSPLIT).map(|_| next()).collect::<Vec<_>>()).unwrap();
        let out = scale_blocks(&hat, &scal);
        for (j, &s) in scal.as_slice().iter().enumerate() {
            for k in 0..SUBDEG {
                let idx = j * SUBDEG + k;
                assert_eq!(out.as_slice()[idx], mulm(s, hat.as_slice()[idx]));
            }
        }
    }

    #[test]
    fn centring_round_trips() {
        let a = CoeffPoly::new(
            &[0, 1, QTILDE - 1, QTILDE / 2, QTILDE / 2 + 1]
                .iter()
                .cycle()
                .take(DTILDE)
                .copied()
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(CoeffPoly::from_centered(&a.centered()).unwrap(), a);
        assert!(a.centered().iter().all(|c| c.abs() <= QTILDE as i64 / 2));
        assert_eq!(a.inf_norm(), QTILDE as i64 / 2);
    }

    /// `from_centered` is the secret path, and it must not divide.
    #[test]
    fn from_centered_is_a_masked_add_over_its_whole_domain() {
        let q = QTILDE as i64;
        for &v in &[0i64, 1, -1, q - 1, -(q - 1), q / 2, -(q / 2)] {
            let poly = CoeffPoly::from_centered(&vec![v; DTILDE]).unwrap();
            let want = (v as i128).rem_euclid(QTILDE as i128) as u64;
            assert_eq!(poly.as_slice()[0], want, "v = {v}");
        }
        // outside `(-q~, q~)` is not a centred value
        assert!(CoeffPoly::from_centered(&vec![q; DTILDE]).is_none());
        assert!(CoeffPoly::from_centered(&vec![-q; DTILDE]).is_none());
        assert!(CoeffPoly::from_centered(&vec![i64::MIN; DTILDE]).is_none());
        // round trip through `centered`, which always lands in range
        let mut next = lcg(1234);
        for _ in 0..64 {
            let a = rand_coeff(&mut next);
            assert_eq!(CoeffPoly::from_centered(&a.centered()).unwrap(), a);
        }
    }

    /// Barrett has to agree with `%` on everything, or the ring moved.
    #[test]
    fn barrett_agrees_with_the_modulus() {
        let mut next = lcg(4242);
        for _ in 0..2000 {
            let (a, b) = (next(), next());
            assert_eq!(mulm(a, b), (a as u128 * b as u128 % QTILDE as u128) as u64);
        }
        for v in [0u128, 1, QTILDE as u128 - 1, QTILDE as u128, u128::MAX >> 2] {
            assert_eq!(redm(v), (v % QTILDE as u128) as u64);
        }
    }
}
