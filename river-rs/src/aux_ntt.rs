//! CRT-NTT backend.
//!
//! RiVeR's own moduli resist the NTT.  At `d = 32` every scheme modulus
//! is `5 mod 8`, so `X^32 + 1` splits into exactly **two** factors of
//! degree 16 — one Cooley-Tukey level, at most a factor of two.  That is
//! not an oversight: the congruence is chosen to hold the number of
//! factors down, because the challenge-difference invertibility premise
//! gets harder as the ring splits further.  The ring carrying almost all
//! the arithmetic is therefore the one with the least usable structure.
//!
//! The way out is to stop transforming modulo `q_hat` at all.  A product
//! in `Z[X]/(X^d+1)` is an *integer* computation; reduction mod `q_hat`
//! is a separate step afterwards.  So carry it over auxiliary
//! NTT-friendly primes, reconstruct the exact integer by CRT, and reduce
//! at the end.
//!
//! **Where this wins, and where it does not.**  An isolated CRT-NTT
//! multiply is *slower* than direct multiplication at `d = 32`: six
//! 32-point transforms cost more than one convolution.  The win is
//! amortising transforms across a matrix product whose matrix is fixed —
//! `G'` is `n_hat × (k_hat + 2N)` and derived from `rho` forever, and it
//! is 76% of per-attempt work at `N = 8` and 88% at `N = 256`.  So
//! [`crate::ring::Ring::mul`] stays schoolbook and only the matrix paths
//! come here.
//!
//! **Sizing.**  Four 32-bit primes give `P = p0·p1·p2·p3 > 2^127`, which
//! covers every profile: products stay in `u64`, which vectorises
//! cleanly, at the cost of four transforms instead of two.  The
//! reconstruction bound is `P > 2·m·d·A^2` for `m` accumulated products
//! of coefficients bounded by `A`.  [`CrtBackend::new`] checks it
//! against the **unsigned** `A = q-1`, not the centred `A = (q-1)/2`,
//! even though [`CrtBackend::split`] centres: the design note records a
//! prototype that sized against the centred bound while feeding unsigned
//! inputs and passed every random test, because the worst case needs
//! every coefficient at the extreme simultaneously.  Checking the looser
//! bound makes the backend correct no matter what reaches it.

use core::marker::PhantomData;

/// The four auxiliary primes, each `= 1 mod 2d = 1 mod 64` and of the
/// form `2^32 - c` with `c < 2^12`, which makes reduction two
/// multiply-adds.  `P = p0·p1·p2·p3 ≈ 2^128`.
pub const AUX_PRIMES: [u64; 4] = [4_294_966_657, 4_294_966_337, 4_294_965_313, 4_294_964_929];
/// `c_i = 2^32 - p_i`.
pub const AUX_C: [u64; 4] = [639, 959, 1983, 2367];

/// Supported ring dimension for the fast path.
pub const SUPPORTED_D: &[usize] = &[32];

const P0: u64 = AUX_PRIMES[0];
const P1: u64 = AUX_PRIMES[1];
const P2: u64 = AUX_PRIMES[2];
const P3: u64 = AUX_PRIMES[3];

/// `P = prod(AUX_PRIMES)`, as a `u128`.
pub const AUX_P: u128 = (P0 as u128) * (P1 as u128) * (P2 as u128) * (P3 as u128);

// ---- modular arithmetic --------------------------------------------------

/// `x mod p` for `p = 2^32 - C` and `x < 2^64`, by the pseudo-Mersenne
/// identity `x ≡ hi·C + lo (mod p)`.
#[inline(always)]
fn reduce_pm<const P: u64, const C: u64>(x: u64) -> u64 {
    const MASK: u64 = (1u64 << 32) - 1;
    // x < 2^64  ->  r1 < 2^32 + C·2^32 < 2^44
    let r1 = (x & MASK) + C * (x >> 32);
    // r1 < 2^44  ->  r2 < 2^32 + C·2^12 < 2^32 + 2^24
    let r2 = (r1 & MASK) + C * (r1 >> 32);
    let mask = 0u64.wrapping_sub((r2 >= P) as u64);
    r2.wrapping_sub(P & mask)
}

/// `(-1)^neg · mag  mod P`, branchlessly, for `mag < 2^64`.
#[inline(always)]
fn signed_mod<const P: u64, const C: u64>(mag: u64, neg: bool) -> u64 {
    let r = reduce_pm::<P, C>(mag);
    // `P - r`, but `0` when `r == 0`
    let nr = (P - r) & 0u64.wrapping_sub((r != 0) as u64);
    let mask = 0u64.wrapping_sub(neg as u64);
    (r & !mask) | (nr & mask)
}

#[inline(always)]
fn mul_mod_p<const P: u64, const C: u64>(a: u64, b: u64) -> u64 {
    debug_assert!(a < P && b < P);
    reduce_pm::<P, C>(a * b)
}

#[inline(always)]
fn add_mod_p<const P: u64>(a: u64, b: u64) -> u64 {
    let s = a + b;
    let mask = 0u64.wrapping_sub((s >= P) as u64);
    s.wrapping_sub(P & mask)
}

#[inline(always)]
fn sub_mod_p<const P: u64>(a: u64, b: u64) -> u64 {
    let d = a.wrapping_sub(b);
    let mask = 0u64.wrapping_sub((a < b) as u64);
    d.wrapping_add(P & mask)
}

/// `v mod P` for `v < 2P`: one masked conditional subtraction.
///
/// Garner's reconstruction crosses between the auxiliary primes, and the
/// four are within 1728 of each other, so a residue mod one is already
/// below twice any other.  The site used to write `v % P`; `P` is a const
/// generic so LLVM strength-reduces it, but "the compiler usually does"
/// is not the property `README.md` claims — these operands are the
/// residues of a secret polynomial.  A masked subtract is the claim
/// itself.
#[inline(always)]
fn narrow_mod_p<const P: u64>(v: u64) -> u64 {
    debug_assert!(v < 2 * P, "narrow_mod_p needs v < 2P");
    let d = v.wrapping_sub(P);
    let mask = 0u64.wrapping_sub((v < P) as u64);
    d.wrapping_add(P & mask)
}

/// `(a · b) mod p` via `u128` — used off the hot path (root search,
/// table construction, CRT constants).
pub fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

pub fn pow_mod(mut base: u64, mut exp: u64, p: u64) -> u64 {
    let mut acc = 1u64;
    base %= p;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, p);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul_mod(base, base, p);
        }
    }
    acc
}

/// Modular inverse via Fermat; `p` must be prime.
pub fn inv_mod(a: u64, p: u64) -> u64 {
    pow_mod(a, p - 2, p)
}

/// Smallest `x` with exact multiplicative order `2d`, i.e.
/// `x^d = -1 (mod p)`.  Requires `2d | p - 1`.
pub fn find_primitive_2d_root(p: u64, d: usize) -> Option<u64> {
    let two_d = (2 * d) as u64;
    if !(p - 1).is_multiple_of(two_d) {
        return None;
    }
    let exponent = (p - 1) / two_d;
    (2..p).find_map(|x| {
        let psi = pow_mod(x, exponent, p);
        (pow_mod(psi, d as u64, p) == p - 1).then_some(psi)
    })
}

#[inline]
fn bitrev(mut k: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (k & 1);
        k >>= 1;
    }
    r
}

/// `zetas[k] = psi^bitrev(k) mod p`, the FIPS 204 twiddle convention.
pub fn make_zeta_table(p: u64, d: usize, psi: u64) -> Vec<u64> {
    let bits = d.trailing_zeros();
    (0..d)
        .map(|k| pow_mod(psi, bitrev(k, bits) as u64, p))
        .collect()
}

// ---- per-prime context ---------------------------------------------------

#[derive(Clone)]
struct Aux<const P: u64, const C: u64> {
    zetas: Vec<u64>,
    inv_d: u64,
    _p: PhantomData<()>,
}

impl<const P: u64, const C: u64> Aux<P, C> {
    fn new(d: usize) -> Self {
        let psi = find_primitive_2d_root(P, d).expect("aux prime does not support this d");
        debug_assert_eq!(pow_mod(psi, d as u64, P), P - 1);
        Self {
            zetas: make_zeta_table(P, d, psi),
            inv_d: inv_mod(d as u64, P),
            _p: PhantomData,
        }
    }

    /// Forward negacyclic NTT: natural order in, bit-reversed out.
    fn ntt(&self, f: &mut [u64]) {
        let d = f.len();
        let mut m = 0usize;
        let mut le = d / 2;
        while le >= 1 {
            let mut st = 0usize;
            while st < d {
                m += 1;
                let z = self.zetas[m];
                for j in st..st + le {
                    let t = mul_mod_p::<P, C>(z, f[j + le]);
                    let fj = f[j];
                    f[j + le] = sub_mod_p::<P>(fj, t);
                    f[j] = add_mod_p::<P>(fj, t);
                }
                st += 2 * le;
            }
            le /= 2;
        }
    }

    /// Inverse negacyclic NTT, including the trailing `1/d`.
    fn intt(&self, f: &mut [u64]) {
        let d = f.len();
        let mut m = d;
        let mut le = 1usize;
        while le < d {
            let mut st = 0usize;
            while st < d {
                m -= 1;
                let z = if self.zetas[m] == 0 {
                    0
                } else {
                    P - self.zetas[m]
                };
                for j in st..st + le {
                    let t = f[j];
                    let u = f[j + le];
                    f[j] = add_mod_p::<P>(t, u);
                    f[j + le] = mul_mod_p::<P, C>(z, sub_mod_p::<P>(t, u));
                }
                st += 2 * le;
            }
            le *= 2;
        }
        for x in f.iter_mut() {
            *x = mul_mod_p::<P, C>(*x, self.inv_d);
        }
    }

    fn forward(&self, input: &[u64]) -> Vec<u64> {
        let mut out = input.to_vec();
        self.ntt(&mut out);
        out
    }
}

/// One polynomial held in all four auxiliary NTT domains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrtNttPoly {
    residues: [Vec<u64>; 4],
}

/// A matrix of polynomials held in the auxiliary NTT domains.
///
/// Opaque, and only [`CrtBackend::mat_to_ntt`] builds one.  Auxiliary
/// residues carry no trace of where they came from: the same four
/// `Vec<u64>`s are a valid transform under *any* `q` with the same `d`,
/// and reconstructing them under a different modulus — or accumulating
/// more columns than the backend's `P` was sized for — is silently wrong
/// rather than detectably wrong.  So the transform carries its context
/// and every consumer checks it.
#[derive(Clone)]
pub struct CrtNttMat {
    rows: Vec<Vec<CrtNttPoly>>,
    q: u64,
    d: usize,
    cols: usize,
}

impl CrtNttMat {
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The modulus this matrix was transformed under.
    pub fn modulus(&self) -> u64 {
        self.q
    }

    pub fn degree(&self) -> usize {
        self.d
    }
}

// ---- backend -------------------------------------------------------------

/// Exact negacyclic multiplication in `R_q = Z_q[X]/(X^d + 1)` through
/// four auxiliary NTT-friendly primes.
#[derive(Clone)]
pub struct CrtBackend {
    q: u64,
    d: usize,
    half_q: u64,
    /// Barrett for `q`, so the CRT reconstruction does not finish with a
    /// variable-time 128-bit division.
    bar: crate::ring::Barrett,
    /// The accumulation `P` was sized for.  Kept, not just checked at
    /// construction: the bound is a property of *how the backend is
    /// used*, so a wider product than this one was built for has to be
    /// refused at the point of use, not assumed away.
    max_terms: usize,
    a0: Aux<P0, { AUX_C[0] }>,
    a1: Aux<P1, { AUX_C[1] }>,
    a2: Aux<P2, { AUX_C[2] }>,
    a3: Aux<P3, { AUX_C[3] }>,
    /// Garner coefficients: `inv(p_j, p_i)` for `j < i`.
    garner: [[u64; 3]; 3],
}

impl CrtBackend {
    /// Build a backend for `R_q` at dimension `d`, sized for
    /// accumulations of up to `max_terms` products.
    ///
    /// Returns `None` when `d` is unsupported or the reconstruction
    /// bound `2 · max_terms · d · (q-1)^2 < P` is violated — in which
    /// case CRT could not recover the exact integer coefficient and the
    /// caller must fall back to schoolbook.  Refusing loudly is the
    /// point: an undersized `P` passes every random test.
    pub fn new(q: u64, d: usize, max_terms: usize) -> Option<Self> {
        if !SUPPORTED_D.contains(&d) {
            return None;
        }
        if !Self::bound_holds(q, d, max_terms) {
            return None;
        }
        let mut garner = [[0u64; 3]; 3];
        for i in 1..4 {
            for j in 0..i {
                garner[i - 1][j] = inv_mod(AUX_PRIMES[j] % AUX_PRIMES[i], AUX_PRIMES[i]);
            }
        }
        Some(Self {
            q,
            d,
            half_q: q / 2,
            bar: crate::ring::Barrett::new(q),
            max_terms,
            a0: Aux::new(d),
            a1: Aux::new(d),
            a2: Aux::new(d),
            a3: Aux::new(d),
            garner,
        })
    }

    /// `2 · max_terms · d · (q-1)^2 < P`, evaluated without overflow.
    pub fn bound_holds(q: u64, d: usize, max_terms: usize) -> bool {
        let a = (q - 1) as u128;
        // a^2 needs 128 bits on its own for a 64-bit q, so compare in
        // two stages rather than forming the product.
        let lhs_small = 2u128 * max_terms as u128 * d as u128;
        match a.checked_mul(a) {
            None => false,
            Some(sq) => match sq.checked_mul(lhs_small) {
                None => false,
                Some(v) => v < AUX_P,
            },
        }
    }

    pub fn d(&self) -> usize {
        self.d
    }

    /// Centre-lift from `[0, q)` and reduce into each auxiliary field.
    fn split(&self, a: &[u64]) -> [Vec<u64>; 4] {
        let q = self.q as i128;
        let mut out = [
            Vec::with_capacity(self.d),
            Vec::with_capacity(self.d),
            Vec::with_capacity(self.d),
            Vec::with_capacity(self.d),
        ];
        // Centre, then reduce with the pseudo-Mersenne identity rather
        // than four `i128::rem_euclid` calls.  The old form was a
        // variable-time 128-bit division per coefficient per prime — on
        // the vector side of every matrix product, which is secret — and
        // four of the slowest instructions on the machine.
        let _ = q;
        for &c in a {
            let (mag, neg) = if c > self.half_q {
                (self.q - c, true)
            } else {
                (c, false)
            };
            out[0].push(signed_mod::<P0, { AUX_C[0] }>(mag, neg));
            out[1].push(signed_mod::<P1, { AUX_C[1] }>(mag, neg));
            out[2].push(signed_mod::<P2, { AUX_C[2] }>(mag, neg));
            out[3].push(signed_mod::<P3, { AUX_C[3] }>(mag, neg));
        }
        out
    }

    fn forward_all(&self, r: &[Vec<u64>; 4]) -> [Vec<u64>; 4] {
        [
            self.a0.forward(&r[0]),
            self.a1.forward(&r[1]),
            self.a2.forward(&r[2]),
            self.a3.forward(&r[3]),
        ]
    }

    fn inverse_all(&self, r: &mut [Vec<u64>; 4]) {
        self.a0.intt(&mut r[0]);
        self.a1.intt(&mut r[1]);
        self.a2.intt(&mut r[2]);
        self.a3.intt(&mut r[3]);
    }

    /// Transform a canonical `[0, q)` polynomial into the auxiliary NTT
    /// domains.  Worth doing once for anything reused — `G'` and `A` are
    /// fixed by `rho` forever.
    pub fn to_ntt(&self, a: &[u64]) -> CrtNttPoly {
        assert_eq!(a.len(), self.d);
        CrtNttPoly {
            residues: self.forward_all(&self.split(a)),
        }
    }

    /// The accumulation width this backend's `P` was sized for.
    pub fn max_terms(&self) -> usize {
        self.max_terms
    }

    /// Transform a matrix for repeated products.
    ///
    /// `None` when the matrix is ragged, has the wrong degree, or is
    /// wider than [`Self::max_terms`] — that last one is the whole point
    /// of carrying the budget: a row of `m` products needs
    /// `2·m·d·(q-1)^2 < P`, and a matrix wider than the backend was built
    /// for reconstructs to the wrong integer with no other symptom.
    pub fn mat_to_ntt(&self, m: &[Vec<Vec<u64>>]) -> Option<CrtNttMat> {
        let cols = m.first()?.len();
        if cols > self.max_terms {
            return None;
        }
        if m.iter().any(|row| row.len() != cols) {
            return None;
        }
        if m.iter().flatten().any(|p| p.len() != self.d) {
            return None;
        }
        Some(CrtNttMat {
            rows: m
                .iter()
                .map(|row| row.iter().map(|p| self.to_ntt(p)).collect())
                .collect(),
            q: self.q,
            d: self.d,
            cols,
        })
    }

    /// Transform a vector for use with [`Self::row_dot`].
    pub fn vec_to_ntt(&self, v: &[Vec<u64>]) -> Option<Vec<CrtNttPoly>> {
        if v.len() > self.max_terms || v.iter().any(|p| p.len() != self.d) {
            return None;
        }
        Some(v.iter().map(|p| self.to_ntt(p)).collect())
    }

    /// `a · b` mod `X^d + 1`, mod `q`.  Inputs and output in `[0, q)`.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let an = self.to_ntt(a);
        self.mul_with_lhs_ntt(&an, b)
    }

    /// `a · b` where `a` is already transformed.
    pub fn mul_with_lhs_ntt(&self, a: &CrtNttPoly, b: &[u64]) -> Vec<u64> {
        assert_eq!(b.len(), self.d);
        let bn = self.forward_all(&self.split(b));
        let mut c = [
            pointwise::<P0, { AUX_C[0] }>(&a.residues[0], &bn[0]),
            pointwise::<P1, { AUX_C[1] }>(&a.residues[1], &bn[1]),
            pointwise::<P2, { AUX_C[2] }>(&a.residues[2], &bn[2]),
            pointwise::<P3, { AUX_C[3] }>(&a.residues[3], &bn[3]),
        ];
        self.inverse_all(&mut c);
        self.crt_combine(&c)
    }

    /// Matrix-vector product with a pre-transformed matrix.
    ///
    /// The vector is transformed once and reused across every output
    /// row, so an `n_hat × m` product costs `m + n_hat` transforms
    /// instead of `n_hat · m` multiplications.
    pub fn mat_vec_ntt(&self, m: &CrtNttMat, v: &[Vec<u64>]) -> Option<Vec<Vec<u64>>> {
        let v_ntt = self.vec_to_ntt(v)?;
        (0..m.rows()).map(|i| self.row_dot(m, i, &v_ntt)).collect()
    }

    /// One row of [`Self::mat_vec_ntt`] — exposed so a caller can
    /// parallelise over rows.  Takes the matrix rather than a bare row so
    /// the tag travels with it.
    pub fn row_dot(&self, m: &CrtNttMat, row: usize, v_ntt: &[CrtNttPoly]) -> Option<Vec<u64>> {
        // A matrix transformed under another modulus or degree would
        // reconstruct to a plausible-looking wrong answer.
        if m.q != self.q || m.d != self.d {
            return None;
        }
        if m.cols != v_ntt.len() || v_ntt.len() > self.max_terms {
            return None;
        }
        let row = m.rows.get(row)?;
        let mut acc: [Vec<u64>; 4] = [
            vec![0u64; self.d],
            vec![0u64; self.d],
            vec![0u64; self.d],
            vec![0u64; self.d],
        ];
        for (a, b) in row.iter().zip(v_ntt.iter()) {
            accumulate::<P0, { AUX_C[0] }>(&mut acc[0], &a.residues[0], &b.residues[0]);
            accumulate::<P1, { AUX_C[1] }>(&mut acc[1], &a.residues[1], &b.residues[1]);
            accumulate::<P2, { AUX_C[2] }>(&mut acc[2], &a.residues[2], &b.residues[2]);
            accumulate::<P3, { AUX_C[3] }>(&mut acc[3], &a.residues[3], &b.residues[3]);
        }
        self.inverse_all(&mut acc);
        Some(self.crt_combine(&acc))
    }

    /// Garner reconstruction to the centred integer in `(-P/2, P/2]`,
    /// then reduction mod `q`.
    ///
    /// The centring stays in `u128` and carries the sign separately.
    /// `P` is just under `2^128`, so it does not fit `i128` — casting
    /// the reconstruction to a signed type wraps for every value above
    /// `2^127`, which is half of them.
    fn crt_combine(&self, r: &[Vec<u64>; 4]) -> Vec<u64> {
        let half_p = AUX_P / 2;
        let mut out = Vec::with_capacity(self.d);
        for (i, &v0) in r[0].iter().enumerate() {
            let v1 = {
                let t = sub_mod_p::<P1>(narrow_mod_p::<P1>(r[1][i]), narrow_mod_p::<P1>(v0));
                mul_mod_p::<P1, { AUX_C[1] }>(t, self.garner[0][0])
            };
            let v2 = {
                let t = sub_mod_p::<P2>(narrow_mod_p::<P2>(r[2][i]), narrow_mod_p::<P2>(v0));
                let t = mul_mod_p::<P2, { AUX_C[2] }>(t, self.garner[1][0]);
                let t = sub_mod_p::<P2>(t, narrow_mod_p::<P2>(v1));
                mul_mod_p::<P2, { AUX_C[2] }>(t, self.garner[1][1])
            };
            let v3 = {
                let t = sub_mod_p::<P3>(narrow_mod_p::<P3>(r[3][i]), narrow_mod_p::<P3>(v0));
                let t = mul_mod_p::<P3, { AUX_C[3] }>(t, self.garner[2][0]);
                let t = sub_mod_p::<P3>(t, narrow_mod_p::<P3>(v1));
                let t = mul_mod_p::<P3, { AUX_C[3] }>(t, self.garner[2][1]);
                let t = sub_mod_p::<P3>(t, narrow_mod_p::<P3>(v2));
                mul_mod_p::<P3, { AUX_C[3] }>(t, self.garner[2][2])
            };
            // x = v0 + p0·(v1 + p1·(v2 + p2·v3))  <  P
            let x = v0 as u128
                + (P0 as u128)
                    * (v1 as u128 + (P1 as u128) * (v2 as u128 + (P2 as u128) * v3 as u128));
            let (negative, magnitude) = if x > half_p {
                (true, AUX_P - x)
            } else {
                (false, x)
            };
            let m = self.bar.reduce(magnitude);
            out.push(if negative && m != 0 { self.q - m } else { m });
        }
        out
    }
}

#[inline]
fn pointwise<const P: u64, const C: u64>(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| mul_mod_p::<P, C>(x, y))
        .collect()
}

#[inline]
fn accumulate<const P: u64, const C: u64>(acc: &mut [u64], a: &[u64], b: &[u64]) {
    for i in 0..acc.len() {
        acc[i] = add_mod_p::<P>(acc[i], mul_mod_p::<P, C>(a[i], b[i]));
    }
}

// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{is_prime, PROFILES, RIVER_N256, RIVER_TOY};

    fn schoolbook(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
        let d = a.len();
        let mut c = vec![0i128; d];
        for i in 0..d {
            for j in 0..d {
                let prod = (a[i] as i128) * (b[j] as i128);
                if i + j < d {
                    c[i + j] += prod;
                } else {
                    c[i + j - d] -= prod;
                }
            }
        }
        c.into_iter()
            .map(|v| v.rem_euclid(q as i128) as u64)
            .collect()
    }

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
    fn aux_primes_are_prime_and_support_d32() {
        for (i, (&p, &c)) in AUX_PRIMES.iter().zip(AUX_C.iter()).enumerate() {
            assert!(is_prime(p), "p{i} = {p} is not prime");
            assert_eq!((p - 1) % 64, 0, "p{i} is not 1 mod 2d");
            assert_eq!(p, (1u64 << 32) - c, "c{i} does not match");
            assert!(c < (1 << 12), "c{i} too large for two-step reduction");
        }
        // distinct, so CRT applies
        let mut seen: Vec<u64> = AUX_PRIMES.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 4, "auxiliary primes are not distinct");
    }

    #[test]
    fn aux_primes_are_the_largest_of_their_form() {
        // Regenerating the set: largest primes = 1 mod 64 below 2^32.
        let mut found = Vec::new();
        let mut n = (1u64 << 32) - 63; // ≡ 1 mod 64
        while found.len() < 4 {
            if is_prime(n) {
                found.push(n);
            }
            n -= 64;
        }
        assert_eq!(found, AUX_PRIMES);
    }

    #[test]
    fn primitive_roots_have_exact_order_2d() {
        for &p in &AUX_PRIMES {
            let psi = find_primitive_2d_root(p, 32).unwrap();
            assert_eq!(pow_mod(psi, 32, p), p - 1);
            assert_eq!(pow_mod(psi, 64, p), 1);
        }
    }

    // Constant-folded on purpose: the point is to fail the build if the
    // prime set is ever edited to something that no longer covers the
    // reconstruction bound, or that no longer fits `u128`.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn the_product_of_the_primes_exceeds_2_to_the_127() {
        // Not a tautology worth a constant-folded assert: it is the
        // margin the reconstruction bound is spent against.
        assert!(AUX_P > 1u128 << 127, "P = {AUX_P} fell below 2^127");
        assert!(AUX_P < u128::MAX, "P must leave room in u128");
    }

    #[test]
    fn bound_covers_every_profile() {
        for par in PROFILES {
            // G' is the widest accumulation: k_hat + 2N columns.
            assert!(
                CrtBackend::bound_holds(par.q_hat, par.d, par.gprime_cols()),
                "{} q_hat with m = {}",
                par.name,
                par.gprime_cols()
            );
            // A is n x ell.
            assert!(
                CrtBackend::bound_holds(par.q(), par.d, par.ell),
                "{} q with m = {}",
                par.name,
                par.ell
            );
        }
    }

    #[test]
    fn bound_refuses_an_oversized_accumulation() {
        // `2 m d (q-1)^2` with `q ~ 2^48` and `m = 2^26` is `~ 2^128`,
        // which is where `P` sits, so take one more doubling to be clear
        // of it.  (`q_hat` at `N256` was 49 bits before the paper
        // published the concrete moduli — see `params::QHAT_49` — so this
        // used to be `1 << 24`.)
        assert!(!CrtBackend::bound_holds(RIVER_N256.q_hat, 32, 1 << 27));
        assert!(CrtBackend::new(RIVER_N256.q_hat, 32, 1 << 27).is_none());
    }

    /// The worst margin the reconstruction bound leaves, measured.
    ///
    /// `2 m d (q-1)^2 < P` has to hold at every profile, and how much room
    /// is left is a property of the *current* profiles, not a constant:
    /// the paper moved every modulus and every rank, so it moved
    /// this too.  At these parameters the tightest case is `A` at `N16`
    /// and `N256` — both carry `q = 61 p` with a 48-bit `p` and
    /// `ell = 59` — and the margin there is 8.26 bits, against the 8.9
    /// the design note quotes.  Still four primes, still comfortable,
    /// but the number in the note is stale and this is what makes that
    /// visible rather than a matter of re-reading the note.
    #[test]
    fn the_worst_reconstruction_margin_is_measured_not_quoted() {
        let mut worst = f64::INFINITY;
        let mut worst_at = "";
        for par in PROFILES {
            for (label, modulus, terms) in [
                ("G'", par.q_hat, par.gprime_cols()),
                ("A", par.q(), par.ell),
            ] {
                let a = (modulus - 1) as f64;
                let need = (2.0 * terms as f64 * par.d as f64).log2() + 2.0 * a.log2();
                let margin = 128.0 - need;
                if margin < worst {
                    worst = margin;
                    worst_at = label;
                }
            }
        }
        assert!(
            (worst - 8.26).abs() < 0.01,
            "worst margin is {worst:.2} bits at {worst_at}, not 8.26 — \
             the profiles moved"
        );
    }

    #[test]
    fn refuses_unsupported_dimensions() {
        assert!(CrtBackend::new(RIVER_TOY.q_hat, 64, 8).is_none());
    }

    #[test]
    fn mul_agrees_with_schoolbook_on_random_inputs() {
        for par in PROFILES {
            for q in [par.q(), par.q_hat] {
                let bk = CrtBackend::new(q, 32, par.gprime_cols()).unwrap();
                let mut r = lcg(0x1234_5678 ^ q);
                for _ in 0..4 {
                    let a: Vec<u64> = (0..32).map(|_| r(q)).collect();
                    let b: Vec<u64> = (0..32).map(|_| r(q)).collect();
                    assert_eq!(bk.mul(&a, &b), schoolbook(&a, &b, q), "q = {q}");
                }
            }
        }
    }

    /// The largest integer magnitude a single product of two polynomials
    /// with every *centred* coefficient equal to `c` can reach: `d · c²`.
    ///
    /// This exists because `q - 1` is the wrong "extreme".  [`split`]
    /// centres, so `q - 1` lifts to `-1` and a product of two saturated
    /// polynomials has coefficients of magnitude at most `d = 32` — about
    /// `2^5`, against a reconstruction budget near `2^127`.  A test using
    /// it exercises *sign* reconstruction, which is worth having, but says
    /// nothing about `P` being large enough.  The extreme for the bound is
    /// a centred coefficient near `q/2`.
    fn product_magnitude(d: u128, c: u128) -> u128 {
        d * c * c
    }

    #[test]
    fn mul_agrees_when_every_centred_coefficient_is_at_the_extreme() {
        // The case an undersized P fails and random tests do not: every
        // centred coefficient at ±q/2 simultaneously.
        for par in PROFILES {
            for q in [par.q(), par.q_hat] {
                let bk = CrtBackend::new(q, 32, par.gprime_cols()).unwrap();
                // `half_q` stays positive under centring; `half_q + 1`
                // lifts to the most negative representative.
                for &c in &[q / 2, q / 2 + 1] {
                    let sat = vec![c; 32];
                    assert_eq!(bk.mul(&sat, &sat), schoolbook(&sat, &sat, q), "q = {q}");
                }
                // A single product is comfortably inside the budget —
                // `d · (q/2)^2` is about `2^103` at the largest `q`, and
                // the accumulation below is the binding case.  The floor
                // is here so this cannot silently degenerate back into a
                // small-value test: with `q - 1` as the input it would be
                // `2^5`.
                let reach = product_magnitude(32, (q / 2) as u128);
                assert!(
                    reach > 1 << 60 && reach < AUX_P,
                    "q = {q}: single product reaches 2^{:.1}, budget 2^{:.1}",
                    (reach as f64).log2(),
                    (AUX_P as f64).log2()
                );
            }
        }
    }

    #[test]
    fn accumulation_agrees_at_the_extreme() {
        // An m-term accumulation with every operand at ±q/2 — the other
        // half of the bound check, and the one that actually approaches
        // `P`: m · d · (q/2)^2 for the widest profile.
        let par = RIVER_N256;
        let q = par.q_hat;
        let m = par.gprime_cols();
        let bk = CrtBackend::new(q, 32, m).unwrap();
        for &c in &[q / 2, q / 2 + 1] {
            let sat = vec![c; 32];
            let row: Vec<Vec<u64>> = (0..m).map(|_| sat.clone()).collect();
            let mat = vec![row.clone()];
            let mat_ntt = bk.mat_to_ntt(&mat).unwrap();
            let got = bk.mat_vec_ntt(&mat_ntt, &row).unwrap();

            let one = schoolbook(&sat, &sat, q);
            let want: Vec<u64> = (0..32)
                .map(|k| (one[k] as u128 * m as u128 % q as u128) as u64)
                .collect();
            assert_eq!(got[0], want, "q = {q}, c = {c}");
        }

        // The accumulated magnitude, stated rather than assumed: this is
        // the number the reconstruction bound is about.  With `q - 1` as
        // the "saturated" input it would be 568·32 = 18176 ≈ 2^14.1, and
        // this test would have been checking nothing about `P`.
        let reach = m as u128 * product_magnitude(32, (q / 2) as u128);
        assert!(
            reach > 1 << 107 && reach < AUX_P,
            "accumulation reaches 2^{:.1}, budget 2^{:.1}",
            (reach as f64).log2(),
            (AUX_P as f64).log2()
        );

        // Why the headroom is ~9 bits and not ~0: `bound_holds` is
        // deliberately conservative, sizing against the *unsigned*
        // `A = q-1` with a factor 2, where the transform actually feeds
        // centred inputs of magnitude at most `q/2`.  That is 8x, and the
        // design note records why — a prototype sized against the centred
        // bound while feeding unsigned inputs passed every random test.
        let checked = 2 * m as u128 * 32 * ((q - 1) as u128).pow(2);
        assert_eq!(checked / reach, 8);
        assert!(checked < AUX_P, "the profile would be refused outright");
        assert!(CrtBackend::bound_holds(q, 32, m));
    }

    #[test]
    fn saturated_unsigned_inputs_reconstruct_with_the_right_sign() {
        // `q - 1` centres to `-1`, so this is the negative-reconstruction
        // case — the one that caught the `i128` wrap in Garner — and is
        // kept separate from the bound tests above precisely because its
        // magnitude is tiny.
        for par in PROFILES {
            for q in [par.q(), par.q_hat] {
                let bk = CrtBackend::new(q, 32, par.gprime_cols()).unwrap();
                let sat = vec![q - 1; 32];
                assert_eq!(bk.mul(&sat, &sat), schoolbook(&sat, &sat, q), "q = {q}");
                assert!(product_magnitude(32, 1) < 1 << 16);
            }
        }
    }

    #[test]
    fn the_transform_refuses_a_wider_accumulation_than_p_allows() {
        // The budget is a property of how the backend is used, so it has
        // to bite at the point of use.  A matrix one column wider than
        // the backend was built for reconstructs to the wrong integer
        // with no other symptom.
        let q = RIVER_TOY.q_hat;
        let bk = CrtBackend::new(q, 32, 8).unwrap();
        assert_eq!(bk.max_terms(), 8);
        let poly = vec![1u64; 32];
        let ok: Vec<Vec<Vec<u64>>> = vec![vec![poly.clone(); 8]];
        let wide: Vec<Vec<Vec<u64>>> = vec![vec![poly.clone(); 9]];
        assert!(bk.mat_to_ntt(&ok).is_some());
        assert!(bk.mat_to_ntt(&wide).is_none());
        // ragged and wrong-degree matrices too
        assert!(bk
            .mat_to_ntt(&[vec![poly.clone(); 4], vec![poly.clone(); 3]])
            .is_none());
        assert!(bk.mat_to_ntt(&[vec![vec![1u64; 16]; 4]]).is_none());
    }

    #[test]
    fn a_transform_from_another_ring_is_refused() {
        // Auxiliary residues carry no trace of their modulus: the same
        // four vectors are a valid transform under any q at this degree,
        // and reconstructing them under another one is silently wrong.
        let poly = vec![7u64; 32];
        let mat: Vec<Vec<Vec<u64>>> = vec![vec![poly.clone(); 4]];
        let a = CrtBackend::new(RIVER_TOY.q_hat, 32, 8).unwrap();
        let b = CrtBackend::new(RIVER_N256.q_hat, 32, 8).unwrap();
        let m_a = a.mat_to_ntt(&mat).unwrap();
        assert_eq!(m_a.modulus(), RIVER_TOY.q_hat);
        assert_eq!((m_a.rows(), m_a.cols(), m_a.degree()), (1, 4, 32));
        let v = vec![poly.clone(); 4];
        assert!(a.mat_vec_ntt(&m_a, &v).is_some());
        assert!(b.mat_vec_ntt(&m_a, &v).is_none());
        // and a vector of the wrong width against a valid matrix
        assert!(a.mat_vec_ntt(&m_a, &vec![poly; 3]).is_none());
    }

    #[test]
    fn x_to_the_d_is_minus_one() {
        let q = RIVER_TOY.q_hat;
        let bk = CrtBackend::new(q, 32, 16).unwrap();
        let mut x = vec![0u64; 32];
        x[1] = 1;
        let mut x31 = vec![0u64; 32];
        x31[31] = 1;
        let mut expected = vec![0u64; 32];
        expected[0] = q - 1;
        assert_eq!(bk.mul(&x, &x31), expected);
    }

    #[test]
    fn mat_vec_agrees_with_the_direct_product() {
        let par = RIVER_TOY;
        let q = par.q_hat;
        let bk = CrtBackend::new(q, 32, 12).unwrap();
        let mut r = lcg(99);
        let mat: Vec<Vec<Vec<u64>>> = (0..4)
            .map(|_| (0..12).map(|_| (0..32).map(|_| r(q)).collect()).collect())
            .collect();
        let v: Vec<Vec<u64>> = (0..12).map(|_| (0..32).map(|_| r(q)).collect()).collect();

        let mat_ntt = bk.mat_to_ntt(&mat).unwrap();
        let got = bk.mat_vec_ntt(&mat_ntt, &v).unwrap();

        for (row, g) in mat.iter().zip(got.iter()) {
            let mut acc = vec![0u64; 32];
            for (a, b) in row.iter().zip(v.iter()) {
                let prod = schoolbook(a, b, q);
                for k in 0..32 {
                    acc[k] = (acc[k] + prod[k]) % q;
                }
            }
            assert_eq!(*g, acc);
        }
    }
}
