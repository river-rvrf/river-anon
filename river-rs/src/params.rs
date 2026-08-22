//! Parameter profiles and derived bounds — port of `river-py/params.py`.
//!
//! Base parameters follow Table `tab:river-final-all-params` of the
//! paper (Appendix "Detailed Parameter
//! Setting").  Every derived bound is computed as `BoundGen` specifies,
//! in the same operation order as the Python, so the two agree to the
//! last float ulp — which matters, because [`RiVeRParams::sigma_a`] and
//! friends are pinned to exact rationals by
//! [`crate::sample::rational_sigma`] and a width off by one unit of
//! `2^-20` moves every mask.
//!
//! ## Provenance
//!
//! Every constant here carries one of three provenance labels:
//!
//! * **Paper** — printed in the current PDF or TeX.
//! * **Derived** — deterministically derived from paper values by a
//!   documented convention, but not printed.
//! * **Repair** — an implementation choice needed to make an ambiguous or
//!   inconsistent part of the paper executable.
//!
//! One thing the paper leaves open, and it is **Derived**:
//!
//! * **Concrete moduli.**  The paper reports only bit lengths for `p`
//!   and `q_hat`.  We take the largest prime below `2^bits` congruent to
//!   `5 mod 8`; that congruence is what makes `X^d + 1` split into
//!   exactly two irreducible factors at `d = 32`, which the
//!   challenge-difference invertibility argument needs.  `q_0 = 61`
//!   already satisfies it.  [`verify_moduli`] re-derives the pinned
//!   values.
//!
//! `(tau_g0, tau_g1)` is **not** in that category: the table prints two
//! decimals and says so in a note, so those are **Paper**, carried here as
//! exact [`Rat`]s.  A test checks that one decimal would *not* reproduce
//! the table's own `B_g0` column, which is why the second decimal
//! matters.
//!
//! ## Response structure
//!
//! `r' = 1`, and `beta_SIS,2` is taken over the response the protocol
//! actually transmits, so the model/protocol split that
//! recorded — two `B_rs`, two `beta_SIS,2` — is gone.  `phi_b` is now a
//! `BoundGen` output rather than a symbol the algorithms used and the
//! parameter generator never produced, and the single outer
//! response width is replaced by the split `(sigma_s, sigma_m)`.
//!
//! ## Exact accept/reject bounds
//!
//! Every bound that decides an acceptance is a [`Rat`], not an `f64`.
//! Each has the shape `K sqrt(M)` with `K` rational and `M` a positive
//! integer, so squaring turns the comparison into one between exact
//! rationals and removes the `sqrt` — and with it the last place where
//! two implementations could disagree about a coefficient sitting on the
//! boundary.  The `f64` accessors that remain are for reporting and for
//! the Gaussian widths, never for an accept/reject test.

#![allow(non_snake_case)]

// ---- prime search --------------------------------------------------------
// Deterministic and reproducible; used only to pin the moduli below.

const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Deterministic Miller-Rabin over a fixed small-prime base set —
/// sufficient well
/// past the 49-bit moduli used here.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &SMALL_PRIMES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    'witness: for &a in &SMALL_PRIMES {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        exp >>= 1;
        base = mul_mod(base, base, m);
    }
    acc
}

/// Largest prime `< 2^bits` congruent to `residue` mod `modulus`.
pub fn largest_prime_below(bits: u32, residue: u64, modulus: u64) -> u64 {
    let mut n = (1u64 << bits) - 1;
    n -= (n - residue) % modulus;
    loop {
        if is_prime(n) {
            return n;
        }
        n -= modulus;
    }
}

/// Pinned moduli, keyed by the bit lengths the paper's table quotes:
/// `p` uses 44 and 48, `q_hat` uses 44, 46, 48 and 49.
///
/// **Paper**, which prints the concrete moduli in
/// `tab:river-concrete-moduli`.  Both tables are reproduced by a rule and
/// the two rules are *different*, which is why guessing was not safe:
///
/// - `p` is the largest prime *below* `2^bits` that is 5 mod 8;
/// - `q_hat` is the smallest prime *above* `2^{bits-1}` that is 5 mod 8.
///
/// This tree previously derived `q_hat` by the `p` rule and so used a
/// value roughly twice the paper's at every profile — admissible against
/// the `hat-q` condition, but not the published one, and `q_hat` enters
/// `b_B` and hence the wire.
pub const P_44: u64 = 17_592_186_043_877;
pub const P_48: u64 = 281_474_976_710_597;
pub const QHAT_44: u64 = 8_796_093_022_237;
pub const QHAT_46: u64 = 35_184_372_088_997;
pub const QHAT_48: u64 = 140_737_488_355_333;
pub const QHAT_49: u64 = 281_474_976_710_677;

/// Smallest prime `> 2^bits` congruent to `residue` mod `modulus`.
pub fn smallest_prime_above(bits: u32, residue: u64, modulus: u64) -> u64 {
    let mut n = (1u64 << bits) + 1;
    n += (residue + modulus - n % modulus) % modulus;
    loop {
        if is_prime(n) {
            return n;
        }
        n += modulus;
    }
}

/// Re-derive every pinned modulus.  Returns the list of mismatches.
pub fn verify_moduli() -> Vec<String> {
    let mut errors = Vec::new();
    for (bits, value, name, below) in [
        (44u32, P_44, "p[44]", true),
        (48, P_48, "p[48]", true),
        (44, QHAT_44, "q_hat[44]", false),
        (46, QHAT_46, "q_hat[46]", false),
        (48, QHAT_48, "q_hat[48]", false),
        (49, QHAT_49, "q_hat[49]", false),
    ] {
        let derived = if below {
            largest_prime_below(bits, 5, 8)
        } else {
            smallest_prime_above(bits - 1, 5, 8)
        };
        if derived != value {
            errors.push(format!("{name}: pinned {value}, derived {derived}"));
        }
        if value % 8 != 5 {
            errors.push(format!("{name} = {value} is not 5 mod 8"));
        }
        if value.next_power_of_two().trailing_zeros() != bits {
            errors.push(format!("{name} = {value} is not a {bits}-bit value"));
        }
    }
    errors
}

// ---- exact rationals -----------------------------------------------------

/// A non-negative exact rational, always in lowest terms.
///
/// Every bound that decides an accept/reject is one of these.  The
/// Python reference uses `fractions.Fraction`; the values in play here
/// have numerators under 60 bits and denominators under 7, so `u128`
/// carries them with room for the comparison products.
///
/// There is no arbitrary-precision fallback on purpose: [`Rat::new`]
/// takes `u128` and every construction site below is a product of
/// profile literals that [`RiVeRParams::check`] has already bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rat {
    num: u128,
    den: u128,
}

const fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Rat {
    /// `num / den`, reduced.  `den == 0` is a programming error and
    /// panics; no caller can reach it with a value from a profile.
    pub const fn new(num: u128, den: u128) -> Rat {
        assert!(den != 0, "Rat with zero denominator");
        let g = gcd_u128(num, den);
        if g == 0 {
            // `num == den == 0` is unreachable given the assert above.
            return Rat { num: 0, den: 1 };
        }
        Rat {
            num: num / g,
            den: den / g,
        }
    }

    pub const fn from_u128(v: u128) -> Rat {
        Rat { num: v, den: 1 }
    }

    pub const fn num(&self) -> u128 {
        self.num
    }

    pub const fn den(&self) -> u128 {
        self.den
    }

    /// `self * v`, or `None` on overflow.
    pub const fn checked_mul_u128(&self, v: u128) -> Option<Rat> {
        match self.num.checked_mul(v) {
            Some(num) => Some(Rat::new(num, self.den)),
            None => None,
        }
    }

    /// `self * other`, or `None` on overflow.
    pub const fn checked_mul(&self, other: Rat) -> Option<Rat> {
        let (Some(num), Some(den)) = (
            self.num.checked_mul(other.num),
            self.den.checked_mul(other.den),
        ) else {
            return None;
        };
        Some(Rat::new(num, den))
    }

    /// `self + other`, or `None` on overflow.
    pub const fn checked_add(&self, other: Rat) -> Option<Rat> {
        let (Some(a), Some(b), Some(den)) = (
            self.num.checked_mul(other.den),
            other.num.checked_mul(self.den),
            self.den.checked_mul(other.den),
        ) else {
            return None;
        };
        match a.checked_add(b) {
            Some(num) => Some(Rat::new(num, den)),
            None => None,
        }
    }

    /// `self * v`.
    ///
    /// # Panics
    ///
    /// On overflow.  Every construction site below is a product of profile
    /// literals that [`RiVeRParams::checked_shapes`] has already bounded,
    /// so this is the same "validate with `check()` first" contract the
    /// other accessors carry — and a loud panic beats the release-mode
    /// alternative, which is a silently wrapped *acceptance bound*.
    pub const fn mul_u128(&self, v: u128) -> Rat {
        match self.checked_mul_u128(v) {
            Some(r) => r,
            None => panic!("Rat::mul_u128 overflows — validate with check() first"),
        }
    }

    /// `self * other`.  Panics on overflow; see [`Self::mul_u128`].
    pub const fn mul(&self, other: Rat) -> Rat {
        match self.checked_mul(other) {
            Some(r) => r,
            None => panic!("Rat::mul overflows — validate with check() first"),
        }
    }

    /// `self + other`.  Panics on overflow; see [`Self::mul_u128`].
    pub const fn add(&self, other: Rat) -> Rat {
        match self.checked_add(other) {
            Some(r) => r,
            None => panic!("Rat::add overflows — validate with check() first"),
        }
    }

    /// `value > self` — the shape every bound check takes.
    ///
    /// Decided as `value · den > num` over `u128`, so no `sqrt` and no
    /// float reach the decision.
    ///
    /// Checked, because `value` is a norm of caller-supplied coefficients
    /// and `den` is a profile constant: a squared infinity norm from a
    /// hand-built proof reaches `2^126`, and `value · den` need not fit.
    /// An overflow can only mean `value · den` is at least `2^128`, which
    /// is above every `num` a `Rat` can hold, so it *is* an exceedance —
    /// wrapping would have reported the opposite.
    pub const fn exceeded_by(&self, value: u128) -> bool {
        match value.checked_mul(self.den) {
            Some(scaled) => scaled > self.num,
            None => true,
        }
    }

    /// `value >= self`.
    pub const fn reached_by(&self, value: u128) -> bool {
        match value.checked_mul(self.den) {
            Some(scaled) => scaled >= self.num,
            None => true,
        }
    }

    /// `floor(num/den)`, exactly.
    ///
    /// The largest integer that does not exceed the bound, for the two
    /// thresholds the paper states directly rather than as `K sqrt(M)`.
    pub const fn floor(&self) -> u128 {
        self.num / self.den
    }

    /// `floor(sqrt(num/den))`, exactly.
    ///
    /// `sqrt` is monotone and `k` is an integer, so `k <= sqrt(x)` iff
    /// `k^2 <= x` iff `k^2 <= floor(x)` — taking the floor first is not
    /// an approximation.  This is how a verifier bound of the form
    /// `K sqrt(M)` becomes the largest coefficient that can pass it,
    /// which is exactly the cap the encoder needs.
    pub const fn floor_sqrt(&self) -> u128 {
        (self.num / self.den).isqrt()
    }

    /// Correctly-rounded `num / den`.
    ///
    /// Round-to-odd on a 55-bit quotient, then one `f64` conversion:
    /// the extra two bits plus the sticky bit make the final
    /// round-to-nearest-even agree with an exact division, which
    /// `num as f64 / den as f64` does not once `num` passes `2^53`.
    /// Only reporting and the security-margin checks use this; nothing
    /// byte-visible does.
    pub fn to_f64(&self) -> f64 {
        if self.num == 0 {
            return 0.0;
        }
        let nbits = 128 - self.num.leading_zeros() as i32;
        let dbits = 128 - self.den.leading_zeros() as i32;
        let mut shift = 55 - (nbits - dbits);
        let headroom = 128 - nbits;
        if shift > headroom {
            shift = headroom;
        }
        if shift < 0 {
            shift = 0;
        }
        let scaled = self.num << shift;
        let q = scaled / self.den;
        let q = if scaled.is_multiple_of(self.den) {
            q
        } else {
            q | 1
        };
        (q as f64) * (2f64).powi(-shift)
    }
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Rat) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rat {
    fn cmp(&self, other: &Rat) -> std::cmp::Ordering {
        (self.num * other.den).cmp(&(other.num * self.den))
    }
}

impl std::fmt::Display for Rat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

// ---- parameter set -------------------------------------------------------

/// One RiVeR parameter profile.  Field names follow the paper; every
/// derived quantity is a method, so a profile is fully described by the
/// literals in [`PROFILES`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiVeRParams {
    pub name: &'static str,

    // ring geometry (fixed across all profiles)
    pub d: usize,
    pub q0: u64,
    pub p: u64,
    pub q_hat: u64,

    // module ranks
    pub n: usize,
    pub ell: usize,
    pub n_hat: usize,
    pub k_hat: usize,

    /// Size of the ring: exactly `N` keys, no padding.
    pub N: usize,

    // challenge / key
    pub w: usize,
    pub gamma: u64,
    pub beta: u64,

    // Rejection sampling.  The paper splits the single outer
    // response into a short block `z_s` (the secret key) and an error
    // block `z_m = (z_key, z_eval)`, with separate widths.  `phi_m` and
    // `phi_b` are shared by every profile.
    //
    // Integers, and that matters: `Rej` folds
    // `exp(-(24 phi + 1) / 2 phi^2)` into an exact threshold, so a float
    // here would change a decision.
    /// Slack for the selector response `f_1`.
    pub phi_a: u64,
    /// Slack for the short response `z_s`.
    pub phi_s: u64,
    /// Slack for the error response `z_m`.  Paper: 32.
    pub phi_m: u64,
    /// Slack for the binary response `z_b`.  Paper: 2.
    pub phi_b: u64,

    // Product-bound calibration, as exact rationals — see the module
    // docs.
    pub tau_g0: Rat,
    pub tau_g1: Rat,

    // repetition calibration exported by the parameter search
    pub epsilon_g_u: f64,

    // bit dropping
    pub K_b: u32,
    pub K_a: u32,
    /// Compression margin `s_c` fixed by the parameter appendix.
    pub s_cmp: u32,

    /// Reduction-only auxiliary block.  Paper: `r' = 1` for every final
    /// profile (it was 3 through the paper).
    pub r_prime: usize,

    // tunables
    pub lam: u32,
    pub max_attempts: u32,

    /// Marks profiles that deliberately violate the security-side
    /// modulus conditions in order to run fast.  Only `TOY` sets it.
    pub insecure_toy: bool,
}

impl RiVeRParams {
    // ---- moduli ----------------------------------------------------------

    /// Outer modulus `q = q_0 · p` (composite; `R_q = R_p × R_{q_0}`).
    ///
    /// # Panics
    ///
    /// If `q_0 · p` leaves `u64`.  Every shipped profile is far from
    /// that, and [`Self::checked_q`] is the form to use on a profile that
    /// has not been validated yet.
    pub const fn q(&self) -> u64 {
        self.q0 * self.p
    }

    /// `q = q_0 · p`, or `None` if that leaves `u64`.
    pub const fn checked_q(&self) -> Option<u64> {
        self.q0.checked_mul(self.p)
    }

    // ---- BoundGen --------------------------------------------------------
    //
    // Each bound below is one entry of BoundGen's output tuple.
    //
    // DISCREPANCY.  The paper gives that tuple in two incompatible
    // orders.  `BoundGen` returns
    //   (r', s_c, phi_a, phi_s, phi_b, phi_m, tau_g0, tau_g1, ...)
    // while all three OOM algorithms parse
    //   (r', s_c, phi_a, phi_b, phi_s, phi_m, tau_g0, tau_g1, ...)
    // — positions 4 and 5 swapped.  That is not cosmetic: `phi_s` is 22
    // to 32 and `phi_b` is 2, so a positional implementation would sample
    // and test the two responses at each other's widths.
    //
    // One re-render was editorial — abstract wording, the
    // supported ring sizes, and citations for the `delta ~ 1.0045`
    // target — and did **not** reconcile these two lines.  So the
    // contradiction stands in the source, and what keeps it out of the
    // arithmetic is that nothing here is positional: every value is a
    // named field, and the meaning follows the authors' clarification
    // (`z_s` answers `r_0 = s` at `sigma_s = phi_s B_s`; `z_m` answers
    // `r_1 = (e_key, e_eval)` at `sigma_m = phi_m eta_m`).

    /// `floor(q_0 / 2)`; bounds the **centred** rounding error.
    ///
    /// The rounding relation itself keeps errors in `[0, q_0-1]`; this is
    /// the centred range the concrete norm bounds use, and the
    /// implementation carries the shift explicitly — see
    /// [`crate::ring::to_centered_error`].
    pub const fn B_e(&self) -> u64 {
        self.q0 / 2
    }

    /// `B_a = gamma sqrt(2w)` as an exact integer when `2w` is a perfect
    /// square, which it is at every published profile (`B_a = 128`).
    ///
    /// `None` means the profile would need a float `B_a`; `sigma_a` and
    /// `B_g0`/`B_g1` are exact only in the `Some` case, and
    /// [`Self::check`] refuses the profile otherwise rather than
    /// silently dropping to floats in a wire-visible place.
    pub fn B_a_exact(&self) -> Option<u64> {
        // Checked from the *first* operation, not from the second: with
        // `w = usize::MAX`, `2 * (w as u64)` overflows before anything
        // downstream gets a chance to notice — a debug panic and a
        // release wrap, inside the function `check` calls to decide
        // whether the profile is admissible at all.
        let two_w = (self.w as u64).checked_mul(2)?;
        let root = two_w.isqrt();
        if root.checked_mul(root)? != two_w {
            return None;
        }
        self.gamma.checked_mul(root)
    }

    /// `B_a` as an `f64`, for reporting.
    pub fn B_a(&self) -> f64 {
        match self.B_a_exact() {
            Some(v) => v as f64,
            None => self.gamma as f64 * ((2 * self.w) as f64).sqrt(),
        }
    }

    /// `B = w gamma beta sqrt(d k_hat)`; scale of the binary mask `r_a`.
    pub fn cal_B(&self) -> f64 {
        self.wgb() as f64 * ((self.d * self.k_hat) as f64).sqrt()
    }

    /// `B_s = w gamma B_e sqrt(d(ell+n))`; scale of the short response.
    ///
    /// the paper regrouped the response: `r_0 = (s, e_key)` is now the
    /// block answered at `sigma_s`, so `B_s` covers `ell + n` ring
    /// elements and carries `B_e` (the bound on `e_key`) rather than
    /// `beta` (the bound on `s` alone).  The previous revision had
    /// `B_s = w gamma beta sqrt(d ell)`, some 43x smaller.
    pub fn B_s(&self) -> f64 {
        self.wgB() as f64 * ((self.d * (self.ell + self.n)) as f64).sqrt()
    }

    /// `eta_m = w gamma B_e sqrt(d)`; scale of the error response.
    ///
    /// Independent of the profile: 86889.28 for all five.
    pub fn eta_m(&self) -> f64 {
        (self.w as u64 * self.gamma * self.B_e()) as f64 * (self.d as f64).sqrt()
    }

    /// `w gamma beta`, the scale of `B` (the binary mask `r_a`).
    const fn wgb(&self) -> u64 {
        self.w as u64 * self.gamma * self.beta
    }

    /// `w gamma B_e`, the scale shared by `B_s` and `eta_m`.
    ///
    /// Both response blocks bound a rounding error;
    /// before it only `eta_m` did.
    #[allow(non_snake_case)]
    const fn wgB(&self) -> u64 {
        self.w as u64 * self.gamma * self.B_e()
    }

    #[allow(non_snake_case)]
    const fn checked_wgB(&self) -> Option<u64> {
        match (self.w as u64).checked_mul(self.gamma) {
            Some(v) => v.checked_mul(self.B_e()),
            None => None,
        }
    }

    /// Product-check threshold for `g_0`, carrying the `(N-1)` variance
    /// factor: `tau_g0 · d(N-1)/3 · (phi_a B_a)^2`.
    ///
    /// Exact: the verifier compares an integer against a rational rather
    /// than against a float whose last ulp could fork a transcript.
    pub fn B_g0(&self) -> Rat {
        self.try_b_g0()
            .expect("B_g0 overflows — validate with check() first")
    }

    /// Product-check threshold for `g_1`: `tau_g1 · d/2 · (phi_a B_a)^2`.
    pub fn B_g1(&self) -> Rat {
        self.try_b_g1()
            .expect("B_g1 overflows — validate with check() first")
    }

    /// `BoundGen`'s lower bound on `K_a`:
    /// `K_b + ceil(log2(w gamma n_hat d)) + s_c`.
    ///
    /// 28 for every published profile, because `ceil(log2(w gamma n_hat d))`
    /// is 20 for all four values of `n_hat` in use (42, 43, 49, 50).
    /// `BoundGen` aborts when `K_a` is below this; [`Self::check`]
    /// reproduces that abort.
    ///
    /// # Panics
    ///
    /// If the product or the sum leaves its type — see
    /// [`Self::checked_k_a_boundgen`].
    pub fn K_a_boundgen(&self) -> u32 {
        self.checked_k_a_boundgen()
            .expect("K_a_boundgen overflows — validate with check() first")
    }

    // ---- Gaussian widths -------------------------------------------------

    /// `sigma_a = phi_a B_a`, exactly (an integer at every profile).
    pub fn sigma_a_exact(&self) -> u64 {
        self.phi_a
            * self
                .B_a_exact()
                .expect("2w is not a perfect square — validate with check() first")
    }

    /// Width of the selector masks `a_i`: `phi_a B_a`.
    pub fn sigma_a(&self) -> f64 {
        self.sigma_a_exact() as f64
    }

    /// Width of the binary mask `r_a`: `phi_b B`.
    ///
    /// **Repair.**  The `Com` figure samples `r_a <- D_B` while its
    /// `Rej_2` call and the communication formula both use `phi_b B`.  A
    /// rejection sampler is only correct when the mask width equals the
    /// sigma in its acceptance test, so we sample at `phi_b B`.
    pub fn sigma_b(&self) -> f64 {
        self.phi_b as f64 * self.cal_B()
    }

    /// Width of the short response: `sigma_s = phi_s B_s`.
    pub fn sigma_s(&self) -> f64 {
        self.phi_s as f64 * self.B_s()
    }

    /// Width of the error response: `sigma_m = phi_m eta_m`.
    pub fn sigma_m(&self) -> f64 {
        self.phi_m as f64 * self.eta_m()
    }

    // ---- verifier bounds -------------------------------------------------
    //
    // Each has the shape `K sqrt(M)` with `K` rational and `M` a positive
    // integer, so squaring turns the comparison into one between exact
    // rationals.  The `*_inf_bound` floats are for reporting and for the
    // codec's field cap, never for an accept/reject test.

    /// Every exact bound, evaluated with checked arithmetic.
    ///
    /// The public accessors below unwrap these and carry the "validate
    /// with `check()` first" contract; [`Self::checked_shapes`] calls
    /// *these*, so `check` can refuse a profile whose bounds would
    /// overflow instead of panicking while deciding whether to.
    fn try_f1_inf_bound_sq(&self) -> Option<Rat> {
        let k = 6u64.checked_mul(self.phi_a)?.checked_mul(self.gamma)? as u128;
        k.checked_mul(k)?
            .checked_mul((self.w as u128).checked_mul(2)?)
            .map(Rat::from_u128)
    }

    fn try_zb_inf_bound_sq(&self) -> Option<Rat> {
        self.try_z_inf_bound_sq(self.phi_b, self.checked_wgb()?, self.k_hat)
    }

    /// `(6 sigma_s)^2 = (6 phi_s w gamma B_e)^2 d(ell+n)`, exactly.
    ///
    /// Bounds `(z_s, z_key)`, not `z_s` alone.
    fn try_zs_inf_bound_sq(&self) -> Option<Rat> {
        // `ell + n` is `s_dim()`, but computed with a checked add: `check`
        // reaches here before it has ruled out `ell = usize::MAX`, and a
        // debug build traps on the wrap.  Every `try_*` is on that path.
        self.try_z_inf_bound_sq(
            self.phi_s,
            self.checked_wgB()?,
            self.ell.checked_add(self.n)?,
        )
    }

    /// `(6 sigma_m)^2 = (6 phi_m w gamma B_e)^2 d`, exactly.
    ///
    /// Unchanged in form by the paper, but it now caps a single ring
    /// element (`z_eval`) rather than the `n + 1` of `(z_key, z_eval)`.
    fn try_zm_inf_bound_sq(&self) -> Option<Rat> {
        let k = 6u64
            .checked_mul(self.phi_m)?
            .checked_mul(self.checked_wgB()?)? as u128;
        k.checked_mul(k)?
            .checked_mul(self.d as u128)
            .map(Rat::from_u128)
    }

    /// `sigma_s^2 = (phi_s w gamma B_e)^2 d(ell+n)`, exactly.
    fn try_sigma_s_sq(&self) -> Option<u128> {
        let ss = self.phi_s.checked_mul(self.checked_wgB()?)? as u128;
        let rows = (self.ell as u128).checked_add(self.n as u128)?;
        ss.checked_mul(ss)?
            .checked_mul((self.d as u128).checked_mul(rows)?)
    }

    /// `sigma_m^2 = (phi_m w gamma B_e)^2 d`, exactly.
    fn try_sigma_m_sq(&self) -> Option<u128> {
        let sm = self.phi_m.checked_mul(self.checked_wgB()?)? as u128;
        sm.checked_mul(sm)?.checked_mul(self.d as u128)
    }

    /// `(6 phi · scale)^2 · d · rows`, the shape `z_b` and `z_s` share.
    fn try_z_inf_bound_sq(&self, phi: u64, scale: u64, rows: usize) -> Option<Rat> {
        let k = 6u64.checked_mul(phi)?.checked_mul(scale)? as u128;
        k.checked_mul(k)?
            .checked_mul((self.d as u128).checked_mul(rows as u128)?)
            .map(Rat::from_u128)
    }

    /// `1.44 (sigma_s^2 d(ell+n) + sigma_m^2 d)`, exactly.
    ///
    /// `1.2^2` is `36/25` as a rational, not the binary float `1.44`.
    fn try_z_l2_bound_sq(&self) -> Option<Rat> {
        let rows = (self.ell as u128).checked_add(self.n as u128)?;
        let d_sn = (self.d as u128).checked_mul(rows)?;
        let inner = self
            .try_sigma_s_sq()?
            .checked_mul(d_sn)?
            .checked_add(self.try_sigma_m_sq()?.checked_mul(self.d as u128)?)?;
        inner.checked_mul(36).map(|v| Rat::new(v, 25))
    }

    fn try_b_g(&self, tau: Rat, width: u128, den: u128) -> Option<Rat> {
        let s = self.checked_sigma_a()? as u128;
        tau.checked_mul(Rat::new(width, den))?
            .checked_mul_u128(s.checked_mul(s)?)
    }

    fn try_b_g0(&self) -> Option<Rat> {
        let width = (self.d as u128).checked_mul(self.N.checked_sub(1)? as u128)?;
        self.try_b_g(self.tau_g0, width, 3)
    }

    fn try_b_g1(&self) -> Option<Rat> {
        self.try_b_g(self.tau_g1, self.d as u128, 2)
    }

    /// `w gamma beta`, checked.
    fn checked_wgb(&self) -> Option<u64> {
        (self.w as u64)
            .checked_mul(self.gamma)?
            .checked_mul(self.beta)
    }

    /// `sigma_a = phi_a B_a`, checked.
    fn checked_sigma_a(&self) -> Option<u64> {
        self.phi_a.checked_mul(self.B_a_exact()?)
    }

    /// `(6 phi_a B_a)^2 = (6 phi_a gamma)^2 · 2w`, exactly.
    ///
    /// # Panics
    ///
    /// On overflow — which [`Self::checked_shapes`] makes unreachable for
    /// a profile `check()` has cleared.  The same holds for every bound
    /// below.
    pub fn f1_inf_bound_sq(&self) -> Rat {
        self.try_f1_inf_bound_sq()
            .expect("f1 bound overflows — validate with check() first")
    }

    /// `||f_1||_inf <= 6 phi_a B_a`.
    pub fn f1_inf_bound(&self) -> f64 {
        6.0 * self.sigma_a()
    }

    /// `(6 phi_b B)^2 = (6 phi_b w gamma beta)^2 · d k_hat`, exactly.
    pub fn zb_inf_bound_sq(&self) -> Rat {
        self.try_zb_inf_bound_sq()
            .expect("zb bound overflows — validate with check() first")
    }

    /// `||z_b||_inf <= 6 phi_b B`.
    pub fn zb_inf_bound(&self) -> f64 {
        6.0 * self.sigma_b()
    }

    /// `(6 sigma_s)^2 = (6 phi_s w gamma beta)^2 · d ell`, exactly.
    pub fn zs_inf_bound_sq(&self) -> Rat {
        self.try_zs_inf_bound_sq()
            .expect("zs bound overflows — validate with check() first")
    }

    /// `||z_s||_inf <= 6 sigma_s`.
    pub fn zs_inf_bound(&self) -> f64 {
        6.0 * self.sigma_s()
    }

    /// `(6 sigma_m)^2 = (6 phi_m w gamma B_e)^2 · d`, exactly.
    pub fn zm_inf_bound_sq(&self) -> Rat {
        self.try_zm_inf_bound_sq()
            .expect("zm bound overflows — validate with check() first")
    }

    /// `||z_eval||_inf <= 6 sigma_m`.
    pub fn zm_inf_bound(&self) -> f64 {
        6.0 * self.sigma_m()
    }

    /// `1.44 (sigma_s^2 d(ell+n) + sigma_m^2 d)`, exactly.
    ///
    /// `1.2^2` is `36/25` as a rational, not the binary float `1.44`.
    pub fn z_l2_bound_sq(&self) -> Rat {
        self.try_z_l2_bound_sq()
            .expect("Euclidean bound overflows — validate with check() first")
    }

    /// `||z||_2 <= 1.2 sqrt(sigma_s^2 d(ell+n) + sigma_m^2 d)`.
    pub fn z_l2_bound(&self) -> f64 {
        let ss = self.sigma_s();
        let sm = self.sigma_m();
        1.2 * (ss * ss * (self.d * (self.ell + self.n)) as f64 + sm * sm * self.d as f64).sqrt()
    }

    /// Compression restart threshold `2^{K_a-1} - w gamma 2^{K_b-1}`.
    pub fn T_cmp(&self) -> i64 {
        (1i64 << (self.K_a - 1)) - (self.w as i64 * self.gamma as i64) * (1i64 << (self.K_b - 1))
    }

    // ---- M-SIS / A-MSIS bounds ------------------------------------------

    /// `2.4 sqrt(sigma_s^2 d(ell+n))`.
    ///
    /// the paper: the `sigma_m` term is gone.  Two accepting forks now
    /// differ only in the `(s, e_key)` block for this bound.
    pub fn beta_sis_1(&self) -> f64 {
        let ss = self.sigma_s();
        2.4 * (ss * ss * (self.d * (self.ell + self.n)) as f64).sqrt()
    }

    /// `2.4 sqrt(sigma_s^2 d(ell+n) + sigma_m^2 d)`.
    pub fn beta_sis_2(&self) -> f64 {
        let ss = self.sigma_s();
        let sm = self.sigma_m();
        2.4 * (ss * ss * (self.d * (self.ell + self.n)) as f64 + sm * sm * self.d as f64).sqrt()
    }

    /// The second term of `beta_SIS`, kept separate so it can be checked.
    ///
    /// `beta_SIS,1 + 2 w gamma sqrt(d(ell beta^2 + (n+r') B_e^2))`.
    pub fn beta_sis_embedded(&self) -> f64 {
        let b_e = self.B_e();
        let inner = (self.ell as u64 * self.beta * self.beta
            + (self.n + self.r_prime) as u64 * b_e * b_e) as f64
            * self.d as f64;
        self.beta_sis_1() + 2.0 * (self.w as u64 * self.gamma) as f64 * inner.sqrt()
    }

    /// `max{4 w gamma beta_SIS,1, beta_SIS,embedded}`.
    ///
    /// The paper notes the first term is the larger for all five
    /// profiles; the tests check that rather than assuming it.
    pub fn beta_sis(&self) -> f64 {
        f64::max(
            4.0 * (self.w as u64 * self.gamma) as f64 * self.beta_sis_1(),
            self.beta_sis_embedded(),
        )
    }

    /// The six A-MSIS block bounds, in extraction block order.
    ///
    /// Widths `(k_hat, 1, N-1, 1, N-1, n_hat)` for
    /// `(z_b, f_0, f_1, g_0, g_1, e_c)`.
    pub fn beta_sel(&self) -> [f64; 6] {
        let s = (4 * self.w as u64 * self.gamma) as f64;
        [
            s * 6.0 * self.sigma_b(),
            s * (self.gamma as f64 + (6 * (self.N - 1)) as f64 * self.sigma_a()),
            s * 6.0 * self.sigma_a(),
            s * self.B_g0().to_f64(),
            s * self.B_g1().to_f64(),
            s * (2f64).powi(self.K_a as i32),
        ]
    }

    /// The five merged bounds used for the A-MSIS *estimate*: `f_0` and
    /// `f_1` become one width-`N` entry carrying the larger bound, which
    /// can only enlarge the admissible solution set.
    pub fn beta_sel_merged(&self) -> [f64; 5] {
        let s = (4 * self.w as u64 * self.gamma) as f64;
        [
            s * 6.0 * self.sigma_b(),
            s * (self.gamma as f64 + (6 * (self.N - 1)) as f64 * self.sigma_a()),
            s * self.B_g0().to_f64(),
            s * self.B_g1().to_f64(),
            s * (2f64).powi(self.K_a as i32),
        ]
    }

    pub fn beta_sel_inf(&self) -> f64 {
        self.beta_sel().into_iter().fold(f64::MIN, f64::max)
    }

    // ---- repetition estimate --------------------------------------------

    /// `tau_rej`, the numerator of `Rej_1`'s repetition constant.
    ///
    /// **Paper.**  The paper parameterises what was
    /// a hard-coded 12: `M_1 = exp(tau_rej/phi + 1/(2 phi^2))`, with
    /// `eps_1` the statistical loss Lemma 3.3 of [DO07] gives for the
    /// chosen value.  Asymptotically it is chosen so `eps_1` is negligible
    /// while `M_1` is polynomial; concretely it is fixed at 12.
    ///
    /// One constant, consumed by **both** the sampler
    /// ([`crate::sample::rej1`], through `oom.rs`) and the three
    /// repetition factors below.  They were separate literals, which meant
    /// the reported estimate could assume a rejection rate the sampler did
    /// not achieve.
    pub const REJ_TAU: u64 = 12;

    /// `exp(tau_rej/phi_a + 1/(2 phi_a^2))`, from Lemma grs(1).
    pub fn mu_a(&self) -> f64 {
        self.mu_rej1(self.phi_a)
    }

    /// `exp(tau_rej/phi_s + 1/(2 phi_s^2))`.
    pub fn mu_s(&self) -> f64 {
        self.mu_rej1(self.phi_s)
    }

    /// `exp(tau_rej/phi_m + 1/(2 phi_m^2))`.
    pub fn mu_m(&self) -> f64 {
        self.mu_rej1(self.phi_m)
    }

    /// `M_1` at one slack, from the shared `tau_rej`.
    fn mu_rej1(&self, phi: u64) -> f64 {
        let p = phi as f64;
        (Self::REJ_TAU as f64 / p + 1.0 / (2.0 * p * p)).exp()
    }

    /// `2 exp(1/(2 phi_b^2))`, from Lemma grs(2).
    ///
    /// The factor 2 is the `<z, v> >= 0` half-space rejection, and the
    /// appendix is explicit that it is therefore not charged again by the
    /// subsequent infinity-norm check.
    pub fn mu_b(&self) -> f64 {
        let p = self.phi_b as f64;
        2.0 * (1.0 / (2.0 * p * p)).exp()
    }

    /// The four `2 d w exp(-18)` tail terms, in `(a, b, s, m)` order.
    ///
    /// The `s` and `m` widths follow the response split: `(ell+n)` and `1`
    ///.  The appendix's own tail paragraph is internally
    /// inconsistent here — one line says `2d(ell+1)exp(-18)` for the `s`
    /// block and the display two paragraphs later says `2d(n+ell)exp(-18)`.
    /// We take the latter, which matches the algorithm's `ell + n`
    /// coefficients.
    pub fn eps_tail(&self) -> [f64; 4] {
        let t = 2.0 * self.d as f64 * (-18f64).exp();
        [
            t * (self.N - 1) as f64,
            t * self.k_hat as f64,
            t * (self.ell + self.n) as f64,
            t,
        ]
    }

    /// `eps_2`: the joint Euclidean response check's failure probability.
    ///
    /// New.  The appendix dominates all `d(ell+n+1)`
    /// response coefficients with a width-`sigma_s` Gaussian — sound
    /// because the same revision requires `sigma_s >= sigma_m` — and
    /// applies the Euclidean tail bound at ratio
    /// `rho = 1.2 sqrt((ell+n+(sigma_m/sigma_s)^2)/(ell+n+1))`, giving
    /// `rho^M exp(M(1-rho^2)/2)` for `M = d(ell+n+1)`.
    ///
    /// The paper states this is below `2^-150` at every final profile; it
    /// is, by about a hundred orders of magnitude, so it does not move the
    /// estimate.  Computed rather than assumed so a profile that *did*
    /// move it would say so.
    pub fn eps_euclidean(&self) -> f64 {
        let m = (self.d * (self.ell + self.n + 1)) as f64;
        let ratio_sq = (self.sigma_m() / self.sigma_s()).powi(2);
        let rho =
            1.2 * (((self.ell + self.n) as f64 + ratio_sq) / (self.ell + self.n + 1) as f64).sqrt();
        // In logs: the direct form underflows to 0.0 well before the bound
        // stops being informative.
        let log2_eps = m * rho.log2() + m * (1.0 - rho * rho) / 2.0 * std::f64::consts::E.log2();
        if log2_eps > -1000.0 {
            (2f64).powf(log2_eps)
        } else {
            0.0
        }
    }

    /// Uniform-low-bits sanity model for the compression check.
    pub fn p_cmp_uniform(&self) -> f64 {
        let t = ((2 * self.T_cmp() - 1) as f64) / (2f64).powi(self.K_a as i32);
        t.powf((self.n_hat * self.d) as f64)
    }

    /// `mu_a mu_b mu_s mu_m`: the four Gaussian rejection samplers.
    pub fn mu_gaussian(&self) -> f64 {
        self.mu_a() * self.mu_b() * self.mu_s() * self.mu_m()
    }

    /// The table's "Repeat bound" column, from the appendix's formula.
    ///
    /// **This closed in the paper.**  Under the table the
    /// appendix's own formula did not reproduce its printed column, and
    /// this tree carried a per-profile `c_pub_model` backsolved *from*
    /// that column — a number with no provenance but the answer it had to
    /// produce.  Against the paper table the components multiply out
    /// to the printed value at all five profiles, so the backsolve is gone
    /// and this is now computed.
    pub fn mu_river(&self) -> f64 {
        self.mu_gaussian() / self.c_pub_from_components()
    }

    /// The denominator of `eq:river-repeat-bound`, from the components:
    /// `(1-eps_a)(1-eps_b)((1-eps_s)(1-eps_m) - eps_2)(1-eps_g)(1-eps_c)`.
    ///
    /// The `eps_2` term entered; before it the four tails
    /// multiplied in flat.
    pub fn c_pub_from_components(&self) -> f64 {
        let [eps_a, eps_b, eps_s, eps_m] = self.eps_tail();
        (1.0 - eps_a)
            * (1.0 - eps_b)
            * ((1.0 - eps_s) * (1.0 - eps_m) - self.eps_euclidean())
            * (1.0 - self.epsilon_g_u)
            * self.p_cmp_uniform()
    }

    // ---- size estimate ---------------------------------------------------

    /// Paper: the exact proof contributes a fixed 13.5 KB to every
    /// profile.
    pub const EXACT_PROOF_KB: f64 = 13.5;

    /// `b_B = ceil(log2(ceil(q_hat / 2^{K_b})))`.
    pub fn b_B(&self) -> u32 {
        ceil_log2(self.q_hat.div_ceil(1u64 << self.K_b))
    }

    /// `h(sigma) = log2(4.13 sigma)`, the entropy model of [C:ESLR23] 2.4.
    fn h(sigma: f64) -> f64 {
        (4.13 * sigma).log2()
    }

    /// The `B`, `x`, `f_1` and `z_b` terms, shared by both layouts.
    fn proof_size_oom_common_bits(&self) -> f64 {
        (self.n_hat * self.d) as f64 * self.b_B() as f64
            + self.challenge_entropy()
            + ((self.N - 1) * self.d) as f64 * Self::h(self.sigma_a())
            + (self.k_hat * self.d) as f64 * Self::h(self.sigma_b())
    }

    /// `|pi_OOM| = L_OOM / 8192` KB, from the communication formula.
    ///
    /// The six terms are the contributions of `B`, `x`, `f_1`, `z_b`,
    /// `(z_s, z_key)` and `z_eval`.  The last two are charged at the
    /// dimensions the algorithm actually transmits: `(ell+n) d h(sigma_s)`
    /// and `d h(sigma_m)`.
    ///
    /// **DISCREPANCY.**  The paper appendix regrouped the
    /// response but left the communication display charging
    /// `ell d h(sigma_s) + (n+1) d h(sigma_m)` — the *previous* layout.
    /// Its printed `|pi_OOM|` column reproduces that stale formula to the
    /// digit, so the published sizes under-count by 0.4–0.6 KB per
    /// profile.  [`Self::proof_size_oom_kb_paper`] evaluates the stale
    /// form; both are pinned by tests.
    pub fn proof_size_oom_kb(&self) -> f64 {
        (self.proof_size_oom_common_bits()
            + (self.s_dim() * self.d) as f64 * Self::h(self.sigma_s())
            + (self.m_dim() * self.d) as f64 * Self::h(self.sigma_m()))
            / 8192.0
    }

    /// The stale display formula, kept so the gap can be pinned.
    pub fn proof_size_oom_kb_paper(&self) -> f64 {
        (self.proof_size_oom_common_bits()
            + (self.ell * self.d) as f64 * Self::h(self.sigma_s())
            + ((self.n + 1) * self.d) as f64 * Self::h(self.sigma_m()))
            / 8192.0
    }

    /// `|pi_RiVeR| = |pi_OOM| + |pi_ex|`, with the paper's fixed `|pi_ex|`.
    pub fn proof_size_total_kb(&self) -> f64 {
        self.proof_size_oom_kb() + Self::EXACT_PROOF_KB
    }

    /// [`Self::proof_size_oom_kb_paper`] plus the paper's fixed `|pi_ex|`.
    pub fn proof_size_total_kb_paper(&self) -> f64 {
        self.proof_size_oom_kb_paper() + Self::EXACT_PROOF_KB
    }

    /// `log2 |C^d_{w,gamma}| = log2 C(d,w) + w log2(2 gamma)`.
    pub fn challenge_entropy(&self) -> f64 {
        log2_binomial(self.d, self.w) + self.w as f64 * ((2 * self.gamma) as f64).log2()
    }

    // ---- dimensions ------------------------------------------------------

    /// Length of the OOM opening `r = (r_0, r_1) = ((s, e_key), e_eval)`.
    ///
    /// The concatenation is unchanged — `s`, then `e_key`, then `e_eval` —
    /// but the paper moved the split between them, so `s_dim + m_dim`
    /// partitions it differently.
    pub const fn r_dim(&self) -> usize {
        self.ell + self.n + 1
    }

    /// Length of `r_0 = (s, e_key)`, responded to at width `sigma_s`.
    ///
    /// the paper moved `e_key` across the split, from the `sigma_m`
    /// block into this one; it was `ell` alone before.
    pub const fn s_dim(&self) -> usize {
        self.ell + self.n
    }

    /// Length of `r_1 = e_eval`, responded to at `sigma_m`.
    ///
    /// One ring element; `n + 1` before it.
    pub const fn m_dim(&self) -> usize {
        1
    }

    /// Length of each derived vector `c_i = (q_0 t_i, q_0 v)`.
    pub const fn c_dim(&self) -> usize {
        self.n + 1
    }

    pub const fn gprime_cols(&self) -> usize {
        self.k_hat + 2 * self.N
    }
}

impl RiVeRParams {
    // ---- consistency -----------------------------------------------------

    /// Violated parameter conditions; empty means the profile is
    /// supported.  Conditions marked "security" are skipped for
    /// `insecure_toy` profiles.
    ///
    /// This *is* `BoundGen`'s abort: [`crate::river::RiVeR::setup`] fails
    /// on a non-empty result rather than proceeding on an unsupported
    /// profile.
    ///
    /// **Total**: it never panics, for any field values at all.  Two
    /// passes in dependency order — [`Self::check_domains`] validates the
    /// profile's own literals and returns early if any fail, because
    /// every derived quantity below assumes them.
    pub fn check(&self) -> Vec<String> {
        let domains = self.check_domains();
        if !domains.is_empty() {
            return domains;
        }
        self.conditions()
    }

    /// The structural and security conditions.  Assumes
    /// [`Self::check_domains`] passed.
    fn conditions(&self) -> Vec<String> {
        let mut e = Vec::new();

        // Primality is not implied by the congruence, and the congruence
        // is only the visible half: 17592186043869 is 5 mod 8 and
        // composite, and without this `Setup` would accept it.  The
        // two-factor splitting argument, and every invertibility claim
        // resting on it, needs `X^d + 1` over a *field*.
        for (label, value) in [("p", self.p), ("q_0", self.q0), ("q_hat", self.q_hat)] {
            if !is_prime(value) {
                e.push(format!("{label} = {value} is not prime"));
            }
        }
        for (label, value) in [("p", self.p), ("q_0", self.q0), ("q_hat", self.q_hat)] {
            if value % 8 != 5 {
                e.push(format!("{label} = {value} is not 5 mod 8"));
            }
        }
        if gcd(self.p, self.q0) != 1 {
            e.push("gcd(p, q_0) != 1, CRT does not apply".into());
        }

        // Correctness-critical: no wraparound in either response block.
        // `q > max{12 sigma_s, 12 sigma_m}`.
        if (self.q() as f64) <= 12.0 * self.sigma_s() {
            e.push("q <= 12 sigma_s (short response may wrap mod q)".into());
        }
        if (self.q() as f64) <= 12.0 * self.sigma_m() {
            e.push("q <= 12 sigma_m (error response may wrap mod q)".into());
        }

        // The product-check thresholds have to admit a nonzero `g`.  An
        // arbitrarily small `tau` is inside the domain — positive, finite,
        // exactly representable — and yields a bound under 1, which only
        // an all-zero `g` can satisfy.  That is a condition rather than a
        // domain rule: it is about what the bound *does*.
        for (label, bound) in [("B_g0", self.B_g0()), ("B_g1", self.B_g1())] {
            if bound < Rat::from_u128(1) {
                e.push(format!(
                    "{label} = {bound} < 1: no nonzero g can pass the product check"
                ));
            }
        }

        // BoundGen's own abort: the compression margin must leave s_c bits.
        if self.K_a < self.K_a_boundgen() {
            e.push(format!(
                "K_a = {} < {} (BoundGen)",
                self.K_a,
                self.K_a_boundgen()
            ));
        }

        // Selector modulus condition.
        let a = 2.0 * (2.0 * self.gamma as f64 + 12.0 * self.sigma_a()).powi(2);
        let b = 2.0 * (self.N * self.N) as f64;
        let c = (2f64).powi(self.K_a as i32 + 1);
        let need = a.max(b).max(c);
        if (self.q_hat as f64) <= need {
            e.push(format!("q_hat <= {need:.4e} (hat-q condition)"));
        }

        if !self.insecure_toy {
            if (self.q_hat as f64) <= self.beta_sel_inf() {
                e.push("q_hat <= ||beta_sel||_inf (security)".into());
            }
            if (self.q() as f64) <= self.beta_sis() {
                e.push("q <= beta_SIS (security)".into());
            }
            if (self.q() as f64) <= self.beta_sis_2() {
                e.push("q <= beta_SIS,2 (security)".into());
            }
            if self.challenge_entropy() < 128.0 {
                e.push("challenge space below 128 bits (security)".into());
            }
        }
        e
    }

    /// The domain of every raw field, checked without touching a derived
    /// quantity.  Run first by [`Self::check`]; separate so a caller can
    /// ask the cheap question on its own.
    ///
    /// Nothing here is a *security* condition — those are in `check`.
    /// These are the conditions under which the security conditions can
    /// be evaluated at all.  Without this pass `check` was neither total
    /// nor fail-closed: `d = 0` panicked inside `K_a_boundgen`, while
    /// `beta = 0`, `N = 0` and `phi_a = 0` all returned "no errors".
    pub fn check_domains(&self) -> Vec<String> {
        let mut e = Vec::new();
        {
            let mut positive = |label: &str, value: usize| {
                if value == 0 {
                    e.push(format!("{label} = 0"));
                }
            };
            positive("d", self.d);
            positive("n", self.n);
            positive("ell", self.ell);
            positive("n_hat", self.n_hat);
            positive("k_hat", self.k_hat);
            positive("N", self.N);
            positive("w", self.w);
            positive("r_prime", self.r_prime);
        }

        // `d - 1` underflows at `d = 0`, so the power-of-two test needs
        // the guard above to have run.
        if self.d != 0 && self.d & (self.d - 1) != 0 {
            e.push("d is not a power of two".into());
        }
        // `log2_binomial(d, w)` computes `d - w`.
        if self.w > self.d {
            e.push(format!("w = {} exceeds d = {}", self.w, self.d));
        }
        // `N - 1` is a row count in the codec; `N = 1` leaves no ring.
        if self.N < 2 {
            e.push(format!("N = {} leaves no ring to hide in", self.N));
        }
        for (label, value) in [("p", self.p), ("q_0", self.q0), ("q_hat", self.q_hat)] {
            if value < 2 {
                e.push(format!("{label} = {value} is not a modulus"));
            }
        }
        if self.gamma == 0 {
            e.push("gamma = 0 (empty challenge space)".into());
        }
        for (label, value) in [
            ("phi_a", self.phi_a),
            ("phi_s", self.phi_s),
            ("phi_m", self.phi_m),
            ("phi_b", self.phi_b),
        ] {
            if value == 0 {
                e.push(format!("{label} = 0 (zero-width mask)"));
            }
        }
        for (label, value) in [("tau_g0", self.tau_g0), ("tau_g1", self.tau_g1)] {
            if value.num() == 0 {
                e.push(format!("{label} = 0 is not a positive factor"));
            }
        }
        if !self.epsilon_g_u.is_finite() || !(0.0..1.0).contains(&self.epsilon_g_u) {
            e.push(format!(
                "epsilon_g_u = {} is outside [0, 1)",
                self.epsilon_g_u
            ));
        }
        // `beta = 0` is an all-zero secret-key distribution.
        if self.beta == 0 {
            e.push("beta = 0 (the secret key would be identically zero)".into());
        }
        if self.max_attempts == 0 {
            e.push("max_attempts = 0 (Eval could never succeed)".into());
        }
        if self.lam == 0 {
            e.push("lam = 0".into());
        }
        // `1 << (K_a - 1)` in `T_cmp`, and `K_a > K_b` is what the
        // compression margin means.
        if self.K_a == 0 || self.K_b == 0 {
            e.push("K_a or K_b is 0".into());
        } else if self.K_a <= self.K_b || self.K_a >= 63 {
            e.push(format!(
                "K_a = {} outside (K_b, 63) with K_b = {}",
                self.K_a, self.K_b
            ));
        }

        // Below this line the checks *evaluate* derived quantities, so
        // nothing above may still be broken.
        if !e.is_empty() {
            return e;
        }

        // `q` gates everything after it: `B_e()` is `q_0 / 2` but
        // `Ring::new` is built over `q`, so a `q` that does not exist
        // cannot be reported by a check that computes one.
        let Some(q) = self.checked_q() else {
            e.push("q_0 · p overflows u64".into());
            return e;
        };

        // `Ring::new` builds a Barrett reduction over a `d(q-1)^2`
        // accumulator, so `q < 2^62` is the condition — a modulus that
        // merely fits `u64` is not enough.
        for (label, value) in [("q", q), ("p", self.p), ("q_hat", self.q_hat)] {
            if value >= 1u64 << 62 {
                e.push(format!("{label} = {value} is at or above 2^62"));
            }
        }
        if !e.is_empty() {
            return e;
        }

        // `sigma_a` and both product-check thresholds are exact only when
        // `2w` is a perfect square.  Dropping to a float `B_a` in a
        // wire-visible place is worse than refusing the profile.
        if self.B_a_exact().is_none() {
            // Report `w`, not `2w`.  The diagnostic used to compute the
            // product it was about to name, unchecked — and the branch it
            // sits in is entered *because* something about `w` is out of
            // range, so `2 * self.w` is exactly the case where it
            // overflows.  `d = w = 1 << 63` clears every preceding shape
            // check and then panicked here, in the function whose whole
            // job is to be total.  The reason is stated instead of the
            // product, which is a better message anyway.
            e.push(format!(
                "w = {}: either 2w is not a perfect square or it leaves u64, \
                 so B_a = gamma sqrt(2w) is not exact",
                self.w
            ));
        }
        if self.checked_k_a_boundgen().is_none() {
            e.push("K_a_boundgen overflows".into());
        }
        if self.N.checked_mul(self.N).is_none() {
            e.push("N^2 overflows usize".into());
        }
        // Every remaining derived shape, in one place — see
        // [`Self::checked_shapes`] for why this is not a per-site list.
        if let Err(msg) = self.checked_shapes() {
            e.push(msg);
        }
        // `Ring::new` lifts a schoolbook accumulator by the smallest
        // multiple of `q` at or above `d(q-1)^2`, which has to fit a
        // *positive* `i128`.  `q < 2^62` above is necessary and not
        // sufficient: at `d = 32` it still admits `2^129`.
        for (label, modulus) in [("q", q), ("p", self.p), ("q_hat", self.q_hat)] {
            if crate::ring::checked_wrap_bias(modulus, self.d).is_none() {
                e.push(format!("d·({label}-1)^2 does not fit a positive i128"));
            }
        }
        e
    }

    /// Every derived shape and scale product, evaluated with checked
    /// arithmetic; `Err` names the first one that leaves its type.
    ///
    /// This is the class fix for a bug that kept reappearing one site at
    /// a time.  Adding a mutation per site only ever closes the site;
    /// every product and sum the accessors above perform is listed here
    /// instead, so the way to reopen the hole is to write a new accessor
    /// without extending this list.
    pub fn checked_shapes(&self) -> Result<(), String> {
        fn over(what: &'static str) -> String {
            format!("{what} overflows")
        }

        // `cal_B`, `B_s` — u64 scale and usize shapes
        let wgb = (self.w as u64)
            .checked_mul(self.gamma)
            .and_then(|v| v.checked_mul(self.beta))
            .ok_or_else(|| over("w · gamma · beta"))?;
        self.d
            .checked_mul(self.k_hat)
            .ok_or_else(|| over("d · k_hat"))?;
        self.d
            .checked_mul(self.ell)
            .ok_or_else(|| over("d · ell"))?;
        // `eta_m` — `w gamma B_e`
        (self.w as u64)
            .checked_mul(self.gamma)
            .and_then(|v| v.checked_mul(self.B_e()))
            .ok_or_else(|| over("w · gamma · B_e"))?;
        // `B_a`
        self.w.checked_mul(2).ok_or_else(|| over("2w"))?;
        // `sigma_a` and, through it, both product-check thresholds
        let sigma_a = self
            .B_a_exact()
            .and_then(|b| self.phi_a.checked_mul(b))
            .ok_or_else(|| over("phi_a · B_a"))?;
        // `B_g0` — `N >= 2` is checked separately, so the saturating form
        // keeps this safe if it has not run yet.
        self.d
            .checked_mul(self.N.saturating_sub(1))
            .ok_or_else(|| over("d · (N - 1)"))?;
        // Every exact accept/reject bound, through the fallible siblings
        // rather than through a hand-copy of their arithmetic.
        //
        // The three `6 phi ...` products used to be checked *from the
        // second operation*: `(6 * self.phi_b).checked_mul(wgb)` computes
        // `6 * phi_b` unchecked first, which panics in debug and wraps in
        // release for a large `phi_b` — inside the function whose job is
        // to decide whether such a profile is admissible.  The bounds are
        // `u128` and grow faster still, so nothing short of evaluating
        // them checked settles it.
        let _ = (sigma_a, wgb);
        for (label, bound) in [
            ("f1 bound", self.try_f1_inf_bound_sq()),
            ("zb bound", self.try_zb_inf_bound_sq()),
            ("zs bound", self.try_zs_inf_bound_sq()),
            ("zm bound", self.try_zm_inf_bound_sq()),
            ("Euclidean bound", self.try_z_l2_bound_sq()),
            ("B_g0", self.try_b_g0()),
            ("B_g1", self.try_b_g1()),
        ] {
            bound.ok_or_else(|| format!("{label} overflows"))?;
        }
        // `r_dim`, and `z_l2_bound_sq`'s `d(n+1)`
        let r_dim = self
            .ell
            .checked_add(self.n)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| over("ell + n + 1"))?;
        self.d
            .checked_mul(r_dim)
            .ok_or_else(|| over("d · (ell + n + 1)"))?;
        // `c_dim` / `m_dim`
        self.n.checked_add(1).ok_or_else(|| over("n + 1"))?;
        // `gprime_cols`
        self.N
            .checked_mul(2)
            .and_then(|v| self.k_hat.checked_add(v))
            .ok_or_else(|| over("k_hat + 2N"))?;
        // `beta_sis_embedded` — `(n + r') B_e^2` over u64
        let b_e = self.B_e();
        self.n
            .checked_add(self.r_prime)
            .and_then(|v| (v as u64).checked_mul(b_e.checked_mul(b_e)?))
            .ok_or_else(|| over("(n + r') · B_e^2"))?;
        Ok(())
    }

    /// `K_a_boundgen`, or `None` if any step leaves its type.
    ///
    /// Both the product `w gamma n_hat d` **and** the sum
    /// `K_b + ceil_log2(v) + s_c`: the latter is `u32` arithmetic, so an
    /// `s_cmp` of `u32::MAX` overflows an addition that the product check
    /// alone does not see.
    pub fn checked_k_a_boundgen(&self) -> Option<u32> {
        let a = (self.w as u64).checked_mul(self.gamma)?;
        let b = a.checked_mul(self.n_hat as u64)?;
        let v = b.checked_mul(self.d as u64)?;
        if v == 0 {
            return None;
        }
        let s = self.K_b.checked_add(ceil_log2(v))?;
        s.checked_add(self.s_cmp)
    }

    /// One-line-per-item human summary, used by the CLI and tests.
    pub fn summary(&self) -> Vec<(&'static str, String)> {
        vec![
            ("name", self.name.to_string()),
            ("N", self.N.to_string()),
            (
                "(phi_a, phi_s)",
                format!("({}, {})", self.phi_a, self.phi_s),
            ),
            ("(n, ell)", format!("({}, {})", self.n, self.ell)),
            (
                "(log2 p, log2 q)",
                format!(
                    "({:.2}, {:.2})",
                    (self.p as f64).log2(),
                    (self.q() as f64).log2()
                ),
            ),
            (
                "(n_hat, k_hat, log2 q_hat)",
                format!(
                    "({}, {}, {:.2})",
                    self.n_hat,
                    self.k_hat,
                    (self.q_hat as f64).log2()
                ),
            ),
            ("B", format!("{:.4}", self.cal_B())),
            ("B_s", format!("{:.4}", self.B_s())),
            ("eta_m", format!("{:.4}", self.eta_m())),
            (
                "(B_g0, B_g1)",
                format!(
                    "({:.6e}, {:.6e})",
                    self.B_g0().to_f64(),
                    self.B_g1().to_f64()
                ),
            ),
            ("beta_SIS,1", format!("{:.6e}", self.beta_sis_1())),
            ("beta_SIS,2", format!("{:.6e}", self.beta_sis_2())),
            ("beta_SIS", format!("{:.6e}", self.beta_sis())),
            ("beta_sel_inf", format!("{:.6e}", self.beta_sel_inf())),
            ("mu_river", format!("{:.4}", self.mu_river())),
            ("|pi_OOM| KB", format!("{:.2}", self.proof_size_oom_kb())),
            ("total KB", format!("{:.2}", self.proof_size_total_kb())),
        ]
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn ceil_log2(v: u64) -> u32 {
    debug_assert!(v > 0);
    if v <= 1 {
        0
    } else {
        64 - (v - 1).leading_zeros()
    }
}

/// `log2 C(n, k)`, exactly for the small shapes used here.
fn log2_binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    let mut acc = 0.0f64;
    for i in 0..k {
        acc += ((n - i) as f64).log2() - ((i + 1) as f64).log2();
    }
    acc
}

// ---- published profiles --------------------------------------------------

/// One row of the paper's parameter table (
/// `tab:river-final-all-params`).  The argument list is wide because the
/// table is: keeping the call sites in column order makes a transcription
/// error visible, which a builder would hide.
#[allow(clippy::too_many_arguments)]
const fn published(
    name: &'static str,
    N: usize,
    phi: (u64, u64),
    n: usize,
    ell: usize,
    p: u64,
    n_hat: usize,
    k_hat: usize,
    q_hat: u64,
    tau: (u128, u128),
    repetition: f64,
) -> RiVeRParams {
    RiVeRParams {
        name,
        d: 32,
        q0: 61,
        p,
        q_hat,
        n,
        ell,
        n_hat,
        k_hat,
        N,
        w: 32,
        gamma: 16,
        beta: 1,
        phi_a: phi.0,
        phi_s: phi.1,
        phi_m: 32,
        phi_b: 2,
        tau_g0: Rat::new(tau.0, 100),
        tau_g1: Rat::new(tau.1, 100),
        epsilon_g_u: repetition,
        K_b: 5,
        K_a: 28,
        s_cmp: 3,
        r_prime: 1,
        lam: 128,
        max_attempts: 1000,
        insecure_toy: false,
    }
}

pub const RIVER_N8: RiVeRParams = published(
    "RiVeR-N8",
    8,
    (32, 26),
    44,
    54,
    P_44,
    42,
    46,
    QHAT_44,
    (314, 268),
    0.007956,
);
pub const RIVER_N16: RiVeRParams = published(
    "RiVeR-N16",
    16,
    (40, 22),
    41,
    59,
    P_48,
    43,
    49,
    QHAT_46,
    (309, 308),
    0.007793,
);
pub const RIVER_N64: RiVeRParams = published(
    "RiVeR-N64",
    64,
    (34, 24),
    44,
    54,
    P_44,
    50,
    51,
    QHAT_48,
    (305, 333),
    0.009060,
);
pub const RIVER_N128: RiVeRParams = published(
    "RiVeR-N128",
    128,
    (24, 34),
    45,
    54,
    P_44,
    50,
    51,
    QHAT_48,
    (309, 358),
    0.007850,
);
pub const RIVER_N256: RiVeRParams = published(
    "RiVeR-N256",
    256,
    (22, 40),
    42,
    59,
    P_48,
    49,
    52,
    QHAT_49,
    (306, 384),
    0.008599,
);

/// `(tau_g0, tau_g1)` exactly as the table displays them, to one decimal
/// place.  Retained so a test can pin which entries the rounded values
/// reproduce — one decimal fails to reproduce the table's own `B_g0`
/// column at `N = 256`, which is why [`RIVER_N8`] and friends carry the
/// two-decimal values instead.
pub const TAU_DISPLAYED: [(usize, (u128, u128)); 5] = [
    (8, (31, 27)),
    (16, (31, 31)),
    (64, (31, 33)),
    (128, (31, 36)),
    (256, (31, 38)),
];

/// Structurally identical to the published profiles (same `d`, `q_0`,
/// `w`, `gamma`, `beta`, radix encoding, and the same split response
/// widths) but with tiny module ranks so the whole pipeline runs in
/// seconds.  Deliberately insecure: it does not meet the M-SIS / A-MSIS
/// modulus conditions.
pub const RIVER_TOY: RiVeRParams = RiVeRParams {
    name: "RiVeR-TOY",
    d: 32,
    q0: 61,
    p: 16_777_213,
    q_hat: 1_099_511_627_581,
    n: 4,
    ell: 6,
    n_hat: 4,
    k_hat: 4,
    N: 4,
    w: 32,
    gamma: 16,
    beta: 1,
    phi_a: 32,
    phi_s: 26,
    phi_m: 32,
    phi_b: 2,
    tau_g0: Rat::new(314, 100),
    tau_g1: Rat::new(268, 100),
    epsilon_g_u: 0.01,
    K_b: 5,
    K_a: 28,
    s_cmp: 3,
    r_prime: 1,
    lam: 128,
    max_attempts: 1000,
    insecure_toy: true,
};

/// The five published profiles plus the toy one.
pub const PROFILES: [RiVeRParams; 6] = [
    RIVER_N8, RIVER_N16, RIVER_N64, RIVER_N128, RIVER_N256, RIVER_TOY,
];

/// The five published profiles.
pub const PUBLISHED: [RiVeRParams; 5] = [RIVER_N8, RIVER_N16, RIVER_N64, RIVER_N128, RIVER_N256];

pub const DEFAULT_PARAMS: RiVeRParams = RIVER_N8;

/// Look up a profile by name.
pub fn get(name: &str) -> Option<RiVeRParams> {
    PROFILES.into_iter().find(|p| p.name == name)
}

// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn moduli_rederive() {
        assert_eq!(verify_moduli(), Vec::<String>::new());
    }

    #[test]
    fn moduli_are_prime_and_split_in_two() {
        for p in PROFILES {
            for (label, v) in [("p", p.p), ("q_0", p.q0), ("q_hat", p.q_hat)] {
                assert!(is_prime(v), "{} {label} = {v} is not prime", p.name);
                assert_eq!(v % 8, 5, "{} {label} = {v} is not 5 mod 8", p.name);
            }
        }
    }

    /// The five rows of `tab:river-final-all-params`, the paper.
    #[test]
    fn profile_inputs_match_the_table() {
        // (N, phi_a, phi_s, n, ell, log2 p, n_hat, k_hat, log2 q_hat)
        let table = [
            (
                8usize, 32u64, 26u64, 44usize, 54usize, 44u32, 42usize, 46usize, 44u32,
            ),
            (16, 40, 22, 41, 59, 48, 43, 49, 46),
            (64, 34, 24, 44, 54, 44, 50, 51, 48),
            (128, 24, 34, 45, 54, 44, 50, 51, 48),
            (256, 22, 40, 42, 59, 48, 49, 52, 49),
        ];
        for (row, p) in table.into_iter().zip(PUBLISHED) {
            let (N, phi_a, phi_s, n, ell, lp, n_hat, k_hat, lqh) = row;
            assert_eq!(p.N, N, "{}", p.name);
            assert_eq!((p.phi_a, p.phi_s), (phi_a, phi_s), "{}", p.name);
            assert_eq!((p.n, p.ell), (n, ell), "{}", p.name);
            assert_eq!((p.n_hat, p.k_hat), (n_hat, k_hat), "{}", p.name);
            assert_eq!(64 - p.p.leading_zeros(), lp, "{} log2 p", p.name);
            assert_eq!(64 - p.q_hat.leading_zeros(), lqh, "{} log2 q_hat", p.name);
        }
    }

    /// the paper prints the concrete moduli outright.
    ///
    /// Before it the paper gave only bit lengths and this tree derived all
    /// three by one rule — largest prime below `2^bits` that is 5 mod 8.
    /// That rule is right for `p` and wrong for `q_hat`, which the paper
    /// takes as the smallest prime *above* `2^{bits-1}`, roughly half the
    /// value.  Since `q_hat` enters `b_B` and hence the wire, the
    /// difference was not cosmetic.
    #[test]
    fn the_concrete_moduli_match_the_published_table() {
        let table = [
            (
                17_592_186_043_877u64,
                1_073_123_348_676_497u64,
                8_796_093_022_237u64,
            ),
            (
                281_474_976_710_597,
                17_169_973_579_346_417,
                35_184_372_088_997,
            ),
            (
                17_592_186_043_877,
                1_073_123_348_676_497,
                140_737_488_355_333,
            ),
            (
                17_592_186_043_877,
                1_073_123_348_676_497,
                140_737_488_355_333,
            ),
            (
                281_474_976_710_597,
                17_169_973_579_346_417,
                281_474_976_710_677,
            ),
        ];
        for ((p_, q_, qh), prof) in table.into_iter().zip(PUBLISHED) {
            assert_eq!(prof.p, p_, "{} p", prof.name);
            assert_eq!(prof.q(), q_, "{} q", prof.name);
            assert_eq!(prof.q_hat, qh, "{} q_hat", prof.name);
            assert_eq!(prof.q(), 61 * prof.p, "{}", prof.name);
        }
        assert!(verify_moduli().is_empty(), "{:?}", verify_moduli());
    }

    /// The paper's shared constants: `phi_m`, `phi_b` and
    /// `r' = 1` are the same at every profile.
    #[test]
    fn revision_constants_are_shared_by_every_profile() {
        for p in PROFILES {
            assert_eq!(p.phi_m, 32, "{} phi_m", p.name);
            assert_eq!(p.phi_b, 2, "{} phi_b", p.name);
            assert_eq!(p.r_prime, 1, "{} r'", p.name);
            assert_eq!((p.d, p.q0, p.w, p.gamma, p.beta), (32, 61, 32, 16, 1));
            assert_eq!((p.K_b, p.K_a, p.s_cmp), (5, 28, 3), "{}", p.name);
            assert_eq!(p.B_e(), 30, "{} B_e", p.name);
        }
    }

    /// **closed in the paper**: `BoundGen` and all three OOM
    /// algorithms now return and parse `(phi_a, phi_b, phi_s, phi_m)`.
    ///
    /// Nothing here was ever positional, so no code path could pick
    /// wrong — this records the size of the gap the naming closed, and
    /// keeps it recorded now that the paper has one order.
    #[test]
    fn named_fields_make_the_boundgen_order_unreachable() {
        for p in PUBLISHED {
            assert!(p.phi_s >= 22 && p.phi_s <= 40, "{}", p.name);
            assert_eq!(p.phi_b, 2, "{}", p.name);
            assert!(p.phi_s >= 11 * p.phi_b, "{} phi_s vs phi_b", p.name);
            // Swapping them would sample `z_s` at width 2 B_s, which no
            // accepted response could reach.
            assert!(p.sigma_s() > 10.0 * p.phi_b as f64 * p.B_s());
        }
    }

    /// `BoundGen`'s scales, against the Python reference.
    #[test]
    fn boundgen_reproduces_the_reference_scales() {
        // (name, cal_B, B_s, sigma_a, sigma_b, sigma_s)
        let table = [
            (
                "RiVeR-N8",
                19643.725919488897,
                860160.0,
                4096.0,
                39287.451838977795,
                22364160.0,
            ),
            (
                "RiVeR-N16",
                20274.16563018069,
                868892.8127220296,
                5120.0,
                40548.33126036138,
                19115641.879884653,
            ),
            (
                "RiVeR-N64",
                20683.786113765535,
                860160.0,
                4352.0,
                41367.57222753107,
                20643840.0,
            ),
            (
                "RiVeR-N128",
                20683.786113765535,
                864537.4328506545,
                3072.0,
                41367.57222753107,
                29394272.716922253,
            ),
            (
                "RiVeR-N256",
                20885.583927676045,
                873226.4695942284,
                2816.0,
                41771.16785535209,
                34929058.78376914,
            ),
            (
                "RiVeR-TOY",
                5792.618751480198,
                274768.0330751742,
                4096.0,
                11585.237502960395,
                7143968.859954529,
            ),
        ];
        for (name, cal_b, b_s, sa, sb, ss) in table {
            let p = get(name).unwrap();
            // Bit-exact: these widths are pinned to rationals downstream,
            // and one ulp moves every mask.
            assert_eq!(p.cal_B(), cal_b, "{name} B");
            assert_eq!(p.B_s(), b_s, "{name} B_s");
            assert_eq!(p.sigma_a(), sa, "{name} sigma_a");
            assert_eq!(p.sigma_b(), sb, "{name} sigma_b");
            assert_eq!(p.sigma_s(), ss, "{name} sigma_s");
            // `eta_m = w gamma B_e sqrt(d)` does not depend on the profile.
            assert_eq!(p.eta_m(), 86889.28127220296, "{name} eta_m");
            assert_eq!(p.sigma_m(), 2780457.000710495, "{name} sigma_m");
        }
    }

    /// Every exact accept/reject bound, against the Python `Fraction`s.
    #[test]
    fn exact_bounds_match_the_reference_rationals() {
        // (name, B_g0, B_g1, f1^2, zb^2, zs^2, zm^2, l2^2)
        type Row = (
            &'static str,
            (u128, u128),
            (u128, u128),
            u128,
            u128,
            u128,
            u128,
            (u128, u128),
        );
        let table: [Row; 6] = [
            (
                "RiVeR-N8",
                (295010566144, 75),
                (17985175552, 25),
                603979776,
                55566139392,
                18005603490201600,
                278313880780800,
                (2258979143578288128, 1),
            ),
            (
                "RiVeR-N16",
                (12960399360, 1),
                (1291845632, 1),
                943718400,
                59190018048,
                13154679521280000,
                278313880780800,
                (1684155220491239424, 1),
            ),
            (
                "RiVeR-N64",
                (194096136192, 5),
                (25227952128, 25),
                681836544,
                61605937152,
                15342052678041600,
                278313880780800,
                (1924863329700937728, 1),
            ),
            (
                "RiVeR-N128",
                (987582431232, 25),
                (13514047488, 25),
                339738624,
                61605937152,
                31104837668044800,
                278313880780800,
                (3941961271062036480, 1),
            ),
            (
                "RiVeR-N256",
                (330008887296, 5),
                (12180258816, 25),
                285474816,
                62813896704,
                43921409310720000,
                278313880780800,
                (5678516037457281024, 1),
            ),
            (
                "RiVeR-TOY",
                (42144366592, 25),
                (17985175552, 25),
                603979776,
                4831838208,
                1837306478592000,
                278313880780800,
                (23873764693377024, 1),
            ),
        ];
        for (name, g0, g1, f1, zb, zs, zm, l2) in table {
            let p = get(name).unwrap();
            assert_eq!(p.B_g0(), Rat::new(g0.0, g0.1), "{name} B_g0");
            assert_eq!(p.B_g1(), Rat::new(g1.0, g1.1), "{name} B_g1");
            assert_eq!(p.f1_inf_bound_sq(), Rat::from_u128(f1), "{name} f1");
            assert_eq!(p.zb_inf_bound_sq(), Rat::from_u128(zb), "{name} zb");
            assert_eq!(p.zs_inf_bound_sq(), Rat::from_u128(zs), "{name} zs");
            assert_eq!(p.zm_inf_bound_sq(), Rat::from_u128(zm), "{name} zm");
            assert_eq!(p.z_l2_bound_sq(), Rat::new(l2.0, l2.1), "{name} l2");
        }
    }

    /// Each squared bound is the square of the float bound it replaces,
    /// so the exact test cannot be a *different* condition.
    #[test]
    fn squared_bounds_agree_with_the_float_forms() {
        for p in PROFILES {
            for (label, sq, flt) in [
                ("f1", p.f1_inf_bound_sq(), p.f1_inf_bound()),
                ("zb", p.zb_inf_bound_sq(), p.zb_inf_bound()),
                ("zs", p.zs_inf_bound_sq(), p.zs_inf_bound()),
                ("zm", p.zm_inf_bound_sq(), p.zm_inf_bound()),
                ("z_l2", p.z_l2_bound_sq(), p.z_l2_bound()),
            ] {
                assert!(
                    close(sq.to_f64().sqrt(), flt, 1e-12),
                    "{} {label}: sqrt({}) = {} vs {}",
                    p.name,
                    sq,
                    sq.to_f64().sqrt(),
                    flt
                );
            }
        }
    }

    /// `2w = 64` at every profile, so `B_a = gamma sqrt(2w) = 128` is an
    /// integer and `sigma_a = phi_a B_a` is exact.
    #[test]
    fn b_a_is_exact_at_every_profile() {
        for p in PROFILES {
            assert_eq!(p.B_a_exact(), Some(128), "{}", p.name);
            assert_eq!(p.sigma_a_exact(), p.phi_a * 128, "{}", p.name);
        }
    }

    #[test]
    fn challenge_space_is_160_bits() {
        for p in PROFILES {
            // `w == d == 32` and each nonzero coefficient has magnitude in
            // [1, gamma], so the count is `(2 gamma)^d`.
            assert_eq!(p.challenge_entropy(), 160.0, "{}", p.name);
        }
    }

    #[test]
    fn boundgen_ka_bound_matches_the_published_profiles() {
        for p in PUBLISHED {
            assert_eq!(p.K_a_boundgen(), 28, "{}", p.name);
            assert_eq!(p.K_a, p.K_a_boundgen(), "{}", p.name);
        }
        assert_eq!(RIVER_TOY.K_a_boundgen(), 24);
    }

    /// `b_B` and the six-term communication formula reproduce the
    /// table's `|pi_OOM|` and total columns to their printed rounding —
    /// through the paper's own display formula.
    ///
    /// **DISCREPANCY.**  the paper regrouped the OOM response but
    /// left the communication display charging
    /// `ell d h(sigma_s) + (n+1) d h(sigma_m)`, which is a different
    /// split.  Its printed column reproduces that stale formula to the
    /// digit, so the published sizes under-count by 0.4–0.6 KB per
    /// profile.  Both are pinned: the paper's against its own column, and
    /// the transmitted layout against what this tree reports.
    #[test]
    fn size_columns_reproduce() {
        // (name, b_B, printed OOM KB, printed total KB, measured OOM, total)
        let table = [
            ("RiVeR-N8", 39u32, 19.6, 33.1, 20.1, 33.6),
            ("RiVeR-N16", 41, 21.0, 34.5, 21.4, 34.9),
            ("RiVeR-N64", 43, 25.0, 38.5, 25.5, 39.0),
            ("RiVeR-N128", 43, 28.5, 42.0, 29.1, 42.6),
            ("RiVeR-N256", 44, 35.6, 49.1, 36.2, 49.7),
        ];
        for (name, b_b, oom, total, measured_oom, measured_total) in table {
            let p = get(name).unwrap();
            assert_eq!(p.b_B(), b_b, "{name} b_B");
            assert!(
                (p.proof_size_oom_kb_paper() - oom).abs() < 0.05,
                "{name}: paper |pi_OOM| = {} vs printed {oom}",
                p.proof_size_oom_kb_paper()
            );
            assert!(
                (p.proof_size_total_kb_paper() - total).abs() < 0.05,
                "{name}: paper total = {} vs printed {total}",
                p.proof_size_total_kb_paper()
            );
            assert!(
                (p.proof_size_oom_kb() - measured_oom).abs() < 0.05,
                "{name}: transmitted |pi_OOM| = {}",
                p.proof_size_oom_kb()
            );
            assert!(
                (p.proof_size_total_kb() - measured_total).abs() < 0.05,
                "{name}: transmitted total = {}",
                p.proof_size_total_kb()
            );
            // The gap is exactly the `n` elements moved between widths.
            let gap = p.proof_size_oom_kb() - p.proof_size_oom_kb_paper();
            assert!((0.4..=0.7).contains(&gap), "{name}: gap {gap}");
        }
    }

    /// The table's "mean attempts" column, 8.3 to 8.6.
    #[test]
    fn repeat_bound_reproduces_the_table() {
        let table = [
            ("RiVeR-N8", 8.3),
            ("RiVeR-N16", 8.4),
            ("RiVeR-N64", 8.6),
            ("RiVeR-N128", 8.6),
            ("RiVeR-N256", 8.5),
        ];
        for (name, printed) in table {
            let p = get(name).unwrap();
            let mu = p.mu_river();
            assert!(
                (mu * 10.0).round() / 10.0 == printed,
                "{name}: mu_river = {mu} does not round to {printed}"
            );
        }
    }

    /// The factor 2 in `mu_b` is the half-space rejection, and the
    /// appendix does not charge it twice.
    #[test]
    fn mu_b_carries_the_half_space_factor() {
        for p in PROFILES {
            assert!(p.mu_b() > 2.0, "{}", p.name);
            assert!(
                close(p.mu_b(), 2.0 * (1.0 / 8.0f64).exp(), 1e-12),
                "{}",
                p.name
            );
        }
    }

    /// **closed in the paper**, and pinned as a closure.
    ///
    /// Under the table the appendix's own denominator did not
    /// reproduce its printed "Repeat bound" column, and this tree carried
    /// a per-profile `c_pub_model` backsolved from that column.  Against
    /// the paper table the components multiply out to the printed
    /// value at every profile, so the backsolve is gone.
    ///
    /// The assertion is on the *printed column*, not on an internal
    /// identity: `mu_river` is defined as the component product, so
    /// comparing it to itself would pass no matter what.
    #[test]
    fn the_repetition_denominator_now_is_the_product_of_its_components() {
        for (p, printed) in PUBLISHED.iter().zip([8.3, 8.4, 8.6, 8.6, 8.5]) {
            let rounded = (p.mu_river() * 10.0).round() / 10.0;
            assert!(
                close(rounded, printed, 1e-12),
                "{}: mu_river {} rounds to {}, paper prints {}",
                p.name,
                p.mu_river(),
                rounded,
                printed
            );
        }
    }

    /// the paper adds `eps_2` and says it is below `2^-150` at every
    /// final profile.  It is, by a wide margin — so the term is carried
    /// for correctness, not because it moves the estimate.
    #[test]
    fn the_euclidean_restart_term_is_negligible_as_the_paper_claims() {
        for p in PUBLISHED {
            let eps = p.eps_euclidean();
            assert!(eps > 0.0 && eps < (2f64).powi(-150), "{}: {eps}", p.name);
        }
    }

    /// The paper states the first `beta_SIS` term dominates at all five
    /// profiles.  Checked, not assumed.
    #[test]
    fn first_beta_sis_term_dominates_as_the_paper_states() {
        for p in PUBLISHED {
            let first = 4.0 * (p.w as u64 * p.gamma) as f64 * p.beta_sis_1();
            assert!(
                first > p.beta_sis_embedded(),
                "{}: {first} vs {}",
                p.name,
                p.beta_sis_embedded()
            );
            assert_eq!(p.beta_sis(), first, "{}", p.name);
        }
    }

    /// Displayed one-decimal `tau` does not reproduce the table's own
    /// `B_g0` column at `N = 256`; the two-decimal values do.
    ///
    /// **Closed in the paper**, which prints two decimals and adds a
    /// table note saying why ("reported to two decimal places to make the
    /// corresponding bounds reproducible from the table").  All ten values
    /// this tree recovered under the one-decimal table match the
    /// published pairs exactly.  The test is kept, and kept in the
    /// direction that can fail: the one-decimal rendering still does not
    /// reproduce `N = 256`, which is what made the recovery necessary.
    #[test]
    fn tau_one_decimal_fails_to_reproduce_its_own_b_g0_column() {
        // The printed `B_g0` column at N = 256.
        let printed = [("RiVeR-N256", 6.6e10)];
        for (name, want) in printed {
            let p = get(name).unwrap();
            let carried = p.B_g0().to_f64();
            assert!(
                close(carried, want, 0.006),
                "{name}: two-decimal tau gives {carried}, printed {want}"
            );
            // Now the displayed one-decimal value, same formula.
            let (_, (t0, _)) = TAU_DISPLAYED.iter().find(|(n, _)| *n == p.N).unwrap();
            let s = p.sigma_a_exact() as u128;
            let rounded = Rat::new(*t0, 10)
                .mul(Rat::new((p.d * (p.N - 1)) as u128, 3))
                .mul_u128(s * s)
                .to_f64();
            assert!(
                !close(rounded, want, 0.006),
                "{name}: one-decimal tau now reproduces {want} — the \
                 paper may have unrounded the column, so re-derive"
            );
        }
    }

    #[test]
    fn published_profiles_satisfy_all_conditions() {
        for p in PROFILES {
            let errors = p.check();
            assert!(errors.is_empty(), "{}: {errors:?}", p.name);
        }
    }

    /// The toy profile is structurally identical and deliberately fails
    /// the security conditions its flag skips.
    #[test]
    fn toy_profile_is_structurally_identical_but_insecure() {
        let mut secured = RIVER_TOY;
        secured.insecure_toy = false;
        let errors = secured.check();
        assert!(
            errors.iter().any(|e| e.contains("security")),
            "TOY passed the security conditions: {errors:?}"
        );
        for field in [
            RIVER_TOY.d,
            RIVER_TOY.w,
            RIVER_TOY.gamma as usize,
            RIVER_TOY.beta as usize,
        ] {
            let _ = field;
        }
        assert_eq!(RIVER_TOY.q0, RIVER_N8.q0);
        assert_eq!(RIVER_TOY.phi_m, RIVER_N8.phi_m);
        assert_eq!(RIVER_TOY.phi_b, RIVER_N8.phi_b);
        assert_eq!(RIVER_TOY.r_prime, RIVER_N8.r_prime);
    }

    #[test]
    fn dimensions_are_consistent() {
        for p in PROFILES {
            // the paper: `r_0 = (s, e_key)` at sigma_s, `r_1 = e_eval`
            // at sigma_m.  The concatenation is unchanged; the split moved.
            assert_eq!(p.s_dim(), p.ell + p.n, "{}", p.name);
            assert_eq!(p.m_dim(), 1, "{}", p.name);
            assert_eq!(p.s_dim() + p.m_dim(), p.r_dim(), "{}", p.name);
            // `c_i = (q_0 t_i, q_0 v)` is `n + 1` regardless of where the
            // response split falls; it used to equal `m_dim` by accident.
            assert_eq!(p.c_dim(), p.n + 1, "{}", p.name);
            assert_eq!(p.gprime_cols(), p.k_hat + 2 * p.N, "{}", p.name);
        }
    }

    /// `check` is total: no field assignment a caller can make turns it
    /// into a panic.
    #[test]
    fn check_never_panics_on_arithmetic_a_caller_can_provoke() {
        let base = RIVER_TOY;
        let mut probes: Vec<RiVeRParams> = Vec::new();
        macro_rules! probe {
            ($field:ident, $value:expr) => {{
                let mut p = base;
                p.$field = $value;
                probes.push(p);
            }};
        }
        probe!(d, 0);
        probe!(d, usize::MAX);
        probe!(d, 33);
        probe!(n, 0);
        probe!(n, usize::MAX);
        probe!(ell, 0);
        probe!(ell, usize::MAX);
        probe!(n_hat, usize::MAX);
        probe!(k_hat, 0);
        probe!(k_hat, usize::MAX);
        probe!(N, 0);
        probe!(N, 1);
        probe!(N, usize::MAX);
        probe!(w, 0);
        probe!(w, usize::MAX);
        probe!(w, 31);
        probe!(gamma, 0);
        probe!(gamma, u64::MAX);
        probe!(beta, 0);
        probe!(beta, u64::MAX);
        probe!(phi_a, 0);
        probe!(phi_a, u64::MAX);
        probe!(phi_s, 0);
        probe!(phi_s, u64::MAX);
        probe!(phi_m, 0);
        probe!(phi_m, u64::MAX);
        probe!(phi_b, 0);
        probe!(phi_b, u64::MAX);
        probe!(q0, 0);
        probe!(q0, 1);
        probe!(q0, u64::MAX);
        probe!(p, 0);
        probe!(p, u64::MAX);
        probe!(q_hat, 0);
        probe!(q_hat, u64::MAX);
        probe!(K_a, 0);
        probe!(K_a, 63);
        probe!(K_a, u32::MAX);
        probe!(K_b, 0);
        probe!(K_b, 40);
        probe!(s_cmp, u32::MAX);
        probe!(r_prime, 0);
        probe!(r_prime, usize::MAX);
        probe!(lam, 0);
        probe!(max_attempts, 0);
        probe!(tau_g0, Rat::from_u128(0));
        probe!(tau_g1, Rat::from_u128(0));
        // Extreme *nonzero* rationals.  Zero is the easy case — the
        // domain pass names it — and these are the ones that used to
        // reach `B_g0`'s `u128` products and wrap there: a huge numerator
        // overflows the product, a huge denominator overflows it on the
        // other side, and neither is caught by "is it positive?".
        probe!(tau_g0, Rat::new(u128::MAX, 1));
        probe!(tau_g1, Rat::new(u128::MAX, 1));
        probe!(tau_g0, Rat::new(1, u128::MAX));
        probe!(tau_g1, Rat::new(1, u128::MAX));
        probe!(tau_g0, Rat::new(u128::MAX, u128::MAX - 1));
        probe!(tau_g1, Rat::new(u128::MAX - 1, 3));
        probe!(tau_g0, Rat::new(1 << 100, 7));
        // Widths large enough that `6 * phi` overflows before the product
        // it feeds — the shape `checked_shapes` used to check from the
        // second operation rather than the first.
        probe!(phi_a, u64::MAX / 5);
        probe!(phi_s, u64::MAX / 5);
        probe!(phi_m, u64::MAX / 5);
        probe!(phi_b, u64::MAX / 5);
        probe!(phi_b, u64::MAX / 7);
        probe!(gamma, u64::MAX / 3);
        probe!(beta, u64::MAX / 3);
        probe!(epsilon_g_u, f64::NAN);
        probe!(epsilon_g_u, 1.0);
        probe!(epsilon_g_u, -1.0);

        // **Coupled** mutations.  Every probe above moves one field, and
        // the guards each field trips are per-field — so a value that is
        // only out of range *relative to another* slips through them all
        // and reaches the arithmetic.  `d = w = 1 << 63` is the case:
        // `w <= d` holds, `d` is a power of two, both are positive, and
        // then `2 * w` overflows inside a diagnostic.
        macro_rules! coupled {
            ($($field:ident = $value:expr),+ $(,)?) => {{
                let mut p = base;
                $(p.$field = $value;)+
                probes.push(p);
            }};
        }
        coupled!(d = 1 << 63, w = 1 << 63);
        coupled!(d = 1 << 62, w = 1 << 62);
        coupled!(d = usize::MAX / 2 + 1, w = usize::MAX / 2 + 1);
        coupled!(w = 2, gamma = u64::MAX);
        coupled!(w = 1 << 40, gamma = 1 << 40, phi_a = 1 << 40);
        coupled!(ell = usize::MAX / 2, n = usize::MAX / 2);
        coupled!(k_hat = usize::MAX / 2, d = 1 << 40);
        coupled!(N = usize::MAX / 2, d = 1 << 40);
        coupled!(beta = u64::MAX, ell = 1 << 40);
        coupled!(phi_s = u64::MAX / 6, beta = 1 << 30);
        coupled!(phi_m = u64::MAX / 6, gamma = 1 << 30);

        for p in probes {
            let errors = p.check();
            assert!(
                !errors.is_empty(),
                "a mutated profile passed check(): {p:?}"
            );
        }
    }

    /// `Rat::to_f64` is correctly rounded past `2^53`, where
    /// `num as f64 / den as f64` is not.
    #[test]
    fn rat_to_f64_is_correctly_rounded() {
        assert_eq!(Rat::new(1, 2).to_f64(), 0.5);
        assert_eq!(Rat::new(0, 7).to_f64(), 0.0);
        assert_eq!(Rat::from_u128(1u128 << 60).to_f64(), (1u128 << 60) as f64);
        // 2^53 + 1 is not representable; halves of it are the interesting
        // case for a naive double rounding.
        let n = (1u128 << 53) + 1;
        assert_eq!(Rat::new(2 * n + 1, 2).to_f64(), ((2 * n + 1) as f64) / 2.0);
        for (num, den) in [
            (295010566144u128, 75u128),
            (1344209420288, 25),
            (36 * 11414309380617216, 25),
        ] {
            let exact = Rat::new(num, den).to_f64();
            let naive = num as f64 / den as f64;
            assert!(
                (exact - naive).abs() <= 4.0 * f64::EPSILON * exact.abs(),
                "{num}/{den}: {exact} vs {naive}"
            );
        }
    }

    /// `check` catches a thin compression margin, a broken modulus, and
    /// a composite that passes the congruence.
    #[test]
    fn check_is_fail_closed_on_the_conditions_boundgen_states() {
        let mut thin = RIVER_N8;
        thin.K_a = 27;
        assert!(thin.check().iter().any(|e| e.contains("BoundGen")));

        let mut broken = RIVER_N8;
        broken.p = 17_592_186_043_871; // 7 mod 8
        assert!(broken.check().iter().any(|e| e.contains("5 mod 8")));

        let mut composite = RIVER_N8;
        composite.p = 17_592_186_043_869; // 5 mod 8, composite
        assert!(composite.check().iter().any(|e| e.contains("not prime")));

        let mut tiny_tau = RIVER_N8;
        tiny_tau.tau_g0 = Rat::new(1, 1u128 << 60);
        assert!(tiny_tau.check().iter().any(|e| e.contains("B_g0")));
    }
}
