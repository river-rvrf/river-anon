//! XOF-driven samplers — port of `river-py/sample.py`.
//!
//! Everything random in RiVeR comes from a SHAKE-256 stream, so a whole
//! transcript is reproducible from its seeds.  That is what makes
//! `../river-py/vectors.json` meaningful across implementations, and it
//! means every routine here is wire-visible: a different rejection loop,
//! a different byte width, a different tie in a rounding, and the bytes
//! move.
//!
//! Determinism notes, carried over from the reference:
//!
//! * The paper never names an XOF for `Expand` / `SamMat`.  This is
//!   SHAKE-256 in counter mode: the stream is the concatenation of
//!   `SHAKE256(seed || counter_le8)` blocks of [`SHAKE_BLOCK`] bytes.
//! * Absorption is injective — every part carries an 8-byte
//!   little-endian length prefix, so `("ab","c")` and `("a","bc")`
//!   cannot collide.
//! * Acceptance probabilities are compared in fixed-point integer form
//!   against an exactly computed threshold; see [`crate::fixed`].

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use crate::fastexp::ExpCtx;
use crate::fixed::{exp_accept, exp_threshold, Int, Nat};

/// SHAKE-256 rate, in bytes.
pub const SHAKE_BLOCK: usize = 136;

/// Fixed-point width of every acceptance test.  **Wire-visible.**
///
/// `gaussian_int` accepts when `u < floor(2^PROB_BITS · exp(-z²/2σ²))`.
/// Once that floor reaches zero the value can never be accepted, so the
/// effective tail cut is `sqrt(2 · PROB_BITS · ln 2)` whatever
/// [`GAUSSIAN_TAILCUT`] says — at 64 bits that is 9.42σ, which is how a
/// declared 14σ cut once silently behaved as 9.42σ.  192 bits puts the
/// hard zero at 16.3σ.
pub const PROB_BITS: u32 = 192;

/// Verifier bound as a multiple of σ: `Rej` returns ⊥ when
/// `||z||_inf > 6σ`.  A *protocol* constant — `beta_sel,inf` and the
/// M-SIS estimate are derived from it.
pub const VERIFIER_TAILCUT: u64 = 6;

/// Tail cut for the discrete Gaussian sampler, as a multiple of σ.
///
/// Not the same constant as [`VERIFIER_TAILCUT`] and deliberately not
/// the same value: that one truncates the *response* and is part of the
/// protocol, this one truncates the *mask* and is an artefact of the
/// sampler.  Sized by union-bounding the per-coefficient tail over every
/// Gaussian coefficient a transcript publishes and requiring the total
/// below `2^-128`.
pub const GAUSSIAN_TAILCUT: u64 = 14;

/// Fixed concrete constant in `M_1 = exp(12/phi + 1/(2 phi^2))`.
///
/// This belongs to the concrete [`rej1`] definition rather than to an
/// individual parameter profile or transcript.
pub const REJ1_CONSTANT: u64 = 12;

/// Denominator [`rational_sigma`] pins Gaussian widths to.
pub const SIGMA_SCALE: u64 = 1 << 20;

// ---- domain separators ---------------------------------------------------
// The paper requires all oracles to be pairwise domain-separated.

pub const DS_EXPAND: &[u8] = b"RiVeR.Expand";
pub const DS_G: &[u8] = b"RiVeR.G";
pub const DS_CHALLENGE: &[u8] = b"RiVeR.H";
pub const DS_COMMIT: &[u8] = b"RiVeR.Com";
pub const DS_EXACT: &[u8] = b"RiVeR.Exact";
pub const DS_DUMMY: &[u8] = b"RiVeR.dummy";
pub const DS_KEYGEN: &[u8] = b"RiVeR.KeyGen";

// ---- absorption ----------------------------------------------------------

/// One argument of [`absorb`].  Mirrors the Python `*parts` variadic,
/// which accepts `bytes` or `int`; an `int` is absorbed as its 8-byte
/// little-endian encoding.
#[derive(Clone, Copy, Debug)]
pub enum Part<'a> {
    Bytes(&'a [u8]),
    Int(u64),
}

impl<'a> From<&'a [u8]> for Part<'a> {
    fn from(b: &'a [u8]) -> Self {
        Part::Bytes(b)
    }
}

/// Injectively hash a domain separator and byte parts to a 64-byte seed.
pub fn absorb(domain: &[u8], parts: &[Part<'_>]) -> [u8; 64] {
    let mut h = Shake256::default();
    h.update(&(domain.len() as u64).to_le_bytes());
    h.update(domain);
    for part in parts {
        match *part {
            Part::Bytes(b) => {
                h.update(&(b.len() as u64).to_le_bytes());
                h.update(b);
            }
            Part::Int(i) => {
                h.update(&8u64.to_le_bytes());
                h.update(&i.to_le_bytes());
            }
        }
    }
    let mut out = [0u8; 64];
    h.finalize_xof().read(&mut out);
    out
}

/// One-shot domain-separated hash to `length` bytes.
pub fn hash_bytes(length: usize, domain: &[u8], parts: &[Part<'_>]) -> Vec<u8> {
    let seed = absorb(domain, parts);
    let mut h = Shake256::default();
    h.update(&seed);
    let mut out = vec![0u8; length];
    h.finalize_xof().read(&mut out);
    out
}

// ---- XOF -----------------------------------------------------------------

/// Unbounded deterministic byte stream: `SHAKE256(seed || counter_le8)`
/// blocks, concatenated.
///
/// Counter mode rather than a squeeze, because the Python reference has
/// no streaming squeeze available and the two must agree byte for byte.
pub struct Xof {
    /// The seed, pre-absorbed.  Counter mode re-derives every block from
    /// `seed || counter`, so the 64-byte absorb would otherwise be paid
    /// once per 136 bytes of output; cloning a state that already holds
    /// it leaves only the counter and the permutation.  Identical output
    /// — a clone is the same sponge state.
    base: Shake256,
    seed: [u8; 64],
    counter: u64,
    buf: [u8; SHAKE_BLOCK],
    pos: usize,
}

impl Xof {
    pub fn new(domain: &[u8], parts: &[Part<'_>]) -> Self {
        Self::from_seed(absorb(domain, parts))
    }

    /// Build directly from a pre-absorbed 64-byte seed.
    pub fn from_seed(seed: [u8; 64]) -> Self {
        let mut base = Shake256::default();
        base.update(&seed);
        Self {
            base,
            seed,
            counter: 0,
            buf: [0u8; SHAKE_BLOCK],
            pos: SHAKE_BLOCK,
        }
    }

    /// The seed this stream was built from.
    pub fn seed(&self) -> &[u8; 64] {
        &self.seed
    }

    fn refill(&mut self) {
        let mut h = self.base.clone();
        h.update(&self.counter.to_le_bytes());
        h.finalize_xof().read(&mut self.buf);
        self.counter += 1;
        self.pos = 0;
    }

    pub fn read_into(&mut self, out: &mut [u8]) {
        let mut written = 0;
        while written < out.len() {
            if self.pos == SHAKE_BLOCK {
                self.refill();
            }
            let take = (SHAKE_BLOCK - self.pos).min(out.len() - written);
            out[written..written + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            written += take;
        }
    }

    pub fn read(&mut self, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        self.read_into(&mut out);
        out
    }

    /// Little-endian unsigned integer from the next `n` bytes, `n <= 8`.
    pub fn uint(&mut self, n: usize) -> u64 {
        debug_assert!(n <= 8);
        let mut b = [0u8; 8];
        self.read_into(&mut b[..n]);
        u64::from_le_bytes(b)
    }

    /// Uniform value in `[0, 2^PROB_BITS)` as three little-endian
    /// limbs — the left side of every acceptance comparison, in the form
    /// the fixed-width path wants and with no allocation.
    pub fn unit_fixed_u192(&mut self) -> [u64; 3] {
        let mut b = [0u8; (PROB_BITS / 8) as usize];
        self.read_into(&mut b);
        let mut out = [0u64; 3];
        for (i, limb) in out.iter_mut().enumerate() {
            *limb = u64::from_le_bytes(b[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        out
    }

    /// The same draw as a [`Nat`].  Kept for the exact path and for
    /// tests; the sampler uses [`Xof::unit_fixed_u192`].
    pub fn unit_fixed(&mut self) -> Nat {
        let mut b = [0u8; (PROB_BITS / 8) as usize];
        self.read_into(&mut b);
        Nat::from_bytes_le(&b)
    }

    pub fn bit(&mut self) -> u8 {
        let mut b = [0u8; 1];
        self.read_into(&mut b);
        b[0] & 1
    }
}

// ---- uniform samplers ----------------------------------------------------

/// Uniform in `[0, modulus)` by rejection on whole bytes.
pub fn uniform_int(xof: &mut Xof, modulus: u64) -> u64 {
    debug_assert!(modulus >= 1);
    let bits = 64 - modulus.leading_zeros();
    let nbytes = bits.div_ceil(8) as usize;
    let mask = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    loop {
        let v = xof.uint(nbytes) & mask;
        if v < modulus {
            return v;
        }
    }
}

pub fn uniform_poly(xof: &mut Xof, modulus: u64, d: usize) -> Vec<u64> {
    (0..d).map(|_| uniform_int(xof, modulus)).collect()
}

pub fn uniform_vec(xof: &mut Xof, modulus: u64, d: usize, len: usize) -> Vec<Vec<u64>> {
    (0..len).map(|_| uniform_poly(xof, modulus, d)).collect()
}

pub fn uniform_matrix(
    xof: &mut Xof,
    modulus: u64,
    d: usize,
    rows: usize,
    cols: usize,
) -> Vec<Vec<Vec<u64>>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| uniform_poly(xof, modulus, d)).collect())
        .collect()
}

/// `SamMat(rho, q, n, m, str)` of Figure 3.
///
/// Deterministic in `(seed, modulus, shape, label)`; both prover and
/// verifier call it and nothing about it is transmitted.
pub fn sam_mat(
    seed: &[u8],
    modulus: u64,
    rows: usize,
    cols: usize,
    d: usize,
    label: &str,
) -> Vec<Vec<Vec<u64>>> {
    // The Python packs `modulus` into its minimal little-endian width and
    // the three shape values into 4 bytes each; every part is then length
    // prefixed by `absorb`.
    let mod_bytes = {
        let bits = 64 - modulus.leading_zeros();
        let n = bits.div_ceil(8) as usize;
        modulus.to_le_bytes()[..n].to_vec()
    };
    let rows_b = (rows as u32).to_le_bytes();
    let cols_b = (cols as u32).to_le_bytes();
    let d_b = (d as u32).to_le_bytes();
    let mut xof = Xof::new(
        DS_EXPAND,
        &[
            Part::Bytes(seed),
            Part::Bytes(label.as_bytes()),
            Part::Bytes(&mod_bytes),
            Part::Bytes(&rows_b),
            Part::Bytes(&cols_b),
            Part::Bytes(&d_b),
        ],
    );
    uniform_matrix(&mut xof, modulus, d, rows, cols)
}

/// A centred value already inside one modulus, as a canonical residue.
///
/// TIMING.  This is on the secret path — `uniform_beta_poly` samples the
/// secret key and `gaussian_poly` samples every mask — so it must not
/// divide and must not branch on the value.  `v >> 63` is the
/// arithmetic-shift sign mask (all ones when `v < 0`, zero otherwise),
/// and the wrapping add is the same instruction sequence either way.
/// `rem_euclid` was both: a division whose latency is operand-dependent
/// on some cores, and a source-level `if` LLVM is free to branch on.
///
/// The precondition — `|v| < modulus` — holds by construction at every
/// call site: `U_beta` draws in `[-beta, beta]` with `2 beta + 1 << q`,
/// and a Gaussian truncates at `tailcut · sigma`, three orders below the
/// smallest modulus in use.  It is a `debug_assert` rather than a check
/// because a runtime branch here would reintroduce exactly what this
/// removes; [`crate::params::RiVeRParams::check`] is where the
/// modulus/width relation is enforced.
#[inline]
pub(crate) fn centered_to_residue(v: i64, modulus: u64) -> u64 {
    debug_assert!(
        v.unsigned_abs() < modulus,
        "centered_to_residue: |{v}| is not below the modulus {modulus}"
    );
    let mask = (v >> 63) as u64; // 0 or !0
    (v as u64).wrapping_add(mask & modulus)
}

/// One draw from `U_beta`: coefficients uniform in `[-beta, beta]`,
/// returned in canonical `[0, modulus)` form.
pub fn uniform_beta_poly(xof: &mut Xof, beta: u64, d: usize, modulus: u64) -> Vec<u64> {
    let span = 2 * beta + 1;
    (0..d)
        .map(|_| {
            let v = uniform_int(xof, span) as i64 - beta as i64;
            centered_to_residue(v, modulus)
        })
        .collect()
}

pub fn uniform_beta_vec(
    xof: &mut Xof,
    beta: u64,
    d: usize,
    len: usize,
    modulus: u64,
) -> Vec<Vec<u64>> {
    (0..len)
        .map(|_| uniform_beta_poly(xof, beta, d, modulus))
        .collect()
}

// ---- discrete Gaussian ---------------------------------------------------

/// Represent a float σ exactly as a fraction `(num, 2^20)`.
///
/// The widths in the paper are irrational (`B_a = gamma·sqrt(2w)`), so
/// they are pinned to an exact rational once — here — rather than
/// letting float rounding leak into accept/reject decisions.
///
/// Python's `round()` is half-to-even, which
/// [`f64::round_ties_even`] reproduces.  A plain `round()` would differ
/// on exact halves and move a test vector.
pub fn rational_sigma(sigma: f64) -> (u64, u64) {
    let num = (sigma * SIGMA_SCALE as f64).round_ties_even();
    (num as u64, SIGMA_SCALE)
}

/// Everything about one Gaussian width that does not change per sample.
///
/// Build it once per field and call [`gaussian_int_ctx`]; the per-sample
/// work is then a uniform draw and a handful of `u64` multiplies.  The
/// convenience form [`gaussian_int`] builds one per call, which costs a
/// setup division the loop below does not.
#[derive(Clone, Copy, Debug)]
pub struct GaussCtx {
    bound: u64,
    span: u64,
    /// `sigma_den²` and `2·sigma_num²`: the exponent is `-z²·a / b`.
    a: u128,
    b: u128,
    exp: ExpCtx,
}

impl GaussCtx {
    /// # Panics
    ///
    /// On a configuration that cannot describe a sampler:
    ///
    /// * a `tailcut` [`PROB_BITS`] cannot reach — see
    ///   [`check_probability_width`], which is what makes an unreachable
    ///   declared cut an error rather than a silent no-op;
    /// * a zero `sigma_num` or `sigma_den`, which are a point mass and a
    ///   division by zero;
    /// * a width whose truncation point does not fit a `u64`.
    ///
    /// These are configuration errors, not input: every width in the scheme
    /// comes from [`crate::params`].  Panicking beats the release-mode
    /// alternative, which was a wrapped `tailcut · sigma_num` — a *smaller*
    /// support that still samples, still verifies against itself, and is a
    /// different distribution from the one the transcript is specified over.
    pub fn new(sigma_num: u64, sigma_den: u64, tailcut: u64) -> Self {
        if let Err(why) = check_probability_width(PROB_BITS, tailcut) {
            panic!("{why}");
        }
        assert!(sigma_num != 0, "sigma_num = 0 is not a Gaussian width");
        assert!(sigma_den != 0, "sigma_den = 0");
        let scaled = tailcut
            .checked_mul(sigma_num)
            .expect("tailcut · sigma_num overflows u64");
        let bound = scaled / sigma_den;
        // `gaussian_int_ctx` draws `uniform_int(span)` and subtracts the
        // bound **in `i64`**, so the whole support has to fit there: the old
        // `u64::MAX / 2` admitted a `span` past `i64::MAX`, where the cast
        // wraps and the subtraction panics in debug or returns a wrong
        // sample in release.  `2 bound + 1 <= i64::MAX` is the real
        // condition, and it is nowhere near binding — the widest width the
        // scheme uses truncates at about `2.5 · 10^8`.
        assert!(
            bound <= (i64::MAX as u64 - 1) / 2,
            "truncation point {bound} does not fit the sampler's i64 support"
        );
        // `2 sigma_num²` is the exponent's denominator and can leave `u128`
        // on its own — the bound check above constrains `tailcut · num`, not
        // `num²`.  Checked for the same reason as the width test: a wrapped
        // denominator is a different distribution that still samples.
        let a = (sigma_den as u128)
            .checked_mul(sigma_den as u128)
            .expect("sigma_den² overflows u128");
        let b = (sigma_num as u128)
            .checked_mul(sigma_num as u128)
            .and_then(|s| s.checked_mul(2))
            .expect("2 · sigma_num² overflows u128");
        Self {
            bound,
            span: 2 * bound + 1,
            a,
            b,
            exp: ExpCtx::new(a, b),
        }
    }

    pub fn bound(&self) -> u64 {
        self.bound
    }
}

/// One sample from `D_sigma`, truncated at `tailcut · sigma`.
///
/// Uniform proposal on the truncated support, accepted with probability
/// `exp(-z²/2σ²)`.  About eleven draws per sample at the default cut.
/// Simple, exact in the accept/reject decision, and — like every sampler
/// here — not constant time.
pub fn gaussian_int(xof: &mut Xof, sigma_num: u64, sigma_den: u64, tailcut: u64) -> i64 {
    gaussian_int_ctx(xof, &GaussCtx::new(sigma_num, sigma_den, tailcut))
}

/// [`gaussian_int`] against a prepared width.
///
/// The acceptance test goes through [`crate::fastexp`], which decides the
/// same predicate as [`crate::fixed::exp_accept`] in fixed width and
/// falls back to it on the roughly one proposal in `2^55` whose bracket
/// is ambiguous.  Same distribution, same XOF consumption, same bytes.
pub fn gaussian_int_ctx(xof: &mut Xof, ctx: &GaussCtx) -> i64 {
    loop {
        let z = uniform_int(xof, ctx.span) as i64 - ctx.bound as i64;
        if z == 0 {
            return 0;
        }
        let m = z.unsigned_abs() as u128;
        let u = xof.unit_fixed_u192();
        if crate::fastexp::accept(&u, m * m, &ctx.exp, ctx.a, ctx.b) {
            return z;
        }
    }
}

pub fn gaussian_poly(
    xof: &mut Xof,
    sigma_num: u64,
    d: usize,
    modulus: u64,
    sigma_den: u64,
) -> Vec<u64> {
    let ctx = GaussCtx::new(sigma_num, sigma_den, GAUSSIAN_TAILCUT);
    (0..d)
        .map(|_| centered_to_residue(gaussian_int_ctx(xof, &ctx), modulus))
        .collect()
}

pub fn gaussian_vec(
    xof: &mut Xof,
    sigma_num: u64,
    d: usize,
    len: usize,
    modulus: u64,
    sigma_den: u64,
) -> Vec<Vec<u64>> {
    (0..len)
        .map(|_| gaussian_poly(xof, sigma_num, d, modulus, sigma_den))
        .collect()
}

/// Assert that [`PROB_BITS`] actually reaches [`GAUSSIAN_TAILCUT`].
///
/// Returns the effective cut `sqrt(2 · bits · ln 2)` in units of σ.  The
/// two constants are independent in the source and coupled in the
/// mathematics: raising the tail cut alone is a silent no-op past the
/// effective one.
pub fn effective_tailcut(bits: u32) -> f64 {
    (2.0 * bits as f64 * core::f64::consts::LN_2).sqrt()
}

/// `ln 2` as an exact rational, so the check below is decided in integers.
const LN2_NUM: u128 = 693_147_180_559_945_309;
const LN2_DEN: u128 = 1_000_000_000_000_000_000;

/// Minimum bits of threshold left below the declared cut.
///
/// At the cut itself the acceptance floor must still be a wide integer, or
/// the *floor* rather than the Gaussian decides the last few σ.
const MIN_HEADROOM_BITS: u128 = 32;

/// Reject a sampler width whose declared tail cut `bits` cannot reach.
///
/// `gaussian_int` accepts when `u < floor(2^bits · exp(-z²/2σ²))`.  Past
/// `|z|/σ = sqrt(2 · bits · ln 2)` that floor is zero, nothing is ever
/// accepted there, and a larger declared cut is silently the same
/// distribution — which is how a declared 14σ once behaved as 9.42σ.
///
/// The paper of the supplement requires
/// `GAUSSIAN_TAILCUT`, `VERIFIER_TAILCUT` and `PROB_BITS` to be independent
/// parameters, and requires a declared cut beyond the precision `PROB_BITS`
/// supports to be an **error** rather than a no-op.  [`GaussCtx::new`] calls
/// this, so no sampler can be built at an unreachable cut.
///
/// Evaluated over a rational `ln 2` in `u128`, so no float rounding decides
/// it.  `river-py`'s `sample._check_probability_width` is the same test.
pub fn check_probability_width(bits: u32, tailcut: u64) -> Result<(), String> {
    // A zero cut is a point mass at 0, not a Gaussian.
    if tailcut == 0 {
        return Err("GAUSSIAN_TAILCUT = 0 is a point mass, not a width".into());
    }
    // `t² / (2 ln 2)`, the bits of threshold the cut spends.  Rounded up, so
    // a value exactly at the boundary is rejected rather than accepted.
    //
    // Checked, because this multiplication *failed open*: at
    // `tailcut = 2^55`, `t² · LN2_DEN` is `2^128 · 5^18`, which is exactly
    // zero mod `2^128`, so `spent` came out 0 and the function returned
    // `Ok` in release.  Any cut large enough to overflow here is many
    // orders past what any `bits` can reach, so overflow *is* the rejection.
    let den = 2 * LN2_NUM;
    let spent = match (tailcut as u128)
        .checked_mul(tailcut as u128)
        .and_then(|t2| t2.checked_mul(LN2_DEN))
    {
        Some(scaled) => scaled.div_ceil(den),
        None => {
            return Err(format!(
                "GAUSSIAN_TAILCUT = {tailcut} is far past any reachable cut"
            ))
        }
    };
    let bits = bits as u128;
    if spent >= bits {
        return Err(format!(
            "PROB_BITS={bits} caps the sampler below GAUSSIAN_TAILCUT={tailcut} \
             (the cut needs {spent} bits of threshold)"
        ));
    }
    if bits - spent < MIN_HEADROOM_BITS {
        return Err(format!(
            "PROB_BITS={bits} leaves only {} bits of threshold at \
             GAUSSIAN_TAILCUT={tailcut}, below the {MIN_HEADROOM_BITS} required",
            bits - spent
        ));
    }
    Ok(())
}

// ---- challenge space -----------------------------------------------------

/// Uniform element of `C^d_{w,gamma}`: exactly `w` of the `d`
/// coefficients nonzero, each uniform in `±[1, gamma]`.
///
/// The paper fixes the *size* of this set but never gives a hash-to-`C`
/// procedure; this is one, and it is uniform over the set.  Positions
/// come from a partial Fisher-Yates shuffle so the sampler always
/// terminates.
pub fn sample_challenge(xof: &mut Xof, d: usize, w: usize, gamma: u64, modulus: u64) -> Vec<u64> {
    let mut positions: Vec<usize> = (0..d).collect();
    let mut chosen = Vec::with_capacity(w);
    for i in 0..w {
        let j = i + uniform_int(xof, (d - i) as u64) as usize;
        positions.swap(i, j);
        chosen.push(positions[i]);
    }
    let mut poly = vec![0u64; d];
    for pos in chosen {
        let magnitude = 1 + uniform_int(xof, gamma) as i64;
        let sign = if xof.bit() != 0 { -1i64 } else { 1i64 };
        poly[pos] = centered_to_residue(sign * magnitude, modulus);
    }
    poly
}

/// `H(...) -> C^d_{w,gamma}`, the OOM Fiat-Shamir challenge.
pub fn challenge_from_hash(
    d: usize,
    w: usize,
    gamma: u64,
    modulus: u64,
    parts: &[Part<'_>],
) -> Vec<u64> {
    let mut xof = Xof::new(DS_CHALLENGE, parts);
    sample_challenge(&mut xof, d, w, gamma, modulus)
}

// ---- rejection sampling (Figure 2) ---------------------------------------
// `true` = reject (the paper's `Rej` returning ⊥), `false` = accept.

/// `u < floor(scale · exp(-(<z,v>·2 - ||v||^2) · sigma_den^2 / (2 sigma_num^2)))`.
///
/// Every product is formed with checked arithmetic.  The exponent is a
/// quadratic form in the response and the width, so at hostile parameters
/// it leaves `i128` well before the argument types run out; a wrapped
/// value would be a plausible-looking acceptance probability for a
/// different distribution rather than a visible failure, which is the one
/// outcome a rejection sampler must not have.
///
/// # Panics
///
/// If `sigma_num` is zero (a zero-width Gaussian has no acceptance rule),
/// or if `(2<z,v> - ||v||^2) sigma_den^2` or `2 sigma_num^2` is not
/// representable.  [`crate::params::RiVeRParams::check`] rejects all of
/// these long before here.
fn p_acc(
    u: &Nat,
    inner_zv: i128,
    norm_v_sq: i128,
    sigma_num: u64,
    sigma_den: u64,
    scale: &Nat,
) -> bool {
    assert!(sigma_num > 0, "sigma_num must be positive");
    let sd = sigma_den as i128;
    let exponent = inner_zv
        .checked_mul(-2)
        .and_then(|x| x.checked_add(norm_v_sq))
        .and_then(|x| x.checked_mul(sd))
        .and_then(|x| x.checked_mul(sd))
        .expect("p_acc: the rejection exponent overflows i128");
    let den = (sigma_num as u128)
        .checked_mul(sigma_num as u128)
        .and_then(|x| x.checked_mul(2))
        .expect("p_acc: 2 sigma_num^2 overflows u128");
    exp_accept(u, &Int::from_i128(exponent), &Nat::from_u128(den), scale)
}

/// `(<z, v>, ||v||^2)` over the integers.
///
/// Both accumulate with checked arithmetic.  `i64` coefficients square to
/// `i128` comfortably, but `d(ell+n)` of them summed do not in general, and
/// a wrapped inner product would flip the sign test in [`rej2`] — turning a
/// rejection into an acceptance rather than into a visible failure.
///
/// # Panics
///
/// If the two slices differ in length, or if either accumulator overflows
/// `i128`.  The length check is not decoration: `zip` would otherwise stop
/// at the shorter slice and silently compute an inner product over a prefix.
fn inner_and_norm(z: &[i64], v: &[i64]) -> (i128, i128) {
    assert_eq!(z.len(), v.len(), "rejection sampling: length mismatch");
    let mut inner: i128 = 0;
    let mut norm: i128 = 0;
    for (&a, &b) in z.iter().zip(v.iter()) {
        inner = (a as i128)
            .checked_mul(b as i128)
            .and_then(|p| inner.checked_add(p))
            .expect("rejection sampling: <z, v> overflows i128");
        norm = (b as i128)
            .checked_mul(b as i128)
            .and_then(|p| norm.checked_add(p))
            .expect("rejection sampling: ||v||^2 overflows i128");
    }
    (inner, norm)
}

fn inf_norm(coeffs: &[i64]) -> i128 {
    coeffs.iter().map(|&c| (c as i128).abs()).max().unwrap_or(0)
}

fn tail_reject(z: &[i64], sigma_num: u64, sigma_den: u64) -> bool {
    inf_norm(z) * sigma_den as i128 > VERIFIER_TAILCUT as i128 * sigma_num as i128
}

/// `24 phi + 1`, the concrete `Rej_1` exponent numerator, or a panic.
///
/// Split out so both the value and its failure are stated once.  The
/// result is asserted to fit `i128` because the caller negates it into
/// [`Int::from_i128`].
fn checked_exponent_num(phi: u64) -> u128 {
    let n = 2u128
        .checked_mul(REJ1_CONSTANT as u128)
        .and_then(|x| x.checked_mul(phi as u128))
        .and_then(|x| x.checked_add(1))
        .filter(|&x| x <= i128::MAX as u128);
    match n {
        Some(x) => x,
        None => panic!("rej1: 24 phi + 1 overflows (phi={phi})"),
    }
}

/// `2 phi^2`, the common denominator of both rejection exponents.
fn checked_two_phi_squared(phi: u64) -> u128 {
    assert!(phi > 0, "phi must be positive");
    match (phi as u128)
        .checked_mul(phi as u128)
        .and_then(|x| x.checked_mul(2))
    {
        Some(x) => x,
        None => panic!("rej: 2 phi^2 overflows (phi={phi})"),
    }
}

/// `Rej_1(z, v, phi, T)` with `sigma = phi · T` as a rational.
///
/// `z` and `v` are flat slices of *centred* integer coefficients.
///
/// The concrete rejection constant is fixed internally at 12.  The sampler
/// and [`crate::params::RiVeRParams::mu_a`] therefore share
/// [`REJ1_CONSTANT`] rather than exposing a per-profile argument.
///
/// # Panics
///
/// On parameters no validated profile produces, and deterministically in
/// release as well as debug — a wrapped exponent is a silently wrong
/// acceptance rate, which is the one failure this procedure must not have.
/// [`crate::params::RiVeRParams::check`] rejects all of these long before
/// here; the contract is for a caller that bypassed it.
///
/// * `phi == 0` — `2 phi^2` is then a zero denominator.
/// * `24 phi + 1` or `2 phi^2` not representable in the checked arithmetic.
/// * `sigma_num == 0`, or `z` and `v` of differing length.
pub fn rej1(xof: &mut Xof, z: &[i64], v: &[i64], phi: u64, sigma_num: u64, sigma_den: u64) -> bool {
    assert!(sigma_num > 0, "sigma_num must be positive");
    let (inner_zv, norm_v_sq) = inner_and_norm(z, v);
    // M = exp(12/phi + 1/(2 phi^2)); over the common denominator
    // `2 phi^2` that exponent is `-(24 phi + 1)`.  Fold 1/M into the
    // threshold scale.
    //
    // Formed in `u128` with checked arithmetic: at `u64` width both
    // products overflow well inside the argument type, and release builds
    // would wrap rather than trap — turning an invalid parameter into a
    // plausible-looking threshold for a different `M`.
    let m_scale = exp_threshold(
        &Int::from_i128(-(checked_exponent_num(phi) as i128)),
        &Nat::from_u128(checked_two_phi_squared(phi)),
        &Nat::pow2(PROB_BITS),
    );
    let u = xof.unit_fixed();
    if !p_acc(&u, inner_zv, norm_v_sq, sigma_num, sigma_den, &m_scale) {
        return true;
    }
    tail_reject(z, sigma_num, sigma_den)
}

/// `Rej_2(z, v, phi, T)`, the optimised sampler.
///
/// Rejects outright when `<z, v> < 0`, which is exactly the half-space
/// hint the Extended M-LWE assumption accounts for — and the factor 2 in
/// `mu_bin` that the repetition appendix has to charge exactly once.
///
/// # Panics
///
/// If `phi` or `sigma_num` is zero, if `2 phi^2` is not representable in
/// `u128`, or if `z` and `v` differ in length.  All are checked before the
/// half-space shortcut, so the contract does not depend on the sign of
/// `<z, v>`.  See [`rej1`] for why this traps rather than wraps.
pub fn rej2(xof: &mut Xof, z: &[i64], v: &[i64], phi: u64, sigma_num: u64, sigma_den: u64) -> bool {
    // Validated before the half-space shortcut below, so the documented
    // panics do not depend on the sign of `<z, v>`: invalid parameters must
    // fail the same way whichever branch the input happens to take.
    let two_phi_sq = checked_two_phi_squared(phi);
    assert!(sigma_num > 0, "sigma_num must be positive");
    let (inner_zv, norm_v_sq) = inner_and_norm(z, v);
    if inner_zv < 0 {
        return true;
    }
    let m_scale = exp_threshold(
        &Int::from_i128(-1),
        &Nat::from_u128(two_phi_sq),
        &Nat::pow2(PROB_BITS),
    );
    let u = xof.unit_fixed();
    if !p_acc(&u, inner_zv, norm_v_sq, sigma_num, sigma_den, &m_scale) {
        return true;
    }
    tail_reject(z, sigma_num, sigma_den)
}

// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xof_stream_is_independent_of_chunking() {
        let mut a = Xof::new(b"t", &[Part::Bytes(b"seed")]);
        let mut b = Xof::new(b"t", &[Part::Bytes(b"seed")]);
        let one = a.read(300);
        let mut two = b.read(100);
        two.extend(b.read(200));
        assert_eq!(one, two);
    }

    #[test]
    fn absorption_is_length_prefixed() {
        let x = hash_bytes(16, b"t", &[Part::Bytes(b"ab"), Part::Bytes(b"c")]);
        let y = hash_bytes(16, b"t", &[Part::Bytes(b"a"), Part::Bytes(b"bc")]);
        assert_ne!(x, y);
    }

    #[test]
    fn uniform_int_stays_in_range_and_is_flat() {
        let mut x = Xof::new(b"t", &[Part::Bytes(b"u")]);
        let vals: Vec<u64> = (0..12200).map(|_| uniform_int(&mut x, 61)).collect();
        assert!(vals.iter().all(|&v| v < 61));
        let min_count = (0..61)
            .map(|k| vals.iter().filter(|&&v| v == k).count())
            .min()
            .unwrap();
        assert!(min_count > 150, "min bin count {min_count}");
    }

    #[test]
    fn uniform_int_handles_a_unit_modulus() {
        // `sample_challenge` hits `uniform_int(xof, 1)` on its last swap.
        let mut x = Xof::new(b"t", &[Part::Bytes(b"unit")]);
        for _ in 0..64 {
            assert_eq!(uniform_int(&mut x, 1), 0);
        }
    }

    #[test]
    fn gaussian_moments_are_close_and_support_is_truncated() {
        let sigma = 1000u64;
        let mut x = Xof::new(b"t", &[Part::Bytes(b"g")]);
        let samples: Vec<i64> = (0..20000)
            .map(|_| gaussian_int(&mut x, sigma, 1, GAUSSIAN_TAILCUT))
            .collect();
        let n = samples.len() as f64;
        let mean = samples.iter().map(|&s| s as f64).sum::<f64>() / n;
        let var = samples
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / n
            - mean * mean;
        assert!(mean.abs() < 40.0, "mean {mean}");
        assert!((var.sqrt() - sigma as f64).abs() / (sigma as f64) < 0.05);
        assert!(samples.iter().all(|&s| s.abs() <= 6 * sigma as i64));
    }

    #[test]
    fn challenge_has_the_declared_weight_and_bound() {
        let q = 1073123348676497u64;
        let mut x = Xof::new(b"t", &[Part::Bytes(b"c")]);
        for _ in 0..20 {
            let c = sample_challenge(&mut x, 32, 32, 16, q);
            let cent: Vec<i64> = c
                .iter()
                .map(|&v| {
                    if v > q / 2 {
                        v as i64 - q as i64
                    } else {
                        v as i64
                    }
                })
                .collect();
            assert_eq!(cent.iter().filter(|&&v| v != 0).count(), 32);
            assert!(cent.iter().all(|&v| v.abs() <= 16));
        }
        let mut x = Xof::new(b"t", &[Part::Bytes(b"c2")]);
        let c = sample_challenge(&mut x, 32, 8, 16, q);
        let nonzero = c.iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero, 8);
    }

    #[test]
    fn rational_sigma_pins_the_width() {
        assert_eq!(rational_sigma(4096.0), (4096 << 20, 1 << 20));
        // half-to-even, as Python's round()
        assert_eq!(rational_sigma(0.5 / SIGMA_SCALE as f64).0, 0);
        assert_eq!(rational_sigma(1.5 / SIGMA_SCALE as f64).0, 2);
    }

    #[test]
    fn prob_bits_reaches_the_declared_tail_cut() {
        let effective = effective_tailcut(PROB_BITS);
        assert!(effective > GAUSSIAN_TAILCUT as f64, "{effective}");
        let headroom = PROB_BITS as f64
            - (GAUSSIAN_TAILCUT * GAUSSIAN_TAILCUT) as f64 / (2.0 * core::f64::consts::LN_2);
        assert!(headroom >= 32.0, "{headroom}");
        // The shipped pair is the one the sampler is actually built at.
        assert!(check_probability_width(PROB_BITS, GAUSSIAN_TAILCUT).is_ok());
    }

    /// An unreachable declared cut is an error,
    /// not a silently unchanged distribution.  The integer check and the
    /// float one have to agree about where the boundary is.
    #[test]
    fn an_unreachable_tail_cut_is_rejected() {
        // 64 bits caps at 9.42σ: the historical silent-no-op case.
        assert!(check_probability_width(64, 14).is_err());
        // Exactly at the effective cut is still a rejection — nothing is
        // ever accepted at the boundary, so it is not a usable width.
        let effective = effective_tailcut(192) as u64; // 16
        assert!(check_probability_width(192, effective).is_err());
        for bits in [64u32, 128, 192, 256] {
            let cut = effective_tailcut(bits);
            for t in 1..=40u64 {
                let ok = check_probability_width(bits, t).is_ok();
                if ok {
                    assert!((t as f64) < cut, "bits={bits} t={t} cut={cut}");
                    assert!(bits as f64 - (t * t) as f64 / (2.0 * core::f64::consts::LN_2) >= 32.0);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "GAUSSIAN_TAILCUT")]
    fn building_a_sampler_at_an_unreachable_cut_panics() {
        // 25σ needs 902 bits of threshold; `PROB_BITS` is 192.
        let _ = GaussCtx::new(30, 1, 25);
    }

    /// The width check has to fail *closed* in release, where `u128`
    /// multiplication wraps instead of panicking.
    #[test]
    fn the_width_check_does_not_fail_open_on_overflow() {
        // `2^55`: `t² · LN2_DEN` is exactly `0 mod 2^128`, so the unchecked
        // form computed `spent = 0` and returned `Ok`.
        assert!(check_probability_width(PROB_BITS, 1u64 << 55).is_err());
        for t in [1u64 << 32, 1 << 40, 1 << 55, 1 << 63, u64::MAX] {
            assert!(check_probability_width(PROB_BITS, t).is_err(), "{t}");
        }
        // a zero cut is a point mass, not a width
        assert!(check_probability_width(PROB_BITS, 0).is_err());
        // and nothing in the reachable range moved: the boundary is still
        // where the float form puts it
        for t in 1..=40u64 {
            let spent = (t * t) as f64 / (2.0 * core::f64::consts::LN_2);
            let want = PROB_BITS as f64 - spent >= 32.0;
            assert_eq!(
                check_probability_width(PROB_BITS, t).is_ok(),
                want,
                "t = {t}"
            );
        }
        assert!(check_probability_width(PROB_BITS, GAUSSIAN_TAILCUT).is_ok());
    }

    #[test]
    #[should_panic(expected = "2 · sigma_num²")]
    fn a_denominator_that_would_wrap_is_rejected() {
        // `tailcut · num` fits and the truncation point is 1, so neither
        // earlier check fires; `2 num²` is about `2^129` and does not fit.
        let _ = GaussCtx::new(u64::MAX, u64::MAX, 1);
    }

    #[test]
    #[should_panic(expected = "sigma_den = 0")]
    fn a_zero_denominator_is_rejected_rather_than_dividing() {
        let _ = GaussCtx::new(30, 0, GAUSSIAN_TAILCUT);
    }

    #[test]
    #[should_panic(expected = "sigma_num = 0")]
    fn a_zero_width_is_rejected() {
        let _ = GaussCtx::new(0, 1, GAUSSIAN_TAILCUT);
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn a_width_that_would_wrap_is_rejected_rather_than_wrapped() {
        // In release this wrapped to a *smaller* support and kept sampling.
        let _ = GaussCtx::new(u64::MAX / 2, 1, GAUSSIAN_TAILCUT);
    }

    /// The support has to fit `i64`, because that is where the proposal is
    /// centred.  A `bound` past `i64::MAX / 2` made `uniform_int(span) as
    /// i64` wrap.
    #[test]
    fn the_support_must_fit_the_i64_the_proposal_is_centred_in() {
        // widest admissible, and one past it
        let widest = (i64::MAX as u64 - 1) / 2;
        assert!(std::panic::catch_unwind(|| GaussCtx::new(widest, 1, 1)).is_ok());
        assert!(std::panic::catch_unwind(|| GaussCtx::new(widest + 1, 1, 1)).is_err());

        // and every width the scheme actually uses is far below it
        for p in crate::params::PROFILES {
            for sigma in [p.sigma_a(), p.sigma_b(), p.sigma_s(), p.sigma_m()] {
                let (num, den) = rational_sigma(sigma);
                let ctx = GaussCtx::new(num, den, GAUSSIAN_TAILCUT);
                assert!(ctx.bound() < 1 << 32, "{}: {}", p.name, ctx.bound());
                // the proposal really does stay inside i64
                assert!(ctx.span <= i64::MAX as u64);
            }
        }
    }

    /// Every width the scheme actually uses builds.
    #[test]
    fn published_widths_all_construct() {
        for p in crate::params::PROFILES {
            for sigma in [p.sigma_a(), p.sigma_b(), p.sigma_s(), p.sigma_m()] {
                let (num, den) = rational_sigma(sigma);
                let ctx = GaussCtx::new(num, den, GAUSSIAN_TAILCUT);
                assert!(ctx.bound() > 0, "{}", p.name);
            }
        }
    }

    #[test]
    #[should_panic(expected = "sigma_num must be positive")]
    fn rej2_validates_the_width_even_on_the_rejecting_half_space() {
        let mut x = Xof::new(b"t", &[Part::Bytes(b"rej2-sigma")]);
        rej2(&mut x, &[1, 0, 0], &[-1, 0, 0], 3, 0, 1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn a_length_mismatch_is_refused_rather_than_silently_truncated() {
        // `zip` would compute the inner product over the shorter prefix and
        // report an acceptance decision about a vector nobody supplied.
        let mut x = Xof::new(b"t", &[Part::Bytes(b"len")]);
        rej2(&mut x, &[1, 2, 3], &[1, 2], 3, 50, 1);
    }

    #[test]
    #[should_panic(expected = "overflows i128")]
    fn the_inner_product_traps_rather_than_wrapping() {
        let big = vec![i64::MAX; 64];
        let mut x = Xof::new(b"t", &[Part::Bytes(b"ovf")]);
        rej2(&mut x, &big, &big, 3, 50, 1);
    }

    #[test]
    #[should_panic(expected = "phi must be positive")]
    fn rej2_validates_phi_even_on_the_rejecting_half_space() {
        // `<z, v> < 0` short-circuits to "reject".  An invalid `phi` must
        // still fail there: a contract that depends on the sign of the
        // inner product is not a contract.
        let mut x = Xof::new(b"t", &[Part::Bytes(b"rej2-phi")]);
        rej2(&mut x, &[1, 0, 0], &[-1, 0, 0], 0, 50, 1);
    }

    #[test]
    #[should_panic(expected = "sigma_num must be positive")]
    fn a_zero_width_has_no_acceptance_rule() {
        p_acc(&Nat::zero(), 1, 1, 0, 1, &Nat::pow2(PROB_BITS));
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn the_acceptance_denominator_traps_rather_than_wrapping() {
        p_acc(&Nat::zero(), 1, 1, u64::MAX, 1, &Nat::pow2(PROB_BITS));
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn the_acceptance_exponent_traps_rather_than_wrapping() {
        // `||v||^2 · sigma_den^2` at these magnitudes leaves `i128`.
        p_acc(
            &Nat::zero(),
            0,
            i128::MAX / 2,
            1,
            u64::MAX,
            &Nat::pow2(PROB_BITS),
        );
    }

    #[test]
    fn the_rejection_exponent_is_exact_where_u64_would_have_wrapped() {
        // At every shipped profile the exponent is small.
        assert_eq!(checked_exponent_num(22), 529);
        assert_eq!(checked_two_phi_squared(22), 968);

        // The fixed constant still has to be widened before multiplication:
        // at the largest admitted argument the mathematical value fits u128,
        // while a u64 implementation wraps.
        let phi = u64::MAX;
        assert_eq!(checked_exponent_num(phi), 24u128 * phi as u128 + 1);
        assert_eq!(
            24u64.wrapping_mul(phi).wrapping_add(1),
            u64::MAX - 22,
            "the u64 form really did wrap here"
        );
    }

    #[test]
    fn rej1_uses_the_single_fixed_concrete_constant() {
        assert_eq!(REJ1_CONSTANT, 12);
        for phi in [22, 24, 26, 32, 34, 40] {
            assert_eq!(
                checked_exponent_num(phi),
                2 * REJ1_CONSTANT as u128 * phi as u128 + 1
            );
        }
    }

    #[test]
    #[should_panic(expected = "overflows")]
    fn the_rejection_denominator_traps_rather_than_wrapping() {
        checked_two_phi_squared(u64::MAX);
    }

    #[test]
    #[should_panic(expected = "phi must be positive")]
    fn a_zero_phi_is_refused_before_it_reaches_a_zero_denominator() {
        checked_two_phi_squared(0);
    }

    #[test]
    fn rej2_rejects_the_negative_half_space_without_reading_the_xof() {
        // The old form of this test read *zero* bytes and asserted the
        // result was zero bytes, which is true of any XOF in any state.
        // What has to hold is that the stream is where it started: the
        // half-space test is decided by `<z, v> < 0` alone, and a `rej2`
        // that consumed randomness first would shift every subsequent
        // draw and move the transcript.
        let mut x = Xof::new(b"t", &[Part::Bytes(b"half")]);
        for _ in 0..4 {
            assert!(rej2(&mut x, &[1, 2, 3], &[1, 0, -1], 3, 50, 1));
        }
        let after = x.read(64);

        let mut fresh = Xof::new(b"t", &[Part::Bytes(b"half")]);
        assert_eq!(after, fresh.read(64), "rej2 consumed XOF state");

        // and the comparison is not vacuous: a call that *does* reach the
        // sampler leaves the stream somewhere else.
        let mut consumed = Xof::new(b"t", &[Part::Bytes(b"half")]);
        assert!(!rej2(&mut consumed, &[1, 2, 3], &[1, 0, 1], 3, 5000, 1));
        assert_ne!(consumed.read(64), after);
    }

    /// The measured acceptance rate is `1/M_1` for the fixed constant 12.
    #[test]
    fn rej1_acceptance_tracks_one_over_m() {
        let phi = 8u64;
        let t = 100i64;
        let mut x = Xof::new(b"t", &[Part::Bytes(b"r")]);
        let trials = 400;
        let mut accepted = 0;
        for _ in 0..trials {
            let v0: Vec<i64> = (0..16)
                .map(|_| gaussian_int(&mut x, 30, 1, GAUSSIAN_TAILCUT))
                .collect();
            let norm = (v0.iter().map(|&c| (c * c) as f64).sum::<f64>())
                .sqrt()
                .max(1.0);
            let v: Vec<i64> = v0
                .iter()
                .map(|&c| (c as f64 * t as f64 / norm) as i64)
                .collect();
            let z: Vec<i64> = (0..16)
                .map(|j| gaussian_int(&mut x, phi * t as u64, 1, GAUSSIAN_TAILCUT) + v[j])
                .collect();
            if !rej1(&mut x, &z, &v, phi, phi * t as u64, 1) {
                accepted += 1;
            }
        }
        let rate = accepted as f64 / trials as f64;
        let expected =
            (-(REJ1_CONSTANT as f64 / phi as f64 + 1.0 / (2.0 * (phi * phi) as f64))).exp();
        assert!((rate - expected).abs() < 0.08, "rate {rate} vs {expected}");
    }
}
