//! The LANES exact proof: `Gen`, `Prove`, `Ver` — port of
//! `river-py/lanes_proof.py`.
//!
//! Figures 3 and 4 of `[ENS20]` with three departures, all of which RiVeR
//! needs:
//!
//! * `k = 1`, i.e. no automorphism (`X -> X^sigma`) stage;
//! * the Hint-MLWE treatment of `[KLSS23]`, which removes the internal
//!   rejection sampling and so contributes no repetition multiplier;
//! * support for proving only *part* of the message ternary, as the hybrid
//!   exact/relaxed framework of `[ESLR23]` requires — here the radix digits
//!   are ternary but `e_eval` and `y_eval` are not.
//!
//! ## Relation proved
//!
//! For a committed message of `N_ex = 6` ring elements carrying `l = 64`
//! scalar slots each — `N_ex l = 384` in all, of which `6 d = 192` are
//! semantic and the rest is padding the linear system constrains to zero:
//!
//! 1. every slot of elements `[alpha_lo, alpha_hi)` is ternary
//!    ([`super::mp`]);
//! 2. the full slot vector `m in Z_q~^{N_ex l}` satisfies a public linear
//!    system `A m = u` over `Z_q~`.
//!
//! Both are checked against the *same* commitment, which is what lets the
//! range argument and the linking equation refer to one witness.
//!
//! ## How the linear part works
//!
//! The verifier's challenge `gamma` compresses `A m = u` into the single
//! scalar `gamma^T (A m - u) = 0`.  That scalar is extracted as the constant
//! coefficient of an NTT-domain element: for slot values `v`,
//!
//! ```text
//! constant_coefficient(slots_to_ntt(v)) = (sum_j v_j) / l   (mod q~)
//! ```
//!
//! so `phi` carries a compensating factor `l` to cancel the `1/l`.  The
//! masking element `g` is sampled with constant coefficient zero so it does
//! not disturb the test, and is committed in `t_{N+1}` so it cannot be
//! chosen later.
//!
//! ## What is transmitted, and what is not
//!
//! Three of the prover's messages — `w`, `v` and `v'` — are **check
//! targets**: each appears in exactly one verification equation, on its
//! own, so that equation determines it from everything else in the proof.
//! Transmitting them is transmitting what the verifier can compute.
//!
//! So `Ver` transmits the challenge `c` instead and *recovers* all three,
//! in the order the transcript needs them:
//!
//! ```text
//! w  := B_0 z - c t_0
//! v  := <b_{N+1}, z> - c t_{N+2} + c f_{N+3} + sum_e alpha_e f(f+c)(f-c)
//! v' := <b_G, z> + sum_e phi_e · <b_e, z> - c (tau + t_g - h)
//! ```
//!
//! then re-derives `alpha`, `gamma` and `c'` over a transcript containing
//! the recovered values, and accepts iff `c' == c`.  Each equality that
//! used to be tested directly is folded into that one comparison — the
//! standard Fiat–Shamir commitment-recovery trade, and sound for the same
//! reason: an adversary who moves any of the three moves the transcript,
//! hence `c'`.
//!
//! `c` costs `d~ = 256` ternary coefficients, 512 bits, against the 19,968
//! the three elements cost. The two further bandwidth optimisations in the
//! paper's size model are also applied:
//!
//! * `t_0` carries only its coefficient-domain high part after dropping
//!   `D = 17` low bits;
//! * the mask and response cover only the `kappa - l~ = 13` non-identity
//!   columns.
//!
//! Those omissions make exact recovery of `w` depend on both omitted
//! quantities. The verifier computes
//!
//! ```text
//! B_0' z - c (2^D t_0,high) = w + c(t_0,low - r_identity)
//! ```
//!
//! and applies one fixed ternary bucket carry per coefficient. That carry
//! costs 2,048 bits (`l~ = 4` rows of `d~ = 256` at 2 bits); its exact
//! perturbation bound is in [`super::params`]. A measured proof is about
//! **13.88 KB**, shown beside the paper's 13.5 KB entropy estimate. Of the
//! concrete overhead, 2,560 bits are `c` plus the hint; the remainder is
//! measured Rice-over-entropy overhead.
//!
//! The displayed perturbation contains `r_identity`, but substituting
//! `t_0 = r_identity + B_0' r_tail` cancels it:
//!
//! ```text
//! t_0,low - r_identity = B_0' r_tail - 2^D t_0,high.
//! ```
//!
//! The hint therefore does not create a special channel for the omitted
//! identity block. It does reveal a deterministic carry depending on the
//! tail opening and public compressed commitment; accounting for that
//! leakage is the fixed-hint composition obligation below.
//!
//! This carry format is an implementation-derived completion of the black-box
//! exact layer. `[ENS20]` gives response compression with a rejection
//! condition and defers commitment-compression hints to Dilithium. This
//! artifact combines that model with rejection-free `[KLSS23]` masking in one
//! concrete wire format. Byte interoperability and algebraic correctness are
//! tested here; the artifact does not supply a reduction for this exact
//! fixed-hint composition.
//! No arbitrary hint-weight cap is imposed: unlike Dilithium's sparse hint
//! format, this format is dense, LANES has no retry here, and the paper gives
//! neither a cap nor a completeness/security argument for one.
//!
//! ## Status
//!
//! Validated behaviourally: honest proofs verify, every tampering the tests
//! apply is rejected, and the whole thing reproduces the reference's bytes
//! — the two `lanes-experimental` vector cases are re-derived from their
//! seeds here, and `sampler_kat.json`'s `lanes_proof` block bisects them.
//! That is **not** a soundness proof.  The RiVeR paper fixes `Pi_ex`'s
//! parameters but does not restate the protocol, so the construction here
//! follows `[ENS20]` directly.

use super::commit::{CommitSecret, Commitment, CommitmentKey, B_G};
use super::mp::{self, ProductProof};
use super::params::{
    sample_challenge, sample_gaussian_vec, sample_uniform_poly, N_EX, N_TILDE, RECOVERY_BITS,
    RECOVERY_BUCKETS, RESPONSE_RANK, SIGMA_Y, Z_INF_BOUND, Z_NORM2_BOUND,
};
use super::ring::{self as lr, CoeffPoly, NttPoly, Slots, DTILDE, LSPLIT, QTILDE, SUBDEG};
use crate::sample::{uniform_int, Part, Xof};

/// Scalar inputs to the linear system: `N_ex · l = 192`.
pub const AN: usize = N_EX * LSPLIT;

/// Fiat–Shamir domain for the three challenges.
const DS_LANES: &[u8] = b"RiVeR.Exact.lanes.fs";

/// `sigma_ex` — everything transmitted.
///
/// `alpha` and `gamma` are recomputed by the verifier from the transcript,
/// and so are the three check targets `w`, `v` and `v'`; `c` is carried
/// because recovering them needs it.  See the module docs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LanesProof {
    pub t_g: NttPoly,
    pub t_mp1: NttPoly,
    pub t_mp2: NttPoly,
    pub h: NttPoly,
    /// The challenge, ternary of weight `w_hat`, coefficient domain.
    pub c: CoeffPoly,
    /// One cyclic carry per `t_0` coefficient, each in `{-1,0,1}`.
    pub hint: Vec<Vec<i64>>,
    /// `z = y + c r_tail`, `RESPONSE_RANK` elements, coefficient domain.
    pub z: Vec<CoeffPoly>,
}

/// A public linear system `A m = u` over `Z_q~`, validated at construction.
///
/// Shape and canonicality are established once, here, so [`linear_terms`]
/// is total and the prover and the verifier cannot be looking at systems of
/// different sizes.  Both of them build this from the *statement*, which is
/// public, so a malformed one is a caller error rather than an attack.
#[derive(Clone, Debug)]
pub struct LinearSystem {
    a: Vec<Vec<u64>>,
    u: Vec<u64>,
}

impl LinearSystem {
    /// `None` unless every row is `AN` canonical residues and there are as
    /// many rows as entries of `u`.
    pub fn new(a: Vec<Vec<u64>>, u: Vec<u64>) -> Option<Self> {
        if a.len() != u.len()
            || a.iter().any(|row| row.len() != AN)
            || a.iter().flatten().chain(u.iter()).any(|&v| v >= QTILDE)
        {
            return None;
        }
        Some(Self { a, u })
    }

    /// Number of constraints, which is how many `gamma` scalars are drawn.
    pub fn rows(&self) -> usize {
        self.u.len()
    }

    /// Row `k` of `A` — for the tests that state what a built system is.
    pub fn row(&self, k: usize) -> Option<&[u64]> {
        self.a.get(k).map(|r| r.as_slice())
    }

    /// Entry `k` of `u`.
    pub fn u_at(&self, k: usize) -> Option<u64> {
        self.u.get(k).copied()
    }
}

/// `phi` (length `AN`) and `<u, gamma>`, from the compression challenge.
///
/// `None` if `gamma` is not one scalar per row.
fn linear_terms(ulp: &LinearSystem, gamma: &[u64]) -> Option<(Vec<u64>, u64)> {
    if gamma.len() != ulp.rows() {
        return None;
    }
    // Each `A[k][i] · gamma[k]` is under `2^58` and there are `2d` rows, so
    // the accumulator is nowhere near `u128` — but it is well past `u64`,
    // which the reference never had to think about.
    let mut phi = vec![0u64; AN];
    for (i, slot) in phi.iter_mut().enumerate() {
        let mut acc: u128 = 0;
        for (k, &g) in gamma.iter().enumerate() {
            acc += ulp.a[k][i] as u128 * g as u128;
        }
        *slot = lr::reduce(LSPLIT as u128 * lr::reduce(acc) as u128);
    }
    let mut u_gamma: u128 = 0;
    for (k, &g) in gamma.iter().enumerate() {
        u_gamma += ulp.u[k] as u128 * g as u128;
    }
    Some((phi, lr::reduce(u_gamma)))
}

/// Element `e`'s block of `phi`, as slot-diagonal scalars.
fn phi_slice(phi: &[u64], element: usize) -> Option<Slots> {
    Slots::new(phi.get(element * LSPLIT..(element + 1) * LSPLIT)?)
}

/// Byte image of NTT-domain elements, for transcript hashing.
fn pack(elements: &[&NttPoly]) -> Vec<u8> {
    let mut out = Vec::with_capacity(elements.len() * DTILDE * 4);
    for el in elements {
        for &c in el.as_slice() {
            out.extend_from_slice(&(c as u32).to_le_bytes());
        }
    }
    out
}

/// Byte image of plain rows used in the transcript (`t_0,high` and the
/// recovered torus quotient of `w`).  Four bytes per value matches the
/// pre-compression NTT packing and the Python reference.
fn pack_rows<'a>(rows: impl IntoIterator<Item = &'a [u64]>) -> Vec<u8> {
    let mut out = Vec::new();
    for row in rows {
        out.reserve(row.len() * 4);
        for &c in row {
            out.extend_from_slice(&(c as u32).to_le_bytes());
        }
    }
    out
}

/// Equal-interval torus quotient used in the recoverable transcript.
#[inline]
fn recovery_high_coefficient(value: u64) -> u64 {
    // `value` comes from the secret mask on the prover path. Do not use
    // `(value * RECOVERY_BUCKETS) / QTILDE`: the crate promises that no
    // secret-derived value reaches a divide. The quotient is 11 bits, so
    // recover it with a fixed number of compare-and-mask steps instead.
    let scaled = value as u128 * RECOVERY_BUCKETS as u128;
    let mut quotient = 0u64;
    let mut bit = RECOVERY_BITS;
    while bit > 0 {
        bit -= 1;
        let place = 1u64 << bit;
        let candidate = quotient | place;
        let take = 0u64.wrapping_sub((scaled >= candidate as u128 * QTILDE as u128) as u64);
        quotient |= place & take;
    }
    quotient
}

fn recovery_high(poly: &CoeffPoly) -> Vec<u64> {
    poly.as_slice()
        .iter()
        .map(|&v| recovery_high_coefficient(v))
        .collect()
}

fn make_recovery_hint(target: &[Vec<u64>], base: &[Vec<u64>]) -> Option<Vec<Vec<i64>>> {
    if target.len() != N_TILDE || base.len() != N_TILDE {
        return None;
    }
    let mask = RECOVERY_BUCKETS - 1;
    target
        .iter()
        .zip(base)
        .map(|(want, have)| {
            if want.len() != DTILDE || have.len() != DTILDE {
                return None;
            }
            want.iter()
                .zip(have)
                .map(|(&a, &b)| match a.wrapping_sub(b) & mask {
                    0 => Some(0),
                    1 => Some(1),
                    x if x == mask => Some(-1),
                    _ => None,
                })
                .collect()
        })
        .collect()
}

fn use_recovery_hint(base: &[Vec<u64>], hint: &[Vec<i64>]) -> Option<Vec<Vec<u64>>> {
    if base.len() != N_TILDE || hint.len() != N_TILDE {
        return None;
    }
    let mask = RECOVERY_BUCKETS - 1;
    base.iter()
        .zip(hint)
        .map(|(have, carries)| {
            if have.len() != DTILDE || carries.len() != DTILDE {
                return None;
            }
            have.iter()
                .zip(carries)
                .map(|(&a, &h)| {
                    if !(-1..=1).contains(&h) {
                        None
                    } else {
                        Some((a.wrapping_add(h as u64)) & mask)
                    }
                })
                .collect()
        })
        .collect()
}

/// Where `alpha`, `gamma` and `c` come from.
///
/// The protocol is three-round: the verifier speaks after the first message
/// (`alpha`), after the product commitments (`gamma`), and after the linear
/// messages (`c`).  RiVeR needs `Pi_ex` non-interactive, so this binds each
/// challenge to the transcript so far by hashing it.
///
/// The transcript is bound in protocol order, so a challenge can never
/// depend on a message that follows it.  Deriving all three from a single
/// fixed stream instead — as a benchmark harness may do, since it only
/// needs prover and verifier to agree — would let the prover choose later
/// messages with the challenges already in hand, and is not a proof.
pub struct Challenges {
    transcript: Vec<Vec<u8>>,
}

impl Challenges {
    /// Start a transcript bound to `statement`.
    pub fn new(statement: &[u8]) -> Self {
        Self {
            transcript: vec![statement.to_vec()],
        }
    }

    /// Append prover messages, in protocol order.
    pub fn absorb(&mut self, parts: &[Vec<u8>]) {
        self.transcript.extend_from_slice(parts);
    }

    fn xof(&self, label: &[u8]) -> Xof {
        let parts: Vec<Part<'_>> = self
            .transcript
            .iter()
            .map(|t| Part::Bytes(t.as_slice()))
            .collect();
        Xof::new(&[DS_LANES, label].concat(), &parts)
    }

    /// The product proof's batching elements.
    pub fn alpha(&self, count: usize) -> Vec<NttPoly> {
        let mut x = self.xof(b".alpha");
        (0..count)
            .map(|_| lr::ntt(&sample_uniform_poly(&mut x)))
            .collect()
    }

    /// The linear system's compression challenge.
    pub fn gamma(&self, count: usize) -> Vec<u64> {
        let mut x = self.xof(b".gamma");
        (0..count).map(|_| uniform_int(&mut x, QTILDE)).collect()
    }

    /// The final ternary challenge `c`.
    pub fn challenge(&self) -> CoeffPoly {
        sample_challenge(&mut self.xof(b".c"))
    }
}

/// `LANES.Prove`.
///
/// `xof` supplies the prover's private randomness (`g`, `y`); `challenges`
/// supplies the verifier's.
///
/// `None` on any input the relation does not admit — a message that is not
/// `N_ex` slot vectors, a range outside them, or a slot the product proof
/// cannot form its cubic coefficients from.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    ck: &CommitmentKey,
    com_pub: &Commitment,
    com_sec: &CommitSecret,
    message_slots: &[Slots],
    ternary_slots: &[Vec<i64>],
    ulp: &LinearSystem,
    alpha_lo: usize,
    alpha_hi: usize,
    xof: &mut Xof,
    challenges: &mut Challenges,
) -> Option<LanesProof> {
    if message_slots.len() != N_EX
        || com_pub.t.len() != N_EX
        || com_pub.t0.len() != N_TILDE
        || !com_sec.is_well_formed()
    {
        return None;
    }
    let r_hat = com_sec.r_hat();

    // `g`: constant coefficient zero, committed in `t_{N+1}`.
    let mut g_coeff = vec![0u64; DTILDE];
    for slot in g_coeff.iter_mut().skip(1) {
        *slot = uniform_int(xof, QTILDE);
    }
    let g = lr::ntt(&CoeffPoly::new(&g_coeff)?);
    let t_g = ck.apply_b(B_G, r_hat)?.add(&g);

    let y = {
        let drawn = sample_gaussian_vec(xof, SIGMA_Y, RESPONSE_RANK);
        #[cfg(test)]
        {
            tests::maybe_scale_mask(drawn)
        }
        #[cfg(not(test))]
        {
            drawn
        }
    };
    let y_hat: Vec<NttPoly> = y.iter().map(lr::ntt).collect();
    let w = (0..N_TILDE)
        .map(|i| ck.apply_b0_tail(i, &y_hat))
        .collect::<Option<Vec<_>>>()?;
    let w_high: Vec<Vec<u64>> = w.iter().map(|p| recovery_high(&lr::intt(p))).collect();
    let b_y = (0..N_EX)
        .map(|i| ck.apply_b_tail(i, &y_hat))
        .collect::<Option<Vec<_>>>()?;

    challenges.absorb(&[
        pack_rows(com_pub.t0.iter().map(|p| p.as_slice())),
        pack(&com_pub.t.iter().collect::<Vec<_>>()),
        pack_rows(w_high.iter().map(Vec::as_slice)),
        pack(&[&t_g]),
    ]);
    let alpha = challenges.alpha(alpha_hi.checked_sub(alpha_lo)?);

    let ProductProof { t_mp1, t_mp2, v } = mp::prove(
        ck,
        ternary_slots,
        &b_y,
        r_hat,
        &y_hat,
        alpha_lo,
        alpha_hi,
        &alpha,
    )?;

    challenges.absorb(&[pack(&[&t_mp1, &t_mp2, &v])]);
    let gamma = challenges.gamma(ulp.rows());
    let (phi, u_gamma) = linear_terms(ulp, &gamma)?;

    // h = g + slotwise( sum_e phi_{e,j} m_{e,j} - <u, gamma> )
    let mut h = g.to_vec();
    for slot in 0..LSPLIT {
        let mut acc: u128 = 0;
        for (elem, m) in message_slots.iter().enumerate() {
            acc += phi[elem * LSPLIT + slot] as u128 * m.as_slice()[slot] as u128;
        }
        let idx = slot * SUBDEG;
        let raw = lr::reduce(h[idx] as u128 + acc);
        h[idx] = lr::reduce((raw + QTILDE - u_gamma) as u128);
    }
    let h = NttPoly::new(&h)?;

    // v' = <b_G, y> + sum_e phi_e · <b_e, y>
    let mut v_prime = ck.apply_b_tail(B_G, &y_hat)?;
    for (elem, by) in b_y.iter().enumerate() {
        v_prime = v_prime.add(&lr::scale_blocks(by, &phi_slice(&phi, elem)?));
    }

    challenges.absorb(&[pack(&[&h, &v_prime])]);
    let c = challenges.challenge();
    let z: Vec<CoeffPoly> = y
        .iter()
        .zip(&com_sec.r()[N_TILDE..])
        .map(|(yi, ri)| yi.add(&lr::mul(&c, ri)))
        .collect();

    let z_hat: Vec<NttPoly> = z.iter().map(lr::ntt).collect();
    let c_hat = lr::ntt(&c);
    let t0_base: Vec<NttPoly> = com_pub.t0.iter().map(|p| lr::ntt(&p.expand())).collect();
    let recovered_base = (0..N_TILDE)
        .map(|i| {
            let u = ck
                .apply_b0_tail(i, &z_hat)?
                .sub(&lr::ntt_mul(&c_hat, &t0_base[i]));
            Some(recovery_high(&lr::intt(&u)))
        })
        .collect::<Option<Vec<_>>>()?;
    let hint = make_recovery_hint(&w_high, &recovered_base)?;

    // The prover applies the verifier's own response bounds, and returns
    // bottom rather than a proof that will be rejected.
    //
    // This was missing: `prove` formed `z` and returned it while `verify`
    // enforced both bounds, so an out-of-bound mask produced a proof that
    // verified as `false` and could not even be serialized (`Coder::rice`
    // refuses a coefficient above its cap).  A prover that can return a
    // proof its own verifier rejects is a defect regardless of how rarely
    // it fires, and it does fire: `Z_INF_BOUND` is a `2^-128` tail bound,
    // not an impossibility.
    //
    // `None` is the existing contract for an exact-layer abort.
    // `RiVeR::eval` discards the whole attempt on it — OOM proof included,
    // because `W` is already bound into the OOM challenge — and retries
    // with fresh randomness.  Both bounds live in
    // [`response_within_bounds`] so the two sides cannot drift apart;
    // `river-py` does the same, in `lanes_proof.response_within_bounds`.
    if !response_within_bounds(&z) {
        return None;
    }

    // `w`, `v` and `v_prime` were computed above because the *transcript*
    // needs them; they are not transmitted, because `Ver` recovers them.
    let _ = (&w, &v, &v_prime);
    Some(LanesProof {
        t_g,
        t_mp1,
        t_mp2,
        h,
        c,
        hint,
        z,
    })
}

/// `LANES.Ver`.
///
/// Total on `com_pub` and `proof`: both come from a peer, so every shape is
/// checked before it is indexed and every failure is `false`.
pub fn verify(
    ck: &CommitmentKey,
    com_pub: &Commitment,
    proof: &LanesProof,
    ulp: &LinearSystem,
    alpha_lo: usize,
    alpha_hi: usize,
    challenges: &mut Challenges,
) -> bool {
    verify_inner(ck, com_pub, proof, ulp, alpha_lo, alpha_hi, challenges) == Some(true)
}

/// Both verifier bounds on the response, in one place.
///
/// [`prove`] calls it before returning and aborts if it fails; [`verify`]
/// calls it before hashing anything.  A single definition is what keeps an
/// honest proof from being rejected by its own verifier.
///
/// * per-coefficient: `|z_i| <= Z_INF_BOUND`, an artifact-derived decoder
///   and verifier cap;
/// * Euclidean: `||z||_2^2 < Z_NORM2_BOUND`, the paper's `2 s sqrt(N_z)`
///   rule at the transmitted rank.
///
/// The Euclidean comparison is **strict**: a response whose squared norm
/// equals the bound is rejected.  That is a choice, and it is the same
/// choice `river-py` makes, which is what matters for interoperability.
pub fn response_within_bounds(z: &[CoeffPoly]) -> bool {
    let mut norm_sq: i128 = 0;
    for poly in z {
        for coeff in poly.centered() {
            if coeff.abs() > Z_INF_BOUND {
                return false;
            }
            norm_sq += (coeff as i128) * (coeff as i128);
        }
    }
    norm_sq < Z_NORM2_BOUND
}

/// The body of [`verify`], with `?` standing in for the reference's blanket
/// `except (KeyError, TypeError, IndexError, ValueError)`.
#[allow(clippy::too_many_arguments)]
fn verify_inner(
    ck: &CommitmentKey,
    com_pub: &Commitment,
    proof: &LanesProof,
    ulp: &LinearSystem,
    alpha_lo: usize,
    alpha_hi: usize,
    challenges: &mut Challenges,
) -> Option<bool> {
    if proof.z.len() != RESPONSE_RANK
        || proof.hint.len() != N_TILDE
        || com_pub.t0.len() != N_TILDE
        || com_pub.t.len() != N_EX
    {
        return Some(false);
    }

    // Both response bounds, before anything is hashed.  Shared with
    // `prove`, which aborts rather than returning a proof this rejects.
    if !response_within_bounds(&proof.z) {
        return Some(false);
    }

    let z_hat: Vec<NttPoly> = proof.z.iter().map(lr::ntt).collect();
    let c_hat = lr::ntt(&proof.c);

    let t0_base: Vec<NttPoly> = com_pub.t0.iter().map(|p| lr::ntt(&p.expand())).collect();
    let w_base = (0..N_TILDE)
        .map(|i| {
            let u = ck
                .apply_b0_tail(i, &z_hat)?
                .sub(&lr::ntt_mul(&c_hat, &t0_base[i]));
            Some(recovery_high(&lr::intt(&u)))
        })
        .collect::<Option<Vec<_>>>()?;
    let w_high = use_recovery_hint(&w_base, &proof.hint)?;

    challenges.absorb(&[
        pack_rows(com_pub.t0.iter().map(|p| p.as_slice())),
        pack(&com_pub.t.iter().collect::<Vec<_>>()),
        pack_rows(w_high.iter().map(Vec::as_slice)),
        pack(&[&proof.t_g]),
    ]);
    let alpha = challenges.alpha(alpha_hi.checked_sub(alpha_lo)?);

    let b_z = (0..N_EX)
        .map(|i| ck.apply_b_tail(i, &z_hat))
        .collect::<Option<Vec<_>>>()?;
    let v = mp::recover_v(
        ck,
        &com_pub.t,
        &alpha,
        &proof.t_mp1,
        &proof.t_mp2,
        &c_hat,
        &z_hat,
        &b_z,
        alpha_lo,
        alpha_hi,
    )?;

    challenges.absorb(&[pack(&[&proof.t_mp1, &proof.t_mp2, &v])]);
    let gamma = challenges.gamma(ulp.rows());

    // The compressed linear relation: `h_0 == 0`.  This one is a genuine
    // check and stays — `h` is transmitted, and one scalar constraint does
    // not determine 128 coefficients.
    if lr::constant_coefficient(&proof.h) != 0 {
        return Some(false);
    }

    let (phi, u_gamma) = linear_terms(ulp, &gamma)?;

    // tau = slotwise(-<u, gamma>) + sum_e phi_e · t_e
    let neg = (QTILDE - u_gamma) % QTILDE;
    let mut tau = lr::slots_to_ntt(&Slots::new(&vec![neg; LSPLIT])?);
    for (elem, t) in com_pub.t.iter().enumerate() {
        tau = tau.add(&lr::scale_blocks(t, &phi_slice(&phi, elem)?));
    }

    let mut lhs = ck.apply_b_tail(B_G, &z_hat)?;
    for (elem, bz) in b_z.iter().enumerate() {
        lhs = lhs.add(&lr::scale_blocks(bz, &phi_slice(&phi, elem)?));
    }

    let inner = tau.add(&proof.t_g).sub(&proof.h);
    let v_prime = lhs.sub(&lr::ntt_mul(&c_hat, &inner));

    challenges.absorb(&[pack(&[&proof.h, &v_prime])]);
    Some(challenges.challenge() == proof.c)
}

#[cfg(test)]
mod tests {
    use super::super::commit::commit;
    use super::*;
    use crate::sample::DS_EXACT;

    const ALPHA_LO: usize = 2;
    const ALPHA_HI: usize = N_EX;

    // Test-only mask scaling, so `prove`'s abort path can be *driven*.
    //
    // The bounds `prove` checks are `2^-128` tail bounds: waiting for an
    // honest response to exceed one is not a test.  This multiplies the
    // drawn mask by a factor a test sets, which is the smallest hook that
    // makes the abort reachable without changing what `prove` computes
    // when the factor is 1 — and 1 is what every other test sees.
    //
    // `#[cfg(test)]` in `prove`, so production has no branch at all.
    // Thread-local rather than a parameter, because adding an argument to
    // `prove` for a test would put it in the port's public shape and in
    // `river-py`'s, which do not need it.
    thread_local! {
        static MASK_SCALE: std::cell::Cell<i64> = const { std::cell::Cell::new(1) };
    }

    pub(super) fn maybe_scale_mask(y: Vec<CoeffPoly>) -> Vec<CoeffPoly> {
        let k = MASK_SCALE.with(|c| c.get());
        if k == 1 {
            return y;
        }
        y.iter()
            .map(|p| {
                let scaled: Vec<i64> = p.centered().iter().map(|&c| c * k).collect();
                CoeffPoly::from_centered(&scaled).expect("scaled mask stays in (-q~, q~)")
            })
            .collect()
    }

    /// Run `f` with the prover's mask scaled by `k`.
    fn with_mask_scale<T>(k: i64, f: impl FnOnce() -> T) -> T {
        MASK_SCALE.with(|c| c.set(k));
        let out = f();
        MASK_SCALE.with(|c| c.set(1));
        out
    }

    fn lcg(seed: u64) -> impl FnMut(u64) -> u64 {
        let mut state = seed;
        move |m| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) % m
        }
    }

    /// A witness with ternary digits and one linear constraint per slot:
    /// element 0's slot `j` is the sum of the ternary slots at `j`.
    fn build(tamper: Option<&str>) -> (Vec<Slots>, Vec<Vec<i64>>, LinearSystem) {
        let mut next = lcg(17);
        let mut slots: Vec<Vec<i64>> = (0..N_EX)
            .map(|e| {
                (0..LSPLIT)
                    .map(|_| {
                        if (ALPHA_LO..ALPHA_HI).contains(&e) {
                            next(3) as i64 - 1
                        } else {
                            next(1000) as i64
                        }
                    })
                    .collect()
            })
            .collect();

        let mut a = vec![vec![0u64; AN]; LSPLIT];
        let u = vec![0u64; LSPLIT];
        for j in 0..LSPLIT {
            a[j][j] = 1;
            for e in ALPHA_LO..ALPHA_HI {
                a[j][e * LSPLIT + j] = QTILDE - 1; // -1
            }
            slots[0][j] = (ALPHA_LO..ALPHA_HI).map(|e| slots[e][j]).sum();
        }
        match tamper {
            Some("digit") => slots[ALPHA_LO][0] += 2,
            Some("linear") => slots[0][0] += 1,
            _ => {}
        }
        let msg: Vec<Slots> = slots
            .iter()
            .map(|s| Slots::from_centered(s).unwrap())
            .collect();
        (msg, slots, LinearSystem::new(a, u).unwrap())
    }

    fn run(tamper: Option<&str>, corrupt: Option<&dyn Fn(&mut LanesProof)>) -> bool {
        let ck = CommitmentKey::new(&[7u8; 32]);
        let (msg, slots, ulp) = build(tamper);
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"proof-test")]);
        let (pub_, sec) = commit(&ck, &msg, &mut xof).unwrap();
        let mut pi = prove(
            &ck,
            &pub_,
            &sec,
            &msg,
            &slots,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            &mut xof,
            &mut Challenges::new(b"selftest"),
        )
        .expect("honest prover");
        if let Some(f) = corrupt {
            f(&mut pi);
        }
        verify(
            &ck,
            &pub_,
            &pi,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            &mut Challenges::new(b"selftest"),
        )
    }

    #[test]
    fn an_honest_proof_verifies() {
        assert!(run(None, None), "honest proof rejected");
    }

    /// The fixed-step quotient agrees with exact division immediately around
    /// every interval boundary, including zero and the wrap at `q~`.
    #[test]
    fn recovery_quotient_is_exact_at_every_bucket_boundary() {
        for bucket in 0..=RECOVERY_BUCKETS {
            let boundary =
                (bucket as u128 * QTILDE as u128).div_ceil(RECOVERY_BUCKETS as u128) as u64;
            let first = boundary.saturating_sub(1).min(QTILDE - 1);
            let last = boundary.saturating_add(1).min(QTILDE - 1);
            for value in first..=last {
                let exact = (value as u128 * RECOVERY_BUCKETS as u128 / QTILDE as u128) as u64;
                assert_eq!(
                    recovery_high_coefficient(value),
                    exact,
                    "bucket {bucket}, value {value}"
                );
            }
        }
    }

    #[test]
    fn a_broken_witness_is_rejected() {
        assert!(!run(Some("digit"), None), "non-ternary digit accepted");
        assert!(
            !run(Some("linear"), None),
            "broken linear relation accepted"
        );
    }

    /// Every transmitted element is bound: moving one coefficient of any of
    /// them must break the proof.
    #[test]
    fn every_field_of_the_proof_is_bound() {
        fn bump(f: impl Fn(&mut LanesProof) -> &mut NttPoly) -> impl Fn(&mut LanesProof) {
            move |pi| {
                let el = f(pi);
                let mut v = el.to_vec();
                v[0] = (v[0] + 1) % QTILDE;
                *el = NttPoly::new(&v).unwrap();
            }
        }
        /// One named tampering of a transmitted element.
        type Tamper<'a> = (&'a str, &'a dyn Fn(&mut LanesProof));
        let fields: [Tamper<'_>; 4] = [
            ("h", &bump(|p| &mut p.h)),
            ("t_g", &bump(|p| &mut p.t_g)),
            ("t_mp1", &bump(|p| &mut p.t_mp1)),
            ("t_mp2", &bump(|p| &mut p.t_mp2)),
        ];
        for (name, f) in fields {
            assert!(!run(None, Some(f)), "tampered {name} accepted");
        }

        // `w`, `v` and `v'` are no longer transmitted, so there is nothing
        // to tamper with — what replaces those cases is that the recovered
        // values feed the transcript, so moving `c` (which every recovery
        // reads) has to break it
        assert!(
            !run(
                None,
                Some(&|pi: &mut LanesProof| {
                    let mut v = pi.c.to_vec();
                    v[0] = (v[0] + 1) % QTILDE;
                    pi.c = CoeffPoly::new(&v).unwrap();
                })
            ),
            "tampered c accepted"
        );
        assert!(
            !run(
                None,
                Some(&|pi: &mut LanesProof| {
                    pi.hint[0][0] = match pi.hint[0][0] {
                        -1 => 0,
                        0 => 1,
                        1 => 0,
                        _ => unreachable!("the prover emits a ternary hint"),
                    };
                })
            ),
            "tampered recovery hint accepted"
        );
        assert!(
            !run(
                None,
                Some(&|pi: &mut LanesProof| {
                    let mut v = pi.z[0].to_vec();
                    v[0] = (v[0] + 1) % QTILDE;
                    pi.z[0] = CoeffPoly::new(&v).unwrap();
                })
            ),
            "tampered z accepted"
        );
        // and an oversized `z` fails the norm check rather than the algebra
        assert!(
            !run(
                None,
                Some(&|pi: &mut LanesProof| {
                    let mut v = pi.z[0].to_vec();
                    v[0] = QTILDE / 2;
                    pi.z[0] = CoeffPoly::new(&v).unwrap();
                })
            ),
            "oversized z accepted"
        );
    }

    /// A proof must not verify against a different statement.
    #[test]
    fn the_statement_is_bound_into_every_challenge() {
        let ck = CommitmentKey::new(&[7u8; 32]);
        let (msg, slots, ulp) = build(None);
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"stmt")]);
        let (pub_, sec) = commit(&ck, &msg, &mut xof).unwrap();
        let pi = prove(
            &ck,
            &pub_,
            &sec,
            &msg,
            &slots,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            &mut xof,
            &mut Challenges::new(b"statement-A"),
        )
        .unwrap();
        assert!(verify(
            &ck,
            &pub_,
            &pi,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            &mut Challenges::new(b"statement-A")
        ));
        assert!(
            !verify(
                &ck,
                &pub_,
                &pi,
                &ulp,
                ALPHA_LO,
                ALPHA_HI,
                &mut Challenges::new(b"statement-B")
            ),
            "statement not bound"
        );
    }

    /// `Ver` is total on a peer's proof: no shape is a panic.
    #[test]
    fn verify_is_total_on_a_malformed_proof() {
        let ck = CommitmentKey::new(&[7u8; 32]);
        let (msg, slots, ulp) = build(None);
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"total")]);
        let (pub_, sec) = commit(&ck, &msg, &mut xof).unwrap();
        let good = prove(
            &ck,
            &pub_,
            &sec,
            &msg,
            &slots,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            &mut xof,
            &mut Challenges::new(b"total"),
        )
        .unwrap();

        let check = |pi: &LanesProof, com: &Commitment, lo, hi| {
            verify(
                ck_ref(&ck),
                com,
                pi,
                &ulp,
                lo,
                hi,
                &mut Challenges::new(b"total"),
            )
        };
        fn ck_ref(ck: &CommitmentKey) -> &CommitmentKey {
            ck
        }
        assert!(check(&good, &pub_, ALPHA_LO, ALPHA_HI));

        // Wrong-length response and recovery hint.
        for len in [0usize, RESPONSE_RANK - 1, RESPONSE_RANK + 1] {
            let mut bad = good.clone();
            bad.z = vec![CoeffPoly::zero(); len];
            assert!(!check(&bad, &pub_, ALPHA_LO, ALPHA_HI), "z len {len}");
        }
        for len in [0usize, N_TILDE - 1, N_TILDE + 1] {
            let mut bad = good.clone();
            bad.hint = vec![vec![0; DTILDE]; len];
            assert!(!check(&bad, &pub_, ALPHA_LO, ALPHA_HI), "hint rows {len}");
        }
        for len in [0usize, DTILDE - 1, DTILDE + 1] {
            let mut bad = good.clone();
            bad.hint[0] = vec![0; len];
            assert!(!check(&bad, &pub_, ALPHA_LO, ALPHA_HI), "hint len {len}");
        }
        for value in [2, -2, i64::MIN, i64::MAX] {
            let mut bad = good.clone();
            bad.hint[0][0] = value;
            assert!(
                !check(&bad, &pub_, ALPHA_LO, ALPHA_HI),
                "hint value {value}"
            );
        }
        // a malformed commitment
        for len in [0usize, N_TILDE - 1, N_TILDE + 1] {
            let mut com = pub_.clone();
            com.t0 = vec![super::super::commit::T0High::zero(); len];
            assert!(!check(&good, &com, ALPHA_LO, ALPHA_HI), "t0 len {len}");
        }
        for len in [0usize, N_EX - 1, N_EX + 1] {
            let mut com = pub_.clone();
            com.t = vec![NttPoly::zero(); len];
            assert!(!check(&good, &com, ALPHA_LO, ALPHA_HI), "t len {len}");
        }
        // and a range the message does not have
        for (lo, hi) in [(ALPHA_HI, ALPHA_LO), (0, N_EX + 1), (N_EX, N_EX + 3)] {
            assert!(!check(&good, &pub_, lo, hi), "range ({lo}, {hi})");
        }
    }

    /// A malformed `CommitSecret` is `None`, not a panic in the prover.
    ///
    /// `apply_b` rejects a short `r_hat`, so the only unguarded index was
    /// `r[i]` in `z = y + c r` — reachable with a full-length `r_hat` and a
    /// short `r`.  `CommitSecret` is opaque now, so this needs the
    /// test-only constructor to reach the guard at all, which is the point.
    #[test]
    fn a_ragged_commitment_secret_is_refused_rather_than_indexed() {
        let ck = CommitmentKey::new(&[7u8; 32]);
        let (msg, slots, ulp) = build(None);
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"ragged")]);
        let (pub_, _) = commit(&ck, &msg, &mut xof).unwrap();
        let bad = super::super::commit::CommitSecret::ragged();
        assert!(!bad.is_well_formed());
        assert!(prove(
            &ck,
            &pub_,
            &bad,
            &msg,
            &slots,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            &mut xof,
            &mut Challenges::new(b"ragged"),
        )
        .is_none());
    }

    /// The linear system validates its own shape.
    #[test]
    fn a_malformed_linear_system_is_refused_at_construction() {
        assert!(LinearSystem::new(vec![vec![0u64; AN]; 4], vec![0u64; 4]).is_some());
        assert!(LinearSystem::new(vec![vec![0u64; AN]; 4], vec![0u64; 3]).is_none());
        assert!(LinearSystem::new(vec![vec![0u64; AN - 1]], vec![0u64; 1]).is_none());
        assert!(LinearSystem::new(vec![vec![QTILDE; AN]], vec![0u64; 1]).is_none());
        assert!(LinearSystem::new(vec![vec![0u64; AN]], vec![QTILDE]).is_none());
        assert!(LinearSystem::new(vec![], vec![]).is_some());

        // and `linear_terms` needs one `gamma` per row
        let ulp = LinearSystem::new(vec![vec![1u64; AN]; 3], vec![1u64; 3]).unwrap();
        assert!(linear_terms(&ulp, &[1, 2, 3]).is_some());
        assert!(linear_terms(&ulp, &[1, 2]).is_none());
        assert!(linear_terms(&ulp, &[1, 2, 3, 4]).is_none());
    }

    /// `phi` carries the compensating factor `l` the slot map needs.
    ///
    /// `constant_coefficient(slots_to_ntt(v)) = (sum_j v_j) / l`, so
    /// without the `l` in `linear_terms` the compressed relation would be
    /// off by that inverse — and would still be *consistent* between prover
    /// and verifier, which is why it needs stating rather than testing by
    /// round trip.
    #[test]
    fn phi_cancels_the_slot_maps_inverse_l() {
        let mut a = vec![vec![0u64; AN]; 1];
        a[0][5] = 3;
        let ulp = LinearSystem::new(a, vec![7]).unwrap();
        let (phi, u_gamma) = linear_terms(&ulp, &[11]).unwrap();
        assert_eq!(phi[5], (LSPLIT as u64 * 3 * 11) % QTILDE);
        assert_eq!(u_gamma, 7 * 11 % QTILDE);
        assert!(phi.iter().enumerate().all(|(i, &v)| i == 5 || v == 0));

        // and the map really does divide by `l`
        let mut v = vec![0u64; LSPLIT];
        v[0] = LSPLIT as u64;
        let hat = lr::slots_to_ntt(&Slots::new(&v).unwrap());
        assert_eq!(lr::constant_coefficient(&hat), 1);
    }

    /// The prover returns bottom rather than a proof `verify` would reject.
    ///
    /// `prove` used to form `z` and return it while `verify` enforced both
    /// bounds, so an out-of-bound mask produced a proof that verified as
    /// `false` and could not even be serialized.  Both bounds now live in
    /// [`response_within_bounds`], `prove` calls it, and `None` is the
    /// existing exact-layer abort contract — `RiVeR::eval` discards the
    /// whole attempt on it.
    ///
    /// **This drives `prove` itself.**  An earlier version of this test
    /// called `prove` once for an honest response and then fed fabricated
    /// bad responses straight to the helper, which meant deleting the
    /// prover-side call left it green — it tested the predicate, not the
    /// regression.  The mask is now scaled past the bound through
    /// [`with_mask_scale`], and what is asserted is that `prove` returns
    /// `None`.
    #[test]
    fn prove_returns_bottom_rather_than_a_proof_verify_would_reject() {
        let ck = CommitmentKey::new(&[7u8; 32]);
        let (msg, slots, ulp) = build(None);

        let run = |scale: i64| -> Option<LanesProof> {
            let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"abort-test")]);
            let (pub_, sec) = commit(&ck, &msg, &mut xof).unwrap();
            with_mask_scale(scale, || {
                prove(
                    &ck,
                    &pub_,
                    &sec,
                    &msg,
                    &slots,
                    &ulp,
                    ALPHA_LO,
                    ALPHA_HI,
                    &mut xof,
                    &mut Challenges::new(b"abort"),
                )
            })
        };

        // Unscaled: an honest proof, within both bounds.
        let honest = run(1).expect("honest prover returns a proof");
        assert!(response_within_bounds(&honest.z));

        // Scaled past the bound: `prove` must return bottom.  Not `verify`
        // returning false, and not a serialization error — bottom, which is
        // what `RiVeR::eval` restarts on.
        assert!(
            run(64).is_none(),
            "prove returned a proof its own verifier rejects"
        );

        // `response_within_bounds` is not vacuous at this magnitude —
        // otherwise the assertion above would pass for the wrong reason.
        // This is an *independent* out-of-bounds response, not the one the
        // hook produced: `with_mask_scale` scales the mask alone before
        // `c · r` is added, so the vector built here is a different one.
        // What it establishes is the predicate's side of the test — that a
        // response of roughly this size is rejected — which is what makes
        // `run(64).is_none()` above attributable to the bound rather than
        // to some unrelated failure inside `prove`.
        let scaled: Vec<CoeffPoly> = honest
            .z
            .iter()
            .map(|p| {
                let big: Vec<i64> = p.centered().iter().map(|&c| c * 64).collect();
                CoeffPoly::from_centered(&big).expect("stays in (-q~, q~)")
            })
            .collect();
        assert!(!response_within_bounds(&scaled));
    }

    /// The two bounds, at and either side of the boundary.
    ///
    /// Separate from the abort test above because these are properties of
    /// the predicate rather than of `prove`, and because the exact
    /// Euclidean equality case has to be *constructed* — no mask scaling
    /// lands on it.
    #[test]
    fn the_response_bounds_are_exact_at_the_boundary() {
        // Infinity: `>` rejects, so the bound itself is admissible.  Built
        // from an otherwise-zero response so the Euclidean test cannot be
        // what decides either case.
        let zero = vec![CoeffPoly::from_centered(&vec![0i64; DTILDE]).unwrap(); RESPONSE_RANK];
        assert!(response_within_bounds(&zero));

        let mut at = zero.clone();
        let mut c = vec![0i64; DTILDE];
        c[0] = Z_INF_BOUND;
        at[0] = CoeffPoly::from_centered(&c).unwrap();
        assert!(
            (Z_INF_BOUND as i128) * (Z_INF_BOUND as i128) < Z_NORM2_BOUND,
            "the single coefficient must not trip the Euclidean test"
        );
        assert!(response_within_bounds(&at), "|z_i| == bound must pass");

        let mut over = zero.clone();
        c[0] = Z_INF_BOUND + 1;
        over[0] = CoeffPoly::from_centered(&c).unwrap();
        assert!(!response_within_bounds(&over), "|z_i| == bound+1 must fail");
        c[0] = -(Z_INF_BOUND + 1);
        over[0] = CoeffPoly::from_centered(&c).unwrap();
        assert!(!response_within_bounds(&over), "the negative side too");

        // Euclidean: `<` rejects equality.  Constructed to land exactly on
        // the bound, spread so no coefficient trips the infinity check.
        let mut rows = vec![vec![0i64; DTILDE]; RESPONSE_RANK];
        let mut left = Z_NORM2_BOUND;
        'fill: for row in rows.iter_mut() {
            for slot in row.iter_mut() {
                if left <= 0 {
                    break 'fill;
                }
                let take = Z_INF_BOUND.min(isqrt_i128(left));
                *slot = take;
                left -= (take as i128) * (take as i128);
            }
        }
        if left > 0 {
            let take = isqrt_i128(left);
            if (take as i128) * (take as i128) == left {
                'top: for row in rows.iter_mut() {
                    for slot in row.iter_mut() {
                        if *slot == 0 {
                            *slot = take;
                            left = 0;
                            break 'top;
                        }
                    }
                }
            }
        }
        assert_eq!(left, 0, "could not land exactly on the bound");

        let exact: Vec<CoeffPoly> = rows
            .iter()
            .map(|r| CoeffPoly::from_centered(r).expect("inside (-q~, q~)"))
            .collect();
        let n2: i128 = exact
            .iter()
            .flat_map(|p| p.centered())
            .map(|v| (v as i128) * (v as i128))
            .sum();
        assert_eq!(n2, Z_NORM2_BOUND);
        assert!(
            !response_within_bounds(&exact),
            "equality must be rejected: both sides must agree on <"
        );

        // ...and one unit below it passes, so the rejection is the
        // equality and not the neighbourhood.
        let mut below = rows.clone();
        'drop: for row in below.iter_mut() {
            for slot in row.iter_mut() {
                if *slot > 0 {
                    let was = (*slot as i128) * (*slot as i128);
                    *slot -= 1;
                    let now = (*slot as i128) * (*slot as i128);
                    assert!(n2 - was + now < Z_NORM2_BOUND);
                    break 'drop;
                }
            }
        }
        let under: Vec<CoeffPoly> = below
            .iter()
            .map(|r| CoeffPoly::from_centered(r).expect("inside (-q~, q~)"))
            .collect();
        assert!(response_within_bounds(&under));
    }

    /// Integer square root of a non-negative `i128`.
    fn isqrt_i128(v: i128) -> i64 {
        if v <= 0 {
            return 0;
        }
        let mut r = (v as f64).sqrt() as i128;
        while r > 0 && r * r > v {
            r -= 1;
        }
        while (r + 1) * (r + 1) <= v {
            r += 1;
        }
        r as i64
    }
}
