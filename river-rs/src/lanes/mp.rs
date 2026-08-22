//! The cubic product proof — port of `river-py/lanes_mp.py`.
//!
//! The product proof of `[ALS20]`, in the form `[ENS20]` Figure 3 uses it.
//!
//! ## What it proves
//!
//! For every slot `m` of the selected message elements, `m(m+1)(m-1) = 0`,
//! i.e. `m in {-1, 0, 1}`.  That is the whole reason RiVeR's rounding-error
//! range is encoded in ternary digits: the range proof reduces to this.
//!
//! ## How
//!
//! With `f = <b_i, z> - c t_i = b_y - c m` — the masked opening of slot `m`
//! — the identity
//!
//! ```text
//! f^3 - c^2 f = b_y^3 - 3c m b_y^2 + c^2 (3m^2 - 1) b_y - c^3 (m^3 - m)
//! ```
//!
//! has a last term that vanishes exactly when `m^3 = m`.  The prover
//! commits to the two intermediate coefficients (`t_{N+2}`, `t_{N+3}`)
//! before seeing `c`, and the verifier recombines them; the check passes iff
//! the `m^3 - m` term is zero.  `alpha` is a random ring element per
//! element, batching the slots.
//!
//! ## The two witness coefficients do not divide
//!
//! The reference forms `3m mod q~` and `(3m^2 - 1) mod q~` with `%`, on the
//! *digits* — which are secret.  Here they go through
//! [`Slots::from_centered`], a masked conditional add, which is exact on the
//! whole alphabet the product proof is about: `3m in {-3, 0, 3}` and
//! `3m^2 - 1 in {-1, 2}`.  [`cubic_coefficients`] returns `None` rather than
//! reducing a slot so far outside that alphabet that the coefficient leaves
//! `(-q~, q~)`, which is a witness no prover should be holding.

use super::commit::{CommitmentKey, B_MP1, B_MP2};
use super::ring::{self as lr, NttPoly, Slots, LSPLIT, QTILDE};

/// `(t_{N+2}, t_{N+3}, v)`, all NTT domain.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ProductProof {
    pub t_mp1: NttPoly,
    pub t_mp2: NttPoly,
    pub v: NttPoly,
}

/// `(3m, 3m^2 - 1)` per slot, as canonical residues.
///
/// `None` unless `slots` is exactly `l` values whose coefficients stay
/// inside `(-q~, q~)`, which every ternary witness does by a factor of
/// `10^8`.
fn cubic_coefficients(slots: &[i64]) -> Option<(Slots, Slots)> {
    if slots.len() != LSPLIT {
        return None;
    }
    let limit = QTILDE as i128;
    let mut three_m = vec![0i64; LSPLIT];
    let mut quad = vec![0i64; LSPLIT];
    for (i, &m) in slots.iter().enumerate() {
        // `3 m^2` leaves `i128` at `m = i64::MIN`, so the guard has to be
        // the arithmetic's rather than a range test after the fact.
        let m = m as i128;
        let a = m.checked_mul(3)?;
        let b = m.checked_mul(m)?.checked_mul(3)?.checked_sub(1)?;
        if a <= -limit || a >= limit || b <= -limit || b >= limit {
            return None;
        }
        three_m[i] = a as i64;
        quad[i] = b as i64;
    }
    Some((
        Slots::from_centered(&three_m)?,
        Slots::from_centered(&quad)?,
    ))
}

/// The range `[alpha_lo, alpha_hi)` against the vectors it indexes.
fn range_ok(lo: usize, hi: usize, alpha: &[NttPoly], bound: usize) -> bool {
    lo <= hi && hi <= bound && alpha.len() == hi - lo
}

/// Produce `(t_{N+2}, t_{N+3}, v)`.
///
/// `ternary_slots` maps element index to its `l` slot values as centred
/// integers; only elements in `[alpha_lo, alpha_hi)` are proved, so the
/// entries outside that range are never read and may be anything.  `alpha`
/// is supplied by the caller so it can be Fiat–Shamir derived.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    ck: &CommitmentKey,
    ternary_slots: &[Vec<i64>],
    b_y: &[NttPoly],
    r_hat: &[NttPoly],
    y_hat: &[NttPoly],
    alpha_lo: usize,
    alpha_hi: usize,
    alpha: &[NttPoly],
) -> Option<ProductProof> {
    if !range_ok(
        alpha_lo,
        alpha_hi,
        alpha,
        ternary_slots.len().min(b_y.len()),
    ) {
        return None;
    }
    let mut t_mp1 = ck.apply_b(B_MP1, r_hat)?;
    let mut t_mp2 = ck.apply_b(B_MP2, r_hat)?;
    let mut v = ck.apply_b_tail(B_MP1, y_hat)?;

    for idx in alpha_lo..alpha_hi {
        let a = &alpha[idx - alpha_lo];
        let by = &b_y[idx];
        let (three_m, quad) = cubic_coefficients(&ternary_slots[idx])?;

        let a_by = lr::ntt_mul(a, by); // alpha b_y
        let a_by2 = lr::ntt_mul(&a_by, by); // alpha b_y^2
        let a_by3 = lr::ntt_mul(&a_by2, by); // alpha b_y^3

        // t_{N+2} -= sum_j 3 m_j · (alpha b_y^2)|_j
        t_mp1 = t_mp1.sub(&lr::scale_blocks(&a_by2, &three_m));
        // t_{N+3} += sum_j (3 m_j^2 - 1) · (alpha b_y)|_j
        t_mp2 = t_mp2.add(&lr::scale_blocks(&a_by, &quad));
        v = v.add(&a_by3);
    }

    t_mp1 = t_mp1.add(&ck.apply_b_tail(B_MP2, y_hat)?);
    Some(ProductProof { t_mp1, t_mp2, v })
}

/// The value `v` the cubic check equates to, recovered from the rest.
///
/// This used to *compare* against a transmitted `v` and return a bit.  `v`
/// is a check target: it appears in one equation, alone, so that equation
/// determines it and transmitting it is transmitting what the verifier can
/// compute.  [`super::proof::verify`] recovers it here, feeds it back into
/// the transcript in the position the prover put it, and lets `c' == c`
/// decide — which is the same test, since an adversary who moves `v` moves
/// the transcript.
///
/// `None` on any shape the reference would have indexed past: `z_hat`,
/// `b_z` and the two commitments all come from a peer in `Ver`.
#[allow(clippy::too_many_arguments)]
pub fn recover_v(
    ck: &CommitmentKey,
    com_t: &[NttPoly],
    alpha: &[NttPoly],
    t_mp1: &NttPoly,
    t_mp2: &NttPoly,
    c_hat: &NttPoly,
    z_hat: &[NttPoly],
    b_z: &[NttPoly],
    alpha_lo: usize,
    alpha_hi: usize,
) -> Option<NttPoly> {
    if !range_ok(alpha_lo, alpha_hi, alpha, com_t.len().min(b_z.len())) {
        return None;
    }
    let b_mp2_z = ck.apply_b_tail(B_MP2, z_hat)?;
    // f_{N+3} = <b_{N+2}, z> - c t_{N+3}
    let f_mp2 = b_mp2_z.sub(&lr::ntt_mul(c_hat, t_mp2));
    let mut total = lr::ntt_mul(c_hat, &f_mp2);

    for idx in alpha_lo..alpha_hi {
        let f = b_z[idx].sub(&lr::ntt_mul(c_hat, &com_t[idx]));
        let term = lr::ntt_mul(&f, &f.add(c_hat));
        let term = lr::ntt_mul(&term, &f.sub(c_hat));
        total = total.add(&lr::ntt_mul(&alpha[idx - alpha_lo], &term));
    }

    // + <b_{N+1}, z> - c t_{N+2}
    total = total.add(&ck.apply_b_tail(B_MP1, z_hat)?);
    Some(total.sub(&lr::ntt_mul(c_hat, t_mp1)))
}

/// `recover_v(...) == v`, for this module's own tests.
#[allow(clippy::too_many_arguments)]
pub fn verify(
    ck: &CommitmentKey,
    com_t: &[NttPoly],
    alpha: &[NttPoly],
    pi: &ProductProof,
    c_hat: &NttPoly,
    z_hat: &[NttPoly],
    b_z: &[NttPoly],
    alpha_lo: usize,
    alpha_hi: usize,
) -> bool {
    recover_v(
        ck, com_t, alpha, &pi.t_mp1, &pi.t_mp2, c_hat, z_hat, b_z, alpha_lo, alpha_hi,
    )
    .is_some_and(|v| v == pi.v)
}

#[cfg(test)]
mod tests {
    use super::super::commit::{commit, CommitmentKey};
    use super::super::params::{
        sample_challenge, sample_gaussian_vec, sample_uniform_poly, KAPPA, N_EX, N_TILDE,
        RESPONSE_RANK, SIGMA_Y,
    };
    use super::*;
    use crate::sample::{Part, Xof, DS_EXACT};

    const ALPHA_LO: usize = 2;
    const ALPHA_HI: usize = N_EX;

    /// One honest run of the sub-protocol, with `ternary` placed at the
    /// proved elements and zeros elsewhere.
    fn run(ternary: &[Vec<i64>]) -> bool {
        let ck = CommitmentKey::new(&[3u8; 32]);
        let mut slots = vec![vec![0i64; LSPLIT]; N_EX];
        for (i, row) in ternary.iter().enumerate() {
            slots[ALPHA_LO + i] = row.clone();
        }
        let msg: Vec<Slots> = slots
            .iter()
            .map(|s| Slots::from_centered(s).unwrap())
            .collect();

        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"mp-test")]);
        let (pub_, sec) = commit(&ck, &msg, &mut xof).unwrap();
        let y = sample_gaussian_vec(&mut xof, SIGMA_Y, RESPONSE_RANK);
        let y_hat: Vec<NttPoly> = y.iter().map(lr::ntt).collect();
        let b_y: Vec<NttPoly> = (0..N_EX)
            .map(|i| ck.apply_b_tail(i, &y_hat).unwrap())
            .collect();

        let alpha: Vec<NttPoly> = (0..ALPHA_HI - ALPHA_LO)
            .map(|_| lr::ntt(&sample_uniform_poly(&mut xof)))
            .collect();
        let pi = prove(
            &ck,
            &slots,
            &b_y,
            sec.r_hat(),
            &y_hat,
            ALPHA_LO,
            ALPHA_HI,
            &alpha,
        )
        .expect("honest prover");

        let c = sample_challenge(&mut xof);
        let c_hat = lr::ntt(&c);
        let z: Vec<_> = (0..RESPONSE_RANK)
            .map(|i| lr::mul(&c, &sec.r()[N_TILDE + i]).add(&y[i]))
            .collect();
        let z_hat: Vec<NttPoly> = z.iter().map(lr::ntt).collect();
        let b_z: Vec<NttPoly> = (0..N_EX)
            .map(|i| ck.apply_b_tail(i, &z_hat).unwrap())
            .collect();

        verify(
            &ck, &pub_.t, &alpha, &pi, &c_hat, &z_hat, &b_z, ALPHA_LO, ALPHA_HI,
        )
    }

    fn good_witness() -> Vec<Vec<i64>> {
        // deterministic ternary, spread over all three values
        (0..ALPHA_HI - ALPHA_LO)
            .map(|e| {
                (0..LSPLIT)
                    .map(|j| ((e * 5 + j * 3) % 3) as i64 - 1)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_ternary_witness_is_accepted() {
        assert!(run(&good_witness()), "honest ternary witness rejected");
    }

    /// Every non-ternary slot value is rejected, at every position.
    ///
    /// This is the clause the whole range proof rests on, so it is tested
    /// as a range rather than at one point.
    #[test]
    fn a_non_ternary_slot_is_rejected() {
        for bad_value in [2i64, -2, 3, -3, 4, 100] {
            for &(elem, slot) in &[(0usize, 0usize), (1, 7), (3, LSPLIT - 1)] {
                let mut bad = good_witness();
                bad[elem][slot] = bad_value;
                assert!(
                    !run(&bad),
                    "slot value {bad_value} accepted at element {elem}, slot {slot}"
                );
            }
        }
    }

    /// `cubic_coefficients` is the identity the proof is built on.
    #[test]
    fn the_cubic_coefficients_are_three_m_and_three_m_squared_minus_one() {
        let cent: Vec<i64> = (0..LSPLIT).map(|j| (j % 3) as i64 - 1).collect();
        let (three_m, quad) = cubic_coefficients(&cent).unwrap();
        for (j, &m) in cent.iter().enumerate() {
            let want_a = (3 * m).rem_euclid(QTILDE as i64) as u64;
            let want_b = (3 * m * m - 1).rem_euclid(QTILDE as i64) as u64;
            assert_eq!(three_m.as_slice()[j], want_a, "3m at {j}");
            assert_eq!(quad.as_slice()[j], want_b, "3m^2-1 at {j}");
        }
        // total: a wrong length, and a slot that would need a reduction
        assert!(cubic_coefficients(&cent[..LSPLIT - 1]).is_none());
        assert!(cubic_coefficients(&vec![i64::MAX; LSPLIT]).is_none());
        assert!(cubic_coefficients(&vec![i64::MIN; LSPLIT]).is_none());
        assert!(cubic_coefficients(&vec![QTILDE as i64; LSPLIT]).is_none());
        // The *quadratic* coefficient is the binding one — `3m` clears
        // `q~` only past `10^8` while `3m^2 - 1` does so past `10^4` — so
        // the admissible range is `|m| <= sqrt((q~ + 1)/3)`, four orders
        // of magnitude wider than the alphabet the proof is about.
        let edge = (0..)
            .take_while(|&m: &i64| 3 * m * m - 1 < QTILDE as i64)
            .last()
            .unwrap();
        // Derived, not pinned: it is `floor(sqrt((q~ + 1)/3))`, and `q~`
        // moved from 29 bits to 26 in the paper, so a literal
        // here would have to be re-typed for every modulus.  What is
        // pinned is the claim the comment above makes — that the
        // admissible range is orders of magnitude wider than `{-1,0,1}`.
        assert_eq!(edge, (((QTILDE as f64 + 1.0) / 3.0).sqrt()) as i64);
        assert!(edge > 1000, "the ternary alphabet has no room: {edge}");
        assert!(cubic_coefficients(&vec![edge; LSPLIT]).is_some());
        assert!(cubic_coefficients(&vec![-edge; LSPLIT]).is_some());
        assert!(cubic_coefficients(&vec![edge + 1; LSPLIT]).is_none());
        assert!(cubic_coefficients(&vec![-edge - 1; LSPLIT]).is_none());
    }

    /// Every wrong shape is `None`/`false`, never an index past the end.
    #[test]
    fn a_wrong_range_is_refused() {
        let ck = CommitmentKey::new(&[3u8; 32]);
        let slots = vec![vec![0i64; LSPLIT]; N_EX];
        let r_hat = vec![NttPoly::zero(); KAPPA];
        let y_hat = vec![NttPoly::zero(); RESPONSE_RANK];
        let b_y = vec![NttPoly::zero(); N_EX];
        let alpha = vec![NttPoly::zero(); 2];

        // alpha's length must match the range
        assert!(prove(&ck, &slots, &b_y, &r_hat, &y_hat, 2, 4, &alpha).is_some());
        assert!(prove(&ck, &slots, &b_y, &r_hat, &y_hat, 2, 5, &alpha).is_none());
        assert!(prove(&ck, &slots, &b_y, &r_hat, &y_hat, 2, 3, &alpha).is_none());
        // and the range must sit inside the vectors it indexes
        assert!(prove(&ck, &slots, &b_y, &r_hat, &y_hat, N_EX, N_EX + 2, &alpha).is_none());
        assert!(prove(&ck, &slots, &b_y, &r_hat, &y_hat, 4, 2, &alpha).is_none());
        // a short randomness vector never reaches the key
        let short = vec![NttPoly::zero(); KAPPA - 1];
        assert!(prove(&ck, &slots, &b_y, &short, &y_hat, 2, 4, &alpha).is_none());
        let short_y = vec![NttPoly::zero(); RESPONSE_RANK - 1];
        assert!(prove(&ck, &slots, &b_y, &r_hat, &short_y, 2, 4, &alpha).is_none());

        let pi = ProductProof {
            t_mp1: NttPoly::zero(),
            t_mp2: NttPoly::zero(),
            v: NttPoly::zero(),
        };
        let com_t = vec![NttPoly::zero(); N_EX];
        let c_hat = NttPoly::zero();
        for (lo, hi) in [(2usize, 5usize), (2, 3), (N_EX, N_EX + 2), (4, 2)] {
            assert!(
                !verify(&ck, &com_t, &alpha, &pi, &c_hat, &y_hat, &com_t, lo, hi),
                "range ({lo}, {hi}) accepted"
            );
        }
        assert!(!verify(
            &ck, &com_t, &alpha, &pi, &c_hat, &short, &com_t, 2, 4
        ));
    }
}
