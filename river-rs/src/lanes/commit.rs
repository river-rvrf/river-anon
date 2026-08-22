//! The `[BDLOP18]` commitment LANES uses — port of
//! `river-py/lanes_commit.py`.
//!
//! The commitment of `[BDLOP18]` at the shape `[ENS20]` Figure 3 wants,
//! exploiting the structure of the key.
//!
//! ## Shape
//!
//! The randomness `r` has `kappa = n~ + l~ + N_ex + 3 = 24` ring elements,
//! split into three roles:
//!
//! ```text
//! r[0 .. n~)                 the identity block of B_0        (7)
//! r[n~ .. n~ + N_ex + 3)     one per committed element        (9)
//! r[kappa - l~ .. kappa)     the shared random tail           (8)
//! ```
//!
//! `B_0` is `[I_{n~} | random]` and each `b_i` is a unit vector at column
//! `n~ + i` plus a random tail in the last `l~` columns.  The identity
//! entries are never materialised: [`CommitmentKey::apply_b0`] adds `r[i]`
//! directly and [`CommitmentKey::apply_b`] adds `r[n~ + i]`, so only the
//! random blocks are stored.
//!
//! This is the latest export's role assignment.  Its unequal `(n~, l~) =
//! (7, 8)` makes the distinction observable: the response rank is
//! `l~ + N_ex + 3 = 17`.
//!
//! Everything lives in the NTT domain; the message occupies one scalar per
//! NTT block ([`super::ring::slots_to_ntt`]). The exception is public `t_0`:
//! its full NTT image is converted back to coefficients and only the high
//! part after `D = 17` power-of-two rounding is transmitted. The opening
//! randomness remains rank 24; only the masked response later drops the
//! seven identity columns.
//!
//! ## Why the two accessors return `Option`
//!
//! `apply_b` is applied to three different vectors — the commitment
//! randomness `r`, the prover's mask `y`, and the *response* `z`, which in
//! `Ver` arrives from a peer.  The reference indexes and slices, so a short
//! `z` is an `IndexError` caught by the blanket `except` in
//! `lanes_proof.verify`; here the length is a precondition of the inner
//! product and the type says so.  A wrong-length `z` is a malformed proof,
//! not a panic and not an inner product over a prefix.

use super::params::{
    sample_gaussian_vec, sample_uniform_poly, t0_power2round, AUX, ELL_TILDE, KAPPA, N_EX, N_TILDE,
    RESPONSE_RANK, SIGMA_R, T0_HIGH_MODULUS, T0_SCALE,
};
use super::ring::{self as lr, CoeffPoly, NttPoly, Slots};
use crate::sample::{Part, Xof, DS_EXACT};

/// The `b`-row carrying the masking element `g` — `t_{N+1}`.
pub const B_G: usize = N_EX;
/// First product-proof commitment — `t_{N+2}`.
pub const B_MP1: usize = N_EX + 1;
/// Second product-proof commitment — `t_{N+3}`.
pub const B_MP2: usize = N_EX + 2;
/// Rows of the `b` block: one per message element, plus the three aux.
pub const B_ROWS: usize = N_EX + AUX;

/// Columns of the stored `B_0` block: everything but the identity.
// **GATED, and label-sensitive.**  This module labels `B_0`'s row count and
// identity width `N_TILDE`, and the per-`b_i` tail width `ELL_TILDE`,
// which is the *opposite* of `crate::exact::rank_roles`: there the
// identity rank is `l~` and the shared tail is `n~`.  The two readings
// coincide only because `n~ = l~ = 4` at the current profile, so nothing
// here evaluates differently and no test can tell them apart — which is
// precisely how such labels get reversed.
//
// Picking a direction is not something this file can validate while the
// layer is gated, so instead the build fails if the ranks ever separate.
// At that point the labelling is a decision to make with the LANES
// manifest in hand, not a rename made on a hunch.
const _: () = assert!(
    N_TILDE == ELL_TILDE,
    "n~ != l~: this module's rank labelling is the opposite of \
     exact::rank_roles and has to be reconciled before it can run"
);

const B0_COLS: usize = KAPPA - N_TILDE;

/// `(B_0, b_0 .. b_8)`, stored as their random blocks only.
pub struct CommitmentKey {
    /// `B_0`'s random block: `n~ x (kappa - n~)`.
    b0: Vec<Vec<NttPoly>>,
    /// `b_i`'s random block: `B_ROWS x l~`, over the last `l~` columns.
    b: Vec<Vec<NttPoly>>,
}

impl CommitmentKey {
    /// Expand the key from `seed`.
    ///
    /// The draw order is the reference's — all of `B_0` row-major, then all
    /// of `b` row-major, off one XOF — because that order *is* the key.
    pub fn new(seed: &[u8]) -> Self {
        let mut xof = Xof::new(&[DS_EXACT, b".lanes.gen"].concat(), &[Part::Bytes(seed)]);
        let b0 = (0..N_TILDE)
            .map(|_| {
                (0..B0_COLS)
                    .map(|_| lr::ntt(&sample_uniform_poly(&mut xof)))
                    .collect()
            })
            .collect();
        let b = (0..B_ROWS)
            .map(|_| {
                (0..ELL_TILDE)
                    .map(|_| lr::ntt(&sample_uniform_poly(&mut xof)))
                    .collect()
            })
            .collect();
        Self { b0, b }
    }

    /// Row `row` of `B_0 r`: `r[row] + sum_j B0[row][j] r[n~ + j]`.
    ///
    /// `None` if `row` is not a row of `B_0` or `r_hat` is not `kappa`
    /// elements.
    pub fn apply_b0(&self, row: usize, r_hat: &[NttPoly]) -> Option<NttPoly> {
        if r_hat.len() != KAPPA {
            return None;
        }
        let acc = self.apply_b0_tail(row, &r_hat[N_TILDE..])?;
        Some(acc.add(&r_hat[row]))
    }

    /// `B_0' tail`, without the identity block.  This is the map applied to
    /// the rank-17 mask and response.
    pub fn apply_b0_tail(&self, row: usize, tail_hat: &[NttPoly]) -> Option<NttPoly> {
        if tail_hat.len() != RESPONSE_RANK {
            return None;
        }
        lr::inner_ntt(self.b0.get(row)?, tail_hat)
    }

    /// `<b_row, r> = r[n~ + row] + sum_j b[row][j] r[kappa - l~ + j]`.
    ///
    /// `None` if `row` is not a `b`-row or `r_hat` is not `kappa` elements.
    pub fn apply_b(&self, row: usize, r_hat: &[NttPoly]) -> Option<NttPoly> {
        if r_hat.len() != KAPPA {
            return None;
        }
        self.apply_b_tail(row, &r_hat[N_TILDE..])
    }

    /// `<b_row, (0_n~ || tail)>`, on the transmitted response rank.
    pub fn apply_b_tail(&self, row: usize, tail_hat: &[NttPoly]) -> Option<NttPoly> {
        if tail_hat.len() != RESPONSE_RANK {
            return None;
        }
        let acc = lr::inner_ntt(self.b.get(row)?, &tail_hat[RESPONSE_RANK - ELL_TILDE..])?;
        Some(acc.add(tail_hat.get(row)?))
    }
}

/// One coefficient-domain high part of `t_0`, after dropping `D = 17` bits.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct T0High([u64; super::ring::DTILDE]);

impl T0High {
    pub fn zero() -> Self {
        Self([0u64; super::ring::DTILDE])
    }

    /// Checked constructor for decoder output.
    pub fn new(values: &[u64]) -> Option<Self> {
        if values.len() != super::ring::DTILDE || values.iter().any(|&v| v >= T0_HIGH_MODULUS) {
            return None;
        }
        let mut out = [0u64; super::ring::DTILDE];
        out.copy_from_slice(values);
        Some(Self(out))
    }

    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }

    pub fn to_vec(&self) -> Vec<u64> {
        self.0.to_vec()
    }

    /// Coefficient-domain representative `2^D t_0,high`.
    pub fn expand(&self) -> CoeffPoly {
        let values = self.0.map(|v| v * T0_SCALE);
        // The current modulus leaves every representative below `q~`, but
        // reduction is the actual wire rule (and what the Python reference
        // applies).  Keeping it here makes a future valid parameter change
        // unable to turn a decoded high part into a verifier panic.
        CoeffPoly::from_reduced(&values).expect("T0High has a fixed length")
    }
}

fn compress_t0(t0_hat: &[NttPoly]) -> Option<Vec<T0High>> {
    if t0_hat.len() != N_TILDE {
        return None;
    }
    t0_hat
        .iter()
        .map(|hat| {
            let coeff = lr::intt(hat);
            let high: Vec<u64> = coeff
                .as_slice()
                .iter()
                .map(|&v| t0_power2round(v).map(|x| x.0))
                .collect::<Option<Vec<_>>>()?;
            T0High::new(&high)
        })
        .collect()
}

/// The public part of a commitment: compressed coefficient-domain `t_0`
/// and NTT-domain `t`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Commitment {
    /// High parts of `B_0 r`, `n~` elements.
    pub t0: Vec<T0High>,
    /// `<b_i, r> + m_i`, `N_ex` elements.
    pub t: Vec<NttPoly>,
}

/// The secret randomness, kept in both domains because both are used:
/// `z = y + c r` is a coefficient-domain product and every commitment
/// application is an NTT-domain one.
///
/// **Opaque.**  Both fields were public, and only one of them was ever
/// length-checked downstream: [`CommitmentKey::apply_b`] rejects an
/// `r_hat` that is not `kappa` elements, but `super::proof::prove` indexes
/// `r[i]` directly to form `z = y + c r`.  A caller assembling this by
/// hand — full-length `r_hat`, short `r` — reached that index.  The scheme
/// layer cannot produce one, but this is an exported primitive and the
/// panic was in the *prover*, on a value it was handed.
///
/// [`commit`] is now the only way to build one, and it builds both vectors
/// at `kappa` from the same draw.
pub struct CommitSecret {
    r: Vec<CoeffPoly>,
    r_hat: Vec<NttPoly>,
}

impl CommitSecret {
    /// The randomness in the coefficient domain, `kappa` elements.
    pub fn r(&self) -> &[CoeffPoly] {
        &self.r
    }

    /// The same, transformed — `kappa` elements.
    pub fn r_hat(&self) -> &[NttPoly] {
        &self.r_hat
    }

    /// Both vectors are `kappa` long.
    ///
    /// A postcondition of [`commit`] rather than something a caller can
    /// break, and checked at the one place that indexes `r` so the claim
    /// is tested rather than argued.
    pub fn is_well_formed(&self) -> bool {
        self.r.len() == KAPPA && self.r_hat.len() == KAPPA
    }

    /// The shape the constructor rules out, so the guard that rejects it
    /// can be exercised.  Tests only — there is no such value in a build.
    #[cfg(test)]
    pub(crate) fn ragged() -> Self {
        Self {
            r: vec![CoeffPoly::zero(); KAPPA - 1],
            r_hat: vec![NttPoly::zero(); KAPPA],
        }
    }
}

/// `LANES.Com`.
///
/// `message` is `N_ex` slot vectors of `l = 64` values each.  `None` on any
/// other count — the commitment key has exactly `N_ex` message rows, and a
/// short message would otherwise commit to a *different, shorter* relation
/// that still verifies against itself.
pub fn commit(
    ck: &CommitmentKey,
    message: &[Slots],
    xof: &mut Xof,
) -> Option<(Commitment, CommitSecret)> {
    if message.len() != N_EX {
        return None;
    }
    let r = sample_gaussian_vec(xof, SIGMA_R, KAPPA);
    let r_hat: Vec<NttPoly> = r.iter().map(lr::ntt).collect();

    let t0_full = (0..N_TILDE)
        .map(|i| ck.apply_b0(i, &r_hat))
        .collect::<Option<Vec<_>>>()?;
    let t0 = compress_t0(&t0_full)?;
    let t = (0..N_EX)
        .map(|i| {
            let mut base = ck.apply_b(i, &r_hat)?;
            lr::add_slots_inplace(&mut base, &message[i]);
            Some(base)
        })
        .collect::<Option<Vec<_>>>()?;

    Some((Commitment { t0, t }, CommitSecret { r, r_hat }))
}

/// Recompute the commitment from the randomness; used only by tests.
pub fn open_check(
    ck: &CommitmentKey,
    public: &Commitment,
    secret: &CommitSecret,
    message: &[Slots],
) -> bool {
    if message.len() != N_EX {
        return false;
    }
    let Some(t0_full) = (0..N_TILDE)
        .map(|i| ck.apply_b0(i, secret.r_hat()))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(t0) = compress_t0(&t0_full) else {
        return false;
    };
    let Some(t) = (0..N_EX)
        .map(|i| {
            let mut base = ck.apply_b(i, secret.r_hat())?;
            lr::add_slots_inplace(&mut base, &message[i]);
            Some(base)
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    t0 == public.t0 && t == public.t
}

#[cfg(test)]
mod tests {
    use super::super::params::T0_LOW_BOUND;
    use super::super::ring::{ntt_to_slots, DTILDE, LSPLIT, QTILDE};
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) % QTILDE
        }
    }

    fn message(next: &mut impl FnMut() -> u64) -> Vec<Slots> {
        (0..N_EX)
            .map(|_| Slots::new(&(0..LSPLIT).map(|_| next()).collect::<Vec<_>>()).unwrap())
            .collect()
    }

    fn test_xof(label: &[u8]) -> Xof {
        Xof::new(DS_EXACT, &[Part::Bytes(label)])
    }

    #[test]
    fn a_commitment_reopens_and_binds_its_message() {
        let ck = CommitmentKey::new(&[1u8; 32]);
        let mut next = lcg(4);
        let msg = message(&mut next);

        let (pub_, sec) = commit(&ck, &msg, &mut test_xof(b"commit-test")).unwrap();
        assert_eq!(pub_.t0.len(), N_TILDE);
        assert_eq!(pub_.t.len(), N_EX);
        assert!(open_check(&ck, &pub_, &sec, &msg), "does not reopen");

        let mut bad = msg.clone();
        let mut moved = bad[0].to_vec();
        moved[0] = (moved[0] + 1) % QTILDE;
        bad[0] = Slots::new(&moved).unwrap();
        assert!(
            !open_check(&ck, &pub_, &sec, &bad),
            "opened to a different message"
        );
    }

    /// The public `t_0` is exactly its high part: the omitted remainder is
    /// centred in `(-2^(D-1), 2^(D-1)]` and reconstructs the full value.
    #[test]
    fn t0_drops_exactly_d_bits_with_a_bounded_remainder() {
        let ck = CommitmentKey::new(&[1u8; 32]);
        let msg = vec![Slots::zero(); N_EX];
        let (public, secret) = commit(&ck, &msg, &mut test_xof(b"compressed-t0")).unwrap();

        for (row, high) in public.t0.iter().enumerate() {
            let full = lr::intt(&ck.apply_b0(row, secret.r_hat()).unwrap());
            let base = high.expand();
            for ((&value, &expanded), &high_part) in full
                .as_slice()
                .iter()
                .zip(base.as_slice())
                .zip(high.as_slice())
            {
                let (want_high, low) = t0_power2round(value).unwrap();
                assert_eq!(high_part, want_high);
                assert!(low > -(T0_LOW_BOUND as i64));
                assert!(low <= T0_LOW_BOUND as i64);
                assert_eq!(
                    (expanded as i128 + low as i128).rem_euclid(QTILDE as i128) as u64,
                    value
                );
            }
        }
    }

    /// Deterministic in the XOF, and a function of the key.
    #[test]
    fn the_commitment_is_determined_by_the_xof_and_the_key() {
        let ck = CommitmentKey::new(&[1u8; 32]);
        let mut next = lcg(4);
        let msg = message(&mut next);

        let (a, _) = commit(&ck, &msg, &mut test_xof(b"commit-test")).unwrap();
        let (b, _) = commit(&ck, &msg, &mut test_xof(b"commit-test")).unwrap();
        assert_eq!(a, b, "same XOF, same commitment");

        let other = CommitmentKey::new(&[2u8; 32]);
        let (c, _) = commit(&other, &msg, &mut test_xof(b"commit-test")).unwrap();
        assert_ne!(a, c, "a different key must commit differently");
    }

    /// The message lands in the slots, and only there.
    ///
    /// Committing the same randomness to zero and to `m` isolates the
    /// message term: the difference has to read back as `m` through the
    /// slot map, which is what makes `t` an *additive* commitment in the
    /// slots rather than merely a function of them.
    #[test]
    fn the_message_lands_in_the_slots() {
        let ck = CommitmentKey::new(&[1u8; 32]);
        let mut next = lcg(4);
        let msg = message(&mut next);
        let zero = vec![Slots::zero(); N_EX];

        let (p0, _) = commit(&ck, &zero, &mut test_xof(b"z")).unwrap();
        let (pm, _) = commit(&ck, &msg, &mut test_xof(b"z")).unwrap();
        for (i, (m, z)) in pm.t.iter().zip(p0.t.iter()).enumerate() {
            assert_eq!(ntt_to_slots(&m.sub(z)), msg[i], "element {i}");
        }
        // and t0 does not carry the message at all
        assert_eq!(p0.t0, pm.t0);
    }

    /// The structured inner products are what the unstored key says.
    ///
    /// `apply_b0` and `apply_b` are the only places the identity blocks
    /// exist, so this is where "the key is `[I | random]`" is checked
    /// rather than assumed.
    #[test]
    fn the_identity_blocks_are_the_unit_vectors_they_stand_for() {
        let ck = CommitmentKey::new(&[3u8; 32]);
        // The unit column of `b_row` is `n~ + row`, and the stored tail
        // starts at `kappa - l~`.  They do not overlap, which is what makes
        // "unit vector *plus* a random tail" a well-defined decomposition.
        const { assert!(N_TILDE + B_ROWS <= KAPPA - ELL_TILDE) };

        // `r = e_k` reads column `k` of the key straight out
        for k in 0..KAPPA {
            let mut r_hat = vec![NttPoly::zero(); KAPPA];
            let mut one = vec![0u64; DTILDE];
            one[0] = 1;
            r_hat[k] = lr::ntt(&CoeffPoly::new(&one).unwrap());

            for row in 0..N_TILDE {
                let want = if k == row {
                    r_hat[k].clone() // the identity entry, stored nowhere
                } else if k >= N_TILDE {
                    ck.b0[row][k - N_TILDE].clone()
                } else {
                    NttPoly::zero()
                };
                assert_eq!(ck.apply_b0(row, &r_hat).unwrap(), want, "B_0[{row}][{k}]");
            }
            for row in 0..B_ROWS {
                let want = if k == N_TILDE + row {
                    r_hat[k].clone()
                } else if k >= KAPPA - ELL_TILDE {
                    ck.b[row][k - (KAPPA - ELL_TILDE)].clone()
                } else {
                    NttPoly::zero()
                };
                assert_eq!(ck.apply_b(row, &r_hat).unwrap(), want, "b[{row}][{k}]");
            }
        }
    }

    /// Every shape the reference would have indexed into is `None`.
    #[test]
    fn a_wrong_shape_is_none_rather_than_a_panic() {
        let ck = CommitmentKey::new(&[1u8; 32]);
        let r_hat = vec![NttPoly::zero(); KAPPA];
        assert!(ck.apply_b0(0, &r_hat).is_some());
        assert!(ck.apply_b(0, &r_hat).is_some());
        assert!(ck.apply_b0(N_TILDE, &r_hat).is_none(), "row past B_0");
        assert!(ck.apply_b(B_ROWS, &r_hat).is_none(), "row past b");
        for len in [0usize, KAPPA - 1, KAPPA + 1] {
            let short = vec![NttPoly::zero(); len];
            assert!(ck.apply_b0(0, &short).is_none(), "len {len}");
            assert!(ck.apply_b(0, &short).is_none(), "len {len}");
        }
        let tail = vec![NttPoly::zero(); RESPONSE_RANK];
        assert!(ck.apply_b0_tail(0, &tail).is_some());
        assert!(ck.apply_b_tail(0, &tail).is_some());
        for len in [0usize, RESPONSE_RANK - 1, RESPONSE_RANK + 1] {
            let wrong = vec![NttPoly::zero(); len];
            assert!(ck.apply_b0_tail(0, &wrong).is_none(), "tail len {len}");
            assert!(ck.apply_b_tail(0, &wrong).is_none(), "tail len {len}");
        }

        let mut next = lcg(4);
        let msg = message(&mut next);
        assert!(commit(&ck, &msg[..N_EX - 1], &mut test_xof(b"x")).is_none());
        let mut long = msg.clone();
        long.push(Slots::zero());
        assert!(commit(&ck, &long, &mut test_xof(b"x")).is_none());

        let (pub_, sec) = commit(&ck, &msg, &mut test_xof(b"x")).unwrap();
        assert!(!open_check(&ck, &pub_, &sec, &msg[..N_EX - 1]));
    }

    /// The randomness `commit` produces is `kappa` in both domains, which
    /// is what makes indexing it downstream total.
    #[test]
    fn the_randomness_is_kappa_in_both_domains() {
        let ck = CommitmentKey::new(&[1u8; 32]);
        let mut next = lcg(4);
        let msg = message(&mut next);
        let (_, sec) = commit(&ck, &msg, &mut test_xof(b"shape")).unwrap();
        assert_eq!((sec.r().len(), sec.r_hat().len()), (KAPPA, KAPPA));
        assert!(sec.is_well_formed());
        // and every `r_hat` really is the transform of its `r`
        for (r, r_hat) in sec.r().iter().zip(sec.r_hat()) {
            assert_eq!(&lr::ntt(r), r_hat);
        }
        assert!(!CommitSecret::ragged().is_well_formed());
    }

    /// Each rank is checked against the role the structure gives it.
    #[test]
    fn the_two_ranks_are_the_roles_the_structure_gives_them() {
        let ck = CommitmentKey::new(&[5u8; 32]);

        // `n~` = rows of `t_0` = width of `B_0`'s identity block
        // Named `n~` here; `exact::rank_roles` calls this role `l~`.
        // Equal at this profile — see the assertion at the top of the
        // module, which is what keeps that from staying invisible.
        assert_eq!(ck.b0.len(), N_TILDE, "B_0 has one row per identity column");
        assert_eq!(
            ck.b0[0].len(),
            KAPPA - N_TILDE,
            "so its identity block is n~ wide"
        );
        let msg = vec![Slots::zero(); N_EX];
        let mut xof = test_xof(b"ranks");
        let (pub_, _) = commit(&ck, &msg, &mut xof).unwrap();
        assert_eq!(pub_.t0.len(), N_TILDE, "t_0 is n~ elements");

        // `l~` = width of the shared random tail
        assert_eq!(ck.b.len(), B_ROWS);
        for row in &ck.b {
            assert_eq!(row.len(), ELL_TILDE, "every b_i draws l~ columns");
        }

        // and the three roles partition the randomness exactly
        assert_eq!(
            KAPPA,
            N_TILDE + B_ROWS + ELL_TILDE,
            "identity block + one-per-element + shared tail = kappa"
        );

        // The Hint-MLWE instance the widths are searched against reads the
        // same way round: secret rank `n~ d~`, samples over the rows that
        // touch the tail.  The paper prints both — "embedded LWE
        // dimension 1024 and 3328 samples" — and they are what decides the
        // assignment, so they are asserted against the *paper's* figures
        // rather than against the ranks they were derived from.
        assert_eq!(N_TILDE * super::super::ring::DTILDE, 1024, "n_LWE");
        assert_eq!(
            (ELL_TILDE + N_EX + AUX) * super::super::ring::DTILDE,
            3328,
            "m_LWE"
        );
    }

    /// The three aux rows are distinct, and are where the protocol says.
    #[test]
    fn the_aux_rows_are_the_three_past_the_message() {
        assert_eq!((B_G, B_MP1, B_MP2), (N_EX, N_EX + 1, N_EX + 2));
        assert_eq!(B_ROWS, N_EX + AUX);
        assert_eq!(B_ROWS, KAPPA - N_TILDE - ELL_TILDE);
        assert_eq!(B0_COLS, KAPPA - N_TILDE);
    }
}
