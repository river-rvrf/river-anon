//! LANES as an exact backend for RiVeR's `Pi_ex` — port of
//! `river-py/lanes_backend.py`.
//!
//! The zero-knowledge alternative to [`crate::exact::OpeningBackend`],
//! which is complete and binding but transmits its witness.
//!
//! ## Mapping RiVeR's relation onto LANES
//!
//! `R^_ex` asks for `e_eval + B_e in [0, q_0-1]^d`, a digit reconstruction, and
//! the link `z_eval = x e_eval + y_eval`.  LANES proves *ternary slots* plus
//! a *public linear system*, so the relation is expressed as:
//!
//! ```text
//! slots    element 0    : y_eval     (32, not ternary)
//!          element 1    : e_eval     (32, not ternary)
//!          elements 2-5 : digits     (128, ternary)
//!
//! ternary  elements [2, 6)   — gives digits in {-1, 0, 1}
//!
//! linear   32 rows  e_eval[i] - sum_j g_j digit'_j[i] = 0
//!          32 rows  sum_b M[i][b] e_eval[b] + y_eval[i] = z_eval[i]
//! ```
//!
//! `M` is the negacyclic multiplication-by-`x` matrix, so the second block
//! is exactly the link equation written out as a linear map.
//!
//! Two consequences of using LANES's *native* encoding:
//!
//! * The digits are carried **centred**, in `{-1, 0, 1}`, because the cubic
//!   product proof certifies `m^3 = m`.  The paper describes digits in
//!   `{0, 1, 2}`; the two are the same encoding shifted by
//!   `sum_j g_j = 30`. Since `e_eval` is centred too, the shifts cancel and
//!   the reconstruction row's constant is zero.
//!   Any implementation of the `[ESLR23]` rounding relation has to centre
//!   for the same reason — the product proof cannot certify `{0,1,2}`
//!   directly.
//!
//!   The *direction* of that shift is where the paper contradicts itself;
//!   this follows the relation, `e_eval = sum_j g_j d_j`.
//!
//! * The link is proved **modulo `q~`**, because that is the only modulus
//!   LANES has.  `q~` is sized against the response
//!   (`q~ > 24 phi_rs B_rs`), so every accepted `z_eval` and every
//!   difference of two of them has a unique centred lift and the
//!   congruence pins the integer.
//!
//! The Python reference additionally provides `field_sizes` and `model_bits`
//! reporting helpers. They are measurement tools rather than protocol; this
//! crate checks the encoded lengths carried by `vectors.json` and reports
//! field sizes through the benchmark binary.

use super::commit::{commit, CommitSecret, Commitment, CommitmentKey, T0High};
use super::params::{N_EX, N_TILDE, RESPONSE_RANK, SIGMA_Y, T0_HIGH_MODULUS, Z_INF_BOUND};
use super::proof::{self, Challenges, LanesProof, LinearSystem, AN};
use super::ring::{NttPoly, Slots, DTILDE, LSPLIT, QTILDE};
use crate::codec::{
    pack_signed, width_for_bound, Coder, Field, FieldValue, Layout, Result as CodecResult,
};
use crate::exact::{
    decompose_poly, lanes_unavailable_reason, ExactParams, ExactWitness, RADIX_WEIGHTS,
};
use crate::params::RiVeRParams;
use crate::sample::{hash_bytes, Part, Xof, DS_EXACT};

/// Witness layout, in message-element indices.  The paper's order,
/// `(y_eval, e_eval, d_0, ..., d_3)`; see [`crate::exact::pack_witness`].
pub const IDX_Y: usize = 0;
pub const IDX_E: usize = 1;
pub const IDX_DIGITS: usize = 2;
/// First element the product proof covers.
pub const ALPHA_LO: usize = IDX_DIGITS;
/// One past the last — `= N_ex`.
pub const ALPHA_HI: usize = IDX_DIGITS + RADIX_WEIGHTS.len();

/// The uniform ring elements `sigma_ex` still carries.
///
/// `w`, `v` and `v'` are gone: each is a check target the verifier
/// recovers, which is what the paper's `(N_ex + alpha + 1) = 10`-element
/// uniform term already assumes.  See [`super::proof`]'s module docs.
const ELEMENTS: [&str; 4] = ["t_g", "t_mp1", "t_mp2", "h"];

/// `(A, u)` over `Z_q~` encoding reconstruction and the link equation.
///
/// `None` if the outer dimension does not fit the slot vector — `d = 32`
/// semantic coefficients into `l = 64` slots, so `d <= l` is the condition
/// and the remaining `N_ex (l - d) = 192` coordinates are padding — or if
/// the statement is the wrong shape.  The padding is not left free: the
/// rows after the link block pin every one of it to zero.
///
/// `x` and `z_eval` are **public** statement data, so the reductions here
/// are the one place this port uses `rem_euclid` rather than the ring's
/// masked add: `z_eval` arrives from a peer as an arbitrary `i64` and has
/// to be reduced rather than refused, exactly as the reference does.
pub fn build_linear_system(d: usize, x_centered: &[i64], z_eval: &[i64]) -> Option<LinearSystem> {
    if d > LSPLIT || x_centered.len() != d || z_eval.len() != d {
        return None;
    }
    let q = QTILDE as i128;
    // `i128`, and the negation inside it.  The wrapped term below is
    // `-x[k + d]`, and `-i64::MIN` is not an `i64` — a panic in debug and a
    // wrapped coefficient in release, on a `pub fn` whose `x` a caller can
    // hand in directly.  `RiVeR::verify` re-encodes `pi_oom.x` against its
    // challenge coder first and so never reaches it, but that is the
    // caller's guard rather than this function's.
    let red = |v: i128| v.rem_euclid(q) as u64;

    let mut a = Vec::with_capacity(2 * d);
    let mut u = Vec::with_capacity(2 * d);

    // reconstruction: e_eval[i] - sum_j g_j centred_digit_j[i] = 0
    for i in 0..d {
        let mut row = vec![0u64; AN];
        row[IDX_E * LSPLIT + i] = 1;
        for (j, &weight) in RADIX_WEIGHTS.iter().enumerate() {
            row[(IDX_DIGITS + j) * LSPLIT + i] = red(-(weight as i128));
        }
        a.push(row);
        u.push(0);
    }

    // link: (x · e_eval)[i] + y_eval[i] = z_eval[i], negacyclic in X^d + 1
    for i in 0..d {
        let mut row = vec![0u64; AN];
        for b in 0..d {
            let k = i as i64 - b as i64;
            let coeff = if k >= 0 {
                x_centered[k as usize] as i128
            } else {
                -(x_centered[(k + d as i64) as usize] as i128)
            };
            row[IDX_E * LSPLIT + b] = red(coeff);
        }
        row[IDX_Y * LSPLIT + i] = 1;
        a.push(row);
        u.push(red(z_eval[i] as i128));
    }

    // **zero padding.**  Each of the `N_ex` message blocks is `l = 64`
    // slots wide and carries `d = 32` semantic coefficients, so `N_ex(l-d)`
    // slots are unused — and *nothing above constrains them*.  Without
    // these rows a prover can commit arbitrary values there and still
    // satisfy every equation, because the reconstruction and link rows only
    // ever index `< d`.  One row per padding coordinate pins it to zero.
    //
    // This is not hypothetical: at `(d, l) = (32, 64)` it is 192 free
    // coordinates out of 384.  `river-py` had the same gap and the same
    // repair; `the_padding_is_constrained_to_zero` below is the
    // discriminating test.
    for element in 0..N_EX {
        for slot in d..LSPLIT {
            let mut row = vec![0u64; AN];
            row[element * LSPLIT + slot] = 1;
            a.push(row);
            u.push(0);
        }
    }

    LinearSystem::new(a, u)
}

/// `(W, z_eval, x)` — everything `Pi_ex.Ver` is told.
pub struct LanesStatement<'a> {
    pub w: &'a Commitment,
    /// `z_eval`, centred integers.
    pub z_eval: &'a [i64],
    /// `x`, centred integers.
    pub x: &'a [i64],
}

/// Prover state carried from `Com` to `Prove`.
pub struct LanesState {
    /// The committed slots, as canonical residues.
    message: Vec<Slots>,
    /// The same slots as centred integers; the product proof reads
    /// `[ALPHA_LO, ALPHA_HI)` of these.
    slots: Vec<Vec<i64>>,
    secret: CommitSecret,
}

impl LanesState {
    /// The centred slot vectors, for the tests that check the digits really
    /// are ternary and really do reconstruct `e_eval`.
    pub fn slots(&self) -> &[Vec<i64>] {
        &self.slots
    }
}

/// `Pi_ex` instantiated with the ported LANES prover.  Zero knowledge.
pub struct LanesBackend {
    ex: ExactParams,
    ck: CommitmentKey,
    /// Bound on `z_eval` in the statement hash — the verifier's own.
    ///
    /// Private for the reason every other derived field here is: it is
    /// fixed by the profile at construction and the statement hash is
    /// computed against it, so a caller who moved it would change which
    /// statements this backend binds to without changing the profile.
    bound_z: i64,
    w_layout: Layout,
    proof_layout: Layout,
    /// `"lanes"` or `"lanes-experimental"` — the name this *instance*
    /// reports, so an artifact recording it reconstructs the same thing.
    name: &'static str,
}

impl LanesBackend {
    pub const NAME: &'static str = "lanes";

    /// The name an *experimental* instance reports.
    ///
    /// Deliberately different from [`LanesBackend::NAME`].  A vector case
    /// or a benchmark row recording `"lanes-experimental"` reconstructs
    /// [`LanesBackend::experimental`] and not the gated production
    /// backend; recording `"lanes"` for both would make an experimental
    /// artifact reconstruct something that refuses to exist.
    pub const EXPERIMENTAL_NAME: &'static str = "lanes-experimental";

    /// **Gated.**  `Err` while [`crate::exact::lanes_unavailable_reason`]
    /// gives a reason — currently the security evidence, not the
    /// parameters, which are the paper's.
    ///
    /// [`LanesBackend::experimental`] is the way past it, under a
    /// different name.
    pub fn new(par: RiVeRParams, seed: &[u8]) -> Result<Self, String> {
        if let Some(reason) = lanes_unavailable_reason() {
            return Err(reason);
        }
        Self::build(par, seed, Self::NAME)
    }

    /// The same backend under the experimental name, ungated.
    ///
    /// It runs at the paper's own parameters; what it does not have is the
    /// security evidence the production name requires.  Use it for
    /// benchmarks and for coverage behind the gate — the alternative, no
    /// coverage at all, is how an unconstrained message-block padding
    /// survived to be found by inspection instead of by a test.
    pub fn experimental(par: RiVeRParams, seed: &[u8]) -> Result<Self, String> {
        Self::build(par, seed, Self::EXPERIMENTAL_NAME)
    }

    fn build(par: RiVeRParams, seed: &[u8], name: &'static str) -> Result<Self, String> {
        let ex = ExactParams::new(&par)?;
        let bound_z = par.zm_inf_bound_sq().floor_sqrt() as i64;

        // Full commitment/proof elements pack uniformly at
        // `ceil(log2 q~) = 26` bits. `t0` has its own high-part domain,
        // 10 bits wide after dropping `D = 17`; `c` and the recovery hint
        // are signed ternary fields, and `z` is Rice-coded.  The widths
        // are read off the constants rather than written out.
        let qt = Coder::uniform(QTILDE);
        let t0_high = Coder::uniform(T0_HIGH_MODULUS);
        let rice_z = Coder::rice(SIGMA_Y.0, SIGMA_Y.1, Z_INF_BOUND);
        // `c` is the transmitted challenge: ternary, so two bits a
        // coefficient.  It replaces `w`, `v` and `v'`, which the verifier
        // recovers — 256 bits against 33,408.
        let ternary = Coder::signed(1);

        let w_layout = Layout::new(vec![
            Field::rows("t0", t0_high, DTILDE, N_TILDE),
            Field::rows("t", qt, DTILDE, N_EX),
        ]);
        let mut fields = vec![
            Field::rows("t0", t0_high, DTILDE, N_TILDE),
            Field::rows("t", qt, DTILDE, N_EX),
        ];
        for name in ELEMENTS {
            fields.push(Field::flat(name, qt, DTILDE));
        }
        fields.push(Field::ring_rows("c", ternary, DTILDE, 1, QTILDE));
        fields.push(Field::rows("hint", ternary, DTILDE, N_TILDE));
        fields.push(Field::ring_rows("z", rice_z, DTILDE, RESPONSE_RANK, QTILDE));

        Ok(Self {
            ex,
            ck: CommitmentKey::new(seed),
            bound_z,
            w_layout,
            proof_layout: Layout::new(fields),
            name,
        })
    }

    /// The name this instance reports: `"lanes"` or
    /// `"lanes-experimental"`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The exact-layer parameters this backend was built for.
    pub fn ex(&self) -> &ExactParams {
        &self.ex
    }

    /// The `z_eval` bound the statement hash is computed against.
    pub fn bound_z(&self) -> i64 {
        self.bound_z
    }

    /// Canonical image of `(W, z_eval, x)`, bound into every FS challenge.
    fn statement_bytes(&self, st: &LanesStatement<'_>) -> Option<Vec<u8>> {
        let w = self.w_encode(st.w).ok()?;
        let x = pack_signed(st.x, 1, 127).ok()?;
        let z = pack_signed(st.z_eval, width_for_bound(self.bound_z).ok()?, self.bound_z).ok()?;
        Some(hash_bytes(
            32,
            &[DS_EXACT, b".lanes.stmt"].concat(),
            &[Part::Bytes(&w), Part::Bytes(&x), Part::Bytes(&z)],
        ))
    }

    fn challenges(&self, st: &LanesStatement<'_>) -> Option<Challenges> {
        Some(Challenges::new(&self.statement_bytes(st)?))
    }

    // -- Pi_ex interface ---------------------------------------------------

    /// `(W, st) <- Pi_ex.Com(w_ex)`; the statement is not known yet.
    ///
    /// `None` when `e_eval` is outside `[0, 60]` — a witness the relation
    /// does not admit — or when a coordinate of `y_eval` leaves
    /// `(-q~, q~)`.  The second cannot happen for a witness this crate
    /// derived: the mask is drawn at `sigma_rs` with tailcut 14 and
    /// `q~ > 24 phi_rs B_rs = 24 sigma_rs`, so `|y_eval| <= 14 sigma_rs`
    /// has room to spare.  `the_mask_always_fits_the_internal_modulus`
    /// checks that for every profile.
    ///
    /// `xof` supplies the prover's private randomness and is **continued**
    /// by [`LanesBackend::prove`]: the reference stores it in the state and
    /// draws `g` and `y` from it after the statement exists, so the two
    /// calls have to share one stream.
    pub fn com(&self, witness: &ExactWitness, xof: &mut Xof) -> Option<(Commitment, LanesState)> {
        let d = self.ex.d();
        // Exactly `d`, not "at most `l`": a short or long semantic input
        // would silently move coefficients into slots the linear system
        // constrains to zero, or leave real ones unconstrained.
        if witness.e_eval.len() != d || witness.y_eval.len() != d || d > LSPLIT {
            return None;
        }
        let canonical: Vec<i64> = witness
            .e_eval
            .iter()
            .map(|&c| c + self.ex.par().B_e() as i64)
            .collect();
        let digits = decompose_poly(&canonical)?; // in {0,1,2}

        // Every block is *padded*: `d = 32` semantic coefficients into an
        // `l = 64` slot vector, the rest zero and constrained so by
        // `build_linear_system`.  Assigning the coefficient vector wholesale
        // (`clone_from`) would leave a 32-entry block where the commitment
        // and the linear system both read 64.
        let mut slots = vec![vec![0i64; LSPLIT]; N_EX];
        slots[IDX_E][..d].copy_from_slice(&witness.e_eval);
        slots[IDX_Y][..d].copy_from_slice(&witness.y_eval);
        for (j, poly) in digits.iter().enumerate() {
            if poly.len() != d {
                return None;
            }
            // -> {-1, 0, 1}: the alphabet the product proof certifies
            for (i, &a) in poly.iter().enumerate() {
                slots[IDX_DIGITS + j][i] = a - 1;
            }
        }

        let message = slots
            .iter()
            .map(|s| Slots::from_centered(s))
            .collect::<Option<Vec<_>>>()?;
        let (public, secret) = commit(&self.ck, &message, xof)?;
        Some((
            public,
            LanesState {
                message,
                slots,
                secret,
            },
        ))
    }

    /// `Pi_ex.Prove`.  Continues `xof` from [`LanesBackend::com`].
    pub fn prove(
        &self,
        st: &LanesStatement<'_>,
        state: &LanesState,
        xof: &mut Xof,
    ) -> Option<LanesProof> {
        let ulp = build_linear_system(self.ex.d(), st.x, st.z_eval)?;
        let mut ch = self.challenges(st)?;
        proof::prove(
            &self.ck,
            st.w,
            &state.secret,
            &state.message,
            &state.slots,
            &ulp,
            ALPHA_LO,
            ALPHA_HI,
            xof,
            &mut ch,
        )
    }

    /// `Pi_ex.Ver`.  Total on `statement` and `proof`.
    pub fn verify(&self, st: &LanesStatement<'_>, sigma: &LanesProof) -> bool {
        let Some(ulp) = build_linear_system(self.ex.d(), st.x, st.z_eval) else {
            return false;
        };
        let Some(mut ch) = self.challenges(st) else {
            return false;
        };
        proof::verify(&self.ck, st.w, sigma, &ulp, ALPHA_LO, ALPHA_HI, &mut ch)
    }

    // -- encoding ----------------------------------------------------------

    pub fn w_encode(&self, w: &Commitment) -> CodecResult<Vec<u8>> {
        self.w_layout.encode(&[as_t0(&w.t0), as_ints(&w.t)])
    }

    pub fn w_decode(&self, data: &[u8]) -> CodecResult<Commitment> {
        let mut f = self.w_layout.decode(data)?.into_iter();
        Ok(Commitment {
            t0: as_t0_high(f.next().unwrap())?,
            t: as_ntt(f.next().unwrap())?,
        })
    }

    /// `W` is all uniform, so this is exact rather than a bound.
    pub fn w_bytes(&self) -> usize {
        self.w_layout.max_bytes()
    }

    /// `pi_ex = (W, sigma_ex)`.
    pub fn proof_encode(&self, w: &Commitment, sigma: &LanesProof) -> CodecResult<Vec<u8>> {
        let mut values = vec![as_t0(&w.t0), as_ints(&w.t)];
        for el in [&sigma.t_g, &sigma.t_mp1, &sigma.t_mp2, &sigma.h] {
            values.push(FieldValue::flat(
                el.as_slice().iter().map(|&c| c as i64).collect(),
            ));
        }
        values.push(FieldValue::Residues(vec![sigma.c.to_vec()]));
        values.push(FieldValue::Ints(sigma.hint.clone()));
        values.push(FieldValue::Residues(
            sigma.z.iter().map(|p| p.to_vec()).collect(),
        ));
        self.proof_layout.encode(&values)
    }

    pub fn proof_decode(&self, data: &[u8]) -> CodecResult<(Commitment, LanesProof)> {
        use crate::codec::CodecError;
        let mut f = self.proof_layout.decode(data)?.into_iter();
        let w_com = Commitment {
            t0: as_t0_high(f.next().unwrap())?,
            t: as_ntt(f.next().unwrap())?,
        };
        let mut single = || -> CodecResult<NttPoly> {
            as_ntt(f.next().unwrap())?
                .into_iter()
                .next()
                .ok_or(CodecError::LengthMismatch)
        };
        let t_g = single()?;
        let t_mp1 = single()?;
        let t_mp2 = single()?;
        let h = single()?;
        let residues = |v: FieldValue| -> CodecResult<Vec<super::ring::CoeffPoly>> {
            match v {
                FieldValue::Residues(r) => r
                    .into_iter()
                    .map(|row| super::ring::CoeffPoly::new(&row).ok_or(CodecError::LengthMismatch))
                    .collect(),
                FieldValue::Ints(_) => unreachable!("layout field is a ring field"),
            }
        };
        let c = residues(f.next().unwrap())?
            .into_iter()
            .next()
            .ok_or(CodecError::LengthMismatch)?;
        let hint = match f.next().unwrap() {
            FieldValue::Ints(rows) => rows,
            FieldValue::Residues(_) => unreachable!("hint is an integer field"),
        };
        let z = match f.next().unwrap() {
            FieldValue::Residues(r) => r
                .into_iter()
                .map(|row| super::ring::CoeffPoly::new(&row).ok_or(CodecError::LengthMismatch))
                .collect::<CodecResult<Vec<_>>>()?,
            FieldValue::Ints(_) => unreachable!("layout field is a ring field"),
        };
        Ok((
            w_com,
            LanesProof {
                t_g,
                t_mp1,
                t_mp2,
                h,
                c,
                hint,
                z,
            },
        ))
    }

    /// The layout `pi_ex` is framed against, for
    /// [`crate::codec::proof_unframe`].
    pub fn proof_layout(&self) -> &Layout {
        &self.proof_layout
    }

    /// Worst-case `|pi_ex|`.
    ///
    /// Rice-coding `z` makes the real length sample-dependent, so this is
    /// an upper bound; measure with [`LanesBackend::proof_encode`].
    pub fn proof_bytes(&self) -> usize {
        self.proof_layout.max_bytes()
    }
}

/// NTT-domain residues as the plain integers a non-ring `Uniform` field
/// wants — the same convention [`crate::exact::OpeningBackend`] uses for
/// `t_0` and `t_1`, and the one the reference's `Layout` applies.
fn as_ints(v: &[NttPoly]) -> FieldValue {
    FieldValue::Ints(
        v.iter()
            .map(|p| p.as_slice().iter().map(|&c| c as i64).collect())
            .collect(),
    )
}

fn as_t0(v: &[T0High]) -> FieldValue {
    FieldValue::Ints(
        v.iter()
            .map(|p| p.as_slice().iter().map(|&c| c as i64).collect())
            .collect(),
    )
}

fn as_t0_high(v: FieldValue) -> CodecResult<Vec<T0High>> {
    match v {
        FieldValue::Ints(rows) => rows
            .into_iter()
            .map(|row| {
                let values: Vec<u64> = row.into_iter().map(|c| c as u64).collect();
                T0High::new(&values).ok_or(crate::codec::CodecError::NonCanonical)
            })
            .collect(),
        FieldValue::Residues(_) => unreachable!("t0 is an integer field"),
    }
}

/// Inverse of [`as_ints`].
///
/// The `Uniform` coder has already established the width and the canonical
/// range, so [`NttPoly::new`] should not be able to fail here — and this
/// still returns an error rather than unwrapping, because the only caller
/// is [`LanesBackend::proof_decode`], which runs on a peer's bytes.  A
/// decoder that panics on input it was handed is a denial of service
/// whether or not the panic is reachable today; the layout is one edit away
fn as_ntt(v: FieldValue) -> CodecResult<Vec<NttPoly>> {
    match v {
        FieldValue::Ints(rows) => rows
            .into_iter()
            .map(|row| {
                let residues: Vec<u64> = row.into_iter().map(|c| c as u64).collect();
                NttPoly::new(&residues).ok_or(crate::codec::CodecError::NonCanonical)
            })
            .collect(),
        FieldValue::Residues(_) => unreachable!("layout field is an integer field"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{RIVER_N8, RIVER_TOY};
    use crate::ring::Ring;
    use crate::sample::{gaussian_int, rational_sigma, uniform_int, GAUSSIAN_TAILCUT};

    // Every test below needs a `LanesBackend`, and builds it through
    // `LanesBackend::experimental`.
    //
    // They used to *skip*: `LanesBackend::new` refused, so each returned
    // early and printed why.  the layer runs at the
    // paper's own parameters and is byte-exact against `river-py`, so
    // nothing is skipped.  Only the production name is gated, on security
    // evidence, and `the_production_name_is_still_gated` asserts that.
    //
    // A test that skips while the code demonstrably works is one gate away
    // from silence — which is how `lanes::ring`'s twiddle tree stayed
    // wrong for as long as it did.

    /// A witness the relation admits, plus the statement it belongs to.
    fn witness_for(par: &RiVeRParams, label: &[u8]) -> (ExactWitness, Vec<i64>, Vec<i64>) {
        let mut x = Xof::new(DS_EXACT, &[Part::Bytes(label)]);
        let e_eval: Vec<i64> = (0..par.d)
            .map(|_| uniform_int(&mut x, par.q0) as i64 - par.B_e() as i64)
            .collect();
        let y_eval: Vec<i64> = (0..par.d)
            .map(|_| {
                let (num, den) = rational_sigma(par.sigma_m());
                gaussian_int(&mut x, num, den, GAUSSIAN_TAILCUT)
            })
            .collect();
        let x_c: Vec<i64> = (0..par.d)
            .map(|i| {
                if i % 3 == 0 {
                    (uniform_int(&mut x, 2 * par.gamma + 1) as i64) - par.gamma as i64
                } else {
                    0
                }
            })
            .collect();
        let prod = Ring::mul_int(
            &x_c.iter().map(|&c| c as i128).collect::<Vec<_>>(),
            &e_eval.iter().map(|&c| c as i128).collect::<Vec<_>>(),
        );
        let z_eval: Vec<i64> = (0..par.d)
            .map(|i| (prod[i] + y_eval[i] as i128) as i64)
            .collect();
        (ExactWitness { e_eval, y_eval }, x_c, z_eval)
    }

    fn run(
        par: RiVeRParams,
        label: &[u8],
    ) -> (
        LanesBackend,
        Commitment,
        LanesProof,
        Vec<i64>,
        Vec<i64>,
        LanesState,
    ) {
        let backend = LanesBackend::experimental(par, &[0x21; 32]).expect("experimental builds");
        let (witness, x_c, z_c) = witness_for(&par, label);
        let mut xof = Xof::new(
            DS_EXACT,
            &[Part::Bytes(b"lanes-backend"), Part::Bytes(label)],
        );
        let (w, state) = backend.com(&witness, &mut xof).unwrap();
        let sigma = {
            let st = LanesStatement {
                w: &w,
                z_eval: &z_c,
                x: &x_c,
            };
            backend.prove(&st, &state, &mut xof).unwrap()
        };
        (backend, w, sigma, x_c, z_c, state)
    }

    #[test]
    fn an_honest_proof_verifies_on_every_profile() {
        for par in crate::params::PROFILES {
            let (backend, w, sigma, x_c, z_c, _) = run(par, par.name.as_bytes());
            let st = LanesStatement {
                w: &w,
                z_eval: &z_c,
                x: &x_c,
            };
            assert!(backend.verify(&st, &sigma), "{}", par.name);
        }
    }

    /// The digits are ternary and reconstruct `e_eval`.
    #[test]
    fn the_committed_digits_are_ternary_and_reconstruct() {
        let par = RIVER_TOY;
        let backend = LanesBackend::experimental(par, &[0x21; 32]).expect("experimental builds");
        let (witness, _, _) = witness_for(&par, b"digits");
        let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"digits")]);
        let (_, state) = backend.com(&witness, &mut xof).unwrap();

        for row in &state.slots()[ALPHA_LO..ALPHA_HI] {
            assert!(row.iter().all(|&a| (-1..=1).contains(&a)), "not ternary");
        }
        for i in 0..par.d {
            let acc: i64 = RADIX_WEIGHTS
                .iter()
                .enumerate()
                .map(|(j, &w)| w * state.slots()[ALPHA_LO + j][i])
                .sum();
            assert_eq!(acc, witness.e_eval[i], "coefficient {i}");
        }
    }

    /// The proof is bound to `(W, z_eval, x)`, one at a time.
    #[test]
    fn the_statement_is_bound() {
        let par = RIVER_TOY;
        let (backend, w, sigma, x_c, z_c, _) = run(par, b"bind");
        assert!(backend.verify(
            &LanesStatement {
                w: &w,
                z_eval: &z_c,
                x: &x_c
            },
            &sigma
        ));

        let mut bad_z = z_c.clone();
        bad_z[0] += 1;
        assert!(
            !backend.verify(
                &LanesStatement {
                    w: &w,
                    z_eval: &bad_z,
                    x: &x_c
                },
                &sigma
            ),
            "wrong z_eval accepted"
        );

        let mut bad_x = x_c.clone();
        let idx = bad_x.iter().position(|&v| v != 0).unwrap();
        bad_x[idx] = -bad_x[idx];
        assert!(
            !backend.verify(
                &LanesStatement {
                    w: &w,
                    z_eval: &z_c,
                    x: &bad_x
                },
                &sigma
            ),
            "wrong challenge accepted"
        );

        let mut bad_w = w.clone();
        let mut moved = bad_w.t0[0].to_vec();
        moved[0] = (moved[0] + 1) % T0_HIGH_MODULUS;
        bad_w.t0[0] = T0High::new(&moved).unwrap();
        assert!(
            !backend.verify(
                &LanesStatement {
                    w: &bad_w,
                    z_eval: &z_c,
                    x: &x_c
                },
                &sigma
            ),
            "wrong commitment accepted"
        );
    }

    /// Zero knowledge in the shape that can be tested: the transmitted
    /// proof round trips, still verifies, and carries no witness field.
    #[test]
    fn the_proof_round_trips_and_carries_no_witness() {
        for par in [RIVER_TOY, RIVER_N8] {
            let (backend, w, sigma, x_c, z_c, _) = run(par, b"bytes");
            let blob = backend.proof_encode(&w, &sigma).unwrap();
            assert!(blob.len() <= backend.proof_bytes(), "{}", par.name);

            let (w2, sigma2) = backend.proof_decode(&blob).unwrap();
            assert_eq!(w2, w, "{}", par.name);
            assert_eq!(sigma2, sigma, "{}", par.name);
            assert!(backend.verify(
                &LanesStatement {
                    w: &w2,
                    z_eval: &z_c,
                    x: &x_c
                },
                &sigma2
            ));

            assert_eq!(backend.w_encode(&w).unwrap().len(), backend.w_bytes());
            assert_eq!(backend.w_decode(&backend.w_encode(&w).unwrap()).unwrap(), w);
        }
    }

    /// `y_eval` always fits `(-q~, q~)`, so `com` never refuses an honest
    /// witness.
    ///
    /// The mask is drawn at `sigma_m` with tailcut `GAUSSIAN_TAILCUT`, and
    /// the paper's `q~ > 24 phi_m eta_m` is `q~ > 24 sigma_m`. The
    /// margin is therefore `24/14`, and it is a *consequence* of the
    /// modulus condition rather than an accident of the profiles — but the
    /// profiles are what ship, so they are what is checked.
    #[test]
    fn the_mask_always_fits_the_internal_modulus() {
        for par in crate::params::PROFILES {
            let reach = GAUSSIAN_TAILCUT as f64 * par.sigma_m();
            assert!(
                reach < QTILDE as f64,
                "{}: 14 sigma_m = {reach} exceeds q~",
                par.name
            );
            // and it follows from the condition `check` enforces
            let ex = ExactParams::new(&par).expect("shipped profile");
            assert!(ex.q_tilde_need() >= 24.0 * par.sigma_m() - 1.0);
        }
    }

    /// The linear system says what the module docs say it says.
    #[test]
    fn the_linear_system_is_reconstruction_then_link() {
        let d = 32usize;
        let x_c: Vec<i64> = (0..d).map(|i| if i == 1 { 5 } else { 0 }).collect();
        let z_eval: Vec<i64> = (0..d).map(|i| i as i64 - 3).collect();
        let ulp = build_linear_system(d, &x_c, &z_eval).unwrap();
        // `2d` semantic rows, then one per padding coordinate.
        assert_eq!(ulp.rows(), 2 * d + N_EX * (LSPLIT - d));

        // reconstruction row `i`: +1 on e_eval[i], -g_j on digit j
        let row = ulp.row(0).unwrap();
        assert_eq!(row[IDX_E * LSPLIT], 1);
        for (j, &weight) in RADIX_WEIGHTS.iter().enumerate() {
            assert_eq!(row[(IDX_DIGITS + j) * LSPLIT], QTILDE - weight as u64);
        }
        assert_eq!(ulp.u_at(0).unwrap(), 0);

        // link row `i`: multiplication by `x` is negacyclic, so the `b`
        // whose index wraps picks up a sign
        let row = ulp.row(d).unwrap(); // i = 0
        assert_eq!(row[IDX_Y * LSPLIT], 1);
        assert_eq!(row[IDX_E * LSPLIT + (d - 1)], QTILDE - 5, "wrapped term");
        assert_eq!(
            ulp.u_at(d).unwrap(),
            (z_eval[0]).rem_euclid(QTILDE as i64) as u64
        );

        // total on a wrong shape
        assert!(build_linear_system(d, &x_c[..d - 1], &z_eval).is_none());
        assert!(build_linear_system(d, &x_c, &z_eval[..d - 1]).is_none());
        assert!(build_linear_system(LSPLIT + 1, &x_c, &z_eval).is_none());

        // and total on the values, not only the shape: the wrapped link
        // term is `-x[k + d]`, and `-i64::MIN` is not an `i64`
        let extreme = vec![i64::MIN; d];
        let ulp = build_linear_system(d, &extreme, &extreme).expect("must reduce, not panic");
        let want = (i64::MIN as i128).rem_euclid(QTILDE as i128) as u64;
        let neg = (-(i64::MIN as i128)).rem_euclid(QTILDE as i128) as u64;
        assert_eq!(ulp.u_at(d).unwrap(), want, "z_eval reduces");
        let link = ulp.row(d).unwrap();
        assert_eq!(link[IDX_E * LSPLIT], want, "the unwrapped term");
        assert_eq!(link[IDX_E * LSPLIT + 1], neg, "the wrapped term");
        for bound in [i64::MAX, i64::MIN + 1] {
            assert!(build_linear_system(d, &vec![bound; d], &vec![bound; d]).is_some());
        }
    }

    /// `Ver` is total on a malformed proof.
    #[test]
    fn verify_is_total_on_peer_input() {
        let par = RIVER_TOY;
        let (backend, w, sigma, x_c, z_c, _) = run(par, b"total");
        let st = |z: &'static [i64]| -> bool {
            backend.verify(
                &LanesStatement {
                    w: &w,
                    z_eval: z,
                    x: &x_c,
                },
                &sigma,
            )
        };
        assert!(!st(&[]), "empty z_eval");

        let mut short = sigma.clone();
        short.z.pop();
        assert!(!backend.verify(
            &LanesStatement {
                w: &w,
                z_eval: &z_c,
                x: &x_c
            },
            &short
        ));

        let mut extreme = sigma.clone();
        extreme.z[0] = super::super::ring::CoeffPoly::new(&vec![QTILDE / 2; DTILDE]).unwrap();
        assert!(!backend.verify(
            &LanesStatement {
                w: &w,
                z_eval: &z_c,
                x: &x_c
            },
            &extreme
        ));

        // arbitrary bytes never decode to a proof
        for len in [0usize, 1, 64, 4096] {
            assert!(
                backend.proof_decode(&vec![0xABu8; len]).is_err(),
                "len {len}"
            );
        }
    }
    /// **The padding is constrained to zero**, and the test discriminates.
    ///
    /// Each of the six message blocks is `l = 64` slots wide and carries
    /// `d = 32` semantic coefficients, so 192 of the 384 slots are unused.
    /// Nothing in the reconstruction or link rows touches them — those only
    /// ever index `< d` — so without the padding rows a prover can commit
    /// arbitrary values there and still satisfy every equation.
    ///
    /// Stated in the direction that can fail: with the rows present, every
    /// padding coordinate appears in the system; without them, `192`
    /// columns are entirely zero, which is the shape of the hole.
    #[test]
    fn the_padding_is_constrained_to_zero() {
        let d = crate::exact::ExactParams::new(&crate::params::RIVER_TOY)
            .unwrap()
            .d();
        assert!(d < LSPLIT, "there is padding to constrain at all");

        let x: Vec<i64> = (0..d).map(|i| (i as i64 % 3) - 1).collect();
        let z: Vec<i64> = (0..d).map(|i| (i as i64 * 11) % 97).collect();
        let ulp = build_linear_system(d, &x, &z).expect("system");

        // Every column is touched by some row.  Restricted to the
        // reconstruction and link rows alone — the first `2d` — exactly the
        // `N_ex (l - d)` padding columns would be untouched, which is the
        // gap these rows close.
        let n_rows = ulp.rows();
        let untouched = |from: usize, to: usize| -> usize {
            (0..AN)
                .filter(|&c| (from..to).all(|k| ulp.row(k).unwrap()[c] == 0))
                .count()
        };
        assert_eq!(untouched(0, n_rows), 0, "a column is unconstrained");
        assert_eq!(
            untouched(0, 2 * d),
            N_EX * (LSPLIT - d),
            "the padding rows are what covers the rest"
        );

        // ...and each padding row is a single 1 at a padding coordinate,
        // with target zero.
        for k in (2 * d)..n_rows {
            let j = k - 2 * d;
            let element = j / (LSPLIT - d);
            let slot = d + j % (LSPLIT - d);
            let mut want = vec![0u64; AN];
            want[element * LSPLIT + slot] = 1;
            assert_eq!(ulp.row(k).unwrap(), want.as_slice(), "padding row {j}");
            assert_eq!(ulp.u_at(k), Some(0));
        }
    }

    /// A witness that is not exactly `d` coefficients is refused, rather
    /// than padded or truncated into slots the system constrains to zero.
    #[test]
    fn com_requires_exactly_d_coefficients() {
        let b = LanesBackend::experimental(crate::params::RIVER_TOY, &[4u8; 32]).unwrap();
        let d = b.ex().d();
        let ok: Vec<i64> = (0..d).map(|i| (i as i64 % 61) - 30).collect();
        for (e, y) in [
            (ok[..d - 1].to_vec(), ok.clone()),
            (ok.clone(), ok[..d - 1].to_vec()),
            ([ok.clone(), vec![0]].concat(), ok.clone()),
        ] {
            let w = ExactWitness {
                e_eval: e,
                y_eval: y,
            };
            let mut xof = Xof::new(b"pad", &[]);
            assert!(b.com(&w, &mut xof).is_none());
        }
    }

    /// The production `lanes` name is gated; everything above runs under
    /// the experimental one.  Asserted so the two cannot drift.
    #[test]
    fn the_production_name_is_still_gated() {
        assert!(LanesBackend::new(crate::params::RIVER_TOY, &[0x21; 32]).is_err());
        assert!(LanesBackend::experimental(crate::params::RIVER_TOY, &[0x21; 32]).is_ok());
        assert_eq!(
            crate::exact::lanes_gate_cause(),
            Some("production-alias-reserved")
        );
    }
}
