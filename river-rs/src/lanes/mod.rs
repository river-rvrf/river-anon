//! The LANES exact backend — port of `river-py/lanes_*.py`.
//!
//! **Optional.**  Nothing in core RiVeR imports this; it exists so the
//! exact layer can be instantiated with the candidate LANES prover instead
//! of [`crate::exact::OpeningBackend`], which reveals its witness.
//!
//! ## The six modules
//!
//! | module | what |
//! |---|---|
//! | [`ring`] | the incomplete NTT over `R_q~`, and the three shapes |
//! | [`params`] | dimensions, Gaussian widths, response bounds, samplers |
//! | [`commit`] | the `[BDLOP18]` commitment at `[ENS20]` Figure 3's shape |
//! | [`mp`] | the `[ALS20]` cubic product proof: committed slots are ternary |
//! | [`proof`] | `Gen`/`Prove`/`Ver`, with the `gamma`-compressed linear part |
//! | [`backend`] | `Pi_ex` assembled — select the candidate with `river::BackendKind::LanesExperimental` |
//!
//! Select the backend and the whole scheme runs on it: `tests/vectors.rs`
//! re-derives the reference's two `lanes-experimental` cases byte for byte, which is
//! what pins the commitment key expansion, the Fiat–Shamir transcript and
//! the wire format against `river-py` rather than against this crate.
//!
//! The paper treats LANES as a black-box exact layer. This implementation
//! follows `[ENS20]` and fixes a concrete compression/recovery and wire-format
//! composition. Interoperability and algebraic correctness are tested; this
//! artifact does not supply a reduction for that exact composition, so the
//! production alias remains reserved.

pub mod backend;
pub mod commit;
pub mod mp;
pub mod params;
pub mod proof;
pub mod ring;
