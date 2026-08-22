//! The LANES exact backend — port of `river-py/lanes_*.py`.
//!
//! **Optional.**  Nothing in core RiVeR imports this; it exists so the
//! exact layer can be instantiated with a real zero-knowledge prover
//! instead of [`crate::exact::OpeningBackend`], which reveals its witness.
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
//! | [`backend`] | `Pi_ex` assembled — select it with `river::BackendKind::Lanes` |
//!
//! Select the backend and the whole scheme runs on it: `tests/vectors.rs`
//! re-derives the reference's two `lanes` cases byte for byte, which is
//! what pins the commitment key expansion, the Fiat–Shamir transcript and
//! the wire format against `river-py` rather than against this crate.
//!
//! That is **not** validation against a specification.  The paper fixes
//! `Pi_ex`'s parameters but does not restate the protocol, so this follows
//! `[ENS20]` directly and nothing checks that RiVeR intended the same
//! reading — the reference took the same reading, and agreeing with it is
//! agreement, not confirmation.  The evidence is behavioural.
//!
//! Its Gaussian widths are the paper's searched Hint-MLWE
//! candidate, which the revision itself gates on an estimator run that has
//! not happened.

pub mod backend;
pub mod commit;
pub mod mp;
pub mod params;
pub mod proof;
pub mod ring;
